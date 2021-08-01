import torch
import torch.nn as nn
from typing import Dict
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.nn.util import masked_max, masked_softmax
from transformers import PreTrainedModel

from .config import LIF3WayAPConfig
from .modules.layers import SeqAttnMat, GatedEncoding, GatedMultifactorSelfAttnEnc, CharCNNEmbedding, WordEmbedding
from sgnlp.models.lif_3way_ap.utils import sequential_weighted_avg


class LIF3WayAPPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LIF3WayAPConfig
    base_model_prefix = "l2af"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class LIF3WayAPModel(LIF3WayAPPreTrainedModel):
    def __init__(self, config: LIF3WayAPConfig):
        super().__init__(config)

        self.config = config

        self.char_embedding = CharCNNEmbedding(**config.char_embedding_args)
        self.word_embedding = WordEmbedding(**config.word_embedding_args)

        self.p_seq_enc = nn.LSTM(**config.p_seq_enc_args)
        self.q_seq_enc = nn.LSTM(**config.q_seq_enc_args)
        self.c_seq_enc = nn.LSTM(**config.c_seq_enc_args)

        self.cartesian_attn_mat = SeqAttnMat(**config.cartesian_attn_mat_args)
        self.pq_attn_mat = SeqAttnMat(**config.pq_attn_mat_args)
        self.pc_attn_mat = SeqAttnMat(**config.pc_attn_mat_args)
        self.qc_attn_mat = SeqAttnMat(**config.cq_attn_mat_args)

        self.gate_qdep_penc = GatedEncoding(**config.gate_qdep_penc_args)
        self.qdep_penc_rnn = nn.LSTM(**config.qdep_penc_rnn_args)
        self.mfa_enc = GatedMultifactorSelfAttnEnc(**config.mfa_enc_args)
        self.mfa_rnn = nn.LSTM(**config.mfa_rnn_args)

        self.variational_dropout = InputVariationalDropout(config.dropout)
        self.dropout = torch.nn.Dropout(config.dropout)

        self.is_qdep_penc = config.is_qdep_penc
        self.is_mfa_enc = config.is_mfa_enc
        self.with_knowledge = config.with_knowledge
        self.is_qc_ap = config.is_qc_ap
        self.shared_rnn = config.shared_rnn
        if not self.with_knowledge and not self.is_qc_ap:
            raise AssertionError

        self.loss = torch.nn.BCELoss()

        self.init_weights()

    def embedding(self, word_tokenized_vector, char_tokenized_vector):
        word_embedding = self.word_embedding(word_tokenized_vector)
        char_embedding = self.char_embedding(char_tokenized_vector)

        # Concatenate and return embedding
        return torch.cat([word_embedding, char_embedding], dim=2)

    def forward(self,
                word_context: torch.LongTensor,
                word_qa: torch.LongTensor,
                word_candidate: torch.LongTensor,
                char_context: torch.LongTensor,
                char_qa: torch.LongTensor,
                char_candidate: torch.LongTensor,
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            word_context (:obj:`torch.LongTensor` of shape :obj:`(batch_size, max_num_words)`):
                Word IDs of context.
            word_qa (:obj:`torch.LongTensor` of shape :obj:`(batch_size, max_num_words)`):
                Word IDs of questions and answers.
            word_candidate (:obj:`torch.LongTensor` of shape :obj:`(batch_size, max_num_words)`):
                Word IDs of candidate question.
            char_context (:obj:`torch.LongTensor` of shape :obj:`(batch_size, max_num_words, max_num_chars)`):
                Character IDs of context.
            char_qa (:obj:`torch.LongTensor` of shape :obj:`(batch_size, max_num_words, max_num_chars)`):
                Character IDs of questions and answers.
            char_candidate (:obj:`torch.LongTensor` of shape :obj:`(batch_size, max_num_words, max_num_chars)`):
                Character IDs of candidate question.
            label (:obj:`torch.IntTensor` of shape :obj:`(batch_size,)`):
                Label for candidate question being a follow-up question.
        """

        if self.with_knowledge:
            embedded_context = self.embedding(word_context, char_context)  # B * T * d
        embedded_all_qa = self.embedding(word_qa, char_qa)  # B * U * d
        embedded_choice = self.embedding(word_candidate, char_candidate)  # B * V * d

        if self.with_knowledge:
            embedded_context = self.variational_dropout(embedded_context)  # B * T * d
        embedded_all_qa = self.variational_dropout(embedded_all_qa)
        embedded_choice = self.variational_dropout(embedded_choice)  # B * V * d

        q_mask = word_qa != 0  # B * U
        c_mask = word_candidate != 0  # B * V

        # Encoding
        if self.with_knowledge:
            # B * T * H
            p_mask = word_context != 0  # B * T
            encoded_context, _ = self.p_seq_enc(embedded_context)
            encoded_context = self.variational_dropout(encoded_context)

        # B * U * H
        if self.shared_rnn:
            encoded_qa, _ = self.p_seq_enc(embedded_all_qa)
        else:
            encoded_qa, _ = self.q_seq_enc(embedded_all_qa)
        encoded_qa = self.variational_dropout(encoded_qa)

        if self.with_knowledge and self.is_qdep_penc:
            # similarity matrix
            _, normalized_attn_mat = self.cartesian_attn_mat(encoded_context,
                                                             encoded_qa,
                                                             q_mask)  # B * T * U
            # question dependent context encoding
            q_aware_context_rep = sequential_weighted_avg(encoded_qa,
                                                          normalized_attn_mat)  # B * T * H

            q_dep_context_enc_rnn_input = torch.cat([encoded_context,
                                                     q_aware_context_rep], 2)  # B * T * 2H

            # gated question dependent context encoding
            gated_qaware_context_rep = self.gate_qdep_penc(q_dep_context_enc_rnn_input)  # B * T * 2H
            encoded_qdep_penc, _ = self.qdep_penc_rnn(gated_qaware_context_rep)  # B * T * H
            encoded_qdep_penc = self.dropout(encoded_qdep_penc)

        # multi factor attentive encoding
        if self.with_knowledge and self.is_mfa_enc:
            if self.is_qdep_penc:
                mfa_enc = self.mfa_enc(encoded_qdep_penc,
                                       p_mask)  # B * T * 2H
            else:
                mfa_enc = self.mfa_enc(encoded_context,
                                       p_mask)  # B * T * 2H
            encoded_context, _ = self.mfa_rnn(mfa_enc)  # B * T * H

        # B * V * H
        if self.shared_rnn:
            encoded_choice, _ = self.p_seq_enc(embedded_choice)
        else:
            encoded_choice, _ = self.c_seq_enc(embedded_choice)
        encoded_choice = self.variational_dropout(encoded_choice)  # B * V * H

        if self.with_knowledge:
            attn_pq, _ = self.pq_attn_mat(encoded_context, encoded_qa, q_mask)  # B * T * U
            combined_pqa_mask = p_mask.unsqueeze(-1) * \
                                q_mask.unsqueeze(1)  # B * T * U
            max_attn_pqa = masked_max(attn_pq, combined_pqa_mask, dim=1)  # B * U
            norm_attn_pqa = masked_softmax(max_attn_pqa, q_mask, dim=-1)  # B * U
            agg_prev_qa = norm_attn_pqa.unsqueeze(1).bmm(encoded_qa).squeeze(1)  # B * H

            attn_pc, _ = self.pc_attn_mat(encoded_context,
                                          encoded_choice, c_mask)  # B * T * V
            combined_pc_mask = p_mask.unsqueeze(-1) * \
                               c_mask.unsqueeze(1)  # B * T * V
            max_attn_pc = masked_max(attn_pc, combined_pc_mask, dim=1)  # B * V
            norm_attn_pc = masked_softmax(max_attn_pc, c_mask, dim=-1)  # B * V
            agg_c = norm_attn_pc.unsqueeze(1).bmm(encoded_choice)  # B * 1 * H

            choice_scores_wk = agg_c.bmm(agg_prev_qa.unsqueeze(-1)).squeeze(-1)  # B * 1

        if self.is_qc_ap:
            attn_qc, _ = self.qc_attn_mat(encoded_qa, encoded_choice, c_mask)  # B * U * V
            combined_qac_mask = q_mask.unsqueeze(-1) * c_mask.unsqueeze(1)  # B * U * V

            max_attn_c = masked_max(attn_qc, combined_qac_mask, dim=1)  # B * V
            max_attn_q = masked_max(attn_qc, combined_qac_mask, dim=2)  # B * U
            norm_attn_c = masked_softmax(max_attn_c, c_mask, dim=-1)  # B * V
            norm_attn_q = masked_softmax(max_attn_q, q_mask, dim=-1)  # B * U
            agg_c_qa = norm_attn_c.unsqueeze(1).bmm(encoded_choice).squeeze(1)  # B * H
            agg_qa_c = norm_attn_q.unsqueeze(1).bmm(encoded_qa).squeeze(1)  # B * H

            choice_scores_nk = agg_c_qa.unsqueeze(1).bmm(agg_qa_c.unsqueeze(
                -1)).squeeze(-1)  # B * 1

        if self.with_knowledge and self.is_qc_ap:
            choice_score = choice_scores_wk + choice_scores_nk
        elif self.is_qc_ap:
            choice_score = choice_scores_nk
        elif self.with_knowledge:
            choice_score = choice_scores_wk
        else:
            raise NotImplementedError

        output = torch.sigmoid(choice_score).squeeze(-1)  # B

        output_dict = {"label_logits": choice_score.squeeze(-1), "label_probs": output}

        if label is not None:
            label = label.long().view(-1)
            loss = self.loss(output, label.float())
            output_dict["loss"] = loss

        return output_dict
