import logging
from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import masked_max, masked_softmax
from allennlp.training.metrics import Auc, F1Measure

from .layers import SeqAttnMat, GatedEncoding, GatedMultifactorSelfAttnEnc
from .util import sequential_weighted_avg

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("lif_3way_ap_model")
class Lif3WayApAllenNlpModel(Model):
    """
    3-way Attentive Pooling Network
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        pseqlevelenc: Seq2SeqEncoder,
        qaseqlevelenc: Seq2SeqEncoder,
        choicelevelenc: Seq2SeqEncoder,
        cartesian_attn: SeqAttnMat,
        pcattnmat: SeqAttnMat,
        gate_qdep_penc: GatedEncoding,
        qdep_penc_rnn: Seq2SeqEncoder,
        mfa_enc: GatedMultifactorSelfAttnEnc,
        mfa_rnn: Seq2SeqEncoder,
        pqaattnmat: SeqAttnMat,
        cqaattnmat: SeqAttnMat,
        initializer: InitializerApplicator,
        dropout: float = 0.3,
        is_qdep_penc: bool = True,
        is_mfa_enc: bool = True,
        with_knowledge: bool = True,
        is_qac_ap: bool = True,
        shared_rnn: bool = True,
    ) -> None:
        super().__init__(vocab)
        self._text_field_embedder = text_field_embedder
        self._pseqlevel_enc = pseqlevelenc
        self._qaseqlevel_enc = qaseqlevelenc
        self._cseqlevel_enc = choicelevelenc

        self._cart_attn = cartesian_attn
        self._pqaattnmat = pqaattnmat
        self._pcattnmat = pcattnmat
        self._cqaattnmat = cqaattnmat

        self._gate_qdep_penc = gate_qdep_penc
        self._qdep_penc_rnn = qdep_penc_rnn
        self._multifactor_attn = mfa_enc
        self._mfarnn = mfa_rnn

        self._with_knowledge = with_knowledge
        self._qac_ap = is_qac_ap
        if not self._with_knowledge:
            if not self._qac_ap:
                raise AssertionError
        self._is_qdep_penc = is_qdep_penc
        self._is_mfa_enc = is_mfa_enc
        self._shared_rnn = shared_rnn

        self._variational_dropout = InputVariationalDropout(dropout)

        self._num_labels = vocab.get_vocab_size(namespace="labels")
        self._auc = Auc()
        self._f1 = F1Measure(positive_label=1)
        self._loss = torch.nn.BCELoss()
        initializer(self)

    def forward(
        self,  # type: ignore
        passage: Dict[str, torch.LongTensor],
        all_qa: Dict[str, torch.LongTensor],
        candidate: Dict[str, torch.LongTensor],
        combined_source: Dict[str, torch.LongTensor],
        label: torch.IntTensor = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        if self._with_knowledge:
            embedded_passage = self._text_field_embedder(passage)  # B * T * d
            passage_len = embedded_passage.size(1)
        embedded_all_qa = self._text_field_embedder(all_qa)  # B * U * d
        embedded_choice = self._text_field_embedder(candidate)  # B * V * d

        if self._with_knowledge:
            embedded_passage = self._variational_dropout(embedded_passage)  # B * T * d
        embedded_all_qa = self._variational_dropout(embedded_all_qa)
        embedded_choice = self._variational_dropout(embedded_choice)  # B * V * d

        all_qa_mask = util.get_text_field_mask(all_qa)  # B * U
        choice_mask = util.get_text_field_mask(candidate)  # B * V

        # Encoding
        if self._with_knowledge:
            # B * T * H
            passage_mask = util.get_text_field_mask(passage)  # B * T
            encoded_passage = self._variational_dropout(
                self._pseqlevel_enc(embedded_passage, passage_mask)
            )
        # B * U * H
        if self._shared_rnn:
            encoded_allqa = self._variational_dropout(
                self._pseqlevel_enc(embedded_all_qa, all_qa_mask)
            )
        else:
            encoded_allqa = self._variational_dropout(
                self._qaseqlevel_enc(embedded_all_qa, all_qa_mask)
            )

        if self._with_knowledge and self._is_qdep_penc:
            # similarity matrix
            _, normalized_attn_mat = self._cart_attn(
                encoded_passage, encoded_allqa, all_qa_mask
            )  # B * T * U
            # question dependent passage encoding
            q_aware_passage_rep = sequential_weighted_avg(
                encoded_allqa, normalized_attn_mat
            )  # B * T * H

            q_dep_passage_enc_rnn_input = torch.cat(
                [encoded_passage, q_aware_passage_rep], 2
            )  # B * T * 2H

            # gated question dependent passage encoding
            gated_qaware_passage_rep = self._gate_qdep_penc(
                q_dep_passage_enc_rnn_input
            )  # B * T * 2H
            encoded_qdep_penc = self._qdep_penc_rnn(
                gated_qaware_passage_rep, passage_mask
            )  # B * T * H

        # multi factor attentive encoding
        if self._with_knowledge and self._is_mfa_enc:
            if self._is_qdep_penc:
                mfa_enc = self._multifactor_attn(
                    encoded_qdep_penc, passage_mask
                )  # B * T * 2H
            else:
                mfa_enc = self._multifactor_attn(
                    encoded_passage, passage_mask
                )  # B * T * 2H
            encoded_passage = self._mfarnn(mfa_enc, passage_mask)  # B * T * H

        # B * V * H
        if self._shared_rnn:
            encoded_choice = self._variational_dropout(
                self._pseqlevel_enc(embedded_choice, choice_mask)
            )  # B * V * H
        else:
            encoded_choice = self._variational_dropout(
                self._cseqlevel_enc(embedded_choice, choice_mask)
            )  # B * V * H

        if self._with_knowledge:
            attn_pq, _ = self._pqaattnmat(
                encoded_passage, encoded_allqa, all_qa_mask
            )  # B * T * U
            combined_pqa_mask = passage_mask.unsqueeze(-1) * all_qa_mask.unsqueeze(
                1
            )  # B * T * U
            max_attn_pqa = masked_max(attn_pq, combined_pqa_mask, dim=1)  # B * U
            norm_attn_pqa = masked_softmax(max_attn_pqa, all_qa_mask, dim=-1)  # B * U
            agg_prev_qa = (
                norm_attn_pqa.unsqueeze(1).bmm(encoded_allqa).squeeze(1)
            )  # B * H

            attn_pc, _ = self._pcattnmat(
                encoded_passage, encoded_choice, choice_mask
            )  # B * T * V
            combined_pc_mask = passage_mask.unsqueeze(-1) * choice_mask.unsqueeze(
                1
            )  # B * T * V
            max_attn_pc = masked_max(attn_pc, combined_pc_mask, dim=1)  # B * V
            norm_attn_pc = masked_softmax(max_attn_pc, choice_mask, dim=-1)  # B * V
            agg_c = norm_attn_pc.unsqueeze(1).bmm(encoded_choice)  # B * 1 * H

            choice_scores_wk = agg_c.bmm(agg_prev_qa.unsqueeze(-1)).squeeze(-1)  # B * 1

        if self._qac_ap:
            attn_qac, _ = self._cqaattnmat(
                encoded_allqa, encoded_choice, choice_mask
            )  # B * U * V
            combined_qac_mask = all_qa_mask.unsqueeze(-1) * choice_mask.unsqueeze(
                1
            )  # B * U * V

            max_attn_c = masked_max(attn_qac, combined_qac_mask, dim=1)  # B * V
            max_attn_qa = masked_max(attn_qac, combined_qac_mask, dim=2)  # B * U
            norm_attn_c = masked_softmax(max_attn_c, choice_mask, dim=-1)  # B * V
            norm_attn_qa = masked_softmax(max_attn_qa, all_qa_mask, dim=-1)  # B * U
            agg_c_qa = norm_attn_c.unsqueeze(1).bmm(encoded_choice).squeeze(1)  # B * H
            agg_qa_c = norm_attn_qa.unsqueeze(1).bmm(encoded_allqa).squeeze(1)  # B * H

            choice_scores_nk = (
                agg_c_qa.unsqueeze(1).bmm(agg_qa_c.unsqueeze(-1)).squeeze(-1)
            )  # B * 1

        if self._with_knowledge and self._qac_ap:
            choice_score = choice_scores_wk + choice_scores_nk
        elif self._qac_ap:
            choice_score = choice_scores_nk
        elif self._with_knowledge:
            choice_score = choice_scores_wk
        else:
            raise NotImplementedError

        output = torch.sigmoid(choice_score).squeeze(-1)  # B

        output_dict = {
            "label_logits": choice_score.squeeze(-1),
            "label_probs": output,
        }

        if label is not None:
            label = label.long().view(-1)
            loss = self._loss(output, label.float())
            self._auc(output, label)
            self._f1(torch.stack([1 - output, output]).T, label)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1 = self._f1.get_metric(reset)
        return {
            "auc": self._auc.get_metric(reset),
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
