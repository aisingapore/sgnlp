from typing import Optional

import torch
import torch.nn as nn
import numpy as np

from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel, BertModel

from .config import LsrConfig
from .modules.encoder import Encoder
from .modules.attention import SelfAttention
from .modules.reasoner import DynamicReasoner
from .modules.reasoner import StructInduction


@dataclass
class LsrModelOutput:
    """
    Output type of :class:`~sgnlp.models.lsr.modeling.LsrModel`

    Args:
        prediction (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, max_h_t_count, num_relations)`):
            Prediction scores for all head to tail entity combinations from the final layer.
            Note that the sigmoid function has not been applied at this point.
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when `labels` is provided ):
            Loss on relation prediction task.
    """
    prediction: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None


class LsrPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LsrConfig
    base_model_prefix = "lsr"

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


class LsrModel(LsrPreTrainedModel):
    """The Latent Structure Refinement Model performs relation classification on all pairs of entity clusters.

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Args:
        config (:class:`~sgnlp.models.lsr.config.LsrConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration.
            Use the :obj:`.from_pretrained` method to load the model weights.

    Example::

        from sgnlp.models.lsr import LsrModel, LsrConfig

        # Method 1: Loading a default model
        config = LsrConfig()
        model = LsrModel(config)

        # Method 2: Loading from pretrained
        config = LsrConfig.from_pretrained('https://sgnlp.blob.core.windows.net/models/lsr/config.json')
        model = LsrModel.from_pretrained('https://sgnlp.blob.core.windows.net/models/lsr/pytorch_model.bin',
                                         config=config)
    """

    def __init__(self, config: LsrConfig):
        super().__init__(config)

        self.config = config

        # Common
        self.dropout = nn.Dropout(config.dropout_rate)
        self.relu = nn.ReLU()

        # Document encoder layers
        if config.use_bert:
            self.bert = BertModel.from_pretrained("bert-base-uncased")
            bert_hidden_size = 768
            self.linear_re = nn.Linear(bert_hidden_size, config.hidden_dim)
        else:
            self.word_emb = nn.Embedding(config.word_embedding_shape[0], config.word_embedding_shape[1])
            if not config.finetune_emb:
                self.word_emb.weight.requires_grad = False
            self.ner_emb = nn.Embedding(13, config.ner_dim, padding_idx=0)
            self.coref_embed = nn.Embedding(config.max_length, config.coref_dim, padding_idx=0)
            self.linear_re = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
            input_size = config.word_embedding_shape[1] + config.coref_dim + config.ner_dim
            self.rnn_sent = Encoder(input_size, config.hidden_dim, config.dropout_emb, config.dropout_rate)

        # Induce latent structure layers
        self.use_struct_att = config.use_struct_att
        if self.use_struct_att:
            self.struct_induction = StructInduction(config.hidden_dim // 2, config.hidden_dim, True)
        self.dropout_gcn = nn.Dropout(config.dropout_gcn)
        self.use_reasoning_block = config.use_reasoning_block
        if self.use_reasoning_block:
            self.reasoner = nn.ModuleList()
            self.reasoner.append(DynamicReasoner(config.hidden_dim, config.reasoner_layer_sizes[0], self.dropout_gcn))
            self.reasoner.append(DynamicReasoner(config.hidden_dim, config.reasoner_layer_sizes[1], self.dropout_gcn))

        # Output layers
        self.dis_embed = nn.Embedding(20, config.distance_size, padding_idx=10)
        self.self_att = SelfAttention(config.hidden_dim)
        self.bili = torch.nn.Bilinear(config.hidden_dim + config.distance_size,
                                      config.hidden_dim + config.distance_size, config.hidden_dim)
        self.linear_output = nn.Linear(2 * config.hidden_dim, config.num_relations)

        self.init_weights()

    def load_pretrained_word_embedding(self, pretrained_word_embedding):
        self.word_emb.weight.data.copy_(torch.from_numpy(pretrained_word_embedding))

    def doc_encoder(self, input_sent, context_seg):
        batch_size = context_seg.shape[0]
        docs_emb = []  # sentence embedding
        docs_len = []
        sents_emb = []

        for batch_no in range(batch_size):
            sent_list = []
            sent_lens = []
            sent_index = ((context_seg[batch_no] == 1).nonzero()).squeeze(
                -1).tolist()  # array of start point for sentences in a document
            pre_index = 0
            for i, index in enumerate(sent_index):
                if i != 0:
                    if i == 1:
                        sent_list.append(input_sent[batch_no][pre_index:index + 1])
                        sent_lens.append(index - pre_index + 1)
                    else:
                        sent_list.append(input_sent[batch_no][pre_index + 1:index + 1])
                        sent_lens.append(index - pre_index)
                pre_index = index

            sents = pad_sequence(sent_list).permute(1, 0, 2)
            sent_lens_t = torch.LongTensor(sent_lens).to(device=self.device)
            docs_len.append(sent_lens)
            sents_output, sent_emb = self.rnn_sent(sents, sent_lens_t)  # sentence embeddings for a document.

            doc_emb = None
            for i, (sen_len, emb) in enumerate(zip(sent_lens, sents_output)):
                if i == 0:
                    doc_emb = emb[:sen_len]
                else:
                    doc_emb = torch.cat([doc_emb, emb[:sen_len]], dim=0)

            docs_emb.append(doc_emb)
            sents_emb.append(sent_emb.squeeze(1))

        docs_emb = pad_sequence(docs_emb).permute(1, 0, 2)  # B * # sentence * Dimension
        sents_emb = pad_sequence(sents_emb).permute(1, 0, 2)

        return docs_emb, sents_emb

    def forward(self, context_idxs, context_pos, context_ner, h_mapping, t_mapping,
                relation_mask, dis_h_2_t, dis_t_2_h, context_seg, node_position, entity_position,
                node_sent_num, all_node_num, entity_num_list, sdp_position, sdp_num_list, context_masks=None,
                context_starts=None, relation_multi_label=None, **kwargs):
        # TODO: current kwargs are ignored, to allow preprocessing to pass in unnecessary arguments
        # TODO: Fix upstream preprocessing such that it is filtered out before passing in.
        """
        Args:
            context_idxs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, max_tokens_length)`):
                Token IDs.
            context_pos (:obj:`torch.LongTensor` of shape :obj:`(batch_size, max_tokens_length)`):
                Coref position IDs.
            context_ner (:obj:`torch.LongTensor` of shape :obj:`(batch_size, max_tokens_length)`):
                NER tag IDs.
            h_mapping (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, h_t_limit, max_tokens_length)`):
                Head entity position mapping.
            t_mapping (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, h_t_limit, max_tokens_length)`):
                Tail entity position mapping.
            relation_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, h_t_limit)`):
                Relation mask. 1 if relation exists in position else 0.
            dis_h_2_t (:obj:`torch.LongTensor` of shape :obj:`(batch_size, h_t_limit)`):
                Distance encoding from head to tail.
            dis_t_2_h (:obj:`torch.LongTensor` of shape :obj:`(batch_size, h_t_limit)`):
                Distance encoding from tail to head.
            context_seg (:obj:`torch.LongTensor` of shape :obj:`(batch_size, max_tokens_length)`):
                Start position of sentences in document. 1 to mark position is start of sentence else 0.
            node_position (:obj:`torch.LongTensor` of shape :obj:`(batch_size, max_node_number, max_tokens_length)`):
                Mention node position.
            entity_position (:obj:`torch.LongTensor` of shape :obj:`(batch_size, max_entity_number, max_tokens_length)`):
                Entity node position. An entity refers to all mentions referring to the same entity.
            node_sent_num (:obj:`torch.LongTensor` of shape :obj:`(batch_size, max_sent_num)`):
                Number of mention nodes in each sentence of a document.
            all_node_num (:obj:`torch.LongTensor` of shape :obj:`(1)`):
                Total number of nodes (mention + MDP) in a document.
            entity_num_list (:obj:`List[int]` of shape :obj:`(batch_size)`):
                Number of entity nodes in each document.
            sdp_position (:obj:`torch.LongTensor` of shape :obj:`(batch_size, max_entity_number, max_tokens_length)`):
                Meta dependency paths (MDP) node position.
            sdp_num_list (:obj:`List[int]` of shape :obj:`(batch_size)`):
                Number of MDP nodes in each document.
            context_masks (:obj:`torch.LongTensor` of shape :obj:`(batch_size, max_length)`, `optional`):
                Mask for padding tokens. Used by bert model only.
            context_starts (:obj:`torch.LongTensor` of shape :obj:`(batch_size, max_length)`, `optional`):
                Tensor indicating start of words. Used by bert model only.
            relation_multi_label (:obj:`torch.LongTensor` of shape :obj:`(batch_size, h_t_limit, num_relations)`):
                Label for all possible head to tail entity relations.

        Returns:
            output (:class:`~sgnlp.models.lsr.modeling.LsrModelOutput`)
        """

        # Step 1: Encode the document
        if self.config.use_bert:
            context_output = self.bert(context_idxs, attention_mask=context_masks)[0]
            context_output = [layer[starts.nonzero().squeeze(1)]
                              for layer, starts in zip(context_output, context_starts)]
            context_output = pad_sequence(context_output, batch_first=True, padding_value=-1)
            context_output = torch.nn.functional.pad(context_output,
                                                     (0, 0, 0, context_idxs.size(-1) - context_output.size(-2)))
            context_output = self.dropout(torch.relu(self.linear_re(context_output)))
            max_doc_len = 512
        else:
            sent_emb = torch.cat(
                [self.word_emb(context_idxs), self.coref_embed(context_pos), self.ner_emb(context_ner)],
                dim=-1)
            docs_rep, sents_rep = self.doc_encoder(sent_emb, context_seg)

            max_doc_len = docs_rep.shape[1]
            context_output = self.dropout(torch.relu(self.linear_re(docs_rep)))

        # Step 2: Extract all node reps of a document graph
        # extract mention node representations
        mention_num_list = torch.sum(node_sent_num, dim=1).tolist()
        max_mention_num = max(mention_num_list)
        mentions_rep = torch.bmm(node_position[:, :max_mention_num, :max_doc_len],
                                 context_output)  # mentions rep
        # extract meta dependency paths (MDP) node representations
        max_sdp_num = max(sdp_num_list)
        sdp_rep = torch.bmm(sdp_position[:, :max_sdp_num, :max_doc_len], context_output)
        # extract entity node representations
        entity_rep = torch.bmm(entity_position[:, :, :max_doc_len], context_output)
        # concatenate all nodes of an instance
        gcn_inputs = []
        all_node_num_batch = []
        for batch_no, (m_n, e_n, s_n) in enumerate(zip(mention_num_list, entity_num_list, sdp_num_list)):
            m_rep = mentions_rep[batch_no][:m_n]
            e_rep = entity_rep[batch_no][:e_n]
            s_rep = sdp_rep[batch_no][:s_n]
            gcn_inputs.append(torch.cat((m_rep, e_rep, s_rep), dim=0))
            node_num = m_n + e_n + s_n
            all_node_num_batch.append(node_num)

        gcn_inputs = pad_sequence(gcn_inputs).permute(1, 0, 2)
        output = gcn_inputs

        # Step 3: Induce the Latent Structure
        if self.use_reasoning_block:
            for i in range(len(self.reasoner)):
                output = self.reasoner[i](output)

        elif self.use_struct_att:
            gcn_inputs, _ = self.struct_induction(gcn_inputs)
            max_all_node_num = torch.max(all_node_num).item()
            assert (gcn_inputs.shape[1] == max_all_node_num)

        node_position = node_position.permute(0, 2, 1)
        output = torch.bmm(node_position[:, :max_doc_len, :max_mention_num], output[:, :max_mention_num])
        context_output = torch.add(context_output, output)

        start_re_output = torch.matmul(h_mapping[:, :, :max_doc_len], context_output)  # aggregation
        end_re_output = torch.matmul(t_mapping[:, :, :max_doc_len], context_output)  # aggregation

        s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
        t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)
        re_rep = self.dropout(self.relu(self.bili(s_rep, t_rep)))
        re_rep = self.self_att(re_rep, re_rep, relation_mask)
        prediction = self.linear_output(re_rep)

        loss = None

        if relation_multi_label is not None:
            loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            loss = torch.sum(loss_fn(prediction, relation_multi_label) * relation_mask.unsqueeze(2)) \
                   / torch.sum(relation_mask)

        return LsrModelOutput(prediction=prediction, loss=loss)
