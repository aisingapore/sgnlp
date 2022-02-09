from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, PreTrainedModel, BertPreTrainedModel
from transformers.file_utils import ModelOutput
from transformers.models.bert.modeling_bert import BertPooler

from sgnlp.models.coupled_hierarchical_transformer.config import DualBertConfig
from sgnlp.models.coupled_hierarchical_transformer.modeling import BertCrossEncoder, ADDBertReturnEncoder, \
    MTBertStancePooler, BertPooler_v2, BertSelfLabelAttention


@dataclass
class DualBertOutput (ModelOutput):
    rumour_loss: float = None
    rumour_logits: torch.Tensor = None
    stance_loss: float = None
    stance_logits: torch.Tensor = None



class DualBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DualBertConfig
    base_model_prefix = "dual_bert"

    def _init_weights(self, module):
        """Initialize the weights"""
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


class DualBert(DualBertPreTrainedModel):
    def __init__(self, config,
                 # rumor_num_labels=2, stance_num_labels=2, max_tweet_num=17, max_tweet_length=30,
                 # convert_size=20
                 ):
        super(DualBert, self).__init__(config)
        self.rumor_num_labels = config.rumor_num_labels
        self.stance_num_labels = config.stance_num_labels
        self.bert = BertModel(config)
        # self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.add_rumor_bert_attention = BertCrossEncoder(config)
        self.add_stance_bert_attention = ADDBertReturnEncoder(config)
        self.max_tweet_num = config.max_tweet_num
        self.max_tweet_length = config.max_tweet_length
        self.stance_pooler = MTBertStancePooler(config)
        # previous version
        # self.rumor_pooler = BertPooler(config)
        # self.add_self_attention = BertSelfLabelAttention(config, stance_num_labels)
        # self.rumor_classifier = nn.Linear(config.hidden_size+stance_num_labels, rumor_num_labels)
        # new version
        # self.rumor_pooler = BertPooler_v2(config.hidden_size+stance_num_labels) # +stance_num_labels
        # self.add_self_attention = BertSelfLabelAttention(config, config.hidden_size+stance_num_labels)
        # self.rumor_classifier = nn.Linear(config.hidden_size+stance_num_labels, rumor_num_labels)
        # Version 3
        # self.rumor_pooler = BertPooler(config)
        # self.add_self_attention = BertSelfLabelAttention(config, config.hidden_size+stance_num_labels)
        # self.rumor_classifier = nn.Linear(config.hidden_size*2+stance_num_labels, rumor_num_labels)
        # Version 4
        self.convert_size = config.convert_size  # 100 pheme seed 42, 100->0.423, 0.509, 75 OK, 32, 50, 64, 80, 90, 120, 128, 200 not good,
        self.rumor_pooler = BertPooler(config)
        self.hybrid_rumor_pooler = BertPooler_v2(config.hidden_size + config.stance_num_labels)
        self.add_self_attention = BertSelfLabelAttention(config, config.hidden_size + config.stance_num_labels)
        self.linear_conversion = nn.Linear(config.hidden_size + config.stance_num_labels, self.convert_size)
        self.rumor_classifier = nn.Linear(config.hidden_size + self.convert_size, config.rumor_num_labels)
        #### self.rumor_classifier = nn.Linear(config.hidden_size, rumor_num_labels)
        self.stance_classifier = nn.Linear(config.hidden_size, config.stance_num_labels)
        #### self.cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)
        # self.apply(self.init_bert_weights)
        self.init_weights()

    def init_bert(self):
        self.bert = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, input_ids1, token_type_ids1, attention_mask1, input_ids2, token_type_ids2, attention_mask2,
                input_ids3, token_type_ids3, attention_mask3, input_ids4, token_type_ids4, attention_mask4,
                attention_mask, rumor_labels=None, task=None, stance_labels=None, stance_label_mask=None):

        output1 = self.bert(input_ids1, token_type_ids1, attention_mask1, output_hidden_states=False)
        output2 = self.bert(input_ids2, token_type_ids2, attention_mask2, output_hidden_states=False)
        output3 = self.bert(input_ids3, token_type_ids3, attention_mask3, output_hidden_states=False)
        output4 = self.bert(input_ids4, token_type_ids4, attention_mask4, output_hidden_states=False)

        sequence_output1 = output1.last_hidden_state
        sequence_output2 = output2.last_hidden_state
        sequence_output3 = output3.last_hidden_state
        sequence_output4 = output4.last_hidden_state

        tmp_sequence = torch.cat((sequence_output1, sequence_output2), dim=1)
        tmp_sequence = torch.cat((tmp_sequence, sequence_output3), dim=1)
        sequence_output = torch.cat((tmp_sequence, sequence_output4), dim=1)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # for stance classification task
        # '''
        # ##add_output_layer = self.add_self_attention(sequence_output, extended_attention_mask)
        add_stance_bert_encoder, stance_attention_probs = self.add_stance_bert_attention(sequence_output,
                                                                                         extended_attention_mask)
        final_stance_text_output = add_stance_bert_encoder[-1]
        stance_attention = stance_attention_probs[-1]
        label_logit_output = self.stance_pooler(final_stance_text_output, self.max_tweet_num, self.max_tweet_length)
        sequence_stance_output = self.dropout(label_logit_output)
        stance_logits = self.stance_classifier(sequence_stance_output)
        # '''

        if task is None:  # for rumor detection task
            # '''
            add_rumor_bert_encoder, rumor_attention_probs = self.add_rumor_bert_attention(final_stance_text_output,
                                                                                          sequence_output,
                                                                                          extended_attention_mask)
            add_rumor_bert_text_output_layer = add_rumor_bert_encoder[-1]
            rumor_attention = rumor_attention_probs[-1]

            # '''  add label attention layer to incorporate stance predictions for rumor verification
            extended_label_mask = stance_label_mask.unsqueeze(1).unsqueeze(2)
            extended_label_mask = extended_label_mask.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_label_mask = (1.0 - extended_label_mask) * -10000.0

            rumor_output = self.rumor_pooler(add_rumor_bert_text_output_layer)
            tweet_level_output = self.stance_pooler(add_rumor_bert_text_output_layer, self.max_tweet_num,
                                                    self.max_tweet_length)
            final_rumor_output = torch.cat((tweet_level_output, stance_logits), dim=-1)  # stance_logits
            combined_layer, attention_probs = self.add_self_attention(final_rumor_output, extended_label_mask)
            hybrid_rumor_stance_output = self.hybrid_rumor_pooler(combined_layer)
            hybrid_conversion_output = self.linear_conversion(hybrid_rumor_stance_output)
            final_rumor_text_output = torch.cat((rumor_output, hybrid_conversion_output), dim=-1)
            rumor_pooled_output = self.dropout(final_rumor_text_output)
            logits = self.rumor_classifier(rumor_pooled_output)
            # '''

            if rumor_labels is not None:
                # alpha = 0.1
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.rumor_num_labels), rumor_labels.view(-1))
                # sim_loss = self.cos_sim(stance_attention, rumor_attention)
                # return loss + alpha*sim_loss
                return loss
            else:
                # return logits
                return logits, attention_probs[:, 0, 0, :]
                # fisrt 0 denotes head, second 0 denotes the first position's attention over all the tweets
        else:
            # for stance classification task

            # label_logit_output = self.stance_pooler(sequence_output)
            '''
            label_logit_output = self.stance_pooler(final_stance_text_output)
            sequence_stance_output = self.dropout(label_logit_output)
            stance_logits = self.stance_classifier(sequence_stance_output)
            '''

            if stance_labels is not None:  # for stance classification task
                loss_fct = CrossEntropyLoss()
                # Only keep active parts of the loss
                if stance_label_mask is not None:
                    active_loss = stance_label_mask.view(-1) == 1
                    # print(active_loss)
                    # print(logits)
                    active_logits = stance_logits.view(-1, self.stance_num_labels)[active_loss]
                    active_labels = stance_labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(stance_logits.view(-1, self.stance_num_labels), stance_labels.view(-1))
                return loss
            else:
                return stance_logits