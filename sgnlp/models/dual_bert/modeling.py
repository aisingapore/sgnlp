import copy
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, PreTrainedModel, BertPreTrainedModel
from transformers.file_utils import ModelOutput
from transformers.models.bert.modeling_bert import BertPooler, BertIntermediate, BertOutput, BertSelfOutput

from sgnlp.models.dual_bert.config import DualBertConfig


@dataclass
class DualBertModelOutput(ModelOutput):
    rumour_loss: float = None
    rumour_logits: torch.Tensor = None
    stance_loss: float = None
    stance_logits: torch.Tensor = None
    attention_probs: nn.Softmax = None


class BertCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super(BertCrossAttentionLayer, self).__init__()
        self.attention = BertCrossAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        attention_output, attention_probs = self.attention(s1_hidden_states, s2_hidden_states, s2_attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BertReturnSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertReturnSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs


class BertReturnAttention(nn.Module):
    def __init__(self, config):
        super(BertReturnAttention, self).__init__()
        self.self = BertReturnSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_probs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


class BertReturnLayer(nn.Module):
    def __init__(self, config):
        super(BertReturnLayer, self).__init__()
        self.attention = BertReturnAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BertCoAttention(nn.Module):
    def __init__(self, config):
        super(BertCoAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        mixed_query_layer = self.query(s1_hidden_states)
        mixed_key_layer = self.key(s2_hidden_states)
        mixed_value_layer = self.value(s2_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + s2_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs


class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super(BertCrossAttention, self).__init__()
        self.self = BertCoAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask):
        self_output, attention_probs = self.self(s1_input_tensor, s2_input_tensor, s2_attention_mask)
        attention_output = self.output(self_output, s1_input_tensor)
        return attention_output, attention_probs


class BertCrossEncoder(nn.Module):
    def __init__(self, config):
        super(BertCrossEncoder, self).__init__()
        layer = BertCrossAttentionLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(1)])

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        all_encoder_layers = []
        all_layer_attentions = []
        for layer_module in self.layer:
            s1_hidden_states, attention_probs = layer_module(s1_hidden_states, s2_hidden_states, s2_attention_mask)
            all_encoder_layers.append(s1_hidden_states)
            all_layer_attentions.append(attention_probs)
        return all_encoder_layers, all_layer_attentions


class BertPooler_v2(nn.Module):
    def __init__(self, hidden_size):
        super(BertPooler_v2, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ADDBertReturnEncoder(nn.Module):
    def __init__(self, config):
        super(ADDBertReturnEncoder, self).__init__()
        layer = BertReturnLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(1)])

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        all_layer_attentions = []
        for layer_module in self.layer:
            hidden_states, attention_probs = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
            all_layer_attentions.append(attention_probs)
        return all_encoder_layers, all_layer_attentions


class MTBertStancePooler(nn.Module):
    def __init__(self, config):
        super(MTBertStancePooler, self).__init__()
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.activation = nn.Tanh()

    def forward(self, hidden_states, max_tweet_num, max_tweet_len):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # max_tweet_len = 20  # the number of words in each tweet (42,12)
        # max_tweet_num = 25  # the number of tweets in each bucket
        max_bucket_num = 4
        max_seq_len = 512

        first_token_tensor = hidden_states[:, 0].unsqueeze(1)

        for i in range(1, max_tweet_num):
            tmp_token_tensor = hidden_states[:, max_tweet_len * i].unsqueeze(1)
            if i == 1:
                tmp_output = torch.cat((first_token_tensor, tmp_token_tensor), dim=1)
            else:
                tmp_output = torch.cat((tmp_output, tmp_token_tensor), dim=1)

        for j in range(1, max_bucket_num):
            for k in range(max_tweet_num):
                tmp_token_tensor = hidden_states[:, max_seq_len * j + max_tweet_len * k].unsqueeze(1)
                tmp_output = torch.cat((tmp_output, tmp_token_tensor), dim=1)

        final_output = tmp_output
        # pooled_output = self.dense(final_output)
        # pooled_output = self.activation(pooled_output)
        return final_output


class BertSelfLabelAttention(nn.Module):
    def __init__(self, config, label_size):
        num_attention_heads = 1
        super(BertSelfLabelAttention, self).__init__()
        if label_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (label_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(label_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(label_size, self.all_head_size)
        self.key = nn.Linear(label_size, self.all_head_size)
        self.value = nn.Linear(label_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs


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

    def forward(self, input_ids_buckets, segment_ids_buckets, input_mask_buckets, input_mask, stance_position,
                stance_label_mask, stance_label_ids=None, rumor_label_ids=None):
        # def forward(self, input_ids1, token_type_ids1, attention_mask1, input_ids2, token_type_ids2, attention_mask2,
        #           input_ids3, token_type_ids3, attention_mask3, input_ids4, token_type_ids4, attention_mask4,
        #          attention_mask, rumor_labels=None, stance_label_ids=None, stance_label_mask=None):

        output = DualBertModelOutput()

        output_sequence = torch.tensor([], device=self.device, dtype=torch.int32)
        # for input_ids, token_type_ids, attention_mask in zip(processed_input["input_ids_buckets"], processed_input[
        #     "segment_ids_buckets"], processed_input["input_mask_buckets"]):
        num_buckets = 4
        for i in range(num_buckets):
            input_ids = input_ids_buckets[:, i].to(self.device)
            token_type_ids = segment_ids_buckets[:, i].to(self.device)
            attention_mask = input_mask_buckets[:, i].to(self.device)

            tmp_model_output = self.bert(input_ids, token_type_ids, attention_mask, output_hidden_states=False)
            output_sequence = torch.cat((output_sequence, tmp_model_output.last_hidden_state), dim=1)

        extended_attention_mask = input_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        extended_attention_mask = extended_attention_mask.to(self.device)

        # for stance classification task
        # '''
        # ##add_output_layer = self.add_self_attention(sequence_output, extended_attention_mask)
        add_stance_bert_encoder, stance_attention_probs = self.add_stance_bert_attention(output_sequence,
                                                                                         extended_attention_mask)
        final_stance_text_output = add_stance_bert_encoder[-1]
        stance_attention = stance_attention_probs[-1]
        label_logit_output = self.stance_pooler(final_stance_text_output, self.max_tweet_num, self.max_tweet_length)
        sequence_stance_output = self.dropout(label_logit_output)
        output.stance_logits = self.stance_classifier(sequence_stance_output)
        # '''

        add_rumor_bert_encoder, rumor_attention_probs = self.add_rumor_bert_attention(final_stance_text_output,
                                                                                      output_sequence,
                                                                                      extended_attention_mask)
        add_rumor_bert_text_output_layer = add_rumor_bert_encoder[-1]
        rumor_attention = rumor_attention_probs[-1]

        # '''  add label attention layer to incorporate stance predictions for rumor verification
        extended_label_mask = stance_label_mask.unsqueeze(1).unsqueeze(2)
        extended_label_mask = extended_label_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_label_mask = (1.0 - extended_label_mask) * -10000.0
        extended_label_mask = extended_label_mask.to(self.device)

        rumor_output = self.rumor_pooler(add_rumor_bert_text_output_layer)
        tweet_level_output = self.stance_pooler(add_rumor_bert_text_output_layer, self.max_tweet_num,
                                                self.max_tweet_length)
        final_rumor_output = torch.cat((tweet_level_output, output.stance_logits), dim=-1)  # stance_logits
        combined_layer, attention_probs = self.add_self_attention(final_rumor_output, extended_label_mask)
        hybrid_rumor_stance_output = self.hybrid_rumor_pooler(combined_layer)
        hybrid_conversion_output = self.linear_conversion(hybrid_rumor_stance_output)
        final_rumor_text_output = torch.cat((rumor_output, hybrid_conversion_output), dim=-1)
        rumor_pooled_output = self.dropout(final_rumor_text_output)
        output.rumour_logits = self.rumor_classifier(rumor_pooled_output)
        # '''

        if rumor_label_ids is not None:
            # alpha = 0.1
            loss_fct = CrossEntropyLoss()
            output.rumour_loss = loss_fct(output.rumour_logits.view(-1, self.rumor_num_labels).to(self.device), rumor_label_ids.view(-1).to(self.device))
            output.attention_probs = attention_probs[:, 0, 0, :]
            # sim_loss = self.cos_sim(stance_attention, rumor_attention)
            # return loss + alpha*sim_loss

        if stance_label_ids is not None:  # for stance classification task
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if stance_label_mask is not None:
                active_loss = stance_label_mask.view(-1) == 1
                # print(active_loss)
                # print(logits)
                active_logits = output.stance_logits.view(-1, self.stance_num_labels)[active_loss].to(self.device)
                active_labels = stance_label_ids.view(-1)[active_loss].to(self.device)
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(output.stance_logits.view(-1, self.stance_num_labels).to(self.device), stance_label_ids.view(-1).to(self.device))
            output.stance_loss = loss

        return output
