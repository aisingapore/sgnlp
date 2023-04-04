import copy
import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput

from .config import BaseConfig, RumourVerificationConfig, StanceClassificationConfig

logger = logging.getLogger(__name__)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:

    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = (
            ACT2FN[config.hidden_act]
            if isinstance(config.hidden_act, str)
            else config.hidden_act
        )

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertEncoderWithSingleLayer(nn.Module):
    """Modified BertEncoder with a single BertLayer."""

    def __init__(self, config):
        super(BertEncoderWithSingleLayer, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertStancePooler(nn.Module):
    def __init__(self, config):
        super(BertStancePooler, self).__init__()
        self.max_tweet_num = config.max_tweet_num
        self.max_tweet_length = config.max_tweet_length
        self.max_tweet_bucket = config.max_tweet_bucket
        self.max_seq_len = config.max_position_embeddings

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0].unsqueeze(1)

        for i in range(1, self.max_tweet_num):
            token = hidden_states[:, self.max_tweet_length * i].unsqueeze(1)
            if i == 1:
                output = torch.cat((first_token_tensor, token), dim=1)
            else:
                output = torch.cat((output, token), dim=1)

        for j in range(1, self.max_tweet_bucket):
            for k in range(self.max_tweet_num):
                token = hidden_states[
                    :, self.max_seq_len * j + self.max_tweet_length * k
                ].unsqueeze(1)
                output = torch.cat((output, token), dim=1)

        return output


@dataclass
class StanceClassificationModelOutput(ModelOutput):
    """Loss and logits of StanceClassificationModel."""

    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


@dataclass
class RumourVerificationModelOutput(ModelOutput):
    """Loss and logits of RumourVerificationModel."""

    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class InitBertWeights:
    """Initialize the model weights. This function is not triggered as part of _init_weights()."""

    config: BaseConfig

    def init_bert_weights(self, module):
        """Initialize the model weights. This function is not triggered as part of _init_weights()."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class StanceClassificationPreTrainedModel(PreTrainedModel):
    """This is used as base class for derived BaseBertModel, StanceClassificationBertModel and StanceClassificationModel."""

    config_class = StanceClassificationConfig
    base_model_prefix = "stance_classification"

    def _init_weights(self, module: nn.Module) -> None:
        pass


class RumourVerificationPreTrainedModel(PreTrainedModel):
    """This is used as base class for derived BaseBertModel, RumourVerificationBertModel and RumourVerificationModel."""

    config_class = RumourVerificationConfig
    base_model_prefix = "rumour_verification"

    def _init_weights(self, module: nn.Module) -> None:
        pass


class BaseBertModel(PreTrainedModel, InitBertWeights):
    """This is used as base class for derived StanceClassificationBertModel and RumourVerificationBertModel.

    The bare Bert Model transformer outputting raw hidden-states without any specific head on top.
    """

    def __init__(self, config):
        super(BaseBertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers: bool = True,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
        )
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class StanceClassificationBertModel(BaseBertModel, StanceClassificationPreTrainedModel):
    """Bert Model for stance classification."""


class RumourVerificationBertModel(BaseBertModel, RumourVerificationPreTrainedModel):
    """Bert Model for rumour verification."""


class StanceClassificationModel(StanceClassificationPreTrainedModel, InitBertWeights):
    """Model for stance classification."""

    def __init__(self, config: StanceClassificationConfig):
        super(StanceClassificationModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = StanceClassificationBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.stance_pooler = BertStancePooler(config)
        self.max_tweet_bucket = config.max_tweet_bucket
        self.add_bert_attention = BertEncoderWithSingleLayer(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, *inputs) -> StanceClassificationModelOutput:
        encoded_outputs = []
        for i in range(self.max_tweet_bucket):
            encoded_output, _ = self.bert(
                inputs[i * 3],  # input_ids
                inputs[i * 3 + 1],  # segment_ids
                inputs[i * 3 + 2],  # input_mask
                output_all_encoded_layers=False,
            )
            encoded_outputs.append(encoded_output)

        attention_mask = inputs[-3]
        labels = inputs[-2]
        label_mask = inputs[-1]

        concatenated_encoded_outputs = torch.cat(encoded_outputs, dim=1)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        add_bert_encoder = self.add_bert_attention(
            concatenated_encoded_outputs, extended_attention_mask
        )
        final_text_output = add_bert_encoder[-1]

        label_logit_output = self.stance_pooler(hidden_states=final_text_output)
        sequence_stance_output = self.dropout(label_logit_output)
        logits = self.classifier(sequence_stance_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if label_mask is not None:
                active_loss = label_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        else:
            loss = None

        return StanceClassificationModelOutput(loss=loss, logits=logits)


class RumourVerificationModel(RumourVerificationPreTrainedModel, InitBertWeights):
    """Model for rumour verification."""

    def __init__(self, config: RumourVerificationConfig):
        super(RumourVerificationModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = RumourVerificationBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.max_tweet_bucket = config.max_tweet_bucket
        self.add_bert_attention = BertEncoderWithSingleLayer(config)
        self.add_bert_pooler = BertPooler(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, *inputs) -> RumourVerificationModelOutput:
        encoded_outputs = []
        for i in range(self.max_tweet_bucket):
            encoded_output, _ = self.bert(
                inputs[i * 3],  # input_ids
                inputs[i * 3 + 1],  # segment_ids
                inputs[i * 3 + 2],  # input_mask
                output_all_encoded_layers=False,
            )
            encoded_outputs.append(encoded_output)

        attention_mask = inputs[-2]
        labels = inputs[-1]

        concatenated_encoded_outputs = torch.cat(encoded_outputs, dim=1)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        add_bert_encoder = self.add_bert_attention(
            concatenated_encoded_outputs, extended_attention_mask
        )
        add_bert_text_output_layer = add_bert_encoder[-1]
        final_text_output = self.add_bert_pooler(add_bert_text_output_layer)

        pooled_output = self.dropout(final_text_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        else:
            loss = None

        return RumourVerificationModelOutput(loss=loss, logits=logits)
