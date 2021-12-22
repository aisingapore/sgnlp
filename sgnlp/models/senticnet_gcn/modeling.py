from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, BertModel
from transformers.file_utils import ModelOutput

from .modules.dynamic_rnn import DynamicLSTM
from .modules.gcn import GraphConvolution
from .config import SenticNetGCNConfig, SenticNetBertGCNConfig


@dataclass
class SenticNetGCNModelOutput(ModelOutput):
    pass


class SenticNetGCNPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for download and loading pretrained models.
    """

    config_class = SenticNetGCNConfig
    base_model_prefix = "senticnet_gcn"

    def _init_weights(self, module):
        pass


class SenticNetGCNModel(SenticNetGCNPreTrainedModel):
    def __init__(self, config: SenticNetGCNConfig) -> None:
        super().__init__(config)
        self.text_lstm = DynamicLSTM(
            config.embed_dim,
            config.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.gc1 = GraphConvolution(2 * config.hidden_dim, 2 * config.hidden_dim)
        self.gc2 = GraphConvolution(2 * config.hidden_dim, 2 * config.hidden_dim)
        self.fc = nn.Linear(2 * config.hidden_dim, config.polarities_dim)
        self.text_embed_dropout = nn.Dropout(config.dropout)
        self.device = config.device

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i, 0]):
                weight[i].append(1 - (aspect_double_idx[i, 0] - j) / context_len)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i, 1] + 1, text_len[i]):
                weight[i].append(1 - (j - aspect_double_idx[i, 1] / context_len))
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight, dtype=torch.float).unsqueeze(2).to(self.device)
        return weight * x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i, 0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i, 1] + 1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(2).to(self.device)
        return mask * x

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)
        text = self.text_embed_dropout(self.embed(text_indices))
        text_out, (_, _) = self.text_lstm(text, text_len)
        x = F.relu(
            self.gc1(
                self.position_weight(text_out, aspect_double_idx, text_len, aspect_len),
                adj,
            )
        )
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1)  # batch_size x 2 * hidden_dim
        output = self.fc(x)
        return output


class SenticNetBertGCNPreTrainedModel(PreTrainedModel):
    config_class = SenticNetBertGCNConfig
    base_model_prefix = "senticnetbert_gcn"

    def _init_weights(self, module):
        pass


class SenticNetBertGCPModel(SenticNetBertGCNPreTrainedModel):
    def __init__(self, config: SenticNetBertGCNConfig) -> None:
        super().__init__()
        self._init_bert_model(config.bert_model)
        self.gc1 = GraphConvolution(config.hidden_dim, config.hidden_dim)
        self.gc2 = GraphConvolution(config.hidden_dim, config.hidden_dim)
        self.gc3 = GraphConvolution(config.hidden_dim, config.hidden_dim)
        self.fc = nn.Linear(config.hidden_dim, config.polarities_dim)
        self.text_embed_dropout = nn.Dropout(config.dropout)
        self.device = config.device
        self.max_seq_len = config.max_seq_len

    def _init_bert_model(self, bert_model: str):
        self.bert = BertModel.from_pretrained(bert_model)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i, 0]):
                weight[i].append(1 - (aspect_double_idx[i, 0] - j) / context_len)
            for j in range(aspect_double_idx[i, 0], min(aspect_double_idx[i, 1] + 1, self.max_seq_len)):
                weight[i].append(0)
            for j in range(aspect_double_idx[i, 1] + 1, text_len[i]):
                weight[i].append(1 - (j - aspect_double_idx[i, 1]) / context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.device)
        return weight * x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i, 0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i, 0], min(aspect_double_idx[i, 1] + 1, self.max_seq_len)):
                mask[i].append(1)
            for j in range(min(aspect_double_idx[i, 1] + 1, self.max_seq_len), seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.device)
        return mask * x

    def forward(self, inputs):
        text_bert_indices, text_indices, aspect_indices, bert_segments_ids, left_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)
        encoder_layer, _ = self.bert(
            text_bert_indices, token_type_ids=bert_segments_ids, output_all_encoded_layers=False
        )
        text_out = encoder_layer
        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc3(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1)  # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output
