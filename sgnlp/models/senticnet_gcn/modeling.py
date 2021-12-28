import pathlib
import pickle
from dataclasses import dataclass
from typing import Optional, Union


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, BertModel
from transformers.file_utils import ModelOutput

from .modules.dynamic_rnn import DynamicLSTM
from .modules.gcn import GraphConvolution
from .config import (
    SenticNetGCNConfig,
    SenticNetGCNBertConfig,
    SenticNetGCNEmbeddingConfig,
    SenticNetGCNBertEmbeddingConfig,
)
from .utils import build_embedding_matrix


@dataclass
class SenticNetGCNModelOutput(ModelOutput):
    """
    Base class for outputs of SenticNetGCNModel.

    Args:
        loss (:obj:`torch.Tensor` of shape `(1,)`, `optional`, return when :obj:`labels` is provided):
            classification loss, typically cross entropy. Loss function used is dependent on what is specified in SenticNetGCNConfig.
        logits (:obj:`torch.Tensor` of shape :obj:`(batch_size, num_classes)`):
            raw logits for each class. num_classes = 3 by default.
    """

    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None


class SenticNetGCNPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for download and loading pretrained models.
    """

    config_class = SenticNetGCNConfig
    base_model_prefix = "senticnetgcn"

    def _init_weights(self, module: nn.Module) -> None:
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
        if config.loss_function == "cross_entropy":
            self.loss_function = nn.CrossEntropyLoss()

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

    def forward(
        self, inputs: dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None
    ) -> SenticNetGCNModelOutput:
        text_indices, aspect_indices, left_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)
        # TODO: How to replace embedding layer here?
        text = self.embedding(text_indices)
        text = self.text_embed_dropout(text_indices)
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
        logits = self.fc(x)

        loss = self.loss_function(logits, labels) if labels is not None else None
        return SenticNetGCNModelOutput(loss=loss, logits=logits)


@dataclass
class SenticNetGCNBertModelOutput(ModelOutput):
    """
    Base class for outputs of SenticNetGCNBertModel.

    Args:
        loss (:obj:`torch.Tensor` of shape `(1,)`, `optional`, return when :obj:`labels` is provided):
            classification loss, typically cross entropy.
            Loss function used is dependent on what is specified in SenticNetGCNBertConfig.
        logits (:obj:`torch.Tensor` of shape :obj:`(batch_size, num_classes)`):
            raw logits for each class. num_classes = 3 by default.
    """

    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None


class SenticNetGCNBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for download and loading pretrained models.
    """

    config_class = SenticNetGCNBertConfig
    base_model_prefix = "senticnetgcnbert"

    def _init_weights(self, module: nn.Module) -> None:
        pass


class SenticNetGCNBertPModel(SenticNetGCNBertPreTrainedModel):
    def __init__(self, config: SenticNetGCNBertConfig) -> None:
        super().__init__()
        self.gc1 = GraphConvolution(config.hidden_dim, config.hidden_dim)
        self.gc2 = GraphConvolution(config.hidden_dim, config.hidden_dim)
        self.gc3 = GraphConvolution(config.hidden_dim, config.hidden_dim)
        self.fc = nn.Linear(config.hidden_dim, config.polarities_dim)
        self.text_embed_dropout = nn.Dropout(config.dropout)
        self.device = config.device
        self.max_seq_len = config.max_seq_len
        self.loss_function = config.loss_function

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

    def forward(self, inputs, labels: torch.Tensor):
        text_bert_indices, text_indices, aspect_indices, bert_segments_ids, left_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)
        # TODO: How to embed in the preprocessor?
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
        logits = self.fc(x)

        loss = self.loss_function(logits, labels) if labels is not None else None
        return SenticNetGCNBertModelOutput(loss=loss, logits=logits)


class SenticNetGCNEmbeddingPreTrainedModel(PreTrainedModel):
    config_class = SenticNetGCNEmbeddingConfig
    base_model_prefix = "senticnetgcnembedding"

    def _init_weights(self, module: nn.Module) -> None:
        pass


class SenticNetGCNEmbeddingModel(SenticNetGCNEmbeddingPreTrainedModel):
    def __init__(self, config: SenticNetGCNEmbeddingConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)

    def load_pretrained_embedding(self, pretrained_embedding_path: Union[str, pathlib.Path]):
        with open(pretrained_embedding_path, "rb") as emb_f:
            embedding_matrix = pickle.load(emb_f)
        embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float)
        self.embed.weight.data.copy_(embedding_tensor)

    @classmethod
    def build_embedding_matrix(
        cls,
        word_vec_file_path: str,
        vocab: dict[str, int],
        embed_dim: int = 300,
    ):
        embedding_matrix = build_embedding_matrix(
            word_vec_file_path=word_vec_file_path, vocab=vocab, embed_dim=embed_dim
        )
        embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float)
        config = SenticNetGCNEmbeddingConfig(vocab_size=vocab, embed_dim=embed_dim)
        senticnetgcn_embed = cls(config)
        senticnetgcn_embed.embed.weight.data.copy_(embedding_tensor)
        return senticnetgcn_embed


class SenticNetGCNBertEmbeddingModel(BertModel):
    """
    The SenticNetGCN Bert Embedding Model used to generate embeddings for model inputs.

    This class inherits from :obj:`BertModel` for weights initalization and utility functions
    from transformers :obj:`PreTrainedModel` class.

    Args:
        config (:obj:`~SenticNetGCNBertEmbeddingConfig`):
            Model configuration class with all parameters required for the model.
            Initializing with a config file does not load
            the weights associated with the model, only the configuration.
            Use the :obj:`.from_pretrained` method to load the model weights.
    """

    def __init__(self, config: SenticNetGCNBertEmbeddingConfig):
        super().__init__(config)
