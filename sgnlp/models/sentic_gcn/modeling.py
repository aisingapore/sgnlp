from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, BertModel
from transformers.file_utils import ModelOutput

from .modules.dynamic_rnn import DynamicLSTM
from .modules.gcn import GraphConvolution
from .config import (
    SenticGCNConfig,
    SenticGCNBertConfig,
    SenticGCNEmbeddingConfig,
    SenticGCNBertEmbeddingConfig,
)
from .utils import build_embedding_matrix


@dataclass
class SenticGCNModelOutput(ModelOutput):
    """
    Base class for outputs of SenticGCNModel.

    Args:
        loss (:obj:`torch.Tensor` of shape `(1,)`, `optional`, return when :obj:`labels` is provided):
            classification loss, typically cross entropy. Loss function used is dependent on what is specified in SenticGCNConfig.
        logits (:obj:`torch.Tensor` of shape :obj:`(batch_size, num_classes)`):
            raw logits for each class. num_classes = 3 by default.
    """

    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None


class SenticGCNPreTrainedModel(PreTrainedModel):
    """
    The SenticGCN Pre-Trained Model used as base class for derived SenticGCN Model.

    This model is the abstract super class for the SenticGCN Model which defines the config
    class types and weights initalization method. This class should not be used or instantiated directly,
    see SenticGCNModel class for usage.
    """

    config_class = SenticGCNConfig
    base_model_prefix = "senticgcn"

    def _init_weights(self, module: nn.Module) -> None:
        pass


class SenticGCNModel(SenticGCNPreTrainedModel):
    """
    The SenticGCN Model for aspect based sentiment analysis.

    This method inherits from :obj:`SenticGCNPreTrainedModel` for weights initalization and utility functions
    from transformer :obj:`PreTrainedModel` class.

    Args:
        config (:obj:`~SenticGCNConfig`):
            Model configuration class with all parameters required for the model.
            Initializing with a config file does not load
            the weights associated with the model, only the configuration.
            Use the :obj:`.from_pretrained` method to load the model weights.
    """

    def __init__(self, config: SenticGCNConfig) -> None:
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
        if config.loss_function == "cross_entropy":
            self.loss_function = nn.CrossEntropyLoss()

    def position_weight(
        self, x: torch.Tensor, aspect_double_idx: torch.Tensor, text_len: torch.Tensor, aspect_len: torch.Tensor
    ) -> torch.Tensor:
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
        weight = torch.tensor(weight, dtype=torch.float).unsqueeze(2).to(x.device)
        return weight * x

    def mask(self, x: torch.Tensor, aspect_double_idx: torch.Tensor) -> torch.Tensor:
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
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(2).to(x.device)
        return mask * x

    def forward(self, inputs: List[torch.Tensor], labels: Optional[torch.Tensor] = None) -> SenticGCNModelOutput:
        text_indices, aspect_indices, left_indices, text_embeddings, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)
        text = self.text_embed_dropout(text_embeddings)
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
        return SenticGCNModelOutput(loss=loss, logits=logits)


@dataclass
class SenticGCNBertModelOutput(ModelOutput):
    """
    Base class for outputs of SenticGCNBertModel.

    Args:
        loss (:obj:`torch.Tensor` of shape `(1,)`, `optional`, return when :obj:`labels` is provided):
            classification loss, typically cross entropy.
            Loss function used is dependent on what is specified in SenticGCNBertConfig.
        logits (:obj:`torch.Tensor` of shape :obj:`(batch_size, num_classes)`):
            raw logits for each class. num_classes = 3 by default.
    """

    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None


class SenticGCNBertPreTrainedModel(PreTrainedModel):
    """
    The SenticGCNBert Pre-Trained Model used as base class for derived SenticGCNBert Model.

    This model is the abstract super class for the SenticGCNBert Model which defines the config
    class types and weights initalization method. This class should not be used or instantiated directly,
    see SenticGCNBertModel class for usage.
    """

    config_class = SenticGCNBertConfig
    base_model_prefix = "senticgcnbert"

    def _init_weights(self, module: nn.Module) -> None:
        pass


class SenticGCNBertModel(SenticGCNBertPreTrainedModel):
    """
    The SenticGCNBert Model for aspect based sentiment analysis.

    This method inherits from :obj:`SenticGCNBertPreTrainedModel` for weights initalization and utility functions
    from transformer :obj:`PreTrainedModel` class.

    Args:
        config (:obj:`~SenticGCNBertConfig`):
            Model configuration class with all parameters required for the model.
            Initializing with a config file does not load
            the weights associated with the model, only the configuration.
            Use the :obj:`.from_pretrained` method to load the model weights.
    """

    def __init__(self, config: SenticGCNBertConfig) -> None:
        super().__init__(config)
        self.gc1 = GraphConvolution(config.hidden_dim, config.hidden_dim)
        self.gc2 = GraphConvolution(config.hidden_dim, config.hidden_dim)
        self.gc3 = GraphConvolution(config.hidden_dim, config.hidden_dim)
        self.fc = nn.Linear(config.hidden_dim, config.polarities_dim)
        self.text_embed_dropout = nn.Dropout(config.dropout)
        self.max_seq_len = config.max_seq_len
        self.loss_function = config.loss_function

    def position_weight(
        self, x: torch.Tensor, aspect_double_idx: torch.Tensor, text_len: torch.Tensor, aspect_len: torch.Tensor
    ) -> torch.Tensor:
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
        weight = torch.tensor(weight).unsqueeze(2).to(x.device)
        return weight * x

    def mask(self, x: torch.Tensor, aspect_double_idx: torch.Tensor) -> torch.Tensor:
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
        mask = torch.tensor(mask).unsqueeze(2).float().to(x.device)
        return mask * x

    def forward(self, inputs: List[torch.Tensor], labels: Optional[torch.Tensor] = None) -> SenticGCNBertModelOutput:
        text_indices, aspect_indices, left_indices, text_embeddings, adj = inputs
        # text_indices, text_
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)

        text_out = text_embeddings
        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc3(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1)  # batch_size x 2*hidden_dim
        logits = self.fc(x)

        loss = self.loss_function(logits, labels) if labels is not None else None
        return SenticGCNBertModelOutput(loss=loss, logits=logits)


class SenticGCNEmbeddingPreTrainedModel(PreTrainedModel):
    """
    The SenticGCN Embedding Pre-Trained Model used as base class for derived SenticGCN Embedding Model.

    This model is the abstract super class for the SenticGCN Embedding Model which defines the config
    class types and weights initalization method. This class should not be used or instantiated directly,
    see SenticGCNEmbeddingModel class for usage.
    """

    config_class = SenticGCNEmbeddingConfig
    base_model_prefix = "senticgcnembedding"

    def _init_weights(self, module: nn.Module) -> None:
        pass


class SenticGCNEmbeddingModel(SenticGCNEmbeddingPreTrainedModel):
    """
    The SenticGCN Embedding Model used to generate embeddings for model inputs.
    By default, the embeddings are generated from the glove.840B.300d embeddings.

    This class inherits from :obj:`SenticGCNEmbeddingPreTrainedModel` for weights initalization and utility functions
    from transformers :obj:`PreTrainedModel` class.

    This class can also be constructed via the SenticGCNEmbeddingModel.build_embedding_matrix class method.

    Args:
        config (:obj:`~SenticGCNEmbeddingConfig`):
            Model configuration class with all parameters required for the model.
            Initializing with a config file does not load
            the weights associated with the model, only the configuration.
            Use the :obj:`.from_pretrained` method to load the model weights.
    """

    def __init__(self, config: SenticGCNEmbeddingConfig) -> None:
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode input token ids using word embedding.

        Args:
            token_ids (torch.Tensor): Tensor of token ids with shape [batch_size, num_words]

        Returns:
            torch.Tensor: return Tensor of embeddings with shape (batch_size, num_words, embed_dim)
        """
        return self.embed(token_ids)

    @classmethod
    def build_embedding_model(
        cls,
        word_vec_file_path: str,
        vocab: Dict[str, int],
        embed_dim: int = 300,
    ):
        """
        This class method is a helper method to construct the embedding model from a file containing word vectors (i.e. GloVe)
        and a vocab dictionary.

        Args:
            word_vec_file_path (str): file path to the word vectors
            vocab (Dict[str, int]): vocab dictionary consisting of words as key and index as values
            embed_dim (int, optional): the embedding dimension. Defaults to 300.

        Returns:
            SenticGCNEmbeddingModel: return an instance of SenticGCNEmbeddingModel
        """
        embedding_matrix = build_embedding_matrix(
            word_vec_file_path=word_vec_file_path, vocab=vocab, embed_dim=embed_dim
        )
        embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float)
        sentic_embed_config = SenticGCNEmbeddingConfig(vocab_size=len(vocab), embed_dim=embed_dim)
        senticgcn_embed = cls(sentic_embed_config)
        senticgcn_embed.embed.weight.data.copy_(embedding_tensor)
        return senticgcn_embed


class SenticGCNBertEmbeddingModel(BertModel):
    """
    The SenticGCN Bert Embedding Model used to generate embeddings for model inputs.

    This class inherits from :obj:`BertModel` for weights initalization and utility functions
    from transformers :obj:`PreTrainedModel` class.

    Args:
        config (:obj:`~SenticGCNBertEmbeddingConfig`):
            Model configuration class with all parameters required for the model.
            Initializing with a config file does not load
            the weights associated with the model, only the configuration.
            Use the :obj:`.from_pretrained` method to load the model weights.
    """

    def __init__(self, config: SenticGCNBertEmbeddingConfig) -> None:
        super().__init__(config)
