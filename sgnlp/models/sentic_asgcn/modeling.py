from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput

from .modules.dynamic_rnn import DynamicLSTM
from .modules.gcn import GraphConvolution
from .config import SenticASGCNConfig


@dataclass
class SenticASGCNModelOutput(ModelOutput):
    pass


class SenticASGCNPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for download and loading pretrained models.
    """

    config_class = SenticASGCNConfig
    base_model_prefix = "sentic_asgcn"

    def _init_weights(self, module):
        pass


class SenticASGCNModel(SenticASGCNPreTrainedModel):
    def __init__(self, config: SenticASGCNConfig):
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
