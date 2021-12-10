from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.file_utils import ModelOutput
from transformers.utils.dummy_pt_objects import PreTrainedModel


@dataclass
class SenticASGCNModelOutput(ModelOutput):
    pass


class SenticASGCNPreTrainedModel(PreTrainedModel):
    # config_class =
    base_model_prefix = "sentic_asgcn"

    def _init_weights(self, module):
        pass


class SenticASGCNModel(SenticASGCNPreTrainedModel):
    def __init__(self, config):
        pass
