import torch
import torch.nn as nn
from transformers import PreTrainedModel


from .config import CSGConfig
from .tokenization import CSGTokenizerFast
from .modules.conv_gec import ConvGEC


class CSGPreTrainedModel(PreTrainedModel):

    config_class = CSGConfig
    base_model_prefix = "csg"

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

class CSGModel(CSGPreTrainedModel):
    def __init__(self, config: CSGConfig):
        super().__init__(config)
        self.config = config
        self.convgec = ConvGEC(config)

    def load_pretrained_embedding(self, pretrained_source_embedding_path, pretrained_target_embedding_path):
        self.convgec.load_pretrained_embedding(pretrained_source_embedding_path, pretrained_target_embedding_path)
        
    def forward(self, source_ids, context_ids, target_ids):
        return self.convgec(source_ids, context_ids, target_ids)
    
