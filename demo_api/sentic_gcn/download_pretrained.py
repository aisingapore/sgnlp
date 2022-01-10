"""Run this script during build time to download the pretrained models and relevant files first"""

from sgnlp.models.sentic_gcn import (
    SenticGCNConfig,
    SenticGCNBertTokenizer,
    SenticGCNBertModel,
    SenticGCNBertPreprocessor
)

config = SenticGCNConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_bert/config.json"
)

tokenizer = SenticGCNBertTokenizer.from_pretrained("bert-base-uncased")


model = SenticGCNBertModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_bert/pytorch_model.bin", 
    config=config
)