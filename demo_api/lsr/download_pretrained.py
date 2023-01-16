"""Run this script during build time to download the pretrained models and relevant files first"""

import allennlp_models.tagging  # Needed by Predictor
from allennlp.predictors.predictor import Predictor
from transformers import cached_path

from sgnlp.models.lsr import LsrModel, LsrConfig


# Downloads pretrained allennlp models
ner = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/fine-grained-ner.2020-06-24.tar.gz"
)
coref = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
)

# Download files and model from azure blob storage
rel2id_path = cached_path("https://storage.googleapis.com/sgnlp-models/models/lsr/rel2id.json")
word2id_path = cached_path(
    "https://storage.googleapis.com/sgnlp-models/models/lsr/word2id.json"
)
ner2id_path = cached_path("https://storage.googleapis.com/sgnlp-models/models/lsr/ner2id.json")
rel_info_path = cached_path(
    "https://storage.googleapis.com/sgnlp-models/models/lsr/rel_info.json"
)

config = LsrConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp-models/models/lsr/v2/config.json"
)
model = LsrModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp-models/models/lsr/v2/pytorch_model.bin",
    config=config,
)
