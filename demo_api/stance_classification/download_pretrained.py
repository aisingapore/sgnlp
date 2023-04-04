"""Run this script during build time to download the pretrained models and relevant files first"""

from sgnlp.models.rumour_stance import (
    StanceClassificationConfig,
    StanceClassificationModel,
)

config = StanceClassificationConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/rumour_stance/stance_classification/config.json"
)

model = StanceClassificationModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/rumour_stance/stance_classification/pytorch_model.bin",
    config=config,
)
