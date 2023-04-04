"""Run this script during build time to download the pretrained models and relevant files first"""

from sgnlp.models.rumour_stance import RumourVerificationConfig, RumourVerificationModel

config = RumourVerificationConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/rumour_stance/rumour_verification/config.json"
)

model = RumourVerificationModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/rumour_stance/rumour_verification/pytorch_model.bin",
    config=config,
)
