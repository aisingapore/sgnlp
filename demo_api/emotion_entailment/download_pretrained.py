"""Run this script during build time to download the pretrained models and relevant files first"""

from sgnlp_models.models.emotion_entailment import (
    RecconEmotionEntailmentConfig,
    RecconEmotionEntailmentTokenizer,
    RecconEmotionEntailmentModel,
)


# Downloads pretrained config, tokenizer and model
config = RecconEmotionEntailmentConfig.from_pretrained(
    "https://sgnlp.blob.core.windows.net/models/reccon_emotion_entailment/config.json"
)
tokenizer = RecconEmotionEntailmentTokenizer.from_pretrained("roberta-base")
model = RecconEmotionEntailmentModel.from_pretrained(
    "https://sgnlp.blob.core.windows.net/models/reccon_emotion_entailment/pytorch_model.bin",
    config=config,
)
