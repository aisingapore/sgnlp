"""Run this script during build time to download the pretrained models and relevant files first"""

from sgnlp.models.emotion_entailment import (
    RecconEmotionEntailmentConfig,
    RecconEmotionEntailmentTokenizer,
    RecconEmotionEntailmentModel,
)


# Downloads pretrained config, tokenizer and model
config = RecconEmotionEntailmentConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp-models/models/reccon_emotion_entailment/config.json"
)
tokenizer = RecconEmotionEntailmentTokenizer.from_pretrained("roberta-base")
model = RecconEmotionEntailmentModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp-models/models/reccon_emotion_entailment/pytorch_model.bin",
    config=config,
)
