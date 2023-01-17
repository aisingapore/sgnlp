from sgnlp.models.span_extraction import (
    RecconSpanExtractionConfig,
    RecconSpanExtractionModel,
    RecconSpanExtractionTokenizer,
)

config = RecconSpanExtractionConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp-models/models/reccon_span_extraction/config.json"
)
model = RecconSpanExtractionModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp-models/models/reccon_span_extraction/pytorch_model.bin",
    config=config,
)
tokenizer = RecconSpanExtractionTokenizer.from_pretrained(
    "mrm8488/spanbert-finetuned-squadv2"
)
