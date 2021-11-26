"""Run this script to download all predtrained weights"""
from sgnlp.models.ufd import (
    UFDEmbeddingConfig,
    UFDEmbeddingModel,
    UFDTokenizer,
    UFDModelBuilder,
)


# Dict with all models grouping
model_builder = UFDModelBuilder()
model_group = model_builder.build_model_group()

# Download and cache xlmr model
embedding_model_name = "xlm-roberta-large"
xlmr_model_config = UFDEmbeddingConfig.from_pretrained(embedding_model_name)
xlmr_model = UFDEmbeddingModel.from_pretrained(embedding_model_name)
xlmr_tokenizer = UFDTokenizer.from_pretrained(embedding_model_name)
