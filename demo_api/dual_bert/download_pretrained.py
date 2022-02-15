from transformers import BertTokenizer

from sgnlp.models.dual_bert import (
    DualBert,
    DualBertConfig,
    DualBertPreprocessor
)

config = DualBertConfig.from_pretrained("https://storage.googleapis.com/sgnlp/models/dual_bert/config.json")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

preprocessor = DualBertPreprocessor(config, tokenizer)

model = DualBert.from_pretrained("https://storage.googleapis.com/sgnlp/models/dual_bert/pytorch_model.bin",
                                 config=config)
