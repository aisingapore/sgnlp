from transformers import BertTokenizer

from sgnlp.models.dual_bert import (
    DualBert,
    DualBertConfig,
    DualBertPreprocessor,
    DualBertPostprocessor
)
config = DualBertConfig.from_pretrained("https://storage.googleapis.com/sgnlp/models/dual_bert/config.json")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
preprocessor = DualBertPreprocessor(config, tokenizer)
model = DualBert.from_pretrained("https://storage.googleapis.com/sgnlp/models/dual_bert/pytorch_model.bin",
                                 config=config)
postprocessor = DualBertPostprocessor()

model.eval()

examples = [
    [
        "#4U9525: Robin names Andreas Lubitz as the copilot in the flight deck who crashed the aircraft.",
        "@thatjohn @mschenk",
        "@thatjohn Have they named the pilot?",
    ]
]

model_inputs = preprocessor(examples)
model_output = model(**model_inputs)
output = postprocessor(model_output, model_inputs["stance_label_mask"])