import torch
from transformers import BertConfig, BertTokenizer

from sgnlp.models.coupled_hierarchical_transformer import (
    DualBert,
    DualBertConfig,
    DualBertPreprocessor,
    DualBertPostprocessor
)

# model_state_dict = torch.load("/Users/nus/Documents/Code/projects/SGnlp/sgnlp/output/pytorch_model.bin")
# model = DualBert.from_pretrained(
#         "bert-base-uncased",
#         state_dict=model_state_dict,
#         rumor_num_labels=3,
#         stance_num_labels=5,
#         max_tweet_num=17,
#         max_tweet_length=30,
#         convert_size=20,
#     )
#
# print("x")
from sgnlp.models.coupled_hierarchical_transformer.train import InputExample

config = DualBertConfig.from_pretrained("https://storage.googleapis.com/sgnlp/models/dual_bert/config.json")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
preprocessor = DualBertPreprocessor(config, tokenizer)
model = DualBert.from_pretrained("https://storage.googleapis.com/sgnlp/models/dual_bert/pytorch_model.bin",
                                 config=config)
postprocessor = DualBertPostprocessor()

model.eval()

# example = [
#     "#4U9525: Robin names Andreas Lubitz as the copilot in the flight deck who crashed the aircraft.",
#     "@thatjohn @mschenk",
#     "@thatjohn Have they named the pilot?",
# ]

examples = [
    [
        "#4U9525: Robin names Andreas Lubitz as the copilot in the flight deck who crashed the aircraft.",
        "@thatjohn @mschenk",
        "@thatjohn Have they named the pilot?",
    ]
]

model_inputs = preprocessor(examples)
# { model_param_1: ..., model_param2: ..., ...}

model_output = model(**model_inputs)
output = postprocessor(model_output, model_inputs["stance_label_mask"])
