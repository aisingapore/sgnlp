import torch
from transformers import BertConfig

from sgnlp.models.coupled_hierarchical_transformer import (
    DualBert,
    prepare_data_for_training
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

preprocessor = DualBertPreprocessor()

config = DualBertConfig.from_pretrained("path to config")
model = DualBert.from_pretrained("/Users/nus/Documents/Code/projects/SGnlp/sgnlp/output/pytorch_model.bin", config=config)

model.eval()

example = [
    "Claim",
    "Response 1",
    "Response 2"
]

model_inputs = preprocessor([example])
# { model_param_1: ..., model_param2: ..., ...}

model(**model_inputs)

