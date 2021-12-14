from typing import List

import torch
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from tokenization import SenticASGCNTokenizer


class SenticASGCNPreprocessor:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer = None,
        tokenizer_name: str = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = SenticASGCNTokenizer.from_pretrained(tokenizer_name)

    def __call__(self, data_batch: List[str]) -> BatchEncoding:
        tokens = self.tokenizer(data_batch, padding=True, return_tensors="pt")
        return tokens
