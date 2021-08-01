from typing import List

import torch
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from sgnlp_models.models.nea.tokenization import NEATokenizer


class NEAPreprocessor:
    """
    Class for preprocesing of a list of raw text to a batch of tensors for the NEAModel to predict on.
    Inject tokenizer instances via the 'tokenizer' input args, or pass in the tokenizer name via the 'tokenizer_name'
    input args to create from_pretrained.
    """
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer = None,
            tokenizer_name: str = None,
            device: torch.device = torch.device('cpu')):
        self.device = device

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = NEATokenizer.from_pretrained(tokenizer_name)

    def __call__(self, data_batch: List[str]) -> BatchEncoding:
        """
        Main method to start preprocessing.

        Args:
            data_batch (List[str]): list of raw text to process

        Returns:
            BatchEncoding: tensor generated from input text
        """
        tokens = self.tokenizer(data_batch, padding=True, return_tensors='pt')
        return tokens
