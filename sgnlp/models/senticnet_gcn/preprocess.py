from typing import List

import torch
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils.dummy_pt_objects import PreTrainedModel

from config import SenticNetGCNEmbeddingConfig, SenticNetGCNBertEmbeddingConfig
from modeling import SenticNetGCNEmbeddingModel, SenticNetGCNBertEmbeddingModel
from tokenization import SenticNetGCNTokenizer, SenticNetGCNBertTokenizer


class SenticNetGCNPreprocessor:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer = None,
        embedding_model: PreTrainedModel = None,
        tokenizer_name: str = None,
        embedding_model_name: str = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = SenticNetGCNTokenizer.from_pretrained(tokenizer_name)

        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            embedding_config = SenticNetGCNEmbeddingConfig.from_pretrained(embedding_model_name)
            self.embedding_model = SenticNetGCNEmbeddingModel.from_pretrained(
                embedding_model_name, config=embedding_config
            ).to(device)

    def __call__(self, data_batch: List[str]) -> BatchEncoding:
        tokens = self.tokenizer(data_batch, padding=True, return_tensors="pt")
        return tokens


class SenticNetGCNBertPreprocessor:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer = None,
        embedding_model: PreTrainedModel = None,
        tokenizer_name: str = None,
        embedding_model_name: str = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = SenticNetGCNBertTokenizer.from_pretrained(tokenizer_name)

        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            embedding_config = SenticNetGCNBertEmbeddingConfig.from_pretrained(embedding_model_name)
            self.embedding_model = SenticNetGCNBertEmbeddingModel.from_pretrained(
                embedding_model_name, config=embedding_config
            ).to(device)

    def __call__(self, data_batch: List[str]) -> BatchEncoding:
        tokens = self.tokenizer(data_batch, padding=True, return_tensors="pt")
        return tokens
