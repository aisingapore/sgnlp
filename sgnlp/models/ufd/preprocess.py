from typing import Dict, List

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding

from .config import UFDEmbeddingConfig
from .tokenization import UFDTokenizer
from .modeling import UFDEmbeddingModel


class UFDPreprocessor:
    """
    Class for preprocessing a raw text to a batch of tensors for the UFDModel to predict on.
    Inject tokenizer and/or embedding model instances via the 'tokenizer' and 'embedding_model' input args,
    or pass in the tokenizer name and/or embedding model name via the 'tokenizer_name' and 'embedding_model_name'
    input args to create from_pretrained.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer = None,
        embedding_model: PreTrainedModel = None,
        tokenizer_name: str = "xlm-roberta-large",
        embedding_model_name: str = "xlm-roberta-large",
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = UFDTokenizer.from_pretrained(tokenizer_name)

        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            embedding_config = UFDEmbeddingConfig.from_pretrained(embedding_model_name)
            self.embedding_model = UFDEmbeddingModel.from_pretrained(
                embedding_model_name, config=embedding_config
            ).to(device)

    def __call__(self, data_batch: List[str]) -> BatchEncoding:
        """
        Main method to start preprocessing.

        Args:
            data_batch (List[str]): list of raw text to process.

        Returns:
            BatchEncoding: return a BatchEncoding instance with key 'input_ids' and embedded values of data batch.
        """
        text_embeddings = self._get_embedding(data_batch)
        mean_features = torch.mean(
            text_embeddings[0], dim=1
        )  # calculate the mean of output layer of embedding model
        return BatchEncoding({"data_batch": mean_features})

    def _get_embedding(self, data_batch: List[str]) -> Dict[str, BatchEncoding]:
        """
        Method to generate tensor from a list of text.

        Args:
            text (List[str]): list of input text.

        Returns:
            Dict[str, BatchEncoding]: tensor generated from input text.
        """
        self.embedding_model.eval()
        with torch.no_grad():
            tokens = self.tokenizer(data_batch, padding=True).to(self.device)
            text_embedding = self.embedding_model(**tokens)
        return text_embedding
