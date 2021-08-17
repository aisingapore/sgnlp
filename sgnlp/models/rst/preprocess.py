import logging
from typing import Dict, List

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding


class RSTPreprocessor:
    """
    Class for preprocessing a list of raw texts to a batch of tensors.
    Inject tokenizer and/or embedding model instances via the 'tokenizer' and 'embedding_model' input args,
    if both tokenzier and embedding model are not provided, then the elmo model from allennlp package will be used.
    """
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer = None,
            embedding_model: PreTrainedModel = None,
            elmo_model_size: str = "Large",
            device: torch.device = torch.device('cpu')):
        self.device = device

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            try:
                from allennlp.modules.elmo import batch_to_ids
                self.tokenizer = batch_to_ids
            except ModuleNotFoundError:
                logging.warning('The package "allennlp" is not installed!')
                logging.warning('To use elmo embedding, please install "allennlp" with "pip install allennlp"')

        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            try:
                from .modules.elmo import initialize_elmo
            except ModuleNotFoundError:
                logging.warning('The package "allennlp" is not installed!')
                logging.warning('To use elmo embedding, please install "allennlp" with "pip install allennlp"')
            self.embedding_model, self.word_dim = initialize_elmo(elmo_model_size)
            self.embedding_model.to(device)
            self.use_elmo = True

    def __call__(self, data_batch: List[str]) -> BatchEncoding:
        """
        Main method to start preprocessing

        Args:
            data_batch (List[str]): list of input texts

        Returns:
            BatchEncoding: return a BatchEncoding instance with key 'data_batch' and embedded values of data batch.
        """
        if self.use_elmo:
            text_embedding = self._get_elmo_embedding(data_batch)
        else:
            text_embedding = self._get_embedding(data_batch)
        return BatchEncoding(text_embedding)

    def _get_embedding(self, data_batch: List[str]) -> torch.Tensor:
        raise NotImplementedError('Embedding method call not implemented.')

    def _get_elmo_embedding(self, data_batch: List[str]) -> Dict[str, List]:
        """
        Method to get elmo embedding from a batch of texts.

        Args:
            data_batch (List[str]): list of input texts

        Returns:
            Dict[str, List]: return a dictionary of elmo embeddings
        """
        character_ids = self.tokenizer(data_batch)
        character_ids.to(self.device)
        embeddings = self.embedding_model(character_ids)
        embeddings['data_batch'] = embeddings.pop('elmo_representations')  # Rename key to data_batch
        return embeddings
