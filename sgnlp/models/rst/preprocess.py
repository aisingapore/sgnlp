import logging
from typing import Dict, List, Tuple

import torch
from transformers import PreTrainedTokenizer
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
            device: torch.device = torch.device('cpu')):
        self.device = device

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            try:
                import nltk
                self.tokenizer = nltk.word_tokenize
            except ModuleNotFoundError:
                logging.warning('The package "nltk" is not installed!')
                logging.warning('Please install "nltk" with "pip install nltk"')

    def __call__(self, data_batch: List[str]):
        """
        Main method to start preprocessing for RST.

        Args:
            data_batch (List[str]): list of input texts

        Returns:
            Tuple[BatchEncoding, List[int]]: return a BatchEncoding instance with key 'data_batch' and embedded values
            of data batch. Also return a list of lengths of each text in the batch.
        """
        character_ids, sentence_lengths = self._get_elmo_char_ids(data_batch)
        return character_ids, sentence_lengths

    def _get_elmo_char_ids(self, data_batch: List[str]):
        """
        Method to get elmo embedding from a batch of texts.

        Args:
            data_batch (List[str]): list of input texts

        Returns:
            Dict[str, List]: return a dictionary of elmo embeddings
        """
        from allennlp.modules.elmo import batch_to_ids
        sentence_lengths = [len(data) for data in data_batch]
        character_ids = batch_to_ids(data_batch)
        character_ids = character_ids.to(self.device)

        return character_ids, sentence_lengths
