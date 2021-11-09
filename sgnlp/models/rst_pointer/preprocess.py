import logging
import numpy as np
from typing import Dict, List, Tuple
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding


class RstPreprocessor:
    """
    Class for preprocessing a list of raw texts to a batch of tensors.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer = None):

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            try:
                import nltk

                self.tokenizer = nltk.word_tokenize
            except ModuleNotFoundError:
                logging.warning('The package "nltk" is not installed!')
                logging.warning('Please install "nltk" with "pip install nltk"')

    def __call__(self, sentences: List[str]):
        """
        Main method to start preprocessing for RST.

        Args:
            sentences (List[str]): list of input texts

        Returns:
            Tuple[BatchEncoding, List[int]]: return a BatchEncoding instance with key 'data_batch' and embedded values
            of data batch. Also return a list of lengths of each text in the batch.
        """
        tokenized_sentences = [
            np.array(self.tokenizer(sentence)) for sentence in sentences
        ]
        character_ids, sentence_lengths = self.get_elmo_char_ids(tokenized_sentences)
        return character_ids, tokenized_sentences, sentence_lengths

    def get_elmo_char_ids(self, tokenized_sentences: List[str]):
        """
        Method to get elmo embedding from a batch of texts.

        Args:
            tokenized_sentences (List[str]): list of input texts

        Returns:
            Dict[str, List]: return a dictionary of elmo embeddings
        """
        from allennlp.modules.elmo import batch_to_ids

        sentence_lengths = [len(data) for data in tokenized_sentences]
        character_ids = batch_to_ids(tokenized_sentences)
        character_ids = character_ids

        return character_ids, sentence_lengths
