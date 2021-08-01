from typing import List, Dict, Optional

from transformers import PreTrainedTokenizer, BatchEncoding

from .tokenization import RecconEmotionEntailmentTokenizer


class RecconEmotionEntailmentPreprocessor:
    """Class to initialise the Preprocessor for RecconEmotionEntailment model.
    Preprocesses inputs and tokenises them so they can be used with RecconEmotionEntailmentModel

    Args:
        tokenizer (Optional[PreTrainedTokenizer], optional): Tokenizer to use for preprocessor. Defaults to None.
        max_length (int, optional): maximum length of truncated tokens. Defaults to 512.
    """

    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 512,
    ):
        self.max_length = max_length
        if tokenizer is None:
            self.tokenizer = RecconEmotionEntailmentTokenizer.from_pretrained(
                "roberta-base"
            )
        else:
            self.tokenizer = tokenizer

    def __call__(self, data_batch: Dict[str, List[str]]) -> BatchEncoding:
        """Preprocess data then tokenize, so it can be used in RecconEmotionEntailmentModel

        Args:
            data_batch (Dict[str, List[str]]): The dictionary should look like this:
                                        {'emotion': ['happiness'],
                                        'target_utterance': ['......'],
                                        'evidence_utterance': ['......'],
                                        'conversation_history': ['......']}
                                The length of each value must be the same

        Returns:

            BatchEncoding: BatchEncoding instance returned from self.tokenizer
        """

        self._check_values_len(data_batch)
        concatenated_batch = self._concatenate_batch(data_batch)
        output = self.tokenizer(
            concatenated_batch,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )

        return output

    def _concatenate_batch(self, data_batch: Dict[str, List[str]]) -> List[str]:
        """Takes in data batch and converts them into a list of string which can be
        used with the tokenizer

        Args:
            data_batch (Dict[str, List[str]]): The dictionary should look like this:
                                        {'emotion': ['happiness'],
                                        'target_utterance': ['......'],
                                        'evidence_utterance': ['......'],
                                        'conversation_history': ['......']}
                                The length of each value must be the same

        Returns:
            List[str]: list of concatenated string for each instance
        """
        concatenated_batch = []

        emotion_batch = data_batch["emotion"]
        target_utterance_batch = data_batch["target_utterance"]
        evidence_utterance_batch = data_batch["evidence_utterance"]
        conversation_history_batch = data_batch["conversation_history"]

        for emotion, target_utterance, evidence_utterance, conversation_history in zip(
            emotion_batch,
            target_utterance_batch,
            evidence_utterance_batch,
            conversation_history_batch,
        ):
            concatenated_instance = self._concatenate_instance(
                emotion, target_utterance, evidence_utterance, conversation_history
            )
            concatenated_batch.append(concatenated_instance)
        return concatenated_batch

    def _concatenate_instance(
        self,
        emotion: str,
        target_utterance: str,
        evidence_utterance: str,
        conversation_history: str,
    ) -> str:
        """Concatenate a single instance into a single string

        Args:
            emotion (str): emotion of instance
            target_utterance (str): target_utterance of instance
            evidence_utterance (str): evidence utterance of instance
            conversation_history (str): conversation history of instance

        Returns:
            str: concated string of a single instance
        """
        concatenated_text = (
            " "
            + emotion
            + " <SEP> "
            + target_utterance
            + " <SEP> "
            + evidence_utterance
            + " <SEP> "
            + conversation_history
        )

        return concatenated_text

    def _check_values_len(self, data_batch: Dict[str, List[str]]):
        """Check if the length of all values in the Dict are the same

        Args:
            data_batch (Dict[str, List[str]]): data_batch input from __call__ method
        """
        values_len = [len(v) for _, v in data_batch.items()]
        unique_len = len(set(values_len))
        assert unique_len == 1, "Length of values are not consistent across"
