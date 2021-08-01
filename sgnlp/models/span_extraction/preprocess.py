from typing import List, Dict, Tuple, Optional, Union

import torch
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers.data.processors.squad import SquadFeatures, SquadExample

from .tokenization import RecconSpanExtractionTokenizer
from .utils import load_examples


class RecconSpanExtractionPreprocessor:
    """Class to initialise the Preprocessor for RecconSpanExtraction model.
    Preprocesses inputs and tokenises them so they can be used with RecconSpanExtractionModel

    Args:
        tokenizer (Optional[PreTrainedTokenizer], optional): Tokenizer to use for preprocessor. Defaults to None.
        max_length (int, optional): maximum length of truncated tokens. Defaults to 512.
    """

    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        if tokenizer is None:
            self.tokenizer = RecconSpanExtractionTokenizer.from_pretrained(
                "mrm8488/spanbert-finetuned-squadv2"
            )
        else:
            self.tokenizer = tokenizer

    def __call__(
        self, data_batch: Dict[str, List[str]]
    ) -> Tuple[
        BatchEncoding,
        List[Dict[str, Union[int, str]]],
        List[SquadExample],
        List[SquadFeatures],
    ]:
        """Preprocess data then tokenize, so it can be used in RecconSpanExtractionModel

        Args:
            data_batch (Dict[str, List[str]]): The dictionary should look like this:
                                        {'emotion': ['happiness'],
                                        'target_utterance': ['......'],
                                        'evidence_utterance': ['......'],
                                        'conversation_history': ['......']}
                                The length of each value must be the same

        Returns:
            Tuple[ BatchEncoding, List[Dict[str, Union[int, str]]], List[SquadExample], List[SquadFeatures] ]:
                1. BatchEncoding output from tokenizer
                2. List of evidence utterances
                3. List of SquadExample output from load_examples() function
                4. List of SquadFeatures output from load_examples() function
        """
        self._check_values_len(data_batch)
        concatenated_batch, evidences = self._concatenate_batch(data_batch)
        dataset, examples, features = load_examples(
            concatenated_batch, self.tokenizer, evaluate=True, output_examples=True
        )

        input_ids = [torch.unsqueeze(instance[0], 0) for instance in dataset]
        attention_mask = [torch.unsqueeze(instance[1], 0) for instance in dataset]
        token_type_ids = [torch.unsqueeze(instance[2], 0) for instance in dataset]

        output = {
            "input_ids": torch.cat(input_ids, axis=0),
            "attention_mask": torch.cat(attention_mask, axis=0),
            "token_type_ids": torch.cat(token_type_ids, axis=0),
        }
        output = BatchEncoding(output)

        return output, evidences, examples, features

    def _concatenate_batch(
        self, data_batch: Dict[str, List[str]]
    ) -> Tuple[List[Dict[str, any]], List[Dict[str, any]]]:
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
            Tuple[List[Dict[str, any], List[Dict[str, any]]]]:
                1. list of concatenated string for each instance
                2. list of evidence utterances for each instance
        """
        concatenated_batch = []
        evidences_batch = []

        emotion_batch = data_batch["emotion"]
        target_utterance_batch = data_batch["target_utterance"]
        evidence_utterance_batch = data_batch["evidence_utterance"]
        conversation_history_batch = data_batch["conversation_history"]

        for i, (
            emotion,
            target_utterance,
            evidence_utterance,
            conversation_history,
        ) in enumerate(
            zip(
                emotion_batch,
                target_utterance_batch,
                evidence_utterance_batch,
                conversation_history_batch,
            )
        ):
            concatenated_qns = (
                "The target utterance is "
                + target_utterance
                + "The evidence utterance is "
                + evidence_utterance
                + "What is the causal span from context that is relevant to the target utterance's emotion "
                + emotion
                + " ?"
            )
            inputs = {
                "id": i,
                "question": concatenated_qns,
                "answers": [{"text": " ", "answer_start": 0}],
                "is_impossible": False,
            }
            instance_dict = {"context": conversation_history, "qas": [inputs]}
            concatenated_batch.append(instance_dict)

            evidence = {"id": i, "evidence": evidence_utterance}
            evidences_batch.append(evidence)

        return concatenated_batch, evidences_batch

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
