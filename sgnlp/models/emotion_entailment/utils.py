from typing import List, Dict
import argparse
import json
import pathlib

import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset

from .tokenization import (
    RecconEmotionEntailmentTokenizer,
)
from .data_class import RecconEmotionEntailmentArguments


def parse_args_and_load_config(
    config_path: str = "config/emotion_entailment_config.json",
) -> RecconEmotionEntailmentArguments:
    """Get config from config file using argparser

    Returns:
        RecconEmotionEntailmentArguments: RecconEmotionEntailmentArguments instance
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=config_path)
    args = parser.parse_args()

    with open(pathlib.Path(__file__).parent / args.config, "r") as f:
        cfg = json.load(f)
    emotion_entailment_args = RecconEmotionEntailmentArguments(**cfg)

    return emotion_entailment_args


class RecconEmotionEntailmentData(torch.utils.data.Dataset):
    """Class to create torch Dataset instance, which is the required data type
    for Transformer's Trainer

    Args:
        dataset (TensorDataset): TensorDataset object
        is_training (bool, optional): Set True if training, set False if evaluating. Defaults to True.
    """

    def __init__(self, dataset: TensorDataset, is_training: bool = True) -> None:
        """Constructor method"""
        self.dataset = dataset
        self.is_training = is_training

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a dictionary of the selected instance for each batch

        Args:
            idx (int): idx to select instances for each batch

        Returns:
            (Dict[str, torch.Tensor]): dictionary containing input_ids, attention_mask and token_type_ids
                    of the selected instance
        """
        item = {}
        in_idx, in_mask, seg_ids, label = self.dataset[idx]
        item["input_ids"] = in_idx
        item["attention_mask"] = in_mask
        item["token_type_ids"] = seg_ids
        if self.is_training:
            item["labels"] = label
        return item

    def __len__(self) -> int:
        """Returns length of dataset

        Returns:
            int: length of the dataset attribute
        """
        return len(self.dataset)


class InputExample(object):
    """Convert pandas dataframe data instance to InputExample instance for
    easier manipulation

    Args:
        guid (int): [description]
        text_a (str): [description]
        text_b (str, optional): Not used for emotion entailment. Defaults to None.
        label (int optional): Contains label of data. Defaults to None.
    """

    def __init__(self, guid: int, text_a: str, text_b: str = None, label: int = None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """Convert InputExample instance to InputFeature instance for
    easier manipulation

    Args:
        input_ids (List[int]): Contains list of input_ids created by tokenizer
        input_mask (List[int]): Contains list of input_mask created by tokenizer.
                                Int needs to be either 1 or 0.
        segment_ids (List[int], optional): Contains list of segment_ids created by tokenizer.
                                            Int needs to be either 1 or 0. Defaults to None.
        label_id (int): Contains label of data.
    """

    def __init__(
        self,
        input_ids: List[int],
        input_mask: List[int],
        segment_ids: List[int],
        label_id: int,
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_df_to_dataset(
    df: pd.DataFrame, max_seq_length: int, tokenizer: RecconEmotionEntailmentTokenizer
) -> TensorDataset:
    """Convert pandas dataframe to TensorDataset

    Args:
        df (pd.DataFrame): DataFrame containing 'text' and 'labels' columns
        max_seq_length (int): max sequence length
        tokenizer (RecconEmotionEntailmentTokenizer): RecconEmotionEntailmentTokenizer from sgnlp

    Returns:
        TensorDataset: A tensordataset object containing all examples
    """
    examples = convert_df_to_examples(df)
    dataset = load_examples(examples, max_seq_length, tokenizer)

    return dataset


def convert_df_to_examples(df: pd.DataFrame) -> List[InputExample]:
    """Convert dataframe to examples which can be fed into dataloader

    Args:
        df (pd.DataFrame): df which contains 'text' and 'labels columns

    Returns:
        List[InputExample]: list of InputExample
    """
    if "text" in df.columns and "labels" in df.columns:
        examples = [
            InputExample(i, text, None, label)
            for i, (text, label) in enumerate(zip(df["text"].astype(str), df["labels"]))
        ]

    return examples


def convert_example_to_feature(
    example_row: List,
    pad_token: int = 0,
    sequence_a_segment_id: int = 0,
    cls_token_segment_id: int = 1,
    pad_token_segment_id: int = 0,
    mask_padding_with_zero: bool = True,
    sep_token_extra: bool = False,
) -> InputFeatures:
    """Method to generate InputFeatures object from individual example row

    Args:
        example_row (List): List of inputs from convert_examples_to_features function
        pad_token (int, optional): Option for pad_token. Defaults to 0.
        sequence_a_segment_id (int, optional): Option for sequence_a_segment_id. Defaults to 0.
        cls_token_segment_id (int, optional): Option for cls_token_segment_id. Defaults to 1.
        pad_token_segment_id (int, optional): Option for pad_token_segment_id Defaults to 0.
        mask_padding_with_zero (bool, optional): Option for mask_padding_with_zero . Defaults to True.
        sep_token_extra (bool, optional): Option for sep_token_extra. Defaults to False.

    Returns:
        InputFeatures: Generated InputFeatures object
    """
    (
        example,
        max_seq_length,
        tokenizer,
        cls_token_at_end,
        cls_token,
        sep_token,
        cls_token_segment_id,
        pad_on_left,
        pad_token_segment_id,
        sep_token_extra,
        pad_token,
        add_prefix_space,
        pad_to_max_length,
    ) = example_row

    if add_prefix_space and not example.text_a.startswith(" "):
        tokens_a = tokenizer.tokenize(" " + example.text_a)
    else:
        tokens_a = tokenizer.tokenize(example.text_a)
    special_tokens_count = 3 if sep_token_extra else 2
    if len(tokens_a) > max_seq_length - special_tokens_count:
        tokens_a = tokens_a[: (max_seq_length - special_tokens_count)]
    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)
    if cls_token_at_end:
        tokens = tokens + [cls_token]
        segment_ids = segment_ids + [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    if pad_to_max_length:
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + (
                [0 if mask_padding_with_zero else 1] * padding_length
            )
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
    return InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=example.label,
    )


def convert_examples_to_features(
    examples: List[InputExample],
    max_seq_length: int,
    tokenizer: RecconEmotionEntailmentTokenizer,
    cls_token_at_end: bool = False,
    sep_token_extra: bool = False,
    pad_on_left: bool = False,
    cls_token: str = "[CLS]",
    sep_token: str = "[SEP]",
    pad_token: int = 0,
    cls_token_segment_id: int = 1,
    pad_token_segment_id: int = 0,
    silent: bool = False,
    add_prefix_space: bool = False,
    pad_to_max_length: bool = True,
) -> List[InputFeatures]:
    """Loop method to process all examples

    Args:
        examples (List[InputExample]):List of InputExample
        max_seq_length (int): Max sequence length
        tokenizer (RecconEmotionEntailmentTokenizer): RecconEmotionEntailmentTokenizer sgnlp
        cls_token_at_end (bool, optional): Option to use cls_token_at_end . Defaults to False.
        sep_token_extra (bool, optional): Option to use sep_token_extra. Defaults to False.
        pad_on_left (bool, optional): Option to use pad_on_left. Defaults to False.
        cls_token (str, optional): cls_token to use. Defaults to "[CLS]".
        sep_token (str, optional): sep_token to use. Defaults to "[SEP]".
        pad_token (int, optional): pad_token to use. Defaults to 0.
        cls_token_segment_id (int, optional): Option for cls_token_segment_id. Defaults to 1.
        pad_token_segment_id (int, optional): Option for pad_token_segment_id. Defaults to 0.
        silent (bool, optional): Option for silent. Defaults to False.
        add_prefix_space (bool, optional): Option to add_prefix_space. Defaults to False.
        pad_to_max_length (bool, optional): Option to pad_to_max_length. Defaults to True.

    Returns:
        List[InputFeatures]: list of InputFeatures
    """
    examples = [
        (
            example,
            max_seq_length,
            tokenizer,
            cls_token_at_end,
            cls_token,
            sep_token,
            cls_token_segment_id,
            pad_on_left,
            pad_token_segment_id,
            sep_token_extra,
            pad_token,
            add_prefix_space,
            pad_to_max_length,
        )
        for example in examples
    ]
    return [
        convert_example_to_feature(example)
        for example in tqdm(examples, disable=silent, position=0, leave=True)
    ]


def load_examples(
    examples: List[InputExample],
    max_seq_length: int,
    tokenizer: RecconEmotionEntailmentTokenizer,
) -> TensorDataset:
    """Load examples from dataframe

    Args:
        examples (List[InputExample]): list of InputFeatures
        max_seq_length (int): max sequence length
        tokenizer (RecconEmotionEntailmentTokenizer): RecconEmotionEntailmentTokenizer from sgnlp

    Returns:
        TensorDataset: A tensordataset object containing all examples
    """
    features = convert_examples_to_features(
        examples,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=True,
        pad_on_left=False,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
        add_prefix_space=True,
        pad_to_max_length=bool(len(examples) > 1),
    )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids
    )
    return dataset


def get_all_evidence_utterance_from_conversation(
    emotion: str, conversation_history: List[str]
) -> Dict[str, List[str]]:
    """Iterate through a conversation history to let each utterance be the evidence
    utterance. The last utterance is treated as the target utterance. Ouput dictionary
    is in a format which can be used with RecconEmotionEntailmentPreprocessor

    Args:
        emotion (str): Emotion of the target utterance
        conversation_history (List[str]): List of utterance in a conversation. The
                                        last utterance is used as the target utterance.

    Returns:
        Dict[str, List[str]]: Dictionary in a format that can be used with RecconEmotionEntailmentPreprocessor
            The dictionary looks like this:
            {'emotion': ['happiness'],
            'target_utterance': ['......'],
            'evidence_utterance': ['......'],
            'conversation_history': ['......']}
    """
    conversation_history_text = " ".join(conversation_history)
    target_utterance = conversation_history[-1]

    output = {
        "emotion": [],
        "target_utterance": [],
        "evidence_utterance": [],
        "conversation_history": [],
    }

    for evidence_utterance in conversation_history:
        output["emotion"].append(emotion)
        output["target_utterance"].append(target_utterance)
        output["evidence_utterance"].append(evidence_utterance)
        output["conversation_history"].append(conversation_history_text)

    return output
