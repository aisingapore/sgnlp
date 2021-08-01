import pytest
import unittest
import pathlib

import pandas as pd
from torch.utils.data import TensorDataset

from sgnlp_models.models.emotion_entailment.utils import (
    convert_df_to_dataset,
    convert_df_to_examples,
    convert_example_to_feature,
    convert_examples_to_features,
    load_examples,
    InputExample,
    InputFeatures,
    RecconEmotionEntailmentData,
)
from sgnlp_models.models.emotion_entailment import RecconEmotionEntailmentTokenizer


TRAINING_OUTPUT_DIR = str(pathlib.Path(__file__).parent)


class EmotionEntailmentUtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer_name = "roberta-base"
        self.tokenizer = RecconEmotionEntailmentTokenizer
        self.example_text = "Emotion <SEP> Target Utterance. <SEP> Evidence Utterance. <SEP> Conversation history."
        self.data_path = TRAINING_OUTPUT_DIR + "/test.csv"

    def test_inputexample(self):
        inputexample = InputExample(None, None, None, None)
        attributes = ["guid", "text_a", "text_b", "label"]
        for attr in attributes:
            self.assertTrue(hasattr(inputexample, attr))

    def test_inputfeatures(self):
        inputfeatures = InputFeatures(None, None, None, None)
        attributes = ["input_ids", "input_mask", "segment_ids", "label_id"]
        for attr in attributes:
            self.assertTrue(hasattr(inputfeatures, attr))

    def test_data(self):
        data = RecconEmotionEntailmentData(None, None)
        attributes = ["dataset", "is_training", "__getitem__", "__len__"]
        for attr in attributes:
            self.assertTrue(hasattr(data, attr))

    @pytest.mark.slow
    def test_convert_example_to_feature(self):
        example = InputExample(
            0,
            self.example_text,
            None,
            1,
        )
        max_seq_length = 20
        tokenizer = self.tokenizer.from_pretrained(self.tokenizer_name)
        cls_token_at_end = False
        sep_token_extra = False
        pad_on_left = False
        cls_token = "[CLS]"
        sep_token = "[SEP]"
        pad_token = 0
        cls_token_segment_id = 1
        pad_token_segment_id = 0
        add_prefix_space = False
        pad_to_max_length = True
        example_row = [
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
        ]

        output = convert_example_to_feature(example_row)
        self.assertIsInstance(output, InputFeatures)

        input_ids = output.input_ids
        self.assertIsInstance(input_ids, list)
        self.assertIsInstance(input_ids[0], int)

        input_mask = output.input_mask
        self.assertIsInstance(input_mask, list)
        self.assertIsInstance(input_mask[0], int)
        valid_numbers = set([0, 1])
        intersection = set(input_mask).intersection(valid_numbers)
        self.assertTrue(len(intersection) > 0)
        self.assertTrue(max(input_mask) <= 1)
        self.assertTrue(min(input_mask) >= 0)

        segment_ids = output.segment_ids
        self.assertIsInstance(segment_ids, list)
        self.assertIsInstance(segment_ids[0], int)
        self.assertTrue(set(segment_ids) == set([0, 1]))

        label_id = output.label_id
        self.assertIsInstance(label_id, int)
        self.assertTrue(label_id in [0, 1])

    @pytest.mark.slow
    def test_convert_examples_to_features(self):
        example = [InputExample(0, self.example_text, None, 1)]
        max_seq_length = 20
        tokenizer = self.tokenizer.from_pretrained(self.tokenizer_name)
        output = convert_examples_to_features(example, max_seq_length, tokenizer)

        self.assertIsInstance(output, list)
        self.assertIsInstance(output[0], InputFeatures)

    @pytest.mark.slow
    def test_load_examples(self):
        example = [InputExample(0, self.example_text, None, 1)]
        max_seq_length = 20
        tokenizer = self.tokenizer.from_pretrained(self.tokenizer_name)
        output = load_examples(example, max_seq_length, tokenizer)

        self.assertIsInstance(output, TensorDataset)

    def test_convert_df_to_examples(self):
        df = pd.read_csv(self.data_path)
        output = convert_df_to_examples(df)
        self.assertIsInstance(output, list)
        self.assertIsInstance(output[0], InputExample)

        guid = output[0].guid
        self.assertIsInstance(guid, int)

        text_a = output[0].text_a
        self.assertIsInstance(text_a, str)

        text_b = output[0].text_b
        self.assertTrue(text_b is None)

        label = output[0].label
        self.assertTrue(label in [0, 1])

    @pytest.mark.slow
    def test_convert_df_to_dataset(self):
        df = pd.read_csv(self.data_path)
        max_seq_length = 20
        tokenizer = self.tokenizer.from_pretrained(self.tokenizer_name)
        output = convert_df_to_dataset(df, max_seq_length, tokenizer)

        self.assertIsInstance(output, TensorDataset)
