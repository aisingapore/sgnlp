import unittest
import pytest
import pickle
import torch
import pathlib

from sgnlp.models.lsr import (
    LsrConfig,
    LsrModel
)

DIR = pathlib.Path(__file__).parent


class TestLsr(unittest.TestCase):
    def setUp(self):
        with open(DIR / 'test_data/sample_preprocessed_input.pickle', 'rb') as f:
            self.test_input_with_labels = pickle.load(f)

        self.test_input_without_labels = self.test_input_with_labels.copy()
        self.test_input_without_labels.pop("relation_multi_label", None)

    def test_model_and_config_init(self):
        config = LsrConfig()
        self.assertIsNotNone(config)

        model = LsrModel(config)
        self.assertIsNotNone(model)

    def test_model_forward_without_labels(self):
        config = LsrConfig()
        model = LsrModel(config)

        output = model(**self.test_input_without_labels)
        self.assertEqual(output.prediction.shape, torch.Size([1, 156, 97]))
        self.assertIsNone(output.loss)

    def test_model_forward_with_labels(self):
        config = LsrConfig()
        model = LsrModel(config)

        output = model(**self.test_input_with_labels)
        self.assertEqual(output.prediction.shape, torch.Size([1, 156, 97]))
        self.assertIsNotNone(output.loss)

    @pytest.mark.slow
    def test_from_pretrained(self):
        config = LsrConfig.from_pretrained("https://sgnlp.blob.core.windows.net/models/lsr/config.json")
        model = LsrModel.from_pretrained("https://sgnlp.blob.core.windows.net/models/lsr/pytorch_model.bin",
                                         config=config)

        output = model(**self.test_input_with_labels)
        self.assertEqual(output.prediction.shape, torch.Size([1, 156, 97]))
        self.assertIsNotNone(output.loss)
