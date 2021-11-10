import unittest

import pytest
import torch

from sgnlp.models.rst_pointer import RstPointerSegmenterConfig, RstPointerSegmenterModel


class TestRstPointerSegmenter(unittest.TestCase):
    def setUp(self):
        self.test_sentence_lengths = [10, 15]
        self.test_tokenized_sentence_ids = torch.zeros((2, 15, 50), dtype=torch.long)
        self.test_labels = [[3, 5], [2, 7, 11]]

    def test_model_and_config_init(self):
        config = RstPointerSegmenterConfig()
        self.assertIsNotNone(config)

        model = RstPointerSegmenterModel(config)
        self.assertIsNotNone(model)

    def test_model_forward(self):
        config = RstPointerSegmenterConfig()
        model = RstPointerSegmenterModel(config)

        output = model(
            self.test_tokenized_sentence_ids,
            self.test_sentence_lengths,
            self.test_labels,
        )

        self.assertEqual(len(output.start_boundaries), 2)
        self.assertEqual(len(output.end_boundaries), 2)
        self.assertIsNotNone(output.loss)

    @pytest.mark.slow
    def test_from_pretrained(self):
        segmenter_config = RstPointerSegmenterConfig.from_pretrained(
            "https://storage.googleapis.com/sgnlp/models/rst_pointer/segmenter/config.json"
        )
        segmenter = RstPointerSegmenterModel.from_pretrained(
            "https://storage.googleapis.com/sgnlp/models/rst_pointer/segmenter/pytorch_model.bin",
            config=segmenter_config,
        )

        segmenter_output = segmenter(
            self.test_tokenized_sentence_ids,
            self.test_sentence_lengths,
            self.test_labels,
        )

        self.assertEqual(len(segmenter_output.start_boundaries), 2)
        self.assertEqual(len(segmenter_output.end_boundaries), 2)
        self.assertIsNotNone(segmenter_output.loss)
