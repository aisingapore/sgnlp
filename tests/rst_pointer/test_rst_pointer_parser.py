import unittest

import pytest
import torch

from sgnlp.models.rst_pointer import RstPointerParserConfig, RstPointerParserModel


class TestRstPointerParser(unittest.TestCase):
    def setUp(self):
        self.test_sentence_lengths = [10, 15]
        self.test_tokenized_sentence_ids = torch.zeros((2, 15, 50), dtype=torch.long)
        self.test_end_boundaries = [[3, 5], [2, 7, 11]]
        self.test_parsing_breaks = [[0], [1, 0]]
        self.test_relation_label = [[10], [29, 30]]

    def test_model_and_config_init(self):
        config = RstPointerParserConfig()
        self.assertIsNotNone(config)

        model = RstPointerParserModel(config)
        self.assertIsNotNone(model)

    def test_model_forward(self):
        config = RstPointerParserConfig()
        model = RstPointerParserModel(config)

        output = model(
            self.test_tokenized_sentence_ids,
            self.test_end_boundaries,
            self.test_sentence_lengths,
            self.test_relation_label,
            self.test_parsing_breaks,
        )

        self.assertEqual(len(output.splits), 2)
        self.assertIsNotNone(output.loss_tree)
        self.assertIsNotNone(output.loss_label)

    @pytest.mark.slow
    def test_from_pretrained(self):
        parser_config = RstPointerParserConfig.from_pretrained(
            "https://storage.googleapis.com/sgnlp-models/models/rst_pointer/parser/config.json"
        )
        parser = RstPointerParserModel.from_pretrained(
            "https://storage.googleapis.com/sgnlp-models/models/rst_pointer/parser/pytorch_model.bin",
            config=parser_config,
        )

        parser_output = parser(
            self.test_tokenized_sentence_ids,
            self.test_end_boundaries,
            self.test_sentence_lengths,
            self.test_relation_label,
            self.test_parsing_breaks,
        )

        self.assertEqual(len(parser_output.splits), 2)
        self.assertIsNotNone(parser_output.loss_tree)
        self.assertIsNotNone(parser_output.loss_label)
