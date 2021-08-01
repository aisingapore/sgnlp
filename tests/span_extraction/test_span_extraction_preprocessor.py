import pytest
import unittest

import torch
from transformers import BatchEncoding
from transformers.data.processors.squad import SquadExample, SquadFeatures

from sgnlp.models.span_extraction import (
    RecconSpanExtractionPreprocessor,
)


class SpanExtractionTestPreprocessor(unittest.TestCase):
    @pytest.mark.slow
    def test_preprocessor(self):
        test_input = {
            "emotion": ["happiness", "sadness"],
            "target_utterance": ["this is target 1", "this is target 2"],
            "evidence_utterance": ["this is evidence 1", "this is evidence 2"],
            "conversation_history": [
                "this is conversation history 1",
                "this is conversation history 2",
            ],
        }
        preprocessor = RecconSpanExtractionPreprocessor()
        output, evidence, examples, features = preprocessor(test_input)

        expected_shape = torch.Size([2, 512])
        batch_size = len(test_input["emotion"])

        self.assertIsInstance(output, BatchEncoding)
        self.assertTrue(
            set(output.keys()) == set(["input_ids", "attention_mask", "token_type_ids"])
        )
        self.assertIsInstance(output["input_ids"], torch.Tensor)
        self.assertIsInstance(output["attention_mask"], torch.Tensor)
        self.assertIsInstance(output["token_type_ids"], torch.Tensor)
        self.assertEqual(output["input_ids"].shape, expected_shape)
        self.assertEqual(output["attention_mask"].shape, expected_shape)
        self.assertEqual(output["token_type_ids"].shape, expected_shape)

        self.assertIsInstance(evidence, list)
        self.assertTrue(len(evidence) == batch_size)
        self.assertIsInstance(evidence[0]["evidence"], str)
        self.assertIsInstance(evidence[0]["id"], int)

        self.assertIsInstance(examples, list)
        self.assertTrue(len(examples) == batch_size)
        self.assertIsInstance(examples[0], SquadExample)

        self.assertIsInstance(features, list)
        self.assertTrue(len(features) == batch_size)
        self.assertIsInstance(features[0], SquadFeatures)
