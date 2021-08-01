import pytest
import unittest

import torch
from transformers import BatchEncoding

from sgnlp.models.emotion_entailment import (
    RecconEmotionEntailmentPreprocessor,
)


class EmotionEntailmentTestPreprocessor(unittest.TestCase):
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
        preprocessor = RecconEmotionEntailmentPreprocessor()
        output = preprocessor(test_input)
        expected_shape = torch.Size([2, 512])

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
