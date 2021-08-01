import unittest
import pytest

import torch

from sgnlp.models.lif_3way_ap import (
    LIF3WayAPConfig,
    LIF3WayAPModel
)


class LIF3WayAPTest(unittest.TestCase):
    def setUp(self):
        self.test_input = {
            "word_context": torch.tensor([[1, 2, 3]]),
            "word_qa": torch.tensor([[1, 2, 3]]),
            "word_candidate": torch.tensor([[1, 2, 3]]),
            "char_context": torch.tensor([[[1, 2, 3, 4, 5], [2, 2, 4, 4, 0], [3, 3, 3, 3, 0]]]),
            "char_qa": torch.tensor([[[1, 2, 3, 4, 5], [2, 2, 4, 4, 0], [3, 3, 3, 3, 0]]]),
            "char_candidate": torch.tensor([[[1, 2, 3, 4, 5], [2, 2, 4, 4, 0], [3, 3, 3, 3, 0]]]),
        }

        self.test_input_with_label = {
            **self.test_input,
            "label": torch.tensor([[1]])
        }

    def test_model_and_config_init(self):
        config = LIF3WayAPConfig()
        self.assertIsNotNone(config)

        model = LIF3WayAPModel(config)
        self.assertIsNotNone(model)

    def test_model_forward(self):
        config = LIF3WayAPConfig()
        model = LIF3WayAPModel(config)

        output = model(**self.test_input)
        self.assertEqual(output["label_logits"].shape, torch.Size([1]))
        self.assertEqual(output["label_probs"].shape, torch.Size([1]))

        output_with_label = model(**self.test_input_with_label)
        self.assertEqual(output_with_label["label_logits"].shape, torch.Size([1]))
        self.assertEqual(output_with_label["label_probs"].shape, torch.Size([1]))
        self.assertIsNotNone(output_with_label["loss"])

    @pytest.mark.slow
    def test_from_pretrained(self):
        config = LIF3WayAPConfig.from_pretrained("")
        model = LIF3WayAPModel.from_pretrained("", config=config)

        output = model(**self.test_input)
        self.assertEqual(output["label_logits"].shape, torch.Size([1]))
        self.assertEqual(output["label_probs"].shape, torch.Size([1]))

        output_with_label = model(**self.test_input_with_label)
        self.assertEqual(output_with_label["label_logits"].shape, torch.Size([1]))
        self.assertEqual(output_with_label["label_probs"].shape, torch.Size([1]))
        self.assertIsNotNone(output_with_label["loss"])
