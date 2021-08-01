import unittest
import pytest

import torch

from sgnlp.models.emotion_entailment import (
    RecconEmotionEntailmentConfig,
    RecconEmotionEntailmentTokenizer,
    RecconEmotionEntailmentModel,
)


class EmotionEntailmentTest(unittest.TestCase):
    def setUp(self):
        self.config = RecconEmotionEntailmentConfig
        self.tokenizer = RecconEmotionEntailmentTokenizer
        self.model = RecconEmotionEntailmentModel
        self.huggingface_pretrained_name = "roberta-base"
        self.batch_size = 13
        self.seq_length = 7
        self.vocab_size = 99
        self.input_ids = torch.randint(
            low=0, high=self.vocab_size, size=[self.batch_size, self.seq_length]
        )
        self.attention_mask = torch.randint(
            low=0, high=2, size=[self.batch_size, self.seq_length]
        )
        self.model_inputs = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }
        self.model_inputs_with_labels = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": torch.ones([self.batch_size]).type(torch.LongTensor),
        }

    def test_config_can_be_init(self):
        config = self.config()
        self.assertIsNotNone(config)

    @pytest.mark.slow
    def test_config_can_load_from_huggingface(self):
        config = self.config.from_pretrained(self.huggingface_pretrained_name)
        self.assertIsNotNone(config)

    @pytest.mark.slow
    def test_tokenizer_can_load_from_huggingface(self):
        tokenizer = self.tokenizer.from_pretrained(self.huggingface_pretrained_name)
        self.assertIsNotNone(tokenizer)

    @pytest.mark.slow
    def test_tokenizer_output(self):
        tokenizer = self.tokenizer.from_pretrained(self.huggingface_pretrained_name)
        output = tokenizer("this is a test", return_tensors="pt")

        self.assertEqual(list(output.keys()), ["input_ids", "attention_mask"])
        self.assertEqual(output["input_ids"].shape, torch.Size([1, 6]))
        self.assertIsInstance(output["input_ids"], torch.Tensor)
        self.assertEqual(output["attention_mask"].shape, torch.Size([1, 6]))
        self.assertIsInstance(output["attention_mask"], torch.Tensor)

    def test_model_can_be_init(self):
        config = self.config()
        model = self.model(config)
        self.assertIsNotNone(model)

    def test_model_forward(self):
        config = self.config()
        model = self.model(config)

        output = model(**self.model_inputs)
        expected_shape = torch.Size([self.batch_size, 2])
        self.assertIsInstance(output["logits"], torch.Tensor)
        self.assertEqual(output["logits"].shape, expected_shape)

        output_with_label = model(**self.model_inputs_with_labels)
        self.assertIsNotNone(output_with_label["loss"])
        self.assertIsInstance(output["logits"], torch.Tensor)
        self.assertEqual(output["logits"].shape, expected_shape)

    @pytest.mark.slow
    def test_from_pretrained(self):
        config = self.config.from_pretrained(
            "https://sgnlp.blob.core.windows.net/models/reccon_emotion_entailment/config.json"
        )
        model = self.model.from_pretrained(
            "https://sgnlp.blob.core.windows.net/models/reccon_emotion_entailment/pytorch_model.bin",
            config=config,
        )
        output = model(**self.model_inputs)
        expected_shape = torch.Size([self.batch_size, 2])
        self.assertIsInstance(output["logits"], torch.Tensor)
        self.assertEqual(output["logits"].shape, expected_shape)

        output_with_label = model(**self.model_inputs_with_labels)
        self.assertIsNotNone(output_with_label["loss"])
        self.assertIsInstance(output["logits"], torch.Tensor)
        self.assertEqual(output["logits"].shape, expected_shape)
