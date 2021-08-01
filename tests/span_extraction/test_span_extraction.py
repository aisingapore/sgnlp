import unittest
import pytest

import torch

from sgnlp.models.span_extraction import (
    RecconSpanExtractionConfig,
    RecconSpanExtractionTokenizer,
    RecconSpanExtractionModel,
)


class SpanExtractionTest(unittest.TestCase):
    def setUp(self):
        self.config = RecconSpanExtractionConfig
        self.tokenizer = RecconSpanExtractionTokenizer
        self.model = RecconSpanExtractionModel
        self.huggingface_pretrained_name = "mrm8488/spanbert-finetuned-squadv2"
        self.batch_size = 13
        self.seq_length = 7
        self.vocab_size = 99
        self.input_ids = torch.randint(
            low=0, high=self.vocab_size, size=[self.batch_size, self.seq_length]
        )
        self.attention_mask = torch.randint(
            low=0, high=2, size=[self.batch_size, self.seq_length]
        )
        self.token_type_ids = torch.randint(
            low=0, high=2, size=[self.batch_size, self.seq_length]
        )
        self.model_inputs = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "token_type_ids": self.token_type_ids,
        }
        self.model_inputs_with_labels = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "token_type_ids": self.token_type_ids,
            "start_positions": torch.zeros([self.batch_size]).type(torch.LongTensor),
            "end_positions": torch.ones([self.batch_size]).type(torch.LongTensor),
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

        self.assertEqual(
            list(output.keys()), ["input_ids", "token_type_ids", "attention_mask"]
        )
        self.assertEqual(output["input_ids"].shape, torch.Size([1, 6]))
        self.assertIsInstance(output["input_ids"], torch.Tensor)
        self.assertEqual(output["attention_mask"].shape, torch.Size([1, 6]))
        self.assertIsInstance(output["attention_mask"], torch.Tensor)
        self.assertEqual(output["token_type_ids"].shape, torch.Size([1, 6]))
        self.assertIsInstance(output["token_type_ids"], torch.Tensor)

    def test_model_can_be_init(self):
        config = self.config()
        model = self.model(config)
        self.assertIsNotNone(model)

    def test_model_forward(self):
        config = self.config()
        model = self.model(config)

        output = model(**self.model_inputs)
        expected_shape = torch.Size([self.batch_size, self.seq_length])
        self.assertIsInstance(output["start_logits"], torch.Tensor)
        self.assertEqual(output["start_logits"].shape, expected_shape)
        self.assertIsInstance(output["end_logits"], torch.Tensor)
        self.assertEqual(output["end_logits"].shape, expected_shape)

        output_with_label = model(**self.model_inputs_with_labels)
        self.assertIsNotNone(output_with_label["loss"])
        self.assertIsInstance(output_with_label["start_logits"], torch.Tensor)
        self.assertEqual(output_with_label["start_logits"].shape, expected_shape)
        self.assertIsInstance(output_with_label["end_logits"], torch.Tensor)
        self.assertEqual(output_with_label["end_logits"].shape, expected_shape)

    @pytest.mark.slow
    def test_from_pretrained(self):
        config = self.config.from_pretrained(
            "https://sgnlp.blob.core.windows.net/models/reccon_span_extraction/config.json"
        )
        model = self.model.from_pretrained(
            "https://sgnlp.blob.core.windows.net/models/reccon_span_extraction/pytorch_model.bin",
            config=config,
        )
        output = model(**self.model_inputs)
        expected_shape = torch.Size([self.batch_size, self.seq_length])
        self.assertIsInstance(output["start_logits"], torch.Tensor)
        self.assertEqual(output["start_logits"].shape, expected_shape)
        self.assertIsInstance(output["end_logits"], torch.Tensor)
        self.assertEqual(output["end_logits"].shape, expected_shape)

        output_with_label = model(**self.model_inputs_with_labels)
        self.assertIsNotNone(output_with_label["loss"])
        self.assertIsInstance(output_with_label["start_logits"], torch.Tensor)
        self.assertEqual(output_with_label["start_logits"].shape, expected_shape)
        self.assertIsInstance(output_with_label["end_logits"], torch.Tensor)
        self.assertEqual(output_with_label["end_logits"].shape, expected_shape)
