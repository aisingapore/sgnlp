import unittest
import pathlib
import shutil
import pytest

import torch
from transformers import PretrainedConfig

from sgnlp.models.nea import (
    NEAConfig,
    NEARegPoolingModel,
    NEARegModel,
    NEABiRegModel,
    NEABiRegPoolingModel,
    NEATokenizer,
)


PARENT_DIR = pathlib.Path(__file__).parent


class NEATest(unittest.TestCase):
    def setUp(self):
        self.config = NEAConfig
        self.reg_model = NEARegModel
        self.reg_pooling_model = NEARegPoolingModel
        self.bi_reg_model = NEABiRegModel
        self.bi_reg_pooling_model = NEABiRegPoolingModel
        self.model_input = torch.ones((2, 20)).int()
        self.model_input_with_label = {
            "input_ids": self.model_input,
            "labels": torch.tensor([1, 1]),
        }

    def test_config_can_be_init(self):
        config = self.config()
        self.assertIsNotNone(config)
        self.assertIsInstance(config, PretrainedConfig)
        self.assertEqual(config.vocab_size, 4000)
        self.assertEqual(config.embedding_dim, 50)
        self.assertEqual(config.dropout, 0.5)
        self.assertEqual(config.cnn_input_dim, 0)
        self.assertEqual(config.cnn_output_dim, 0)
        self.assertEqual(config.cnn_kernel_size, 0)
        self.assertEqual(config.cnn_padding, 0)
        self.assertEqual(config.rec_layer_type, "lstm")
        self.assertEqual(config.rec_input_dim, 50)
        self.assertEqual(config.rec_output_dim, 300)
        self.assertEqual(config.aggregation, "mot")
        self.assertEqual(config.linear_input_dim, 300)
        self.assertEqual(config.linear_output_dim, 1)
        self.assertEqual(config.skip_init_bias, False)
        self.assertEqual(config.loss_function, "mse")

    def test_reg_model_can_be_init(self):
        config = self.config()
        model = self.reg_model(config=config)
        self.assertIsNotNone(model)

    def test_reg_pooling_model_can_be_init(self):
        config = self.config()
        model = self.reg_pooling_model(config=config)
        self.assertIsNotNone(model)

    def test_bi_reg_model_can_be_init(self):
        config = self.config(linear_input_dim=600)
        model = self.bi_reg_model(config=config)
        self.assertIsNotNone(model)

    def test_bi_reg_pooling_model_can_be_init(self):
        config = self.config(linear_input_dim=600)
        model = self.bi_reg_pooling_model(config=config)
        self.assertIsNotNone(model)

    def test_reg_model_forward_pass(self):
        config = self.config()
        model = self.reg_model(config=config)

        output = model(self.model_input)
        self.assertIsInstance(output["logits"], torch.Tensor)
        self.assertEqual(output["logits"].shape, torch.Size([2, 1]))

        output_with_label = model(**self.model_input_with_label)
        self.assertIsInstance(output_with_label["logits"], torch.Tensor)
        self.assertEqual(output_with_label["logits"].shape, torch.Size([2, 1]))
        self.assertIsNotNone(output_with_label["loss"])

    def test_reg_pooling_model_forward_pass(self):
        config = self.config()
        model = self.reg_pooling_model(config=config)

        output = model(self.model_input)
        self.assertIsInstance(output["logits"], torch.Tensor)
        self.assertEqual(output["logits"].shape, torch.Size([2, 1]))

        output_with_label = model(**self.model_input_with_label)
        self.assertIsInstance(output_with_label["logits"], torch.Tensor)
        self.assertEqual(output_with_label["logits"].shape, torch.Size([2, 1]))
        self.assertIsNotNone(output_with_label["loss"])

    def test_bi_reg_model_forward_pass(self):
        config = self.config(linear_input_dim=600)
        model = self.bi_reg_model(config=config)

        output = model(self.model_input)
        self.assertIsInstance(output["logits"], torch.Tensor)
        self.assertEqual(output["logits"].shape, torch.Size([2, 1]))

        output_with_label = model(**self.model_input_with_label)
        self.assertIsInstance(output_with_label["logits"], torch.Tensor)
        self.assertEqual(output_with_label["logits"].shape, torch.Size([2, 1]))
        self.assertIsNotNone(output_with_label["loss"])

    def test_bi_reg_pooling_model_forward_pass(self):
        config = self.config(linear_input_dim=600)
        model = self.bi_reg_pooling_model(config=config)

        output = model(self.model_input)
        self.assertIsInstance(output["logits"], torch.Tensor)
        self.assertEqual(output["logits"].shape, torch.Size([2, 1]))

        output_with_label = model(**self.model_input_with_label)
        self.assertIsInstance(output_with_label["logits"], torch.Tensor)
        self.assertEqual(output_with_label["logits"].shape, torch.Size([2, 1]))
        self.assertIsNotNone(output_with_label["loss"])

    @pytest.mark.slow
    def test_from_pretrained(self):
        config = self.config.from_pretrained(
            "https://sgnlp.blob.core.windows.net/models/nea/config.json"
        )
        model = self.reg_pooling_model.from_pretrained(
            "https://sgnlp.blob.core.windows.net/models/nea/pytorch_model.bin",
            config=config,
        )

        output = model(self.model_input)
        self.assertIsInstance(output["logits"], torch.Tensor)
        self.assertEqual(output["logits"].shape, torch.Size([2, 1]))

        output_with_label = model(**self.model_input_with_label)
        self.assertIsInstance(output_with_label["logits"], torch.Tensor)
        self.assertEqual(output_with_label["logits"].shape, torch.Size([2, 1]))
        self.assertIsNotNone(output_with_label["loss"])


class NEAIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.config = NEAConfig
        self.tokenizer = NEATokenizer
        self.vocab_path = PARENT_DIR / "test_data/vocab"
        self.reg_model = NEARegModel
        self.reg_pooling_model = NEARegPoolingModel
        self.bi_reg_model = NEABiRegModel
        self.bi_reg_pooling_model = NEABiRegPoolingModel

        # for initialising linear bias
        self.y_train = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5])

        # for loading embedding
        self.emb_matrix = torch.ones((4000, 50))

        # train tokenizer to get the vocab artifacts
        train_path = str(PARENT_DIR / "test_data/train.tsv")
        vocab_dir = str(self.vocab_path)
        nea_tokenizer = NEATokenizer(train_file=train_path, train_vocab=True)
        nea_tokenizer.save_pretrained(vocab_dir)

    def test_reg_model_integration(self):
        config = self.config()
        model = self.reg_model(config=config)
        model.initialise_linear_bias(self.y_train)
        model.load_pretrained_embedding(self.emb_matrix)
        tokenizer = self.tokenizer.from_pretrained(self.vocab_path)

        inputs = tokenizer("this is a test", return_tensors="pt")["input_ids"]
        output = model(inputs)
        self.assertIsInstance(output["logits"], torch.Tensor)
        self.assertEqual(output["logits"].shape, torch.Size([1, 1]))

        inputs_with_labels = {"input_ids": inputs, "labels": torch.Tensor([0.9])}
        output_with_label = model(**inputs_with_labels)
        self.assertIsInstance(output_with_label["logits"], torch.Tensor)
        self.assertEqual(output_with_label["logits"].shape, torch.Size([1, 1]))
        self.assertIsNotNone(output_with_label["loss"])

    def test_reg_pooling_model_integration(self):
        config = self.config()
        model = self.reg_pooling_model(config=config)
        model.initialise_linear_bias(self.y_train)
        model.load_pretrained_embedding(self.emb_matrix)
        tokenizer = self.tokenizer.from_pretrained(self.vocab_path)

        inputs = tokenizer("this is a test", return_tensors="pt")["input_ids"]
        output = model(inputs)
        self.assertIsInstance(output["logits"], torch.Tensor)
        self.assertEqual(output["logits"].shape, torch.Size([1, 1]))

        inputs_with_labels = {"input_ids": inputs, "labels": torch.Tensor([0.9])}
        output_with_label = model(**inputs_with_labels)
        self.assertIsInstance(output_with_label["logits"], torch.Tensor)
        self.assertEqual(output_with_label["logits"].shape, torch.Size([1, 1]))
        self.assertIsNotNone(output_with_label["loss"])

    def test_bi_reg_model_integration(self):
        config = self.config(linear_input_dim=600)
        model = self.bi_reg_model(config=config)
        model.initialise_linear_bias(self.y_train)
        model.load_pretrained_embedding(self.emb_matrix)
        tokenizer = self.tokenizer.from_pretrained(self.vocab_path)

        inputs = tokenizer("this is a test", return_tensors="pt")["input_ids"]
        output = model(inputs)
        self.assertIsInstance(output["logits"], torch.Tensor)
        self.assertEqual(output["logits"].shape, torch.Size([1, 1]))

        inputs_with_labels = {"input_ids": inputs, "labels": torch.Tensor([0.9])}
        output_with_label = model(**inputs_with_labels)
        self.assertIsInstance(output_with_label["logits"], torch.Tensor)
        self.assertEqual(output_with_label["logits"].shape, torch.Size([1, 1]))
        self.assertIsNotNone(output_with_label["loss"])

    def test_bi_reg_pooling_model_integration(self):
        config = self.config(linear_input_dim=600)
        model = self.bi_reg_pooling_model(config=config)
        model.initialise_linear_bias(self.y_train)
        model.load_pretrained_embedding(self.emb_matrix)
        tokenizer = self.tokenizer.from_pretrained(self.vocab_path)

        inputs = tokenizer("this is a test", return_tensors="pt")["input_ids"]
        output = model(inputs)
        self.assertIsInstance(output["logits"], torch.Tensor)
        self.assertEqual(output["logits"].shape, torch.Size([1, 1]))

        inputs_with_labels = {"input_ids": inputs, "labels": torch.Tensor([0.9])}
        output_with_label = model(**inputs_with_labels)
        self.assertIsInstance(output_with_label["logits"], torch.Tensor)
        self.assertEqual(output_with_label["logits"].shape, torch.Size([1, 1]))
        self.assertIsNotNone(output_with_label["loss"])

    def tearDown(self) -> None:
        shutil.rmtree(self.vocab_path)
