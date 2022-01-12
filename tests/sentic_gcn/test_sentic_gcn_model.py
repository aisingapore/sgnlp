import unittest

import torch
from transformers import PretrainedConfig, PreTrainedModel, BertConfig, BertModel

from sgnlp.models.sentic_gcn.config import (
    SenticGCNConfig,
    SenticGCNBertConfig,
    SenticGCNEmbeddingConfig,
    SenticGCNBertEmbeddingConfig,
)
from sgnlp.models.sentic_gcn.modeling import (
    SenticGCNModel,
    SenticGCNModelOutput,
    SenticGCNBertModel,
    SenticGCNBertModelOutput,
    SenticGCNEmbeddingModel,
    SenticGCNBertEmbeddingModel,
)


DEVICE = torch.device("cpu")


class TestSenticGCNConfigTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SenticGCNConfig()

    def test_pretrained_config_base_class(self):
        self.assertTrue(issubclass(self.config.__class__, PretrainedConfig))

    def test_default_params(self):
        self.assertEqual(self.config.embed_dim, 300)
        self.assertEqual(self.config.hidden_dim, 300)
        self.assertEqual(self.config.dropout, 0.3)
        self.assertEqual(self.config.polarities_dim, 3)
        self.assertEqual(self.config.loss_function, "cross_entropy")


class TestSenticGCNBertConfigTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SenticGCNBertConfig()

    def test_pretrained_config_base_class(self):
        self.assertTrue(issubclass(self.config.__class__, PretrainedConfig))

    def test_default_params(self):
        self.assertEqual(self.config.embed_dim, 300)
        self.assertEqual(self.config.hidden_dim, 768)
        self.assertEqual(self.config.max_seq_len, 85)
        self.assertEqual(self.config.polarities_dim, 3)
        self.assertEqual(self.config.dropout, 0.3)
        self.assertEqual(self.config.loss_function, "cross_entropy")


class TestSenticGCNEmbeddingConfigTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SenticGCNEmbeddingConfig()

    def test_pretrained_config_base_class(self):
        self.assertTrue(issubclass(self.config.__class__, PretrainedConfig))

    def test_default_params(self):
        self.assertEqual(self.config.vocab_size, 17662)
        self.assertEqual(self.config.embed_dim, 300)


class TestSenticGCNBertEmbeddingConfigTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SenticGCNBertEmbeddingConfig()

    def test_pretrained_config_base_class(self):
        self.assertTrue(issubclass(self.config.__class__, PretrainedConfig))
        self.assertTrue(issubclass(self.config.__class__, BertConfig))


class TestSenticGCNModel(unittest.TestCase):
    def setUp(self) -> None:
        config = SenticGCNConfig()
        self.model = SenticGCNModel(config=config)

    def test_pretrained_model_base_class(self):
        self.assertTrue(issubclass(self.model.__class__, PreTrainedModel))

    def test_config_class(self):
        self.assertEqual(self.model.config_class, SenticGCNConfig)

    def test_base_model_prefix(self):
        self.assertEqual(self.model.base_model_prefix, "senticgcn")

    def test_forward_pass(self):
        text_indices = torch.zeros(
            [1, 10],
            dtype=torch.float32,
            device=DEVICE,
        )
        for i in range(0, 3):
            text_indices[0][i] = 1

        aspect_indices = torch.zeros([1, 10], dtype=torch.float32, device=DEVICE)
        aspect_indices[0][0] = 1

        left_indices = torch.zeros([1, 10], dtype=torch.float32, device=DEVICE)
        left_indices[0][0] = 1
        left_indices[0][1] = 1

        input_tensors = [
            text_indices,
            aspect_indices,
            left_indices,
            torch.zeros([1, 10, 300], dtype=torch.float32, device=DEVICE),
            torch.zeros([1, 3, 3], dtype=torch.float32, device=DEVICE),
        ]

        self.model.to(DEVICE)
        self.model.eval()
        result = self.model(input_tensors)

        self.assertEqual(type(result), SenticGCNModelOutput)
        self.assertEqual(type(result.logits), torch.Tensor)
        self.assertEqual(result.logits.shape, torch.Size([1, 3]))


class TestSenticGCNBertModel(unittest.TestCase):
    def setUp(self) -> None:
        config = SenticGCNBertConfig()
        self.model = SenticGCNBertModel(config=config)

    def test_pretrained_model_base_class(self):
        self.assertTrue(issubclass(self.model.__class__, PreTrainedModel))

    def test_config_class(self):
        self.assertEqual(self.model.config_class, SenticGCNBertConfig)

    def test_base_model_prefix(self):
        self.assertEqual(self.model.base_model_prefix, "senticgcnbert")

    def test_forward_pass(self):
        input_tensors = [
            torch.ones([1, 85], dtype=torch.float32, device=DEVICE),
            torch.ones([1, 85], dtype=torch.float32, device=DEVICE),
            torch.ones([1, 85], dtype=torch.float32, device=DEVICE),
            torch.ones([1, 85, 768], dtype=torch.float32, device=DEVICE),
            torch.ones([1, 85, 85], dtype=torch.float32, device=DEVICE),
        ]

        self.model.to(DEVICE)
        self.model.eval()
        result = self.model(input_tensors)

        self.assertEqual(type(result), SenticGCNBertModelOutput)
        self.assertEqual(type(result.logits), torch.Tensor)
        self.assertEqual(result.logits.shape, torch.Size([1, 3]))


class TestSenticGCNEmbeddingModel(unittest.TestCase):
    def setUp(self) -> None:
        config = SenticGCNEmbeddingConfig()
        self.model = SenticGCNEmbeddingModel(config=config)

    def test_pretrained_model_base_class(self):
        self.assertTrue(issubclass(self.model.__class__, PreTrainedModel))

    def test_config_class(self):
        self.assertEqual(self.model.config_class, SenticGCNEmbeddingConfig)

    def test_base_model_prefix(self):
        self.assertEqual(self.model.base_model_prefix, "senticgcnembedding")

    def test_forward_pass(self):
        input_tensor = torch.ones([1, 100], dtype=torch.long, device=DEVICE)
        self.model.to(DEVICE)
        self.model.eval()
        result = self.model(input_tensor)

        self.assertEqual(type(result), torch.Tensor)
        self.assertEqual(result.shape, torch.Size([1, 100, 300]))


class TestSenticGCNBertEmbeddingModel(unittest.TestCase):
    def setUp(self) -> None:
        config = SenticGCNBertEmbeddingConfig()
        self.model = SenticGCNBertEmbeddingModel(config=config)

    def test_pretrained_Bert_base_class(self):
        self.assertTrue(issubclass(self.model.__class__, BertModel))
        self.assertTrue(issubclass(self.model.__class__, PreTrainedModel))
