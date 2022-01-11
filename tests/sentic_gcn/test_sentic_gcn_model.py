import pytest
import unittest

from transformers import PretrainedConfig, BertConfig

from sgnlp.models.sentic_gcn import SenticGCNConfig, SenticGCNBertConfig
from sgnlp.models.sentic_gcn.config import SenticGCNBertEmbeddingConfig, SenticGCNEmbeddingConfig


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
