import pytest
import unittest

import torch
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    PreTrainedTokenizer,
    XLMRobertaConfig,
    XLMRobertaModel,
    XLMRobertaTokenizer
)

from sgnlp_models.models.ufd import (
    UFDAdaptorGlobalConfig,
    UFDAdaptorDomainConfig,
    UFDCombineFeaturesMapConfig,
    UFDClassifierConfig,
    UFDAdaptorGlobalModel,
    UFDAdaptorDomainModel,
    UFDCombineFeaturesMapModel,
    UFDClassifierModel,
    UFDMaxDiscriminatorModel,
    UFDMinDiscriminatorModel,
    UFDDeepInfoMaxLossModel,
    UFDEmbeddingConfig,
    UFDEmbeddingModel,
    UFDTokenizer,
    UFDModel
)


DEVICE = torch.device('cpu')

# Config Test Cases


class TestUFDAdaptorGlobalConfigTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.config = UFDAdaptorGlobalConfig()

    def test_pretrained_config_base_class(self):
        self.assertTrue(issubclass(self.config.__class__, PretrainedConfig))

    def test_default_params(self):
        self.assertEqual(self.config.in_dim, 1024)
        self.assertEqual(self.config.dim_hidden, 1024)
        self.assertEqual(self.config.out_dim, 1024)
        self.assertEqual(self.config.initrange, 0.1)


class TestUFDAdaptorDomainConfigTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.config = UFDAdaptorDomainConfig()

    def test_pretrained_config_base_class(self):
        self.assertTrue(issubclass(self.config.__class__, PretrainedConfig))

    def test_default_params(self):
        self.assertEqual(self.config.in_dim, 1024)
        self.assertEqual(self.config.dim_hidden, 1024)
        self.assertEqual(self.config.out_dim, 1024)
        self.assertEqual(self.config.initrange, 0.1)


class TestUFDCombineFeaturesMapConfigTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.config = UFDCombineFeaturesMapConfig()

    def test_pretrained_config_base_class(self):
        self.assertTrue(issubclass(self.config.__class__, PretrainedConfig))

    def test_default_params(self):
        self.assertEqual(self.config.embed_dim, 1024)
        self.assertEqual(self.config.initrange, 0.1)


class TestUFDClassifierConfigTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.config = UFDClassifierConfig()

    def test_pretrained_config_base_class(self):
        self.assertTrue(issubclass(self.config.__class__, PretrainedConfig))

    def test_default_params(self):
        self.assertEqual(self.config.embed_dim, 1024)
        self.assertEqual(self.config.num_class, 2)
        self.assertEqual(self.config.initrange, 0.1)


class TestUFDEmbeddingConfigTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.config = UFDEmbeddingConfig()

    def test_pretrained_config_base_class(self):
        self.assertTrue(issubclass(self.config.__class__, PretrainedConfig))
        self.assertTrue(issubclass(self.config.__class__, XLMRobertaConfig))


# Model Test Cases


class TestUFDAdaptorGlobalTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config = UFDAdaptorGlobalConfig()
        self.model = UFDAdaptorGlobalModel(config=config)

    def test_pretrained_model_base_class(self):
        self.assertTrue(issubclass(self.model.__class__, PreTrainedModel))

    def test_config_class(self):
        self.assertEqual(self.model.config_class, UFDAdaptorGlobalConfig)

    def test_base_model_prefix(self):
        self.assertEqual(self.model.base_model_prefix, 'UFDAdaptorGlobal')

    def test_forward_pass(self):
        input_tensor = torch.ones([1, 1024], dtype=torch.float32, device=DEVICE)

        self.model.to(DEVICE)
        self.model.eval()
        result = self.model(input_tensor)

        self.assertEqual(len(result), 2)
        self.assertEqual(type(result[0]), torch.Tensor)
        self.assertEqual(type(result[1]), torch.Tensor)
        self.assertEqual(result[0].shape, torch.Size([1, 1024]))
        self.assertEqual(result[1].shape, torch.Size([1, 1024]))


class TestUFDAdaptorDomainModelTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config = UFDAdaptorDomainConfig()
        self.model = UFDAdaptorDomainModel(config=config)

    def test_pretrained_model_base_class(self):
        self.assertTrue(issubclass(self.model.__class__, PreTrainedModel))

    def test_config_class(self):
        self.assertEqual(self.model.config_class, UFDAdaptorDomainConfig)

    def test_base_model_prefix(self):
        self.assertEqual(self.model.base_model_prefix, 'UFDAdaptorDomain')

    def test_forward_pass(self):
        input_tensor = torch.ones([1, 1024], dtype=torch.float32, device=DEVICE)

        self.model.to(DEVICE)
        self.model.eval()
        result = self.model(input_tensor)

        self.assertEqual(len(result), 2)
        self.assertEqual(type(result[0]), torch.Tensor)
        self.assertEqual(type(result[1]), torch.Tensor)
        self.assertEqual(result[0].shape, torch.Size([1, 1024]))
        self.assertEqual(result[1].shape, torch.Size([1, 1024]))


class TestUFDCombineFeaturesMapModelTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config = UFDCombineFeaturesMapConfig()
        self.model = UFDCombineFeaturesMapModel(config=config)

    def test_pretrained_model_base_class(self):
        self.assertTrue(issubclass(self.model.__class__, PreTrainedModel))

    def test_config_class(self):
        self.assertEqual(self.model.config_class, UFDCombineFeaturesMapConfig)

    def test_base_model_prefix(self):
        self.assertEqual(self.model.base_model_prefix, 'UFDCombineFeaturesMap')

    def test_forward_pass(self):
        input_tensor = torch.ones([1, 2 * 1024], dtype=torch.float32, device=DEVICE)

        self.model.to(DEVICE)
        self.model.eval()
        result = self.model(input_tensor)

        self.assertEqual(len(result), 1)
        self.assertEqual(type(result), torch.Tensor)
        self.assertEqual(result.shape, torch.Size([1, 1024]))


class TestUFDClassifierModelTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config = UFDClassifierConfig()
        self.model = UFDClassifierModel(config=config)

    def test_pretrained_model_base_class(self):
        self.assertTrue(issubclass(self.model.__class__, PreTrainedModel))

    def test_config_class(self):
        self.assertEqual(self.model.config_class, UFDClassifierConfig)

    def test_base_model_prefix(self):
        self.assertEqual(self.model.base_model_prefix, 'UFDClassifier')

    def test_forward_pass(self):
        input_tensor = torch.ones([1, 1024], dtype=torch.float32, device=DEVICE)

        self.model.to(DEVICE)
        self.model.eval()
        result = self.model(input_tensor)

        self.assertEqual(len(result), 1)
        self.assertEqual(type(result), torch.Tensor)
        self.assertEqual(result.shape, torch.Size([1, 2]))


class TestUFDMaxDiscriminatorModelTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.model = UFDMaxDiscriminatorModel()

    def test_forward_pass(self):
        g_input = torch.ones([1, 1024], dtype=torch.float32, device=DEVICE)
        d_input = torch.ones([1, 1024], dtype=torch.float32, device=DEVICE)

        self.model.to(DEVICE)
        self.model.eval()
        result = self.model(g_input, d_input)

        self.assertEqual(len(result), 1)
        self.assertEqual(type(result), torch.Tensor)
        self.assertEqual(result.shape, torch.Size([1, 1]))


class TestUFDMinDiscriminatorModelTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.model = UFDMinDiscriminatorModel()

    def test_forward_pass(self):
        g_input = torch.ones([1, 1024], dtype=torch.float32, device=DEVICE)
        d_input = torch.ones([1, 1024], dtype=torch.float32, device=DEVICE)

        self.model.to(DEVICE)
        self.model.eval()
        result = self.model(g_input, d_input)

        self.assertEqual(len(result), 1)
        self.assertEqual(type(result), torch.Tensor)
        self.assertEqual(result.shape, torch.Size([1, 1]))


class TestUFDDeepInfoMaxLossModelTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.model = UFDDeepInfoMaxLossModel()

    def test_forward_pass(self):
        x_input = torch.ones([1, 1024], dtype=torch.float32, device=DEVICE)
        x_n_input = torch.ones([1, 1024], dtype=torch.float32, device=DEVICE)
        f_g_input = torch.ones([1, 1024], dtype=torch.float32, device=DEVICE)
        fg_n_input = torch.ones([1, 1024], dtype=torch.float32, device=DEVICE)
        f_d_input = torch.ones([1, 1024], dtype=torch.float32, device=DEVICE)
        fd_n_input = torch.ones([1, 1024], dtype=torch.float32, device=DEVICE)
        y_g_input = torch.ones([1, 1024], dtype=torch.float32, device=DEVICE)
        y_d_input = torch.ones([1, 1024], dtype=torch.float32, device=DEVICE)
        yd_n_input = torch.ones([1, 1024], dtype=torch.float32, device=DEVICE)

        self.model.to(DEVICE)
        self.model.eval()
        result = self.model(
            x_input, x_n_input,
            f_g_input, fg_n_input,
            f_d_input, fd_n_input,
            y_g_input, y_d_input, yd_n_input
        )

        self.assertEqual(result.ndim, 0)
        self.assertEqual(type(result), torch.Tensor)


class TestUFDEmbeddingModelTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config = UFDEmbeddingConfig()
        self.model = UFDEmbeddingModel(config=config)

    def test_pretrained_XLMRobertaModel_base_class(self):
        self.assertTrue(issubclass(self.model.__class__, XLMRobertaModel))
        self.assertTrue(issubclass(self.model.__class__, PreTrainedModel))


class TestUFDModelTestCase(unittest.TestCase):
    def setUp(self) -> None:
        ad_config = UFDAdaptorDomainConfig()
        ad_model = UFDAdaptorDomainModel(config=ad_config)
        ag_config = UFDAdaptorGlobalConfig()
        ag_model = UFDAdaptorGlobalModel(config=ag_config)
        feat_maper_config = UFDCombineFeaturesMapConfig()
        feat_maper_model = UFDCombineFeaturesMapModel(config=feat_maper_config)
        classifier_config = UFDClassifierConfig()
        classifier_model = UFDClassifierModel(config=classifier_config)
        self.model = UFDModel(
            adaptor_domain=ad_model,
            adaptor_global=ag_model,
            feature_maper=feat_maper_model,
            classifier=classifier_model)

    def test_forward_pass(self):
        input_tensor = torch.ones([1, 1024], dtype=torch.float32, device=DEVICE)

        self.model.to(DEVICE)
        self.model.eval()
        result = self.model(input_tensor)

        self.assertEqual(len(result), 1)
        self.assertEqual(type(result.logits), torch.Tensor)
        self.assertEqual(result.logits.shape, torch.Size([1, 2]))


# Tokenizer

@pytest.mark.slow
class TestUFDTokenizerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = UFDTokenizer.from_pretrained("xlm-roberta-large")

    def test_pretrained_XLMRobertaTokenizer_base_class(self):
        self.assertTrue(issubclass(self.tokenizer.__class__, XLMRobertaTokenizer))
        self.assertTrue(issubclass(self.tokenizer.__class__, PreTrainedTokenizer))


if __name__ == "__main__":
    unittest.main()
