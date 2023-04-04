import unittest

from sgnlp.models.rumour_stance import (
    RumourVerificationModel,
    RumourVerificationPreprocessor,
    RumourVerificationTokenizer,
    StanceClassificationModel,
    StanceClassificationPreprocessor,
    StanceClassificationTokenizer,
)
from sgnlp.models.rumour_stance.modules.optimization import BertAdam
from sgnlp.models.rumour_stance.train import (
    RumourVerificationModelTrainer,
    StanceClassificationModelTrainer,
)
from sgnlp.models.rumour_stance.utils import (
    load_rumour_verification_config,
    load_stance_classification_config,
)


class TestModelTrainer(unittest.TestCase):

    model_cfg = {
        "stance": {
            "model": StanceClassificationModel,
            "trainer": StanceClassificationModelTrainer,
            "preprocessor": StanceClassificationPreprocessor,
            "tokenizer": StanceClassificationTokenizer,
            "load_config": load_stance_classification_config,
        },
        "rumour": {
            "model": RumourVerificationModel,
            "trainer": RumourVerificationModelTrainer,
            "preprocessor": RumourVerificationPreprocessor,
            "tokenizer": RumourVerificationTokenizer,
            "load_config": load_rumour_verification_config,
        },
    }

    def _get_models_and_train_configs(self, model_type: str):
        config = self.model_cfg[model_type]["load_config"]()
        trainer = self.model_cfg[model_type]["trainer"](config)
        model = trainer._create_model()
        tokenizer = self.model_cfg[model_type]["tokenizer"].from_pretrained(
            "bert-base-uncased",
            do_lower_case=False,
        )
        preprocessor = self.model_cfg[model_type]["preprocessor"](
            tokenizer=tokenizer,
        )
        _, train_configs = trainer._get_dataloader_and_configs(
            model=model,
            preprocessor=preprocessor,
        )
        return model, train_configs

    def setUp(self) -> None:
        for model_type in ("stance", "rumour"):
            model, train_configs = self._get_models_and_train_configs(model_type)
            self.model_cfg[model_type]["init_model"] = model
            self.model_cfg[model_type]["train_configs"] = train_configs

    def test_create_model(self):
        for model_type in ("stance", "rumour"):
            self.assertIsInstance(
                self.model_cfg[model_type]["init_model"],
                self.model_cfg[model_type]["model"],
            )

    def test_get_dataloader_and_configs(self):
        for model_type in ("stance", "rumour"):
            model_train_configs = self.model_cfg[model_type]["train_configs"]
            self.assertIsInstance(model_train_configs["optimizer"], BertAdam)
            self.assertIsInstance(model_train_configs["num_train_steps"], int)
            self.assertIsInstance(model_train_configs["t_total"], int)
