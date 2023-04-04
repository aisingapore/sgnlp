import unittest

import torch

from sgnlp.models.rumour_stance import (
    RumourVerificationConfig,
    RumourVerificationModel,
    RumourVerificationPostprocessor,
    RumourVerificationPreprocessor,
    RumourVerificationTokenizer,
    StanceClassificationConfig,
    StanceClassificationModel,
    StanceClassificationPostprocessor,
    StanceClassificationPreprocessor,
    StanceClassificationTokenizer,
)
from sgnlp.models.rumour_stance.utils import set_device_and_seed


class TestPostprocessor(unittest.TestCase):
    model_cfg = {
        "stance": {
            "config": StanceClassificationConfig,
            "model": StanceClassificationModel,
            "preprocessor": StanceClassificationPreprocessor,
            "postprocessor": StanceClassificationPostprocessor,
            "tokenizer": StanceClassificationTokenizer,
            "config_file": "https://storage.googleapis.com/sgnlp/models/rumour_stance/stance_classification/config.json",
            "model_file": "https://storage.googleapis.com/sgnlp/models/rumour_stance/stance_classification/pytorch_model.bin",
        },
        "rumour": {
            "config": RumourVerificationConfig,
            "model": RumourVerificationModel,
            "preprocessor": RumourVerificationPreprocessor,
            "postprocessor": RumourVerificationPostprocessor,
            "tokenizer": RumourVerificationTokenizer,
            "config_file": "https://storage.googleapis.com/sgnlp/models/rumour_stance/rumour_verification/config.json",
            "model_file": "https://storage.googleapis.com/sgnlp/models/rumour_stance/rumour_verification/pytorch_model.bin",
        },
    }

    def setUp(self) -> None:
        set_device_and_seed()

        for model_type in ("stance", "rumour"):
            config = self.model_cfg[model_type]["config"].from_pretrained(
                self.model_cfg[model_type]["config_file"]
            )

            model = self.model_cfg[model_type]["model"].from_pretrained(
                self.model_cfg[model_type]["model_file"],
                config=config,
            )
            tokenizer = self.model_cfg[model_type]["tokenizer"].from_pretrained(
                "bert-base-uncased",
                do_lower_case=False,
            )

            preprocessor = self.model_cfg[model_type]["preprocessor"](
                tokenizer=tokenizer
            )

            postprocessor = self.model_cfg[model_type]["postprocessor"]()

            inputs = [
                "This is a rumor",
                "Yes, I agree.",
            ]

            processed_inputs = preprocessor(inputs)

            with torch.no_grad():
                outputs = model(*processed_inputs)

            if model_type == "stance":
                self.model_cfg[model_type]["predictions"] = postprocessor(
                    inputs, outputs
                )
            else:
                self.model_cfg[model_type]["predictions"] = postprocessor(outputs)

    def test_call(self):
        self.assertIn(
            self.model_cfg["rumour"]["predictions"]["pred"], {"FALSE", "TRUE", "UNVERIFIED"}
        )

        self.assertEqual(len(self.model_cfg["stance"]["predictions"]["preds"]), 2)
        for pred in self.model_cfg["stance"]["predictions"]["preds"]:
            self.assertIn(pred, {"DENY", "SUPPORT", "QUERY", "COMMENT"})
