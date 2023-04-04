import unittest

from torch.utils.data import RandomSampler, SequentialSampler

from sgnlp.models.rumour_stance import (
    RumourVerificationPreprocessor,
    RumourVerificationTokenizer,
    StanceClassificationPreprocessor,
    StanceClassificationTokenizer,
)
from sgnlp.models.rumour_stance.modules.thread import ThreadPreprocessor


class TestThreadPreprocessor(unittest.TestCase):
    model_cfg = {
        "stance": {
            "train_config": {
                "path": "sgnlp/models/rumour_stance/stance_input/stance_train.tsv",
                "sampler": RandomSampler,
            },
            "dev_config": {
                "path": "sgnlp/models/rumour_stance/stance_input/stance_dev.tsv",
                "sampler": SequentialSampler,
            },
            "test_config": {
                "path": "sgnlp/models/rumour_stance/stance_input/stance_test.tsv",
                "sampler": SequentialSampler,
            },
        },
        "rumour": {
            "train_config": {
                "path": "sgnlp/models/rumour_stance/rumour_input/rumour_train.tsv",
                "sampler": RandomSampler,
            },
            "dev_config": {
                "path": "sgnlp/models/rumour_stance/rumour_input/rumour_dev.tsv",
                "sampler": SequentialSampler,
            },
            "test_config": {
                "path": "sgnlp/models/rumour_stance/rumour_input/rumour_test.tsv",
                "sampler": SequentialSampler,
            },
        },
    }

    def setUp(self) -> None:
        lines = [["A", "B"], "C"]
        self.inference_threads = ThreadPreprocessor.from_api(lines)
        for model_type in ("stance", "rumour"):
            self.model_cfg[model_type]["train_threads"] = ThreadPreprocessor.from_file(
                self.model_cfg[model_type]["train_config"]["path"]
            )
            self.model_cfg[model_type]["dev_threads"] = ThreadPreprocessor.from_file(
                self.model_cfg[model_type]["dev_config"]["path"]
            )
            self.model_cfg[model_type]["test_threads"] = ThreadPreprocessor.from_file(
                self.model_cfg[model_type]["test_config"]["path"]
            )

    def test_create_threads_from_inference_inputs(self):
        self.assertEqual(self.inference_threads[0].text, ["a", "b"])
        self.assertEqual(self.inference_threads[0].label, ["1", "1"])

    def test_create_threads_from_file(self):
        for model_type in ("stance", "rumour"):
            self.assertEqual(len(self.model_cfg[model_type]["train_threads"]), 272)
            self.assertEqual(len(self.model_cfg[model_type]["dev_threads"]), 25)
            self.assertEqual(len(self.model_cfg[model_type]["test_threads"]), 28)


class TestPreprocessor(unittest.TestCase):
    model_cfg = {
        "stance": {
            "preprocessor": StanceClassificationPreprocessor,
            "tokenizer": StanceClassificationTokenizer,
        },
        "rumour": {
            "preprocessor": RumourVerificationPreprocessor,
            "tokenizer": RumourVerificationTokenizer,
        },
    }

    def setUp(self) -> None:
        for model_type in ("stance", "rumour"):
            tokenizer = self.model_cfg[model_type]["tokenizer"].from_pretrained(
                "bert-base-uncased",
                do_lower_case=False,
            )
            preprocessor = self.model_cfg[model_type]["preprocessor"](
                tokenizer=tokenizer,
            )
            self.model_cfg[model_type][
                "train_dataloader"
            ] = preprocessor.get_train_dataloader()
            self.model_cfg[model_type][
                "dev_dataloader"
            ] = preprocessor.get_dev_dataloader()
            self.model_cfg[model_type][
                "test_dataloader"
            ] = preprocessor.get_test_dataloader()

    def test_get_train_dataloader(self):
        for model_type in ("stance", "rumour"):
            self.assertIsInstance(
                self.model_cfg[model_type]["train_dataloader"].sampler, RandomSampler
            )
            self.assertEqual(
                len(self.model_cfg[model_type]["train_dataloader"].dataset), 272
            )

    def test_get_dev_dataloader(self):
        for model_type in ("stance", "rumour"):
            self.assertIsInstance(
                self.model_cfg[model_type]["dev_dataloader"].sampler, SequentialSampler
            )
            self.assertEqual(
                len(self.model_cfg[model_type]["dev_dataloader"].dataset), 25
            )

    def test_get_test_dataloader(self):
        for model_type in ("stance", "rumour"):
            self.assertIsInstance(
                self.model_cfg[model_type]["test_dataloader"].sampler, SequentialSampler
            )
            self.assertEqual(
                len(self.model_cfg[model_type]["test_dataloader"].dataset), 28
            )
