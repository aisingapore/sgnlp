import os
import pathlib
import pytest
import shutil
import unittest

import torch

from sgnlp.models.ufd import (
    UFDAdaptorGlobalModel,
    UFDAdaptorDomainModel,
    UFDCombineFeaturesMapModel,
    UFDClassifierModel,
    UFDArguments,
)
from sgnlp.models.ufd.utils import (
    create_unsupervised_models,
    load_trained_models,
    create_classifiers,
    get_source2target_domain_mapping,
    generate_train_val_dataset,
)

PARENT_DIR = str(pathlib.Path(__file__).parent)


class TestUFDUtilsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        cfg = {
            "verbose": 1,
            "device": "cpu",
            "data_folder": PARENT_DIR + "/test_data",
            "model_folder": "https://storage.googleapis.com/sgnlp/models/ufd",
            "cache_folder": PARENT_DIR + "/test_cache",
            "embedding_model_name": "xlm-roberta-large",
            "train_args": {
                "unsupervised_dataset_filename": "raw.txt",
                "train_filename": "train.txt",
                "val_filename": "sampled.txt",
                "train_cache_filename": "train_dataset.pickle",
                "val_cache_filename": "val_dataset.pickle",
                "learning_rate": 0.00001,
                "seed": 0,
                "unsupervised_model_batch_size": 1,
                "unsupervised_epochs": 3,
                "in_dim": 32,
                "dim_hidden": 64,
                "out_dim": 24,
                "initrange": 0.1,
                "classifier_epochs": 3,
                "classifier_batch_size": 1,
                "num_class": 2,
                "source_language": "en",
                "source_domains": ["books", "dvd", "music"],
                "target_domains": ["books", "dvd", "music"],
                "target_languages": ["de", "fr", "jp"],
                "warmup_epochs": 0,
            },
            "eval_args": {
                "result_folder": "result/",
                "result_filename": "results.log",
                "test_filename": "test.txt",
                "eval_batch_size": 8,
                "config_filename": "config.json",
                "model_filename": "pytorch_model.bin",
                "source_language": "en",
                "source_domains": ["books", "dvd", "music"],
                "target_domains": ["books", "dvd", "music"],
                "target_languages": ["de", "fr", "jp"],
            },
        }
        self.cfg = UFDArguments(**cfg)

    def test_create_unsupervised_models_base_class(self):
        ufd_ad, ufd_ag, ufd_cfm = create_unsupervised_models(self.cfg)
        self.assertEqual(ufd_ad.__class__, UFDAdaptorDomainModel)
        self.assertEqual(ufd_ag.__class__, UFDAdaptorGlobalModel)
        self.assertEqual(ufd_cfm.__class__, UFDCombineFeaturesMapModel)

    def test_create_unsupervised_models_params(self):
        ufd_ad, ufd_ag, ufd_cfm = create_unsupervised_models(self.cfg)
        self.assertEqual(ufd_ad.lin1.in_features, self.cfg.train_args["in_dim"])
        self.assertEqual(ufd_ad.lin1.out_features, self.cfg.train_args["dim_hidden"])
        self.assertEqual(ufd_ad.lin2.in_features, self.cfg.train_args["dim_hidden"])
        self.assertEqual(ufd_ad.lin2.out_features, self.cfg.train_args["out_dim"])
        self.assertEqual(ufd_ag.lin1.in_features, self.cfg.train_args["in_dim"])
        self.assertEqual(ufd_ag.lin1.out_features, self.cfg.train_args["dim_hidden"])
        self.assertEqual(ufd_ag.lin2.in_features, self.cfg.train_args["dim_hidden"])
        self.assertEqual(ufd_ag.lin2.out_features, self.cfg.train_args["out_dim"])
        self.assertEqual(ufd_cfm.fc.in_features, 2 * self.cfg.train_args["in_dim"])
        self.assertEqual(ufd_cfm.fc.out_features, self.cfg.train_args["in_dim"])

    @pytest.mark.slow
    def test_load_trained_models(self):
        ufd_ad, ufd_ag, ufd_cfm, ufd_cls = load_trained_models(
            self.cfg, "books", "de", "music"
        )
        self.assertEqual(ufd_ad.__class__, UFDAdaptorDomainModel)
        self.assertEqual(ufd_ag.__class__, UFDAdaptorGlobalModel)
        self.assertEqual(ufd_cfm.__class__, UFDCombineFeaturesMapModel)
        self.assertEqual(ufd_cls.__class__, UFDClassifierModel)

    def test_create_classifiers(self):
        classifiers = create_classifiers(self.cfg)
        self.assertEqual(
            len(classifiers.keys()), len(self.cfg.train_args["source_domains"])
        )
        for k, v in classifiers.items():
            self.assertTrue(k in self.cfg.train_args["source_domains"])
            self.assertEqual(v["model"].__class__, UFDClassifierModel)
            self.assertEqual(v["criterion"].__class__, torch.nn.CrossEntropyLoss)
            self.assertEqual(v["optimizer"].__class__, torch.optim.Adam)

    def test_get_source2target_domain_mapping(self):
        mappings = get_source2target_domain_mapping(
            self.cfg.train_args["source_domains"], self.cfg.train_args["target_domains"]
        )
        self.assertEqual(
            len(mappings.keys()), len(self.cfg.train_args["source_domains"])
        )
        for k, v in mappings.items():
            self.assertTrue(k in self.cfg.train_args["source_domains"])
            self.assertTrue(len(v), len(self.cfg.train_args["target_domains"]) - 1)
            self.assertTrue(k not in v)

    @pytest.mark.slow
    def test_generate_train_val_dataset(self):
        os.makedirs(self.cfg.cache_folder, exist_ok=True)
        train_data, valid_data = generate_train_val_dataset(self.cfg)
        all_train_keys = ["raw"] + self.cfg.train_args["source_domains"]
        self.assertEqual(list(train_data.keys()).sort(), all_train_keys.sort())
        self.assertEqual(
            list(valid_data.keys()).sort(),
            self.cfg.train_args["target_languages"].sort(),
        )

        for k, v in train_data.items():
            self.assertIsNotNone(v)
            self.assertEqual(len(train_data[k]), 5)

        for k, v in valid_data.items():
            self.assertIsNotNone(v)
            for k1, v1 in valid_data[k].items():
                self.assertIsNotNone(v1)
                self.assertEqual(
                    len(valid_data[k]), len(self.cfg.train_args["target_domains"])
                )
                self.assertEqual(len(valid_data[k][k1]), 5)

        train_cache_path = pathlib.Path(self.cfg.cache_folder).joinpath(
            self.cfg.train_args["train_cache_filename"]
        )
        valid_cache_path = pathlib.Path(self.cfg.cache_folder).joinpath(
            self.cfg.train_args["val_cache_filename"]
        )

        self.assertEqual(
            (str(train_cache_path), train_cache_path.is_file()),
            (str(train_cache_path), True),
        )
        self.assertEqual(
            (str(valid_cache_path), valid_cache_path.is_file()),
            (str(valid_cache_path), True),
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.cfg.cache_folder, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
