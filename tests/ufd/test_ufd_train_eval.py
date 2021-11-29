import os
import pathlib
import pytest
import shutil
import unittest

from sgnlp.models.ufd import UFDArguments
from sgnlp.models.ufd.train import train
from sgnlp.models.ufd.eval import evaluate

PARENT_DIR = str(pathlib.Path(__file__).parent)


class TestTrainTestCase(unittest.TestCase):
    def setUp(self) -> None:
        cfg = {
            "verbose": 0,
            "device": "cpu",
            "data_folder": PARENT_DIR + "/test_data",
            "model_folder": PARENT_DIR + "/output",
            "cache_folder": PARENT_DIR + "/cache",
            "embedding_model_name": "xlm-roberta-large",
            "train_args": {
                "unsupervised_dataset_filename": "raw.txt",
                "train_filename": "train.txt",
                "val_filename": "sampled.txt",
                "train_cache_filename": "train_dataset.pickle",
                "val_cache_filename": "val_dataset.pickle",
                "learning_rate": 0.00001,
                "seed": 0,
                "unsupervised_model_batch_size": 2,
                "unsupervised_epochs": 4,
                "in_dim": 1024,
                "dim_hidden": 1024,
                "out_dim": 1024,
                "initrange": 0.1,
                "classifier_epochs": 4,
                "classifier_batch_size": 2,
                "num_class": 2,
                "source_language": "en",
                "source_domains": ["books", "dvd", "music"],
                "target_domains": ["books", "dvd", "music"],
                "target_languages": ["de", "fr", "jp"],
                "warmup_epochs": 1,
            },
            "eval_args": {
                "result_folder": PARENT_DIR + "/result",
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

    def test_train(self):
        os.makedirs(self.cfg.model_folder, exist_ok=True)
        (
            adaptor_loss_log,
            train_loss_log,
            train_acc_log,
            val_loss_log,
            val_acc_log,
            best_val_loss_log,
            best_val_acc_log,
        ) = train(self.cfg)

        self.assertIsNotNone(adaptor_loss_log)
        self.assertIsNotNone(train_loss_log)
        self.assertIsNotNone(train_acc_log)
        self.assertIsNotNone(val_loss_log)
        self.assertIsNotNone(val_acc_log)
        self.assertIsNotNone(best_val_loss_log)
        self.assertIsNotNone(best_val_acc_log)

        self.assertEqual(
            len(adaptor_loss_log), self.cfg.train_args["unsupervised_epochs"]
        )
        self.assertEqual(
            len(train_loss_log.keys()), self.cfg.train_args["unsupervised_epochs"]
        )
        self.assertEqual(
            len(train_acc_log.keys()), self.cfg.train_args["unsupervised_epochs"]
        )
        self.assertEqual(
            len(val_loss_log.keys()), self.cfg.train_args["unsupervised_epochs"]
        )
        self.assertEqual(
            len(val_acc_log.keys()), self.cfg.train_args["unsupervised_epochs"]
        )

        for ep in range(self.cfg.train_args["unsupervised_epochs"]):
            for dom in self.cfg.train_args["source_domains"]:
                self.assertIsNotNone(train_loss_log[ep][dom])
                self.assertIsNotNone(train_acc_log[ep][dom])
                self.assertTrue(
                    len(train_loss_log[ep][dom]),
                    self.cfg.train_args["unsupervised_epochs"],
                )
                self.assertTrue(
                    len(train_acc_log[ep][dom]),
                    self.cfg.train_args["unsupervised_epochs"],
                )
                self.assertTrue(
                    len(val_loss_log[ep][dom]),
                    self.cfg.train_args["unsupervised_epochs"],
                )
                self.assertTrue(
                    len(val_acc_log[ep][dom]),
                    self.cfg.train_args["unsupervised_epochs"],
                )

        self.assertEqual(len(best_val_loss_log), 18)
        self.assertEqual(len(best_val_acc_log), 18)

    @pytest.mark.slow
    def test_eval(self):
        self.cfg.model_folder = "https://storage.googleapis.com/sgnlp/models/ufd"
        self.cfg.eval_args["source_language"] = "en"
        self.cfg.eval_args["source_domains"] = ["books"]
        self.cfg.eval_args["target_languages"] = ["de"]
        self.cfg.eval_args["target_domains"] = ["music"]
        os.makedirs(self.cfg.eval_args["result_folder"], exist_ok=True)
        evaluate(self.cfg)

        result_file = (
            pathlib.Path(self.cfg.eval_args["result_folder"])
            / self.cfg.eval_args["result_filename"]
        )

        self.assertEqual(
            (str(result_file), result_file.is_file()), (str(result_file), True)
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.cfg.model_folder, ignore_errors=True)
        shutil.rmtree(self.cfg.eval_args["result_folder"], ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
