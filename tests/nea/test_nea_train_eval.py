import pathlib
import shutil
import unittest
import pytest

from sgnlp_models.models.nea import train, evaluate, NEAArguments


PARENT_DIR = pathlib.Path(__file__).parent


class NEATrainTest(unittest.TestCase):
    def setUp(self):
        args = {
            "model_type": "regp",
            "emb_path": str(PARENT_DIR / "test_data/embeddings.w2v.txt"),
            "preprocess_data_args": {
                "train_path": str(PARENT_DIR / "test_data/train.tsv"),
                "dev_path": str(PARENT_DIR / "test_data/dev.tsv"),
                "test_path": str(PARENT_DIR / "test_data/test.tsv"),
                "prompt_id": 1,
                "maxlen": 0,
                "to_lower": True,
                "score_index": 6,
            },
            "train_args": {
                "output_dir": str(PARENT_DIR / "test_output/"),
                "overwrite_output_dir": True,
                "seed": 0,
                "num_train_epochs": 1,
                "per_device_train_batch_size": 32,
                "per_device_eval_batch_size": 32,
                "learning_rate": 0.001,
                "optimizer_type": "rmsprop",
                "optimizer_epsilon": 1e-6,
                "logging_strategy": "epoch",
                "evaluation_strategy": "epoch",
                "save_total_limit": 3,
                "no_cuda": False,
                "metric_for_best_model": "qwk",
                "load_best_model_at_end": True,
                "report_to": "none",
            },
            "tokenizer_args": {
                "vocab_train_file": str(PARENT_DIR / "test_data/train.tsv"),
                "save_folder": str(PARENT_DIR / "nea_tokenizer"),
            },
        }
        self.args = NEAArguments(**args)

    @pytest.mark.slow
    def test_train(self):
        train(self.args)

        output_dir = pathlib.Path(self.args.train_args["output_dir"])
        nea_tokenizer_path = pathlib.Path(self.args.tokenizer_args["save_folder"])

        self.assertTrue(pathlib.Path(output_dir / "config.json").exists())
        self.assertTrue(pathlib.Path(output_dir / "training_args.bin").exists())
        self.assertTrue(pathlib.Path(output_dir / "pytorch_model.bin").exists())

        self.assertTrue(
            pathlib.Path(nea_tokenizer_path / "special_tokens_map.json").exists()
        )
        self.assertTrue(
            pathlib.Path(nea_tokenizer_path / "tokenizer_config.json").exists()
        )
        self.assertTrue(pathlib.Path(nea_tokenizer_path / "vocab.txt").exists())


class NEAEvalTest(unittest.TestCase):
    def setUp(self):
        args = {
            "model_type": "regp",
            "emb_path": str(PARENT_DIR / "test_data/embeddings.w2v.txt"),
            "preprocess_data_args": {
                "train_path": str(PARENT_DIR / "test_data/train.tsv"),
                "dev_path": str(PARENT_DIR / "test_data/dev.tsv"),
                "test_path": str(PARENT_DIR / "test_data/test.tsv"),
                "prompt_id": 1,
                "vocab_size": 4000,
                "maxlen": 0,
                "to_lower": True,
                "score_index": 6,
            },
            "eval_args": {
                "results_path": str(PARENT_DIR / "test_output/result.txt"),
                "trainer_args": {
                    "output_dir": str(PARENT_DIR / "test_output/"),
                    "report_to": "none",
                },
            },
            "tokenizer_args": {
                "vocab_train_file": str(PARENT_DIR / "test_data/train.tsv"),
                "save_folder": str(PARENT_DIR / "nea_tokenizer"),
            },
        }
        self.args = NEAArguments(**args)

    @pytest.mark.slow
    def test_eval(self):
        evaluate(self.args)

        self.assertTrue(pathlib.Path(self.args.eval_args["results_path"]).exists())

    def tearDown(self):
        shutil.rmtree(self.args.eval_args["trainer_args"]["output_dir"])
        shutil.rmtree(self.args.tokenizer_args["save_folder"])
