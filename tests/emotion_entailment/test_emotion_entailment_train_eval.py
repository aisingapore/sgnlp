import os
import pathlib
import shutil
import unittest
import pytest

from sgnlp.models.emotion_entailment import (
    train,
    evaluate,
    RecconEmotionEntailmentArguments,
)


TRAINING_OUTPUT_DIR = str(pathlib.Path(__file__).parent)


class EmotionEntailmentTrainTest(unittest.TestCase):
    def setUp(self):
        args = {
            "model_name": "roberta-base",
            "x_train_path": str(pathlib.Path(TRAINING_OUTPUT_DIR) / "train.csv"),
            "x_valid_path": str(pathlib.Path(TRAINING_OUTPUT_DIR) / "valid.csv"),
            "max_seq_length": 20,
            "train_args": {
                "output_dir": str(pathlib.Path(TRAINING_OUTPUT_DIR) / "output"),
                "overwrite_output_dir": True,
                "evaluation_strategy": "steps",
                "per_device_train_batch_size": 5,
                "per_device_eval_batch_size": 5,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-5,
                "weight_decay": 0,
                "adam_epsilon": 1e-8,
                "max_grad_norm": 1,
                "num_train_epochs": 1,
                "lr_scheduler_type": "linear",
                "warmup_ratio": 0,
                "save_strategy": "steps",
                "save_steps": 1,
                "no_cuda": False,
                "seed": 0,
                "fp16": False,
                "load_best_model_at_end": True,
                "report_to": "none",
            },
        }
        self.args = RecconEmotionEntailmentArguments(**args)

    @pytest.mark.slow
    def test_train(self):
        train(self.args)

        output_dir = pathlib.Path(self.args.train_args["output_dir"]) / "checkpoint-1/"

        self.assertTrue((output_dir / "config.json").exists())
        self.assertTrue((output_dir / "optimizer.pt").exists())
        self.assertTrue((output_dir / "pytorch_model.bin").exists())
        self.assertTrue((output_dir / "scheduler.pt").exists())
        self.assertTrue((output_dir / "trainer_state.json").exists())
        self.assertTrue((output_dir / "training_args.bin").exists())


class EmotionEntailmentEvalTest(unittest.TestCase):
    def setUp(self):
        args = {
            "model_name": "roberta-base",
            "max_seq_length": 20,
            "eval_args": {
                "trained_model_dir": str(
                    pathlib.Path(TRAINING_OUTPUT_DIR) / "output/checkpoint-1"
                ),
                "x_test_path": str(pathlib.Path(TRAINING_OUTPUT_DIR) / "test.csv"),
                "results_path": str(
                    pathlib.Path(TRAINING_OUTPUT_DIR)
                    / "output/classification_result.txt"
                ),
                "report_to": "none",
                "per_device_eval_batch_size": 8,
                "no_cuda": False,
            },
        }
        self.args = RecconEmotionEntailmentArguments(**args)

    @pytest.mark.slow
    def test_eval(self):
        evaluate(self.args)

        self.assertTrue(pathlib.Path(self.args.eval_args["results_path"]).exists())

    def tearDown(self):
        shutil.rmtree(str(pathlib.Path(TRAINING_OUTPUT_DIR) / "output"))
