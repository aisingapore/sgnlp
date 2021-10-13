import pathlib
import shutil
import unittest

import pytest

from sgnlp.models.span_extraction import (
    train_model,
    evaluate,
    RecconSpanExtractionArguments,
)

TRAINING_OUTPUT_DIR = str(pathlib.Path(__file__).parent)


class SpanExtractionTrainTest(unittest.TestCase):
    def setUp(self):
        args = {
            "model_name": "mrm8488/spanbert-finetuned-squadv2",
            "train_data_path": str(pathlib.Path(TRAINING_OUTPUT_DIR) / "train.json"),
            "val_data_path": str(pathlib.Path(TRAINING_OUTPUT_DIR) / "valid.json"),
            "test_data_path": str(pathlib.Path(TRAINING_OUTPUT_DIR) / "test.json"),
            "max_seq_length": 20,
            "doc_stride": 20,
            "max_query_length": 20,
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
                "warmup_ratio": 0.06,
                "no_cuda": False,
                "seed": 0,
                "fp16": False,
                "load_best_model_at_end": True,
                "label_names": ["start_positions", "end_positions"],
                "report_to": "none",
            },
        }
        self.args = RecconSpanExtractionArguments(**args)

    @pytest.mark.slow
    def test_train(self):
        train_model(self.args)

        output_dir = pathlib.Path(self.args.train_args["output_dir"]) / "checkpoint-1/"

        self.assertTrue((output_dir / "config.json").exists())
        self.assertTrue((output_dir / "optimizer.pt").exists())
        self.assertTrue((output_dir / "pytorch_model.bin").exists())
        self.assertTrue((output_dir / "scheduler.pt").exists())
        self.assertTrue((output_dir / "trainer_state.json").exists())
        self.assertTrue((output_dir / "training_args.bin").exists())


class SpanExtractionEvalTest(unittest.TestCase):
    def setUp(self):
        args = {
            "model_name": "mrm8488/spanbert-finetuned-squadv2",
            "test_data_path": str(pathlib.Path(TRAINING_OUTPUT_DIR) / "test.json"),
            "eval_args": {
                "trained_model_dir": str(
                    pathlib.Path(TRAINING_OUTPUT_DIR) / "output/checkpoint-1"
                ),
                "results_path": str(pathlib.Path(TRAINING_OUTPUT_DIR) / "output"),
                "batch_size": 5,
                "n_best_size": 20,
                "null_score_diff_threshold": 0.0,
                "sliding_window": False,
                "no_cuda": False,
                "max_answer_length": 200,
                "report_to": "none",
            },
        }
        self.args = RecconSpanExtractionArguments(**args)

    @pytest.mark.slow
    def test_eval(self):
        evaluate(self.args)

        path = pathlib.Path(self.args.eval_args["results_path"])
        self.assertTrue((path / "nbest_predictions_text.json").exists())
        self.assertTrue((path / "null_odds_text.json").exists())
        self.assertTrue((path / "predictions_text.json").exists())
        self.assertTrue((path / "results.txt").exists())

    def tearDown(self):
        shutil.rmtree(str(pathlib.Path(TRAINING_OUTPUT_DIR) / "output"))
