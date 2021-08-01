from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class RecconEmotionEntailmentArguments:
    model_name: str = field(
        default="roberta-base",
        metadata={"help": "Pretrained model to use for training"},
    )
    x_train_path: str = field(
        default="data/dailydialog_classification_train_with_context.csv",
        metadata={"help": "Path of training data"},
    )
    x_valid_path: str = field(
        default="data/dailydialog_classification_valid_with_context.csv",
        metadata={"help": "Path of validation data"},
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "Batch size for training"},
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"},
    )
    train_args: Dict[str, Any] = field(
        default_factory=lambda: {
            "output_dir": "output/",
            "overwrite_output_dir": True,
            "evaluation_strategy": "steps",
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-5,
            "weight_decay": 0,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1,
            "num_train_epochs": 12,
            "lr_scheduler_type": "linear",
            "warmup_ratio": 0.06,
            "no_cuda": False,
            "seed": 0,
            "fp16": False,
            "load_best_model_at_end": True,
            "report_to": "none",
        },
        metadata={"help": "Arguments for training Reccon Emotion Entailment models."},
    )
    eval_args: Dict[str, Any] = field(
        default_factory=lambda: {
            "trained_model_dir": "output/",
            "x_test_path": "data/dailydialog_classification_test_with_context.csv",
            "results_path": "output/classification_result.txt",
            "per_device_eval_batch_size": 8,
            "no_cuda": False,
        },
        metadata={"help": "Arguments for evaluating Reccon Emotion Entailment models."},
    )

    def __post_init__(self):
        # Model
        assert self.model_name in [
            "roberta-base",
            "roberta-large",
        ], "Invalid model type!"

        # Training
        assert self.batch_size > 0, "batch_size must be positive!"
        assert self.max_seq_length > 0, "max_seq_length must be positive!"
        assert isinstance(
            self.train_args, dict
        ), "train_args must be represented as a Dictionary."
        assert self.train_args["seed"] >= 0, "Random seed must be positive!"
        assert (
            self.train_args["num_train_epochs"] > 0
        ), "num_train_epochs must be at least 1."
        assert (
            self.train_args["per_device_train_batch_size"] > 0
        ), "per_device_train_batch_size must be at least 1."
        assert (
            self.train_args["per_device_eval_batch_size"] > 0
        ), "per_device_eval_batch_size must be at least 1."
        assert self.train_args["learning_rate"] > 0, "learning_rate must be positive."
        assert self.train_args["warmup_ratio"] >= 0, "warmup_ratio must be at least 0."
        assert self.train_args["max_grad_norm"] > 0, "max_grad_norm must be positive."

        # Eval
        assert isinstance(
            self.eval_args, dict
        ), "eval_args must be represented as a Dictionary."
        assert (
            self.train_args["per_device_eval_batch_size"] > 0
        ), "per_device_eval_batch_size must be at least 1."
