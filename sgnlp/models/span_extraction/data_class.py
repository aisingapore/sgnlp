from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class RecconSpanExtractionArguments:
    model_name: str = field(
        default="mrm8488/spanbert-finetuned-squadv2",
        metadata={"help": "Pretrained model to use for training"},
    )
    train_data_path: str = field(
        default="data/subtask1/fold1/dailydialog_qa_train_with_context.json",
        metadata={"help": "Path of training data"},
    )
    val_data_path: str = field(
        default="data/subtask1/fold1/dailydialog_qa_valid_with_context.json",
        metadata={"help": "Path of validation data"},
    )
    test_data_path: str = field(
        default="data/subtask1/fold1/dailydialog_qa_test_with_context.json",
        metadata={"help": "Path of validation data"},
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"},
    )
    doc_stride: int = field(
        default=512,
        metadata={"help": "Document stride"},
    )
    max_query_length: int = field(
        default=512,
        metadata={"help": "Maximum query length"},
    )

    train_args: Dict[str, Any] = field(
        default_factory=lambda: {
            "output_dir": "output/",
            "overwrite_output_dir": True,
            "evaluation_strategy": "steps",
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-5,
            "weight_decay": 0,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1,
            "num_train_epochs": 12,
            "warmup_ratio": 0.06,
            "no_cuda": False,
            "seed": 0,
            "fp16": False,
            "load_best_model_at_end": True,
            "label_names": ["start_positions", "end_positions"],
            "report_to": "none",
        },
        metadata={"help": "Arguments for training Reccon Span Extraction models."},
    )
    eval_args: Dict[str, Any] = field(
        default_factory=lambda: {
            "trained_model_dir": "output/",
            "results_path": "result/",
            "batch_size": 16,
            "n_best_size": 20,
            "null_score_diff_threshold": 0.0,
            "sliding_window": False,
            "no_cuda": False,
            "max_answer_length": 200,
        },
        metadata={"help": "Arguments for evaluating Reccon Span Extraction models."},
    )

    def __post_init__(self):
        # Model
        assert self.model_name in [
            "mrm8488/spanbert-finetuned-squadv2",
            "roberta-base",
        ], "Invalid model type!"

        # Training
        assert self.max_seq_length > 0, "max_seq_length must be positive."
        assert self.doc_stride > 0, "doc_stride must be positive."
        assert self.max_query_length > 0, "max_query_length must be positive."

        assert isinstance(
            self.train_args, Dict
        ), "train_args must be represented as a Dictionary."
        assert self.train_args["seed"] >= 0, "Random seed must be at least 0."
        assert (
            self.train_args["num_train_epochs"] > 0
        ), "num_train_epochs must be at least 1."
        assert (
            self.train_args["per_device_train_batch_size"] > 0
        ), "per_device_train_batch_size must be at least 1."
        assert (
            self.train_args["per_device_eval_batch_size"] > 0
        ), "per_device_eval_batch_size must be at least 1."
        assert (
            self.train_args["gradient_accumulation_steps"] > 0
        ), "gradient_accumulation_steps must be positive."
        assert self.train_args["learning_rate"] > 0, "learning_rate must be positive."
        assert self.train_args["warmup_ratio"] >= 0, "warmup_ratio must be at least 0."
        assert self.train_args["weight_decay"] >= 0, "weight_decay must be at least 0."
        assert self.train_args["max_grad_norm"] > 0, "max_grad_norm must be positive."
        assert self.train_args["adam_epsilon"] >= 0, "adam_epsilon must be at least 0."

        # Eval
        assert isinstance(
            self.eval_args, Dict
        ), "eval_args must be represented as a Dictionary."
        assert self.eval_args["n_best_size"] >= 1, "n_best_size must be at least 1."
        assert (
            self.eval_args["null_score_diff_threshold"] >= 0
        ), "null_score_diff_threshold must be at least 0."
        assert (
            self.eval_args["max_answer_length"] >= 1
        ), "max_answer_length must be at least 1."
