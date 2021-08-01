from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class NEAArguments:
    use_wandb: bool = field(
        default=False, metadata={"help": "Use weight and biases for training logs."}
    )
    wandb_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "project": "nea",
            "tags": ["nea"],
            "name": "nea_train_run",
        },
        metadata={"help": "Config for weight and biases."},
    )
    model_type: str = field(
        default="regp", metadata={"help": "The NEA model type to create."}
    )
    emb_path: str = field(
        default="embeddings.w2v.txt", metadata={"help": "File path to embeddings."}
    )
    preprocess_data_args: Dict[str, Any] = field(
        default_factory=lambda: {
            "train_path": "data/fold_0/train.tsv",
            "dev_path": "data/fold_0/dev.tsv",
            "test_path": "data/fold_0/test.tsv",
            "prompt_id": 1,
            "maxlen": 0,
            "to_lower": True,
            "score_index": 6,
        },
        metadata={"help": "Arguments for data preprocessing stage."},
    )
    tokenizer_args: Dict[str, Any] = field(
        default_factory=lambda: {
            "azure_path": "https://sgnlp.blob.core.windows.net/models/nea/",
            "files": [
                "special_tokens_map.json",
                "vocab.txt",
                "tokenizer_config.json",
            ],
            "vocab_train_file": "data/fold_0/train.tsv",
            "save_folder": "nea_tokenizer",
        },
        metadata={"help": "Arguments for NEA tokenizer."},
    )
    train_args: Dict[str, Any] = field(
        default_factory=lambda: {
            "output_dir": "output/",
            "overwrite_output_dir": True,
            "seed": 0,
            "num_train_epochs": 50,
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
        metadata={"help": "Arguments for training NEA models."},
    )
    eval_args: Dict[str, Any] = field(
        default_factory=lambda: {
            "results_path": "output/result.txt",
            "trainer_args": {
                "output_dir": "output/",
                "report_to": "none",
            },
        },
        metadata={"help": "Arguments for evaluating NEA models."},
    )
    preprocess_raw_dataset_args: Dict[str, Any] = field(
        default_factory=lambda: {
            "data_folder": "data/",
            "input_file": "training_set_rel3.tsv",
        },
        metadata={"help": "Arguments for preprocessing raw data set files."},
    )
    preprocess_embedding_args: Dict[str, Any] = field(
        default_factory=lambda: {
            "raw_embedding_file": "release/En_vectors.txt",
            "preprocessed_embedding_file": "embeddings.w2v.txt",
        },
        metadata={"help": "Arguments for embedding steps."},
    )

    def __post_init__(self):
        # Wandb
        assert (
            self.wandb_config.get("project", None) is not None
        ), "Valid project name must be defined to use wandb."
        if self.wandb_config.get("tags", None) is not None:
            assert isinstance(
                self.wandb_config.get("tags"), List
            ), "Tags must be represented by a list of tags."

        # Model
        assert self.model_type in [
            "reg",
            "regp",
            "breg",
            "bregp",
        ], "Invalid model type!"

        # Preprocess data
        assert self.preprocess_data_args["prompt_id"] in [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ], "Invalid prompt_id!"
        assert self.preprocess_data_args["maxlen"] >= 0, "Maxlen must be positive!"

        # Tokenizer

        # Training
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
        assert self.train_args["optimizer_type"] in [
            "rmsprop",
            "adagrad",
            "adamax",
            "adadelta",
        ], f"optimizer_type {self.train_args['optimizer_type']} not supported."
        assert (
            self.train_args["save_total_limit"] > 0
        ), "save_total_limit must be positive."

        # Eval
        assert isinstance(
            self.eval_args, Dict
        ), "eval_args must be represented as a Dictionary."
        assert isinstance(
            self.eval_args["trainer_args"], Dict
        ), "trainer_args must be represented as a Dictionary."

        # Preprocess Raw Dataset
        assert isinstance(
            self.preprocess_raw_dataset_args, Dict
        ), "preprocess_raw_dataset must be represented as a Dictionary."

        # Preprocess embedding
        assert isinstance(
            self.preprocess_embedding_args, Dict
        ), "preprocess_embedding_args must be represented as a Dictionary."
