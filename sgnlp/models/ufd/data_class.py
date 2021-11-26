from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class UFDArguments:
    verbose: bool = field(
        default=False, metadata={"help": "Enable verbose logging messages."}
    )
    device: str = field(
        default="cuda", metadata={"help": "Pytorch device type to set for training."}
    )
    data_folder: str = field(
        default="data/", metadata={"help": "Folder path to dataset."}
    )
    model_folder: str = field(
        default="model/", metadata={"help": "Folder path to model weights."}
    )
    cache_folder: str = field(
        default="cache/", metadata={"help": "Folder path for caching."}
    )
    embedding_model_name: str = field(
        default="xlm_roberta_large",
        metadata={"help": "Name of HuggingFace model used for embedding model."},
    )
    use_wandb: bool = field(
        default=False, metadata={"help": "Use weight and biases for training logs."}
    )
    wandb_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "project": "ufd",
            "tags": ["ufd"],
            "name": "ufd_train_run",
        },
        metadata={"help": "Config for weight and biases."},
    )
    train_args: Dict[str, Any] = field(
        default_factory=lambda: {
            "unsupervised_dataset_filename": "raw.0.6.txt",
            "train_filename": "train.txt",
            "val_filename": "sampled.txt",
            "train_cache_filename": "train_dataset.pickle",
            "val_cache_filename": "val_dataset.pickle",
            "learning_rate": 0.00001,
            "seed": 0,
            "unsupervised_model_batch_size": 16,
            "unsupervised_epochs": 30,
            "in_dim": 1024,
            "dim_hidden": 1024,
            "out_dim": 1024,
            "initrange": 0.1,
            "classifier_epochs": 60,
            "classifier_batch_size": 16,
            "num_class": 2,
            "source_language": "en",
            "source_domains": ["books", "dvd", "music"],
            "target_domains": ["books", "dvd", "music"],
            "target_languages": ["de", "fr", "jp"],
            "warmup_epochs": 5,
        },
        metadata={"help": "Arguments for training UFD models."},
    )
    eval_args: Dict[str, Any] = field(
        default_factory=lambda: {
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
        }
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

        # device
        assert self.device in [
            "cpu",
            "cuda",
        ], f"Invalid {self.device} device! Must be either 'cpu', or 'cuda' only."

        # Training
        assert (
            self.train_args["learning_rate"] > 0
        ), f"Invalid {self.train_args['learning_rate']} learning_rate, must be positive only."
        assert (
            self.train_args["seed"] >= 0
        ), f"Invalid {self.train_args['seed']} random seed, must be positive only."
        assert (
            self.train_args["unsupervised_model_batch_size"] >= 1
        ), "unsupervised_model_batch_size must be at least 1."
        assert (
            self.train_args["unsupervised_epochs"] >= 1
        ), "unsupervised_epochs must be at least 1."
        assert self.train_args["in_dim"] >= 1, "in_dim must be at least 1."
        assert self.train_args["dim_hidden"] >= 1, "dim_hidden must be at least 1."
        assert self.train_args["out_dim"] >= 1, "out_dim must be at least 1."
        assert (
            self.train_args["initrange"] >= 0 and self.train_args["initrange"] <= 1
        ), "initrange must be between 0 and 1."
        assert (
            self.train_args["classifier_epochs"] >= 1
        ), "classifier_epochs must be at least 1."
        assert (
            self.train_args["classifier_batch_size"] >= 1
        ), "classifier_batch_size must be at least 1."
        assert self.train_args["num_class"] >= 1, "num_class must be at least 1."
        assert isinstance(
            self.train_args["source_language"], str
        ), "source_language must be a string."
        assert isinstance(
            self.train_args["source_domains"], List
        ), "source_domains must be a list of strings."
        assert isinstance(
            self.train_args["target_domains"], List
        ), "target_domains must be a list of strings."
        assert isinstance(
            self.train_args["target_languages"], List
        ), "target_languages must be a list of strings."
        assert self.train_args["warmup_epochs"] >= 0, "warmup_epochs must be positive."

        # Eval
        assert (
            self.eval_args["eval_batch_size"] >= 1
        ), "eval_batch_size must be at least 1."
        assert isinstance(
            self.eval_args["source_language"], str
        ), "source_language must be a string."
        assert isinstance(
            self.eval_args["source_domains"], List
        ), "source_domains must be a list of strings."
        assert isinstance(
            self.eval_args["target_domains"], List
        ), "target_domains must be a list of strings."
        assert isinstance(
            self.eval_args["target_languages"], List
        ), "target_languages must be a list of strings."
