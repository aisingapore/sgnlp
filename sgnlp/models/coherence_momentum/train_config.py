from dataclasses import dataclass, field


@dataclass
class CoherenceMomentumTrainConfig:
    train_file: str = field(metadata={"help": "Train file path."})
    dev_file: str = field(metadata={"help": "Dev file path."})
    test_file: str = field(metadata={"help": "Test file path."})
    eval_file: str = field(metadata={"help": "Eval file path."})
    output_dir: str = field(metadata={"help": "Output directory."})
    DATA_TYPE_CHOICES = ["multiple", "single"]
    data_type: str = field(
        metadata={"choices": DATA_TYPE_CHOICES, "help": "Data format."}
    )
    lr_start: float = field(default=5e-06)
    lr_end: float = field(default=1e-06)
    lr_anneal_epochs: int = field(default=50)
    eval_interval: int = field(default=1000)
    seed: int = field(default=100)
    batch_size: int = field(default=1)
    train_steps: int = field(default=200)
    num_checkpoints: int = field(
        default=5, metadata={"help": "Number of best checkpoints to save"}
    )

    def __post_init__(self):
        assert (
            self.data_type in self.DATA_TYPE_CHOICES
        ), f"Data type must be one of {self.DATA_TYPE_CHOICES}"
