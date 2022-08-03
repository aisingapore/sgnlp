from dataclasses import dataclass, field


@dataclass
class CoherenceMomentumTrainConfig:
    train_file: str = field(metadata={"help": "Train file path."})
    dev_file: str = field(metadata={"help": "Dev file path."})
    test_file: str = field(metadata={"help": "Test file path."})
    eval_file: str = field(metadata={"help": "Eval file path."})
    DATA_TYPE_CHOICES = ["multiple", "single"]
    data_type: str = field(
        metadata={"choices": DATA_TYPE_CHOICES, "help": "Data format."}
    )

    def __post_init__(self):
        assert (
            self.data_type in self.DATA_TYPE_CHOICES
        ), f"Data type must be one of {self.DATA_TYPE_CHOICES}"
