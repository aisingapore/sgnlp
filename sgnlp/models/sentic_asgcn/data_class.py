from dataclasses import dataclass, field


@dataclass
class SenticASGCNTrainArgs:
    initializer: str = field(
        default="xavier_uniform", metadata={"help": "Type of initalizer to use."}
    )
    optimizer: str = field(
        default="adam", metadata={"help": "Type of optimizer to use."}
    )
    learning_rate: float = field(
        default=0.001, metadata={"help": "Default learning rate for training."}
    )
    l2reg: float = field(default=0.00001, metadata={"help": "Default l2reg value."})
    epochs: int = field(default=100, metadata={"help": "Number of epochs to train."})
    batch_size: int = field(default=32, metadata={"help": "Training batch size."})
    log_step: int = field(default=5, metadata={"help": "Default log step."})
    embed_dim: int = field(
        default=300, metadata={"help": "Number of neurons for embed layer."}
    )
    hidden_dim: int = field(
        default=300, metadata={"help": "Number of neurons for hidden layer."}
    )
    dropout: float = field(
        default=0.3, metadata={"help": "Default value for dropout percentages."}
    )
    polarities_dim: int = field(
        default=3, metadata={"help": "Default dimension for polarities."}
    )
    save: bool = field(
        default=True, metadata={"help": "Flag to indicate if results should be saved."}
    )
    seed: int = field(
        default=776, metadata={"help": "Default random seed for training."}
    )
    device: str = field(
        default="cuda", metadata={"help": "Type of compute device to use for training."}
    )

    def __post_init__(self):
        assert self.initializer in [
            "xavier_uniform",
            "xavier_uniform",
            "orthogonal",
        ], "Invalid initializer type!"
        assert self.optimizer in [
            "adadelta",
            "adagrad",
            "adam",
            "adamax",
            "asgd",
            "rmsprop",
            "sgd",
        ], "Invalid optimizer"
        assert self.device in ["cuda", "cpu"], "Invalid device type."
