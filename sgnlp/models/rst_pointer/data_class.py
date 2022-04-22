from dataclasses import dataclass, field


@dataclass
class RstPointerSegmenterTrainArgs:
    train_data_dir: str = field(metadata={"help": "Training data directory."})
    test_data_dir: str = field(metadata={"help": "Test data directory."})
    save_dir: str = field(metadata={"help": "Directory to save the model."})
    hidden_dim: int = field(default=64, metadata={"help": "Hidden dimension size."})
    rnn: str = field(default="GRU", metadata={"help": "RNN type."})
    num_rnn_layers: int = field(default=6, metadata={"help": "Number of RNN layers."})
    use_bilstm: bool = field(
        default=True, metadata={"help": "Use BI-LSTM for encoding."}
    )
    lr: float = field(default=0.01, metadata={"help": "Learning rate."})
    dropout: float = field(default=0.2, metadata={"help": "Dropout rate."})
    weight_decay: float = field(default=0.0001, metadata={"help": "Weight decay."})
    seed: int = field(default=550, metadata={"help": "Random seed."})
    batch_size: int = field(default=80, metadata={"help": "Batch size."})
    lr_decay_epoch: int = field(
        default=10, metadata={"help": "Learning rate decay epoch."}
    )
    use_batch_norm: bool = field(default=True, metadata={"help": "Use BN for RNN."})
    elmo_size: str = field(
        default="Large", metadata={"help": "Elmo size. ['Large', 'Medium', 'Small']"}
    )
    epochs: int = field(default=100, metadata={"help": "Number of epochs."})

    def __post_init__(self):
        assert self.rnn in ["GRU", "LSTM"], "Invalid RNN type!"
        assert self.elmo_size in ["Large", "Medium", "Small"], "Invalid Elmo size!"


# TODO: Separate train and model args
@dataclass
class RstPointerParserTrainArgs:
    train_data_dir: str = field(metadata={"help": "Training data directory."})
    test_data_dir: str = field(metadata={"help": "Test data directory."})
    save_dir: str = field(metadata={"help": "Directory to save the model."})
    gpu_id: int = field(default=0, metadata={"help": "GPU ID for training."})
    elmo_size: str = field(
        default="Large", metadata={"help": "Elmo size. ['Large', 'Medium', 'Small']"}
    )
    batch_size: int = field(default=64, metadata={"help": "Batch size."})
    hidden_size: int = field(default=64, metadata={"help": "Hidden size of RNN."})
    num_rnn_layers: int = field(default=6, metadata={"help": "Number of RNN layers."})
    dropout_e: float = field(
        default=0.33, metadata={"help": "Dropout rate for encoder."}
    )
    dropout_d: float = field(
        default=0.5, metadata={"help": "Dropout rate for decoder."}
    )
    dropout_c: float = field(
        default=0.5, metadata={"help": "Dropout rate for classifier."}
    )
    input_is_word: bool = field(
        default=True, metadata={"help": "Whether the encoder input is word or EDU."}
    )
    atten_model: str = field(
        default="Dotproduct",
        metadata={"help": "Attention mode. ['Dotproduct', 'Biaffine']"},
    )
    classifier_input_size: int = field(
        default=64, metadata={"help": "Input size of relation classifier."}
    )
    classifier_hidden_size: int = field(
        default=64, metadata={"help": "Hidden size of relation classifier."}
    )
    classifier_bias: bool = field(
        default=True, metadata={"help": "Whether classifier has bias."}
    )
    seed: int = field(default=550, metadata={"help": "Random seed."})
    epochs: int = field(default=300, metadata={"help": "Number of epochs."})
    lr: float = field(default=0.001, metadata={"help": "Learning rate."})
    lr_decay_epoch: int = field(
        default=10, metadata={"help": "Learning rate decay epoch."}
    )
    weight_decay: float = field(default=0.0005, metadata={"help": "Weight decay rate."})
    highorder: bool = field(
        default=True, metadata={"help": "Whether to incorporate highorder information."}
    )

    # model_config_path: str = field(default=None,
    #                                metadata={"help": "Path to model config. Uses default if not provided."})

    def __post_init__(self):
        assert self.elmo_size in ["Large", "Medium", "Small"], "Invalid Elmo size!"
        assert self.atten_model in [
            "Dotproduct",
            "Biaffine",
        ], "Invalid Attention model!"
