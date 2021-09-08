from dataclasses import dataclass, field


@dataclass
class RstPointerSegmenterTrainArgs:
    train_data_dir: str = field(default=None, metadata={"help": "Training data directory."})
    test_data_dir: str = field(default=None, metadata={"help": "Test data directory."})
    save_dir: str = field(default=None, metadata={"help": "Directory to save the model."})
    use_polyaxon: bool = field(default=False, metadata={"help": "Use Polyaxon to track the training progress."})
    hdim: int = field(default=64, metadata={"help": "Hidden dimension size."})
    rnn: str = field(default="GRU", metadata={"help": "RNN type."})
    rnnlayers: int = field(default=6, metadata={"help": "Number of RNN layers."})
    fine: bool = field(default=False, metadata={"help": "Fine tune word embedding."})
    isbi: bool = field(default=True, metadata={"help": "Use BI-LSTM for encoding."})
    lr: float = field(default=0.01, metadata={"help": "Learning rate."})
    dout: float = field(default=0.2, metadata={"help": "Dropout rate."})
    wd: float = field(default=0.0001, metadata={"help": "Weight decay."})
    seed: int = field(default=550, metadata={"help": "Random seed."})
    bsize: int = field(default=80, metadata={"help": "Batch size."})
    lrdepoch: int = field(default=10, metadata={"help": "Learning rate decay epoch."})
    isbarnor: bool = field(default=True, metadata={"help": "Use BN for RNN."})
    iscudnn: bool = field(default=True, metadata={"help": "Use CuDNN for RNN."})
    elmo_size: str = field(default="Large", metadata={"help": "Elmo size. ['Large', 'Medium', 'Small']"})
    epochs: int = field(default=100, metadata={"help": "Number of epochs."})

    def __post_init__(self):
        assert (self.rnn in ["GRU", "LSTM"]), "Invalid RNN type!"
        assert (self.elmo_size in ['Large', 'Medium', 'Small']), "Invalid Elmo size!"
        if self.use_polyaxon:
            try:
                import polyaxon_client  # noqa: F401
            except ImportError:
                raise ImportError("polyaxon_client is required for using Polyaxon to track the training progress.")
        if not self.use_polyaxon:
            assert (self.save_dir is not None), "Please specify a directory to save the model."


@dataclass
class RstPointerParserTrainArgs:
    train_data_dir: str = field(default=None, metadata={"help": "Training data directory."})
    test_data_dir: str = field(default=None, metadata={"help": "Test data directory."})
    save_dir: str = field(default=None, metadata={"help": "Directory to save the model."})
    use_polyaxon: bool = field(default=False, metadata={"help": "Use Polyaxon to track the training progress."})
    gpu_id: int = field(default=0, metadata={"help": "GPU ID for training."})
    elmo_size: str = field(default="Large", metadata={"help": "Elmo size. ['Large', 'Medium', 'Small']"})
    batch_size: int = field(default=64, metadata={"help": "Batch size."})
    hidden_size: int = field(default=64, metadata={"help": "Hidden size of RNN."})
    rnn_layers: int = field(default=6, metadata={"help": "Number of RNN layers."})
    dropout_e: float = field(default=0.33, metadata={"help": "Dropout rate for encoder."})
    dropout_d: float = field(default=0.5, metadata={"help": "Dropout rate for decoder."})
    dropout_c: float = field(default=0.5, metadata={"help": "Dropout rate for classifier."})
    input_is_word: bool = field(default=True, metadata={"help": "Whether the encoder input is word or EDU."})
    atten_model: str = field(default='Dotproduct', metadata={"help": "Attention mode. ['Dotproduct', 'Biaffine']"})
    classifier_input_size: int = field(default=64, metadata={"help": "Input size of relation classifier."})
    classifier_hidden_size: int = field(default=64, metadata={"help": "Hidden size of relation classifier."})
    classifier_bias: bool = field(default=True, metadata={"help": "Whether classifier has bias."})
    seed: int = field(default=550, metadata={"help": "Random seed."})
    eval_size: int = field(default=600, metadata={"help": "Number of evaluation data."})
    epochs: int = field(default=300, metadata={"help": "Number of epochs."})
    lr: float = field(default=0.001, metadata={"help": "Learning rate."})
    lr_decay_epoch: int = field(default=10, metadata={"help": "Learning rate decay epoch."})
    weight_decay: float = field(default=0.0005, metadata={"help": "Weight decay rate."})
    highorder: bool = field(default=True, metadata={"help": "Whether to incorporate highorder information."})

    def __post_init__(self):
        assert (self.elmo_size in ['Large', 'Medium', 'Small']), "Invalid Elmo size!"
        assert (self.atten_model in ['Dotproduct', 'Biaffine']), "Invalid Attention model!"
        if self.use_polyaxon:
            try:
                import polyaxon_client  # noqa: F401
            except ImportError:
                raise ImportError("polyaxon_client is required for using Polyaxon to track the training progress.")
        if not self.use_polyaxon:
            assert (self.save_dir is not None), "Please specify a directory to save the model."
