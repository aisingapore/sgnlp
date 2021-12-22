from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SenticNetGCNTrainArgs:
    raw_dataset_files: List[str] = field(default=list, metadata={"help": "List of raw dataset to process."})
    senticnet_word_file_path: str = field(
        default="./senticNet/senticnet_word.txt", metadata={"help": "SenticNet word file path."}
    )
    spacy_pipeline: str = field(
        default="en_core_web_sm", metadata={"help": "Type of spacy pipeline to load for processor."}
    )
    dataset_train: Dict[str, str] = field(
        default=dict,
        metadata={"help": "Dictionary containing 3 file paths to the raw, graph and tree train datasets."},
    )
    dataset_test: Dict[str, str] = field(
        default=dict,
        metadata={"help": "Dictionary containing 3 file paths to the raw, graph and tree test datasets."},
    )
    word_vec_file_path: str = field(
        default="glove/glove.840B.300d.txt",
        metadata={"help": "File path to word vector."},
    )
    save_embedding_matrix: bool = field(
        default=True,
        metadata="Flag to indicate if embedding matrix should be saved. Flag is ignored if 'saved_embedding_matrix_file_path' is populated and valid.",
    )
    saved_embedding_matrix_file_path: str = field(
        default="embedding/embeddings.pickle",
        metadata={
            "help": "Full path of saved embedding matrix, if file exists, embeddings will be generated from file instead of generated from word vector and vocab."
        },
    )
    initializer: str = field(default="xavier_uniform", metadata={"help": "Type of initalizer to use."})
    optimizer: str = field(default="adam", metadata={"help": "Type of optimizer to use."})
    learning_rate: float = field(default=0.001, metadata={"help": "Default learning rate for training."})
    l2reg: float = field(default=0.00001, metadata={"help": "Default l2reg value."})
    epochs: int = field(default=100, metadata={"help": "Number of epochs to train."})
    batch_size: int = field(default=32, metadata={"help": "Training batch size."})
    log_step: int = field(default=5, metadata={"help": "Default log step."})
    embed_dim: int = field(default=300, metadata={"help": "Size of embedding."})
    hidden_dim: int = field(default=300, metadata={"help": "Number of neurons for hidden layer."})
    dropout: float = field(default=0.3, metadata={"help": "Default value for dropout percentages."})
    polarities_dim: int = field(default=3, metadata={"help": "Default dimension for polarities."})
    save: bool = field(default=True, metadata={"help": "Flag to indicate if results should be saved."})
    seed: int = field(default=776, metadata={"help": "Default random seed for training."})
    device: str = field(default="cuda", metadata={"help": "Type of compute device to use for training."})

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
