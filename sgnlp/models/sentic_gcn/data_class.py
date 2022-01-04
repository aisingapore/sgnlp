from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SenticGCNTrainArgs:
    model: str = field(default="senticgcn", metadata={"help": "Option to choose which model to train."})
    tokenizer: str = field(
        default="senticgcn",
        metadata={"help": "Option to choose which tokenizer to use for training preprocessing."},
    )
    senticnet_word_file_path: str = field(
        default="./senticNet/senticnet_word.txt", metadata={"help": "SenticNet word file path."}
    )
    spacy_pipeline: str = field(
        default="en_core_web_sm", metadata={"help": "Type of spacy pipeline to load for processor."}
    )
    dataset_train: str = field(
        default="train.raw",
        metadata={"help": "File path to train dataset."},
    )
    dataset_test: str = field(
        default="test.raw",
        metadata={"help": "File path to test dataset."},
    )
    valset_ratio: float = field(
        default=0.0,
        metadata={
            "help": """
                Ratio of train dataset to be split for validation.
                If value is set to 0, test dataset is set as validation dataset as well."""
        },
    )
    word_vec_file_path: str = field(
        default="glove/glove.840B.300d.txt",
        metadata={"help": "File path to word vector."},
    )
    save_embedding_matrix: bool = field(
        default=True,
        metadata={
            "help": """Flag to indicate if embedding matrix should be saved.
                    If 'saved_embedding_matrix_file_path' is populated and valid, it will be overwritten if flag is set to True.
                    """
        },
    )
    saved_embedding_matrix_file_path: str = field(
        default="embedding/embeddings.pickle",
        metadata={
            "help": """Full path of saved embedding matrix, if file exists and 'save_embedding_matrix' flag is set to False.
                    Embeddings will be generated from file instead of generated from word vector and vocab."""
        },
    )
    save_state_dict: bool = field(
        default=True, metadata={"help": "Flag to indicate if best model state_dict should be saved."}
    )
    saved_state_dict_folder_path: str = field(
        default="/state_dict", metadata={"help": "Folder to save model state_dict."}
    )
    save_preprocessed_senticnet: str = field(
        default=True,
        metadata={
            "help": """Flag to indicate if senticnet dictionary should be saved during preprocess step.
                    If 'saved_preprocessed_senticnet_file_path' is populated and valid, it will be overwritten if flag is set to True."""
        },
    )
    saved_preprocessed_senticnet_file_path: str = field(
        default="senticnet/senticnet.pickle",
        metadata={
            "help": """File path to saved preprocessed senticnet, if file exists and 'save_preprocessed_senticnet' flag is set to False.
                    SenticNet will be loaded from file instead of generated from raw senticnet files."""
        },
    )
    initializer: str = field(default="xavier_uniform", metadata={"help": "Type of initalizer to use."})
    optimizer: str = field(default="adam", metadata={"help": "Type of optimizer to use."})
    loss_function: str = field(default="cross_entropy", metadata={"help": "Loss function for training/eval."})
    learning_rate: float = field(default=0.001, metadata={"help": "Default learning rate for training."})
    l2reg: float = field(default=0.00001, metadata={"help": "Default l2reg value."})
    epochs: int = field(default=100, metadata={"help": "Number of epochs to train."})
    batch_size: int = field(default=32, metadata={"help": "Training batch size."})
    log_step: int = field(default=5, metadata={"help": "Number of train steps to log results."})
    embed_dim: int = field(default=300, metadata={"help": "Size of embedding."})
    hidden_dim: int = field(default=300, metadata={"help": "Number of neurons for hidden layer."})
    dropout: float = field(default=0.3, metadata={"help": "Default value for dropout percentages."})
    polarities_dim: int = field(default=3, metadata={"help": "Default dimension for polarities."})
    save: bool = field(default=True, metadata={"help": "Flag to indicate if results should be saved."})
    seed: int = field(default=776, metadata={"help": "Default random seed for training."})
    device: str = field(default="cuda", metadata={"help": "Type of compute device to use for training."})
    repeats: int = field(default=10, metadata={"help": "Number of times to repeat train loop."})
    patience: int = field(
        default=5, metadata={"help": "Number of train epoch without improvements prior to early stopping."}
    )

    def __post_init__(self):
        assert self.model in ["senticgcn", "senticgcnbert"]
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
        assert self.repeats > 1, "Repeats value must be at least 1."
        assert self.patience > 1, "Patience value must be at least 1."
        assert 0 >= self.valset_ratio < 1, "Valset_ratio must be greater or equals to 0 and less than 1."
