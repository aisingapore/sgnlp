from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class SenticGCNTrainArgs:
    """
    Data class for training config for both SenticGCNModel and SenticGCNBertModel
    """

    # External resources (e.g. Senticnet file, GloVe word vectors, etc)
    senticnet_word_file_path: str = field(
        default="./senticNet/senticnet_word.txt", metadata={"help": "SenticNet word file path."}
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
    spacy_pipeline: str = field(
        default="en_core_web_sm", metadata={"help": "Type of spacy pipeline to load for processor."}
    )
    word_vec_file_path: str = field(
        default="glove/glove.840B.300d.txt",
        metadata={"help": "File path to word vector."},
    )

    # Dataset specific config
    dataset_train: list = field(
        default_factory=list,
        metadata={"help": "List of file path to train dataset(s)."},
    )
    dataset_test: list = field(
        default_factory=list,
        metadata={"help": "List of file path to test dataset(s)."},
    )
    valset_ratio: float = field(
        default=0.0,
        metadata={
            "help": """
                Ratio of train dataset to be split for validation.
                If value is set to 0, test dataset is set as validation dataset as well."""
        },
    )

    # Model specific config
    model: str = field(default="senticgcn", metadata={"help": "Option to choose which model to train."})
    save_best_model: bool = field(
        default=True,
        metadata={
            "help": """Flag to indicate if best model should be saved during training.
                    Applies to both bert and non-bert SenticGCN models."""
        },
    )
    save_model_path: str = field(
        default="senticgcn",
        metadata={
            "help": """Folder path to save trained model using the save_pretrained method.
                    Applies to both bert and non-bert SenticGCN models."""
        },
    )

    # Tokenizer specific config
    tokenizer: str = field(
        default="senticgcn_tokenizer",
        metadata={
            "help": """Option to choose which tokenizer to use for training preprocessing.
                        Value will be used to create tokenizer via the from_pretrained method."""
        },
    )
    train_tokenizer: bool = field(
        default=False,
        metadata={
            "help": """Flag to indicate if tokenizer should be trained on train and test input dataset.
                        Only applies to non-bert SenticGCN tokenizer."""
        },
    )
    save_tokenizer: bool = field(
        default=False,
        metadata={
            "help": """Flag to indicate if tokenizer should be saved using the save_pretrained method.
                            Only applies to non-bert SenticGCN tokenizer."""
        },
    )
    save_tokenizer_path: str = field(
        default="senticgcn_tokenizer",
        metadata={
            "help": """Folder path to save pretrained tokenizer using the save_pretrained method.
                            Only applies to non-bert SenticGCN tokenizer."""
        },
    )

    # Embedding specific config
    embedding_model: str = field(
        default="senticgcn_embed_model",
        metadata={
            "help": """Option to choose which embeding model to use for training preprocessing.
                    For non-bert model, value should point to a pretraine model folder.
                    'config.json' and 'pytorch_model.bin' will be used to create the config and embedding model
                    via the from_pretrained method.
                    Ignore if 'build_embedding_model' flag is set, only affects non-bert SenticGCN embedding model.
                    For bert model, value should be model name used to download from huggingface model hub."""
        },
    )
    build_embedding_model: bool = field(
        default=False,
        metadata={
            "help": """Flag to indicate if embedding model should be built from input word vectors.
                    Only applies to non-bert SenticGCN embedding models.
                    Word vectors to train on is indicated in 'word_vec_file_path' config."""
        },
    )
    save_embedding_model: bool = field(
        default=False,
        metadata={
            "help": """Flag to indicate if embedding model should be saved using the save_pretrained method.
                            Only applies to non-bert SenticGCN embedding model."""
        },
    )
    save_embedding_model_path: str = field(
        default="senticgcn_embed_model",
        metadata={
            "help": """Folder path to save pretrained embedding model using the save_pretrained method.
                        Only applies to non-bert SenticGCN embeddding model."""
        },
    )

    # Training results
    save_results: bool = field(default=True, metadata={"help": "Flag to indicate if results should be saved."})
    save_results_folder: str = field(default="results", metadata={"help": "Folder location to save results pickle."})

    initializer: str = field(default="xavier_uniform_", metadata={"help": "Type of initalizer to use."})
    optimizer: str = field(default="adam", metadata={"help": "Type of optimizer to use."})
    loss_function: str = field(default="cross_entropy", metadata={"help": "Loss function for training/eval."})
    learning_rate: float = field(default=0.001, metadata={"help": "Default learning rate for training."})
    l2reg: float = field(default=0.00001, metadata={"help": "Default l2reg value."})
    epochs: int = field(default=100, metadata={"help": "Number of epochs to train."})
    batch_size: int = field(default=16, metadata={"help": "Training batch size."})
    log_step: int = field(default=5, metadata={"help": "Number of train steps to log results."})
    embed_dim: int = field(default=300, metadata={"help": "Size of embedding."})
    hidden_dim: int = field(default=300, metadata={"help": "Number of neurons for hidden layer."})
    dropout: float = field(default=0.3, metadata={"help": "Default value for dropout percentages."})
    polarities_dim: int = field(default=3, metadata={"help": "Default dimension for polarities."})
    save_results: bool = field(default=True, metadata={"help": "Flag to indicate if results should be saved."})
    seed: int = field(default=776, metadata={"help": "Default random seed for training."})
    device: str = field(default="cuda", metadata={"help": "Type of compute device to use for training."})
    repeats: int = field(default=10, metadata={"help": "Number of times to repeat train loop."})
    patience: int = field(
        default=5, metadata={"help": "Number of train epoch without improvements prior to early stopping."}
    )
    max_len: int = field(default=85, metadata={"help": "Max length to pad for bert tokenizer."})
    eval_args: Dict[str, Any] = field(
        default_factory=lambda: {
            "model": "senticgcn",
            "model_path": "",
            "tokenizer": "senticgcn",
            "embedding_model": "senticgcn",
            "config_filename": "config.json",
            "model_filename": "pytorch_model.bin",
            "test_filename": "",
            "senticnet": "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticnet.pickle",
            "spacy_pipeline": "en_core_web_sm",
            "result_folder": "./eval_result/",
            "eval_batch_size": 16,
            "seed": 776,
            "device": "cpu",
        }
    )

    def __post_init__(self):
        # Model
        assert self.model in ["senticgcn", "senticgcnbert"], "Invalid model type!"

        assert self.initializer in [
            "xavier_uniform_",
            "xavier_normal_",
            "orthogonal_",
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
        assert self.repeats >= 1, "Repeats value must be at least 1."
        assert self.patience >= 1, "Patience value must be at least 1."
        assert 0 >= self.valset_ratio < 1, "Valset_ratio must be greater or equals to 0 and less than 1."
        assert self.max_len > 0, "Max_len must be greater than 0."

        # Assign sub dataset columns name
        self.data_cols = ["text_indices", "aspect_indices", "left_indices", "text_embeddings", "sdat_graph"]
