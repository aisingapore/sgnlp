from dataclasses import dataclass, field


@dataclass
class BaseArguments:
    """Store arguments for the training and evaluation of StanceClassificationModel or RumourVerificationModel."""

    train_path: str = field(
        metadata={"help": "Path of training data file"},
    )
    dev_path: str = field(
        metadata={"help": "Path of development data file"},
    )
    test_path: str = field(
        metadata={"help": "Path of test data file"},
    )
    output_dir: str = field(
        metadata={
            "help": "Path of model weights, configuration and evaluation results directory"
        },
    )
    bert_model: str = field(
        metadata={
            "help": "Name of pretrained BERT model for model configs and tokenizer"
        },
    )
    do_lower_case: bool = field(
        metadata={"help": "Whether or not to lowercase the input when tokenizing"},
    )
    batch_size: int = field(
        metadata={"help": "Batch size for model training and evaluation"},
    )
    max_tweet_length: int = field(
        metadata={"help": "Number of words in each tweet"},
    )
    max_tweet_num: int = field(
        metadata={"help": "Number of tweets in each bucket"},
    )
    max_tweet_bucket: int = field(
        metadata={"help": "Number of buckets in each thread"},
    )
    max_seq_length: int = field(
        metadata={"help": "Maximum sequence length of the inputs"},
    )
    no_cuda: bool = field(
        metadata={"help": "Whether to not use CUDA even when it is available or not"},
    )
    fp16: bool = field(
        metadata={"help": "Whether to use 16-bit float precision instead of 32-bit"},
    )
    seed: int = field(
        metadata={"help": "Random seed for initialization"},
    )
    num_train_epochs: int = field(
        metadata={"help": "Number of training epochs"},
    )
    learning_rate: float = field(
        metadata={"help": "Learning rate"},
    )
    gradient_accumulation_steps: int = field(
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass"
        },
    )
    loss_scale: int = field(
        metadata={
            "help": "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True."
        },
    )
    warmup_proportion: float = field(
        metadata={
            "help": "Proportion of training to perform linear learning rate warmup for"
        },
    )

    def __post_init__(self):
        assert self.bert_model in [
            "bert-base-uncased",
            "bert-large-uncased",
            "bert-base-cased",
            "bert-large-cased",
            "bert-base-multilingual-uncased",
            "bert-base-multilingual-cased",
            "bert-base-chinese",
        ], "Invalid model type!"
        assert self.batch_size > 0, "batch_size must be at least 1!"
        assert self.max_tweet_length > 0, "max_tweet_length must be at least 1."
        assert self.max_tweet_num > 0, "max_tweet_num must be at least 1."
        assert self.max_tweet_bucket > 0, "max_tweet_bucket must be at least 1."
        assert self.max_seq_length > 0, "max_seq_length must be at least 1!"
        assert self.seed >= 0, "Random seed must be positive!"
        assert self.num_train_epochs > 0, "num_train_epochs must be at least 1."
        assert self.learning_rate > 0, "learning_rate must be positive!"
        assert (
            self.gradient_accumulation_steps > 0
        ), "gradient_accumulation_steps must be at least 1."
        assert self.warmup_proportion > 0, "warmup_proportion must be positive!"
