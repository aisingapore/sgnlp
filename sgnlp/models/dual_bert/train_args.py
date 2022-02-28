from dataclasses import dataclass, field


@dataclass
class CustomDualBertTrainConfig:
    # required
    data_dir: str = field(
        # TEMP
        default='/polyaxon-data/workspace/atenzer/SD_baseline/rumor_data/semeval17/split_0/',
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."})
    data_dir2: str = field(
        # TEMP
        default='/polyaxon-data/workspace/atenzer/SD_baseline/rumor_data/semeval17_stance/split_0/',
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."})
    bert_model: str = field(
        default='bert-base-uncased',
        metadata={"help": "Bert pre-trained model selected in the list: bert-base-uncased,\n"
                          "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased,\n"
                          "bert-base-multilingual-cased, bert-base-chinese."})
    task_name: str = field(
        default='semeval17',
        metadata={"help": "The name of the task to train."})
    task_name2: str = field(
        default='semeval17_stance',
        metadata={"help": "The name of the task to train."})
    rumor_num_labels: int = field(
        default=3,
        metadata={"help": "num of labels for Rumour classification"})
    stance_num_labels: int = field(
        default=5,
        metadata={"help": "num of labels for Stance detection"})
    output_dir: str = field(
        default='/polyaxon-data/workspace/atenzer/SD_baseline/output_release/semeval17_multitask_output_DB_9/',
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."})

    ## Other parameters
    max_seq_length: int = field(
        default=512,
        metadata={"help": "The maximum total input sequence length after WordPiece tokenization. \n}"
                          "Sequences longer than this will be truncated, and sequences shorter \n"
                          "than this will be padded."})
    do_train: bool = field(
        default=True,
        metadata={"help": "Whether to run training."})
    do_eval: bool = field(
        default=True,
        metadata={"help": "Whether to run eval on the dev set."})
    do_lower_case: bool = field(
        default=True,
        metadata={"help": "Set this flag if you are using an uncased model."})
    train_batch_size: int = field(
        default=1,
        metadata={"help": "Total batch size for training."})
    eval_batch_size: int = field(
        default=1,
        metadata={"help": "Total batch size for eval."})
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for Adam."})
    num_train_epochs: float = field(
        default=30.0,
        metadata={"help": "Total number of training epochs to perform."})
    warmup_proportion: float = field(
        default=0.1,
        metadata={"help": "Proportion of training to perform linear learning rate warmup for."
                          "E.g., 0.1 = 10%% of training."})
    no_cuda: bool = field(
        default=False,
        metadata={"help": "Whether not to use CUDA when available"})
    local_rank: int = field(
        default=-1,
        metadata={"help": "local_rank for distributed training on gpus"})
    seed: int = field(
        default=42,
        metadata={"help": "random seed for initialization"})
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."})
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit float precision instead of 32-bit"})
    mt_model: str = field(
        default='DB',
        metadata={"help": 'model name'})
    max_tweet_num: int = field(
        default=17,
        metadata={"help": "the maximum number of tweets"})
    max_tweet_length: int = field(
        default=30,
        metadata={"help": "the maximum length of each tweet"})
    convert_size: int = field(
        default=20,
        metadata={"help": "conversion size"})
