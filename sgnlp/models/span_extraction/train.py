import math
import json

from transformers import Trainer
from transformers import TrainingArguments

from .config import RecconSpanExtractionConfig
from .tokenization import RecconSpanExtractionTokenizer
from .modeling import RecconSpanExtractionModel
from .data_class import RecconSpanExtractionArguments
from .utils import parse_args_and_load_config, load_examples, RecconSpanExtractionData


def train(cfg: RecconSpanExtractionArguments):
    """
    Method for training RecconSpanExtractionModel.

    Args:
        config (:obj:`RecconSpanExtractionArguments`):
            RecconSpanExtractionArguments config load from config file.

    Example::

            import json
            from sgnlp.models.span_extraction import train
            from sgnlp.models.span_extraction.utils import parse_args_and_load_config

            cfg = parse_args_and_load_config('config/span_extraction_config.json')
            train(cfg)
    """
    config = RecconSpanExtractionConfig.from_pretrained(cfg.model_name)
    tokenizer = RecconSpanExtractionTokenizer.from_pretrained(cfg.model_name)
    model = RecconSpanExtractionModel.from_pretrained(cfg.model_name, config=config)

    with open(cfg.train_data_path, "r") as train_file:
        train_json = json.load(train_file)

    with open(cfg.val_data_path, "r") as val_file:
        val_json = json.load(val_file)

    load_train_exp_args = {
        "examples": train_json,
        "tokenizer": tokenizer,
        "max_seq_length": cfg.max_seq_length,
        "doc_stride": cfg.doc_stride,
        "max_query_length": cfg.max_query_length,
    }
    load_valid_exp_args = {
        "examples": val_json,
        "tokenizer": tokenizer,
        "max_seq_length": cfg.max_seq_length,
        "doc_stride": cfg.doc_stride,
        "max_query_length": cfg.max_query_length,
    }
    train_dataset = load_examples(**load_train_exp_args)
    val_dataset = load_examples(**load_valid_exp_args)

    t_total = (
        len(train_dataset)
        // cfg.train_args["gradient_accumulation_steps"]
        * cfg.train_args["num_train_epochs"]
    )
    cfg.train_args["eval_steps"] = int(
        len(train_dataset) / cfg.train_args["per_device_train_batch_size"]
    )
    cfg.train_args["warmup_steps"] = math.ceil(t_total * cfg.train_args["warmup_ratio"])

    training_args = TrainingArguments(**cfg.train_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=RecconSpanExtractionData(train_dataset),
        eval_dataset=RecconSpanExtractionData(val_dataset),
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    cfg = parse_args_and_load_config()
    train(cfg)
