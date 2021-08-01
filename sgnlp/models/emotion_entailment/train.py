import math

import pandas as pd
from transformers import Trainer, TrainingArguments

from .config import RecconEmotionEntailmentConfig
from .tokenization import RecconEmotionEntailmentTokenizer
from .modeling import RecconEmotionEntailmentModel
from .data_class import RecconEmotionEntailmentArguments
from .utils import (
    RecconEmotionEntailmentData,
    convert_df_to_dataset,
    parse_args_and_load_config,
)


def train(cfg: RecconEmotionEntailmentArguments):
    """
    Method for training RecconEmotionEntailmentModel.

    Args:
        config (:obj:`RecconEmotionEntailmentArguments`):
            RecconEmotionEntailmentArguments config load from config file.

    Example::

            import json
            from sgnlp.models.emotion_entailment import train
            from sgnlp.models.emotion_entailment.utils import parse_args_and_load_config

            cfg = parse_args_and_load_config('config/emotion_entailment_config.json')
            train(cfg)
    """

    config = RecconEmotionEntailmentConfig.from_pretrained(cfg.model_name)
    tokenizer = RecconEmotionEntailmentTokenizer.from_pretrained(cfg.model_name)
    model = RecconEmotionEntailmentModel.from_pretrained(cfg.model_name, config=config)

    train_df = pd.read_csv(cfg.x_train_path)
    val_df = pd.read_csv(cfg.x_valid_path)
    train_dataset = convert_df_to_dataset(
        df=train_df, max_seq_length=cfg.max_seq_length, tokenizer=tokenizer
    )
    val_dataset = convert_df_to_dataset(
        df=val_df, max_seq_length=cfg.max_seq_length, tokenizer=tokenizer
    )

    cfg.len = len(train_df)
    cfg.train_args["eval_steps"] = (
        cfg.len / cfg.train_args["per_device_train_batch_size"]
    )
    cfg.train_args["warmup_steps"] = math.ceil(
        (
            cfg.len
            // cfg.train_args["gradient_accumulation_steps"]
            * cfg.train_args["num_train_epochs"]
        )
        * cfg.train_args["warmup_ratio"]
    )

    train_args = TrainingArguments(**cfg.train_args)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=RecconEmotionEntailmentData(train_dataset),
        eval_dataset=RecconEmotionEntailmentData(val_dataset),
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    cfg = parse_args_and_load_config()
    train(cfg)
