import math

import pandas as pd
from transformers import Trainer, TrainingArguments

from .config import RecconEmotionEntailmentConfig
from .data_class import RecconEmotionEntailmentArguments
from .modeling import RecconEmotionEntailmentModel
from .tokenization import RecconEmotionEntailmentTokenizer
from .utils import (
    RecconEmotionEntailmentData,
    convert_df_to_dataset,
    parse_args_and_load_config,
)


def train_model(train_config: RecconEmotionEntailmentArguments):
    """
    Method for training RecconEmotionEntailmentModel.

    Args:
        train_config (:obj:`RecconEmotionEntailmentArguments`):
            RecconEmotionEntailmentArguments config load from config file.

    Example::

        import json
        from sgnlp.models.emotion_entailment import train
        from sgnlp.models.emotion_entailment.utils import parse_args_and_load_config

        config = parse_args_and_load_config('config/emotion_entailment_config.json')
        train(config)
    """

    config = RecconEmotionEntailmentConfig.from_pretrained(train_config.model_name)
    tokenizer = RecconEmotionEntailmentTokenizer.from_pretrained(
        train_config.model_name
    )
    model = RecconEmotionEntailmentModel.from_pretrained(
        train_config.model_name, config=config
    )

    train_df = pd.read_csv(train_config.x_train_path)
    val_df = pd.read_csv(train_config.x_valid_path)
    train_dataset = convert_df_to_dataset(
        df=train_df, max_seq_length=train_config.max_seq_length, tokenizer=tokenizer
    )
    val_dataset = convert_df_to_dataset(
        df=val_df, max_seq_length=train_config.max_seq_length, tokenizer=tokenizer
    )

    train_config.len = len(train_df)
    train_config.train_args["eval_steps"] = (
        train_config.len / train_config.train_args["per_device_train_batch_size"]
    )
    train_config.train_args["warmup_steps"] = math.ceil(
        (
            train_config.len
            // train_config.train_args["gradient_accumulation_steps"]
            * train_config.train_args["num_train_epochs"]
        )
        * train_config.train_args["warmup_ratio"]
    )

    train_args = TrainingArguments(**train_config.train_args)
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
    train_model(cfg)
