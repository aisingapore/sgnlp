import logging
import random

import numpy as np
import torch

from .data_class import NEAArguments
from .utils import (
    init_model,
    get_model_friendly_scores,
    pad_sequences_from_list,
    get_emb_matrix,
    NEATrainingArguments,
    NEATrainer,
    NEADataset,
    build_compute_metrics_fn,
    train_and_save_tokenizer,
    load_train_dev_dataset,
    parse_args_and_load_config,
    process_results,
)


logging.basicConfig(level=logging.DEBUG)


def set_seed(seed: int) -> None:
    """Help function to set random seed

    Args:
        seed (int): the seed number to set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(cfg: NEAArguments) -> None:
    """Method for training Neural Essay Assessor model. Saves trained model
    weights.

    NEA evaluate method requires the use of NLTK 'punkt' package.
    Please download the required pacakges as shown in example below prior to running the evaluate method.

    Args:
        config (:obj:`NEAArgument`):
            NEAArgument Config load from configuration file.

    Example::
        # Download NLTK package
        import nltk
        nltk.download('punkt')

        # From Code
        import json
        from sgnlp_models.models.nea.utils import parse_args_and_load_config
        from sgnlp_models.models.nea import train
        cfg = parse_args_and_load_config('config/nea_config.json')
        train(cfg)
    """
    logging.info(f"Training arguments: {cfg}")

    if cfg.use_wandb:
        try:
            import wandb
        except ImportError:
            raise (
                "wandb package not installed! Please install wandb first and try again."
            )
        wandb.init(**cfg.wandb_config)

    set_seed(cfg.train_args["seed"])

    (train_x, train_y, train_pmt), (
        dev_x,
        dev_y,
        dev_pmt,
    ) = load_train_dev_dataset(cfg)

    # tokenize data
    tokenizer = train_and_save_tokenizer(cfg)
    train_x = tokenizer(train_x)["input_ids"]
    dev_x = tokenizer(dev_x)["input_ids"]

    # pad sequences
    train_x = pad_sequences_from_list(train_x)
    dev_x = pad_sequences_from_list(dev_x)

    # convert to model friendly score
    train_y = get_model_friendly_scores(train_y, train_pmt)
    dev_y = get_model_friendly_scores(dev_y, dev_pmt)

    # Prepare data
    train_data = NEADataset(train_x, train_y)
    dev_data = NEADataset(dev_x, dev_y)

    # get_vocab from tokenizer
    vocab = tokenizer.get_vocab()
    emb_matrix = get_emb_matrix(vocab, cfg.emb_path)

    # trainer
    model = init_model(cfg)
    model.initialise_linear_bias(train_y)
    model.load_pretrained_embedding(emb_matrix)

    training_args = NEATrainingArguments(**cfg.train_args)
    compute_metrics_fn = build_compute_metrics_fn(cfg)
    trainer = NEATrainer(
        model=model,
        compute_metrics=compute_metrics_fn,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
    )
    trainer.train()
    trainer.save_model()

    best_result = trainer.evaluate()
    best_result = process_results(best_result)
    logging.info(f"Best model performance on dev data: {best_result}")


if __name__ == "__main__":
    cfg = parse_args_and_load_config()
    train(cfg)
