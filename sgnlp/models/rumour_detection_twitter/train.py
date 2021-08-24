import argparse
import json
import logging
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import set_seed

from .config import RumourDetectionTwitterConfig
from .modeling import RumourDetectionTwitterModel
from .modules.optimizer.scheduler import WarmupScheduler
from .tokenization import RumourDetectionTwitterTokenizer
from .utils import load_datasets


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the rumour detection model.")

    # load the path for the train args config
    parser.add_argument("--train_args_config", type=str, required=True)
    args = parser.parse_args()

    return args


def train(args):
    # Load train args
    with open(args.train_args_config, "r") as f:
        train_args = json.load(f)

    # Log the experiment name and info
    expt_name = train_args["experiment_name"]
    expt_num = train_args["experiment_number"]
    logger.info(f"Experiment {expt_name}-{expt_num}")

    # Default to cpu if GPU is unavailable
    device = (
        torch.device("cuda")
        if train_args["use_gpu"] and torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Create the output dir. Raises an error if the folder already exists
    experiment_output_dir = os.path.join(
        train_args["output_dir"],
        train_args["experiment_name"],
        train_args["experiment_number"],
    )
    assert not os.path.exists(
        experiment_output_dir
    ), "Experiment folder exists. Please change experiment name and/r experiment number"
    os.makedirs(experiment_output_dir)

    # Save train arguments
    with open(os.path.join(experiment_output_dir, "train_args.json"), "w") as f:
        json.dump(train_args, f, sort_keys=True, indent=4)

    # Set seed if provided
    if train_args["seed"] is not None:
        set_seed(train_args["seed"])

    # Load config if provided
    if train_args.get("model_config_path") is not None:
        config = RumourDetectionTwitterConfig.from_json_file(
            json_file=train_args["model_config_path"]
        )
    else:
        config = RumourDetectionTwitterConfig()
    config.save_pretrained(experiment_output_dir)
    model = RumourDetectionTwitterModel(config)

    # Load pretrained embeddings if provided
    if train_args["pretrained_embedding_path"] is not None:
        model.load_pretrained_embedding(train_args["pretrained_embedding_path"])

    # Move model to device
    model.train().to(device)

    # Load and transform datasets
    train_dataloader, val_dataloader, _ = load_datasets(train_args)

    # Set up the optimizer
    assert (
        train_args["optim"] == "sgd" or train_args["optim"] == "adam"
    ), "Only sgd and adam optimizers are supported"
    if train_args["optim"] == "sgd":
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=train_args["lr"],
            momentum=train_args["momentum"],
            nesterov=True,
        )
    elif train_args["optim"] == "adam":
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=(config.d_model) ** (-0.5),
            betas=(train_args["beta1"], train_args["beta2"]),
        )
        scheduler = WarmupScheduler(
            optimizer,
            step_size=train_args["scheduler_step_size"],
            n_warmup_steps=train_args["n_warmup_steps"],
        )

    for epoch in range(train_args["num_epoch"]):

        logger.info(f"starting epoch {epoch}")

        for i, batch in enumerate(tqdm(train_dataloader)):

            token_ids = torch.stack(batch["tweet_token_ids"]).to(device).transpose(0, 1)
            time_delay_ids = batch["time_delay_ids"].to(device)
            structure_ids = (
                torch.stack(batch["structure_ids"]).transpose(0, 1).to(device)
            )
            token_attention_mask = (
                torch.stack(batch["token_attention_mask"])
                .transpose(0, 1)
                .type(torch.Tensor)
                .to(device)
            )
            post_attention_mask = (
                batch["post_attention_mask"].type(torch.Tensor).to(device)
            )

            target_ids = batch["label"].to(device)

            loss = model(
                token_ids=token_ids,
                time_delay_ids=time_delay_ids,
                structure_ids=structure_ids,
                token_attention_mask=token_attention_mask,
                post_attention_mask=post_attention_mask,
                labels=target_ids,
            ).loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % train_args["log_frequency"] == 0:
                logger.info(
                    f"epoch: {epoch} batch: {i} loss: {loss} lr: {scheduler.get_last_lr()}"
                )

                # TODO try to integrate this with the logger
                with open(os.path.join(experiment_output_dir, "loss_log"), "a") as f:
                    f.write(
                        "epoch: {} batch: {} loss: {} lr: {}\n".format(
                            epoch, i, loss, scheduler.get_last_lr()
                        )
                    )

            # Each step gradient update is treated as 1 step
            scheduler.step()

        if epoch % train_args["save_model_frequency"] == 0:
            epoch_output_dir = os.path.join(
                experiment_output_dir, "epoch-" + str(epoch)
            )
            os.makedirs(epoch_output_dir)
            model.save_pretrained(epoch_output_dir)


if __name__ == "__main__":
    args = parse_args()
    train(args)
