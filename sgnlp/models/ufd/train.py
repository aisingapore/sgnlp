import logging
import pathlib
import pickle
import random
from typing import Dict, List, Tuple
from itertools import product

import os
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from .data_class import UFDArguments
from .modeling import (
    UFDAdaptorGlobalModel,
    UFDAdaptorDomainModel,
    UFDCombineFeaturesMapModel,
    UFDClassifierModel,
    UFDDeepInfoMaxLossModel,
)
from .utils import (
    create_unsupervised_models,
    create_classifiers,
    get_source2target_domain_mapping,
    generate_train_val_dataset,
    parse_args_and_load_config,
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


def train_adaptor(
    train_data: List,
    batch_size: int,
    device: torch.device,
    adaptor_global: UFDAdaptorGlobalModel,
    adaptor_domain: UFDAdaptorDomainModel,
    optim: Adam,
    loss_optim: Adam,
    loss_fn: UFDDeepInfoMaxLossModel,
) -> float:
    """Train step method for unsupervised training

    Args:
        train_data (List): raw dataset embeddings
        batch_size (int): unsupervised training batch size
        device (torch.device): torch device type
        adaptor_global (UFDAdaptorGlobalModel): adaptor global model
        adaptor_domain (UFDAdaptorDomainModel): adaptor domain model
        optim (Adam): Adam optimizer
        loss_optim (Adam): Adam optimizer
        loss_fn (UFDDeepInfoMaxLossModel): Deep Info Max Loss function

    Returns:
        int: loss calculated for the train step
    """
    t_loss = 0
    data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    for i, pairs in enumerate(data):
        x = pairs[:, :, 1:]
        x = torch.squeeze(x, dim=1)
        x = x.to(device)

        optim.zero_grad()
        loss_optim.zero_grad()

        y_g, f_g = adaptor_global(x)
        y_d, f_d = adaptor_domain(x)

        x_n = torch.cat((x[1:], x[0].unsqueeze(0)), dim=0)
        fg_n = torch.cat((f_g[1:], f_g[0].unsqueeze(0)), dim=0)
        fd_n = torch.cat((f_d[1:], f_d[0].unsqueeze(0)), dim=0)
        yd_n = torch.cat((y_d[1:], y_d[0].unsqueeze(0)), dim=0)

        loss_a = loss_fn(x, x_n, f_g, fg_n, f_d, fd_n, y_g, y_d, yd_n)
        t_loss += loss_a.item()

        loss_a.backward()
        optim.step()
        loss_optim.step()

    return t_loss


def train_classifier(
    train_data: List,
    batch_size: int,
    device: torch.device,
    optimizer: Adam,
    optim_fusion: Adam,
    adaptor_global: UFDAdaptorGlobalModel,
    adaptor_domain: UFDAdaptorDomainModel,
    model: UFDClassifierModel,
    maper: UFDCombineFeaturesMapModel,
    criterion: torch.nn.CrossEntropyLoss,
    cross_domain=False,
) -> float:
    """Train step method for classifier training

    Args:
        train_data (List): train dataset embeddings
        batch_size (int): supervised training batch size
        device (torch.device): torch device type
        optimizer (Adam): Adam optimizer
        optim_fusion (Adam): Adam optimizer
        adaptor_global (UFDAdaptorGlobalModel): adaptor global model
        adaptor_domain (UFDAdaptorDomainModel): adaptor domain model
        model (UFDClassifierModel): classifier model
        maper (UFDCombineFeaturesMapModel): combine feature map model
        criterion (torch.nn.CrossEntropyLoss): cross entropy loss criterion
        cross_domain (bool, optional): cross domain flag. Defaults to False.

    Returns:
        float: loss calculated for the train step
    """
    train_loss = 0
    train_acc = 0
    data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    for i, pairs in enumerate(data):
        text = pairs[:, :, 1:]
        index = pairs[:, :, :1]
        text = torch.squeeze(text, dim=1)
        index = index.view(-1)
        index = index.long()

        optimizer.zero_grad()
        optim_fusion.zero_grad()
        text, index = text.to(device), index.to(device)

        if cross_domain:
            global_f, _ = adaptor_global(text)
            output = model(global_f)
        else:
            global_f, _ = adaptor_global(text)
            domain_f, _ = adaptor_domain(text)
            features = torch.cat((global_f, domain_f), dim=1)
            output = model(maper(features))

        loss = criterion(output, index)
        train_loss += loss.item()
        loss.backward(retain_graph=True)
        optimizer.step()
        optim_fusion.step()
        train_acc += (output.argmax(1) == index).sum().item()

    return train_loss / len(train_data), train_acc / len(train_data)


def test_classifier(
    test_data: List,
    batch_size: int,
    device: torch.device,
    adaptor_global: UFDAdaptorGlobalModel,
    adaptor_domain: UFDAdaptorDomainModel,
    model: UFDClassifierModel,
    maper: UFDCombineFeaturesMapModel,
    criterion: torch.nn.CrossEntropyLoss,
    cross_domain=False,
) -> float:
    """Test step method for classifier training

    Args:
        test_data (List): test/validation dataset embeddings
        batch_size (int): supervised training batch size
        device (torch.device): torch device type
        adaptor_global (UFDAdaptorGlobalModel): adaptor global model
        adaptor_domain (UFDAdaptorDomainModel): adaptor domain model
        model (UFDClassifierModel): classifier model
        maper (UFDCombineFeaturesMapModel): combine feature map model
        criterion (torch.nn.CrossEntropyLoss): cross entropy loss criterion
        cross_domain (bool, optional): cross domain flag. Defaults to False.

    Returns:
        float: loss calcaulted for the test step
    """
    loss = 0
    acc = 0
    data = DataLoader(test_data, batch_size=batch_size)
    for i, pairs in enumerate(data):
        text = pairs[:, :, 1:]
        index = pairs[:, :, :1]
        text = torch.squeeze(text, dim=1)
        index = index.view(-1)
        index = index.long()
        text, index = text.to(device), index.to(device)
        with torch.no_grad():
            if cross_domain:
                global_f, _ = adaptor_global(text)
                output = model(global_f)
            else:
                global_f, _ = adaptor_global(text)
                domain_f, _ = adaptor_domain(text)
                features = torch.cat((global_f, domain_f), dim=1)
                output = model(maper(features))
            crit_loss = criterion(output, index)
            loss += crit_loss.item()
            acc += (output.argmax(1) == index).sum().item()

    return loss / len(test_data), acc / len(test_data)


def save_models(
    cfg: UFDArguments,
    full_combi_name: str,
    adaptor_global: UFDAdaptorGlobalModel,
    adaptor_domain: UFDAdaptorDomainModel,
    maper: UFDCombineFeaturesMapModel,
    classifier: UFDClassifierModel,
) -> None:
    """Help function to save best models

    Args:
        cfg (UFDArguments): UFDArguments config load from configuration file.
        full_combi_name (str): model name
        adaptor_global (UFDAdaptorGlobalModel): adaptor global model
        adaptor_domain (UFDAdaptorDomainModel): adaptor domain model
        maper (UFDCombineFeaturesMapModel): combine feature map model
        classifier (UFDClassifierModel): classifier model
    """
    if not os.path.isdir(cfg.model_folder):
        os.mkdir(cfg.model_folder)

    ag_path = str(
        pathlib.Path(cfg.model_folder).joinpath(
            full_combi_name + "_" + "adaptor_global"
        )
    )
    ad_path = str(
        pathlib.Path(cfg.model_folder).joinpath(
            full_combi_name + "_" + "adaptor_domain"
        )
    )
    maper_path = str(
        pathlib.Path(cfg.model_folder).joinpath(full_combi_name + "_" + "maper")
    )
    cls_path = str(
        pathlib.Path(cfg.model_folder).joinpath(full_combi_name + "_" + "classifier")
    )

    adaptor_global.save_pretrained(ag_path)
    adaptor_domain.save_pretrained(ad_path)
    maper.save_pretrained(maper_path)
    classifier.save_pretrained(cls_path)


def train(cfg: UFDArguments) -> Tuple[Dict]:
    """Main train method consisting of unsupervised training at the outer train loop and individual
    classifier supervised training in the inner train loop.

    Args:
        config (:obj:`UFDArguments`):
            UFDArguments Config load from configuration file.

    Returns:
        :obj:`Tuple[Dict]`: all train/validation loss and acc records.

    Example::
        from sgnlp.models.ufd import parse_args_and_load_config
        from sgnlp.models.ufd import train
        cfg = parse_args_and_load_config
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

    device = torch.device(cfg.device)

    sourcedomain2targetdomains = get_source2target_domain_mapping(
        cfg.train_args["source_domains"], cfg.train_args["target_domains"]
    )

    # Init unsupervised models
    (
        adaptor_domain_model,
        adaptor_global_model,
        combine_features_map_model,
    ) = create_unsupervised_models(cfg)

    # Add model watch to wandb
    if cfg.use_wandb:
        wandb.watch(adaptor_domain_model, log="all", idx=1)
        wandb.watch(adaptor_global_model, log="all", idx=2)
        wandb.watch(combine_features_map_model, log="all", idx=3)
        classifier_id = 3

    # Init optimizers and loss functions
    optim_fusion = Adam(
        combine_features_map_model.parameters(), lr=cfg.train_args["learning_rate"]
    )
    loss_fn = UFDDeepInfoMaxLossModel().to(device)
    optim = Adam(
        [
            {"params": adaptor_domain_model.parameters()},
            {"params": adaptor_global_model.parameters()},
        ],
        lr=cfg.train_args["learning_rate"],
    )
    loss_optim = Adam(loss_fn.parameters(), lr=cfg.train_args["learning_rate"])

    # Generate train/val dataset
    train_data, valid_data = generate_train_val_dataset(cfg)

    # Init all logs record
    adaptor_loss_log = []
    train_loss_log = {}
    train_acc_log = {}
    val_loss_log = {}
    val_acc_log = {}
    best_val_loss_log = {}
    best_val_acc_log = {}

    # Outer loop training unsupervised models
    for epoch in range(cfg.train_args["unsupervised_epochs"]):
        a_loss = train_adaptor(
            train_data["raw"],
            cfg.train_args["unsupervised_model_batch_size"],
            device,
            adaptor_global_model,
            adaptor_domain_model,
            optim,
            loss_optim,
            loss_fn,
        )

        if cfg.use_wandb:
            wandb.log({"adaptor-loss": a_loss})

        adaptor_loss_log.append(a_loss)
        train_loss_log[epoch] = {}
        train_acc_log[epoch] = {}
        val_loss_log[epoch] = {}
        val_acc_log[epoch] = {}

        # Inner loop training all classifiers models
        classifiers = create_classifiers(cfg)
        for domain in classifiers.keys():

            # Add model watch for classifier models
            if cfg.use_wandb:
                classifier_id += 1
                wandb.watch(classifiers[domain]["model"], log="all", idx=classifier_id)

            train_loss_log[epoch][domain] = []
            train_acc_log[epoch][domain] = []
            val_loss_log[epoch][domain] = {}
            val_acc_log[epoch][domain] = {}
            for ep in range(cfg.train_args["classifier_epochs"]):
                if ep > cfg.train_args["warmup_epochs"]:
                    train_loss, train_acc = train_classifier(
                        train_data[domain],
                        cfg.train_args["classifier_batch_size"],
                        device,
                        classifiers[domain]["optimizer"],
                        optim_fusion,
                        adaptor_global_model,
                        adaptor_domain_model,
                        classifiers[domain]["model"],
                        combine_features_map_model,
                        classifiers[domain]["criterion"],
                    )

                    train_loss_log[epoch][domain].append(train_loss)
                    train_acc_log[epoch][domain].append(train_acc)

                    target_domains = sourcedomain2targetdomains[domain]
                    for tlang, tdom in list(
                        product(cfg.train_args["target_languages"], target_domains)
                    ):
                        combi_name = tlang + "_" + tdom
                        full_combi_name = domain + "_" + combi_name

                        if full_combi_name not in best_val_loss_log.keys():
                            best_val_loss_log[full_combi_name] = (
                                float("inf"),
                                epoch,
                                ep,
                            )
                        if full_combi_name not in best_val_acc_log.keys():
                            best_val_acc_log[full_combi_name] = (
                                float("-inf"),
                                epoch,
                                ep,
                            )

                        if (
                            ep == cfg.train_args["warmup_epochs"] + 1
                        ):  # Init list for new keys
                            val_loss_log[epoch][domain][combi_name] = []
                            val_acc_log[epoch][domain][combi_name] = []

                        val_loss, val_acc = test_classifier(
                            valid_data[tlang][tdom],
                            cfg.train_args["classifier_batch_size"],
                            device,
                            adaptor_global_model,
                            adaptor_domain_model,
                            classifiers[domain]["model"],
                            combine_features_map_model,
                            classifiers[domain]["criterion"],
                        )

                        if cfg.use_wandb:
                            wandb.log(
                                {
                                    f"{full_combi_name}-train-loss": train_loss,
                                    f"{full_combi_name}-train-acc": train_acc,
                                    f"{full_combi_name}-val-loss": val_loss,
                                    f"{full_combi_name}-val-acc": val_acc,
                                }
                            )

                        val_loss_log[epoch][domain][combi_name].append(val_loss)
                        val_acc_log[epoch][domain][combi_name].append(val_acc)

                        if val_loss < best_val_loss_log[full_combi_name][0]:
                            best_val_loss_log[full_combi_name] = (val_loss, epoch, ep)
                            best_val_acc_log[full_combi_name] = (val_acc, epoch, ep)

                            logging.info(
                                f"Found new best for {full_combi_name}, \
                                Epoch {epoch}, ep {ep}, Loss {val_loss:.3f}, Acc {val_acc:.3f}"
                            )

                            save_models(
                                cfg,
                                full_combi_name,
                                adaptor_global_model,
                                adaptor_domain_model,
                                combine_features_map_model,
                                classifiers[domain]["model"],
                            )

                            logging.info("New best models saved")

    if cfg.verbose:
        with open(cfg.cache_folder + "adaptor_loss.pickle", "wb") as handle:
            pickle.dump(adaptor_loss_log, handle)
        with open(cfg.cache_folder + "train_loss.pickle", "wb") as handle:
            pickle.dump(train_loss_log, handle)
        with open(cfg.cache_folder + "train_acc.pickle", "wb") as handle:
            pickle.dump(train_acc_log, handle)
        with open(cfg.cache_folder + "val_loss.pickle", "wb") as handle:
            pickle.dump(val_loss_log, handle)
        with open(cfg.cache_folder + "val_acc.pickle", "wb") as handle:
            pickle.dump(val_acc_log, handle)
        with open(cfg.cache_folder + "best_val_loss.pickle", "wb") as handle:
            pickle.dump(best_val_loss_log, handle)
        with open(cfg.cache_folder + "best_val_acc.pickle", "wb") as handle:
            pickle.dump(best_val_acc_log, handle)

    return (
        adaptor_loss_log,
        train_loss_log,
        train_acc_log,
        val_loss_log,
        val_acc_log,
        best_val_loss_log,
        best_val_acc_log,
    )


if __name__ == "__main__":
    cfg = parse_args_and_load_config()
    (
        adaptor_loss_log,
        train_loss_log,
        train_acc_log,
        val_loss_log,
        val_acc_log,
        best_val_loss_log,
        best_val_acc_log,
    ) = train(cfg)
