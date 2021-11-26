import argparse
import json
import logging
import os
import pathlib
import pickle
from typing import Dict, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam

from .config import (
    UFDAdaptorGlobalConfig,
    UFDAdaptorDomainConfig,
    UFDCombineFeaturesMapConfig,
    UFDClassifierConfig,
    UFDEmbeddingConfig,
)
from .data_class import UFDArguments
from .tokenization import UFDTokenizer
from .modeling import (
    UFDAdaptorGlobalModel,
    UFDAdaptorDomainModel,
    UFDCombineFeaturesMapModel,
    UFDClassifierModel,
    UFDEmbeddingModel,
)


logging.basicConfig(level=logging.DEBUG)


def create_unsupervised_models(
    cfg: UFDArguments,
) -> Tuple[UFDAdaptorDomainModel, UFDAdaptorGlobalModel, UFDCombineFeaturesMapModel]:
    """Helper function to create the unsupervised model group.

    Args:
        cfg (UFDArguments): UFDArguments config load from configuration file

    Returns:
        Tuple[UFDAdaptorDomainModel, UFDAdaptorGlobalModel, UFDCombineFeaturesMapModel]:
            return the initialize unsupervised model group.
    """
    device = torch.device(cfg.device)
    adaptor_domain_config = UFDAdaptorDomainConfig(
        in_dim=cfg.train_args["in_dim"],
        dim_hidden=cfg.train_args["dim_hidden"],
        out_dim=cfg.train_args["out_dim"],
        initrange=cfg.train_args["initrange"],
    )
    adaptor_global_config = UFDAdaptorGlobalConfig(
        in_dim=cfg.train_args["in_dim"],
        dim_hidden=cfg.train_args["dim_hidden"],
        out_dim=cfg.train_args["out_dim"],
        initrange=cfg.train_args["initrange"],
    )
    combine_features_map_config = UFDCombineFeaturesMapConfig(
        embed_dim=cfg.train_args["in_dim"], initrange=cfg.train_args["initrange"]
    )
    return (
        UFDAdaptorDomainModel(adaptor_domain_config).to(device),
        UFDAdaptorGlobalModel(adaptor_global_config).to(device),
        UFDCombineFeaturesMapModel(combine_features_map_config).to(device),
    )


def load_trained_models(
    cfg: UFDArguments,
    source_domain: str,
    target_language: str,
    target_domain: str,
) -> Tuple[
    UFDAdaptorDomainModel,
    UFDAdaptorGlobalModel,
    UFDCombineFeaturesMapModel,
    UFDClassifierModel,
]:
    """Helper function to load pretrained config and model weights for both supervised and unsupervised
        models.

    Args:
        cfg (UFDArguments): UFDArguments config load from configuration file
        source_domain (str): name of source domain
        target_language (str): name of target language
        target_domain (str): name of target domain

    Returns:
        Tuple[ UFDAdaptorDomainModel, UFDAdaptorGlobalModel, UFDCombineFeaturesMapModel, UFDClassifierModel, ]:
            return all supervised, unsupervised models with pretrained weights loaded.
    """
    device = torch.device(cfg.device)
    full_combi = (
        cfg.model_folder
        + "/"
        + source_domain
        + "_"
        + target_language
        + "_"
        + target_domain
    )
    adaptor_domain_model_dir = full_combi + "_adaptor_domain/"
    adaptor_global_model_dir = full_combi + "_adaptor_global/"
    maper_model_dir = full_combi + "_maper/"
    classifier_model_dir = full_combi + "_classifier/"

    config_filename = cfg.eval_args["config_filename"]
    model_filename = cfg.eval_args["model_filename"]

    adaptor_domain_config = UFDAdaptorDomainConfig.from_pretrained(
        adaptor_domain_model_dir + config_filename
    )
    adaptor_global_config = UFDAdaptorGlobalConfig.from_pretrained(
        adaptor_global_model_dir + config_filename
    )
    maper_config = UFDCombineFeaturesMapConfig.from_pretrained(
        maper_model_dir + config_filename
    )
    classifier_config = UFDClassifierConfig.from_pretrained(
        classifier_model_dir + config_filename
    )
    return (
        UFDAdaptorDomainModel.from_pretrained(
            adaptor_domain_model_dir + model_filename, config=adaptor_domain_config
        ).to(device),
        UFDAdaptorGlobalModel.from_pretrained(
            adaptor_global_model_dir + model_filename, config=adaptor_global_config
        ).to(device),
        UFDCombineFeaturesMapModel.from_pretrained(
            maper_model_dir + model_filename, config=maper_config
        ).to(device),
        UFDClassifierModel.from_pretrained(
            classifier_model_dir + model_filename, config=classifier_config
        ).to(device),
    )


def load_unlabelled(filename: str) -> List:
    """Helper function to load unlabelled dataset for unsupervised training.

    Args:
        filename (str): filename of dataset

    Returns:
        List: list of dataset by line.
    """
    data = []
    with open(filename, "r") as F:
        for line in F:
            # 0 is used in the research code
            data.append([0, line.strip()])
    return data


def load_labelled(filename: str) -> List:
    """Helper function to load labelled dataset for supervised training.

    Args:
        filename (str): filename of dataset

    Returns:
        List: list of dataset by line.
    """
    data = []
    with open(filename, "r") as F:
        for line in F:
            data.append(line.split("\t"))
    return data


def extract_embeddings(
    cfg: UFDArguments, dataset: List, tokenizer: UFDTokenizer, model: UFDEmbeddingModel
) -> List:
    """Helper function to extract embeddings with the UFD embedding model.

    Args:
        cfg (UFDArguments): UFDArguments config load from configuration file
        dataset (List): list of dataset by line
        tokenizer (UFDTokenizer): UFD tokenizer class instance
        model (UFDEmbeddingModel): UFD embedding model class instance

    Returns:
        List: return list of generated embeddings.
    """
    device = torch.device(cfg.device)
    model.eval()
    embeded_data = []
    with torch.no_grad():
        for pair in dataset:
            tokens = tokenizer(pair[1])["input_ids"].to(device)
            last_layer_features = model(tokens)[0]  # 1 * length * embedding_size
            mean_features = torch.mean(last_layer_features, dim=1)  # 1 * embedding_size
            tem_label = torch.tensor([[float(pair[0])]]).to(device)
            new_pair = torch.cat((tem_label, mean_features), dim=1)
            embeded_data.append(new_pair)
    return embeded_data


def create_train_embeddings(
    cfg: UFDArguments, tokenizer: UFDTokenizer, model: UFDEmbeddingModel
) -> Dict:
    """Helper function to generate training dataset for supervised and unsupervised training.

    Args:
        cfg (UFDArguments): UFDArguments config load from configuration file
        tokenizer (UFDTokenizer): UFD tokenizer class instance
        model (UFDEmbeddingModel): UFD embedding model class instance

    Returns:
        Dict: dictionary of dataset embeddings for supervised and unsupervised dataset
    """
    embeddings_dict = {}
    source_domains_list = cfg.train_args["source_domains"] + ["raw"]
    for source_domain in source_domains_list:
        if source_domain == "raw":
            # assume will only have one source language
            filepath = (
                f"{cfg.data_folder}/{cfg.train_args['unsupervised_dataset_filename']}"
            )
            dataset = load_unlabelled(filepath)
        else:
            filepath = f"{cfg.data_folder}/{cfg.train_args['source_language']}/{source_domain}/{cfg.train_args['train_filename']}"
            dataset = load_labelled(filepath)
        dataset_embedding = extract_embeddings(cfg, dataset, tokenizer, model)
        embeddings_dict[source_domain] = dataset_embedding
    return embeddings_dict


def create_val_test_embeddings(
    cfg: UFDArguments,
    tokenizer: UFDTokenizer,
    model: UFDEmbeddingModel,
    dataset_type: str,
) -> Dict:
    """Helper function to generate validation dataset for supervised and unsupervised training.

    Args:
        cfg (UFDArguments): UFDArguments config load from configuration file
        tokenizer (UFDTokenizer): UFD tokenizer class instance
        model (UFDEmbeddingModel): UFD embedding model class instance

    Returns:
        Dict: dictionary of dataset embeddings for supervised and unsupervised dataset
    """
    embeddings_dict = {}
    if dataset_type == "valid":
        target_languages_list = cfg.train_args["target_languages"]
        target_domains_list = cfg.train_args["target_domains"]
        filename = cfg.train_args["val_filename"]
    elif dataset_type == "test":
        target_languages_list = cfg.eval_args["target_languages"]
        target_domains_list = cfg.eval_args["target_domains"]
        filename = cfg.eval_args["test_filename"]
    for target_language in target_languages_list:
        embeddings_dict[target_language] = {}
        for target_domain in target_domains_list:
            filepath = f"{cfg.data_folder}/{target_language}/{target_domain}/{filename}"
            dataset = load_labelled(filepath)
            dataset_embedding = extract_embeddings(cfg, dataset, tokenizer, model)
            embeddings_dict[target_language][target_domain] = dataset_embedding
    return embeddings_dict


def create_dataset_embedding(cfg: UFDArguments, dataset_type: str) -> Dict:
    """Main helper wrapper function to generate datasets.

    Args:
        cfg (UFDArguments): UFDArguments config load from configuration file
        dataset_type (str): type of dataset to generate embeddings

    Raises:
        Exception: raise exception for unknown dataset types

    Returns:
        Dict: return dictionary of dataset embeddings.
    """
    device = torch.device(cfg.device)
    config = UFDEmbeddingConfig.from_pretrained(cfg.embedding_model_name)
    model = UFDEmbeddingModel.from_pretrained(
        cfg.embedding_model_name, config=config
    ).to(device)
    tokenizer = UFDTokenizer.from_pretrained(cfg.embedding_model_name)

    if dataset_type == "train":
        dataset_embedding_dict = create_train_embeddings(
            cfg,
            tokenizer,
            model,
        )
    elif dataset_type in ["test", "valid"]:
        dataset_embedding_dict = create_val_test_embeddings(
            cfg, tokenizer, model, dataset_type
        )
    else:
        raise Exception(
            "Invalid value for dataset_type, dataset_type should be train/ valid/ test"
        )

    return dataset_embedding_dict


def create_classifiers(cfg: UFDArguments) -> Dict:
    """Helper function to generate all classifier models, criterion and optimizer.
        One set of classifier, criterion and optimizer required per source domain.

    Args:
        cfg (UFDArguments): UFDArguments config load from configuration file

    Returns:
        Dict: return dictionary of models, criterion and optimizer group by source domain.
    """
    device = torch.device(cfg.device)
    config = UFDClassifierConfig(
        embed_dim=cfg.train_args["out_dim"],
        num_class=cfg.train_args["num_class"],
        initrange=cfg.train_args["initrange"],
    )
    classifiers = {}
    for domain in cfg.train_args["source_domains"]:
        classifiers[domain] = {}
        classifiers[domain]["model"] = UFDClassifierModel(config).to(device)
        classifiers[domain]["criterion"] = nn.CrossEntropyLoss().to(device)
        classifiers[domain]["optimizer"] = Adam(
            classifiers[domain]["model"].parameters(),
            lr=cfg.train_args["learning_rate"],
        )
    return classifiers


def get_source2target_domain_mapping(
    source_domains: List[str], target_domains: List[str]
) -> Dict[str, List[str]]:
    """Helper function to return cross domains keys for each domains.

    Args:
        source_domains (List[str]): list of all source domains
        target_domains (List[str]): list of all target domains

    Returns:
        Dict[str, List[str]]: return a list of cross domains per domain keys.
    """
    mapping = {}
    for dom in source_domains:
        mapping[dom] = [d for d in target_domains if d != dom]
    return mapping


def generate_train_val_dataset(cfg: UFDArguments) -> Tuple[Dict[str, float]]:
    """Helper function to generate train and validation datasets.
        Load dataset object from cache if available, else call the dataset embeddings creation methods.

    Args:
        cfg (UFDArguments): UFDArguments config load from configuration file

    Returns:
        Tuple(Dict[str, float]): return the generated train and validation dictionaries.
    """
    use_cache = (
        "train_cache_filename" in cfg.train_args.keys()
        and "val_cache_filename" in cfg.train_args.keys()
    )

    if use_cache:
        train_cache_path = str(
            pathlib.Path(cfg.cache_folder).joinpath(
                cfg.train_args["train_cache_filename"]
            )
        )
        if os.path.isfile(train_cache_path):
            with open(train_cache_path, "rb") as handle:
                train_data = pickle.load(handle)
            logging.info("Train data loaded from cache")

            for source_domain in cfg.train_args["source_domains"]:
                assert (
                    source_domain in train_data.keys()
                ), "Source domain key does not exist in cached data, consider deleting the cache and rerun the code"
        else:
            train_data = create_dataset_embedding(cfg, dataset_type="train")
            if not os.path.isdir(cfg.cache_folder):
                os.mkdir(cfg.cache_folder)
            with open(train_cache_path, "wb") as handle:
                pickle.dump(train_data, handle)
            logging.info("Train data saved in cache")

        valid_cache_path = str(
            pathlib.Path(cfg.cache_folder).joinpath(
                cfg.train_args["val_cache_filename"]
            )
        )
        if os.path.isfile(valid_cache_path):
            with open(valid_cache_path, "rb") as handle:
                valid_data = pickle.load(handle)
            logging.info("Validation data loaded from cache")

            for target_language in cfg.train_args["target_languages"]:
                assert (
                    target_language in valid_data.keys()
                ), "Target language key does not exist in cached data, consider deleting the cache and rerun the code"

                for target_domain in cfg.train_args["target_domains"]:
                    assert (
                        target_domain in valid_data[target_language].keys()
                    ), "Target domain key does not exist in cached data, consider deleting the cache and rerun the code"

        else:
            valid_data = create_dataset_embedding(cfg, dataset_type="valid")
            with open(valid_cache_path, "wb") as handle:
                pickle.dump(valid_data, handle)
            logging.info("Validation data saved in cache")

    else:
        train_data = create_dataset_embedding(cfg, dataset_type="train")
        valid_data = create_dataset_embedding(cfg, dataset_type="valid")

    return train_data, valid_data


def parse_args_and_load_config(config_path: str = "config/ufd_config.json"):
    """Args parser helper method

    Returns:
        UFDArguments: UFDArguments instance with parsed args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=config_path)
    args = parser.parse_args()
    with open(args.config, "r") as cfg_file:
        cfg = json.load(cfg_file)
    ufd_args = UFDArguments(**cfg)
    return ufd_args
