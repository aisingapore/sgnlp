import argparse
import json
import logging
import pickle
import random
import pathlib
import requests
import urllib
from typing import Dict, List, Union

import numpy as np
import torch
from torch.utils.data import random_split, Dataset
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from data_class import SenticGCNTrainArgs


def parse_args_and_load_config(
    config_path: str = "config/senticnet_gcn_config.json",
) -> SenticGCNTrainArgs:
    """Get config from config file using argparser

    Returns:
        SenticGCNTrainArgs: SenticGCNTrainArgs instance populated from config
    """
    parser = argparse.ArgumentParser(description="SenticASGCN Training")
    parser.add_argument("--config", type=str, default=config_path)
    args = parser.parse_args()

    cfg_path = pathlib.Path(__file__).parent / args.config
    with open(cfg_path, "r") as cfg_file:
        cfg = json.load(cfg_file)

    sentic_asgcn_args = SenticGCNTrainArgs(**cfg)
    return sentic_asgcn_args


def set_random_seed(seed: int = 776) -> None:
    """Helper method to set random seeds for python, numpy and torch

    Args:
        seed (int, optional): seed value to set. Defaults to 776.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def download_tokenizer_files(
    base_url: str,
    save_folder: str,
    files: list[str] = ["special_tokens_map.json", "tokenizer_config.json", "vocab.pkl"],
) -> None:
    """
    Helper method to download files from online storage.

    Args:
        base_url (str): Url string to storage folder.
        save_folder (str): Local folder to save downloaded files. Folder will be created if it does not exists.
    """
    file_paths = [urllib.parse.urljoin(base_url, file_name) for file_name in files]
    for file_path in file_paths:
        download_url_file(file_path, save_folder)


def download_url_file(url: str, save_folder: str) -> None:
    """
    Helper method to download and save url file.

    Args:
        url (str): Url of file to download.
        save_folder (str): Folder to save downloaded file. Will be created if it does not exists.
    """
    save_folder_path = pathlib.Path(save_folder)
    save_folder_path.mkdir(exist_ok=True)
    fn_start_pos = url.rfind("/") + 1
    file_name = url[fn_start_pos:]
    save_file_path = save_folder_path.joinpath(file_name)
    req = requests.get(url)
    if req.status_code == requests.codes.ok:
        with open(save_file_path, "wb") as f:
            for data in req:
                f.write(data)
    else:
        logging.error(f"Fail to request files from {url}.")


def load_word_vec(word_vec_file_path: str, vocab: Dict[str, int], embed_dim: int = 300) -> Dict[str, np.asarray]:
    """
    Helper method to load word vectors from file (e.g. GloVe) for each word in vocab.

    Args:
        word_vec_file_path (str): full file path to word vectors.
        vocab (Dict[str, int]): dictionary of vocab word as key and word index as values.
        embed_dim (int, optional): embedding dimension. Defaults to 300.

    Returns:
        Dict[str, np.asarray]: dictionary with words as key and word vectors as values.
    """
    with open(word_vec_file_path, "r", encoding="utf-8", newline="\n", errors="ignore") as fin:
        word_vec = {}
        for line in fin:
            tokens = line.rstrip().split()
            word, vec = " ".join(tokens[:-embed_dim]), tokens[-embed_dim:]
            if word in vocab.keys():
                word_vec[word] = np.asarray(vec, dtype="float32")
    return word_vec


def build_embedding_matrix(
    word_vec_file_path: str,
    vocab: Dict[str, int],
    embed_dim: int = 300,
    save_embed_matrix: bool = False,
    save_embed_file_path: str = None,
) -> np.ndarray:
    """
    Helper method to generate an embedding matrix.

    Args:
        word_vec_file_path (str): full file path to word vectors.
        vocab (Dict[str, int]): dictionary of vocab word as key and word index as values.
        embed_dim (int, optional): embedding dimension. Defaults to 300.
        save_embed_matrix (bool, optional): flag to indicate if . Defaults to False.
        save_embed_directory (str, optional): [description]. Defaults to None.

    Returns:
        np.array: numpy array of embedding matrix
    """
    embedding_matrix = np.zeros((len(vocab), embed_dim))
    embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(embed_dim), 1 / np.sqrt(embed_dim), (1, embed_dim))
    word_vec = load_word_vec(word_vec_file_path, vocab, embed_dim)
    for word, idx in vocab.items():
        vec = word_vec.get(word)
        if vec is not None:
            embedding_matrix[idx] = vec

    if save_embed_matrix:
        save_file_path = pathlib.Path(save_embed_file_path)
        if not save_file_path.exists():
            save_file_path.parent.mkdir(exist_ok=True)
        with open(save_file_path, "wb") as fout:
            pickle.dump(embedding_matrix, fout)

    return embedding_matrix


class ABSADataset(object):
    """
    Data class to hold dataset for training.
    """

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def generate_senticgcn_dataset(cfg: SenticGCNTrainArgs) -> dict[str, torch.Tensor]:
    # TODO: add senticgcn dataset prep
    pass


def generate_senticgcn_bert_dataset(cfg: SenticGCNTrainArgs) -> dict[str, torch.Tensor]:
    # TODO: add senticgcn bert dataset prep
    pass


def generate_train_val_dataset(cfg: SenticGCNTrainArgs) -> dict[str, torch.Tensor]:
    if cfg.model == "senticgcn":
        return generate_senticgcn_dataset(cfg)
    elif cfg.model == "senticgcnbert":
        return generate_senticgcn_bert_dataset(cfg)


class ABSADatasetReader:
    def __init__(
        self,
        config: SenticGCNTrainArgs,
        tokenizer: PreTrainedTokenizer,
    ):
        self.cfg = config
        self.tokenizer = tokenizer
        self.embedding_matrix = build_embedding_matrix(
            config.word_vec_file_path,
            tokenizer.vocab,
            config.embed_dim,
            config.save_embedding_matrix,
            config.saved_embedding_matrix_file_path,
        )
        self.train_data = ABSADataset(ABSADatasetReader.__read_data__(self.cfg.dataset_train, tokenizer))
        self.test_data = ABSADataset(ABSADatasetReader.__read_data__(self.cfg.dataset_test, tokenizer))
        if config.valset_ratio:
            valset_len = int(len(self.train_data) * config.valset_ratio)
            self.train_data, self.val_data = random_split(
                self.train_data, (len(self.train_data) - valset_len, valset_len)
            )
        else:
            self.val_data = self.test_data

    @staticmethod
    def __read_data__(datasets: Dict[str, str], tokenizer: PreTrainedTokenizer):
        # Read raw data, graph data and tree data
        with open(datasets["raw"], "r", encoding="utf-8", newline="\n", errors="ignore") as fin:
            lines = fin.readlines()
        with open(datasets["graph"], "rb") as fin_graph:
            idx2graph = pickle.load(fin_graph)

        # Prep all data
        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].lower().strip()
            text_indices = tokenizer(f"{text_left} {aspect} {text_right}")
            context_indices = tokenizer(f"{text_left} {text_right}")
            aspect_indices = tokenizer(aspect)
            left_indices = tokenizer(text_left)
            polarity = int(polarity) + 1
            dependency_graph = idx2graph[i]

            data = {
                "text_indices": text_indices,
                "context_indices": context_indices,
                "aspect_indices": aspect_indices,
                "left_indices": left_indices,
                "polarity": polarity,
                "dependency_graph": dependency_graph,
            }
            all_data.append(data)
        return all_data
