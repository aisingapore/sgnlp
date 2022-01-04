import argparse
from collections import namedtuple
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


def load_and_process_senticnet(config: SenticGCNTrainArgs) -> dict[str, float]:
    """
    Helper method to load and process senticnet. Default is SenticNet 5.0.
    If a saved preprocess senticnet file is available, and save flag is set to false, it will be loaded from file instead.
    Source:
    https://github.com/BinLiang-NLP/Sentic-GCN/tree/main/senticnet-5.0

    Args:
        config (SenticGCNTrainArgs): SenticGCN training config

    Returns:
        dict[str, float]: return dictionary with concept word as keys and intensity as values.
    """
    saved_senticnet_file_path = pathlib.Path(config.saved_preprocessed_senticnet_file_path)
    if saved_senticnet_file_path.exists() and not config.save_preprocessed_senticnet:
        with open(saved_senticnet_file_path, "r") as f:
            sentic_dict = pickle.load(f)
    else:
        senticnet_file_path = pathlib.Path(config.senticnet_word_file_path)
        sentic_dict = {}
        with open(senticnet_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items = line.split("\t")
                if "_" in items[0]:
                    continue  # skip words with '_'
                sentic_dict[items[0]] = items[-1]
        if config.save_preprocessed_senticnet:
            with open(saved_senticnet_file_path, "wb") as f:
                pickle.dump(sentic_dict, f)
    return sentic_dict


class SenticGCNDatasetGenerator(Dataset):
    def __init__(
        self,
        dataset_type: str,
        config: SenticGCNTrainArgs,
        tokenizer: PreTrainedTokenizer,
    ):
        self.config = config
        self.tokenizer = tokenizer

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def _read_raw_dataset(self, dataset_type: str) -> list[namedtuple]:
        """
        Private helper method to read raw dataset files based on requested type (e.g. Train or Test).

        Args:
            dataset_type (str): Type of dataset files to read. Train or Test.

        Returns:
            list[namedtuple]: list of namedtuples consisting of the full text, the aspect and polarity.
        """
        file_path = self.config.dataset_train["raw"] if dataset_type == "train" else self.config.dataset_test["raw"]
        RawDataSet = namedtuple("RawDataSet", ["text", "aspect", "polarity"])
        with open(file_path, "r", encoding="utf-8", newline="\n", errors="ignore") as f:
            lines = f.readlines()
        output = []
        for i in range(0, len(lines), 3):
            output.append(
                RawDataSet(lines[i].lower().strip(), lines[i + 1].lower().strip(), lines[i + 2].lower().strip())
            )
        return output

    def _read_dependency_senticnet_graph(self, dataset_type: str) -> dict[str, np.ndarray]:
        """
        Private helpder method to read senticnet graph dataset based on requested type (i.e. Train or Test).

        Args:
            dataset_type (str): Type of dataset files to read. Train or Test.

        Returns:
            dict[str, np.ndarray]: dictionary with
        """
        file_path = (
            self.config.dataset_train["dependency_sencticnet_graph"]
            if dataset_type == "train"
            else self.config.dataset_test["dependency_sencticnet_graph"]
        )
        with open(file_path, "rb") as f:
            graph = pickle.load(f)
        return graph

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
