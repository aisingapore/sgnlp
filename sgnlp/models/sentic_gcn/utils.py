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


def load_and_process_senticnet(
    senticnet_file_path: str = None,
    save_preprocessed_senticnet: bool = False,
    saved_preprocessed_senticnet_file_path: str = "senticnet.pkl",
) -> dict[str, float]:
    """
    Helper method to load and process senticnet. Default is SenticNet 5.0.
    If a saved preprocess senticnet file is available, and save flag is set to false, it will be loaded from file instead.
    Source:
    https://github.com/BinLiang-NLP/Sentic-GCN/tree/main/senticnet-5.0

    Args:
        senticnet_file_path (str): File path to senticnet 5.0 file.
        save_preprocessed_senticnet (bool): Flag to indicate if processed senticnet should be saved.
        saved_preprocessed_senticnet_file_path: (str): File path to saved preprocessed senticnet file.

    Returns:
        dict[str, float]: return dictionary with concept word as keys and intensity as values.
    """
    saved_senticnet_file_path = pathlib.Path(saved_preprocessed_senticnet_file_path)
    if saved_senticnet_file_path.exists() and not save_preprocessed_senticnet:
        with open(saved_senticnet_file_path, "r") as f:
            sentic_dict = pickle.load(f)
    else:
        senticnet_file_path = pathlib.Path(senticnet_file_path)
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
        if save_preprocessed_senticnet:
            saved_senticnet_file_path.parent.mkdir(exist_ok=True)
            with open(saved_senticnet_file_path, "wb") as f:
                pickle.dump(sentic_dict, f)
    return sentic_dict


def generate_dependency_adj_matrix(text: str, aspect: str, senticnet: dict[str, float], spacy_pipeline) -> np.ndarray:
    """
    Helper method to generate senticnet depdency adj matrix.

    Args:
        text (str): input text to process
        aspect (str): aspect from input text
        senticnet (dict[str, float]): dictionary of preprocessed senticnet. See load_and_process_senticnet()
        spacy_pipeline : Spacy pretrained pipeline (e.g. 'en_core_web_sm')

    Returns:
        np.ndarray: return ndarry representing adj matrix.
    """
    document = spacy_pipeline(text)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype("float32")
    for token in document:
        sentic = float(senticnet[str(token)]) + 1.0 if str(token) in senticnet else 0
        if str(token) in aspect:
            sentic += 1.0
        if token.i < seq_len:
            matrix[token.i][token.i] = 1.0 * sentic
            for child in token.children:
                if str(child) in aspect:
                    sentic += 1.0
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1.0 * sentic
                    matrix[child.i][token.i] = 1.0 * sentic
    return matrix


class SenticGCNDataset(Dataset):
    """
    Data class for SenticGCN dataset.
    """

    def __init__(self, data: list[dict[str, torch.Tensor]]) -> None:
        self.data = data

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.data[index]

    def __len__(self):
        return len(self.data)


class SenticGCNDatasetGenerator:
    def __init__(self, config: SenticGCNTrainArgs):
        self.config = config

    def _read_raw_dataset(self, dataset_type: str) -> list[namedtuple]:
        """
        Private helper method to read raw dataset files based on requested type (e.g. Train or Test).

        Args:
            dataset_type (str): Type of dataset files to read. Train or Test.

        Returns:
            list[namedtuple]: list of namedtuples consisting of the full text, the aspect and polarity.
        """
        file_path = self.config.dataset_train if dataset_type == "train" else self.config.dataset_test
        RawDataSet = namedtuple("RawDataSet", ["text", "aspect", "polarity"])
        with open(file_path, "r", encoding="utf-8", newline="\n", errors="ignore") as f:
            lines = f.readlines()
        output = []
        for i in range(0, len(lines), 3):
            output.append(
                RawDataSet(lines[i].lower().strip(), lines[i + 1].lower().strip(), lines[i + 2].lower().strip())
            )
        return output

    # @staticmethod
    # def __read_data__(datasets: Dict[str, str], tokenizer: PreTrainedTokenizer):
    #     # Read raw data, graph data and tree data
    #     with open(datasets["raw"], "r", encoding="utf-8", newline="\n", errors="ignore") as fin:
    #         lines = fin.readlines()
    #     with open(datasets["graph"], "rb") as fin_graph:
    #         idx2graph = pickle.load(fin_graph)

    #     # Prep all data
    #     all_data = []
    #     for i in range(0, len(lines), 3):
    #         text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
    #         aspect = lines[i + 1].lower().strip()
    #         polarity = lines[i + 2].lower().strip()
    #         text_indices = tokenizer(f"{text_left} {aspect} {text_right}")
    #         context_indices = tokenizer(f"{text_left} {text_right}")
    #         aspect_indices = tokenizer(aspect)
    #         left_indices = tokenizer(text_left)
    #         polarity = int(polarity) + 1
    #         dependency_graph = idx2graph[i]

    #         data = {
    #             "text_indices": text_indices,
    #             "context_indices": context_indices,
    #             "aspect_indices": aspect_indices,
    #             "left_indices": left_indices,
    #             "polarity": polarity,
    #             "dependency_graph": dependency_graph,
    #         }
    #         all_data.append(data)
    #     return all_data
