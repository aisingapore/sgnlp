import argparse
import json
import logging
import pickle
import random
import pathlib
import requests
import urllib
from typing import Dict, Tuple, Union

import numpy as np
import spacy
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


def pad_and_truncate(
    sequence: list[float],
    max_len: int,
    dtype: str = "int64",
    padding: str = "post",
    truncating: str = "post",
    value: int = 0,
):
    """
    Helper method for padding and truncating text and aspect segment.

    Args:
        sequence (list[float]): input sequence of indices
        max_len (int): maximum len to pad
        dtype (str, optional): data type to cast indices. Defaults to "int64".
        padding (str, optional): type of padding, 'pre' or 'post'. Defaults to "post".
        truncating (str, optional): type of truncating, 'pre' or 'post'. Defaults to "post".
        value (int, optional): value used for padding. Defaults to 0.

    Returns:
        [type]: [description]
    """
    seq_arr = (np.ones(max_len) * value).astype(dtype)
    trunc = sequence[-max_len:] if truncating == "pre" else sequence[:max_len]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == "post":
        seq_arr[: len(trunc)] = trunc
    else:
        seq_arr[-len(trunc) :] = trunc
    return seq_arr


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
) -> Dict[str, float]:
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
        Dict[str, float]: return dictionary with concept word as keys and intensity as values.
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


def generate_dependency_adj_matrix(text: str, aspect: str, senticnet: Dict[str, float], spacy_pipeline) -> np.ndarray:
    """
    Helper method to generate senticnet depdency adj matrix.

    Args:
        text (str): input text to process
        aspect (str): aspect from input text
        senticnet (Dict[str, float]): dictionary of preprocessed senticnet. See load_and_process_senticnet()
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

    def __init__(self, data: list[Dict[str, torch.Tensor]]) -> None:
        self.data = data

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.data[index]

    def __len__(self):
        return len(self.data)


class SenticGCNDatasetGenerator:
    """
    Main dataset generator class to preprocess raw dataset file.
    """

    def __init__(self, config: SenticGCNTrainArgs, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.senticnet = load_and_process_senticnet(
            config.senticnet_word_file_path,
            config.save_preprocessed_senticnet,
            config.saved_preprocessed_senticnet_file_path,
        )
        self.spacy_pipeline = spacy.load(config.spacy_pipeline)
        self.tokenizer = tokenizer
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if config.device is None
            else torch.device(config.device)
        )

    def _read_raw_dataset(self, dataset_type: str) -> list[str]:
        """
        Private helper method to read raw dataset files based on requested type (e.g. Train or Test).

        Args:
            dataset_type (str): Type of dataset files to read. Train or Test.

        Returns:
            list[str]: list of str consisting of the full text, aspect and polarity index.
        """
        file_path = self.config.dataset_train if dataset_type == "train" else self.config.dataset_test
        with open(file_path, "r", encoding="utf-8", newline="\n", errors="ignore") as f:
            lines = f.readlines()
        return lines

    def _generate_senticgcn_dataset(self, raw_data: list[str]) -> Dict[str, Union[BatchEncoding, torch.Tensor]]:
        """
        Data preprocess method to generate all indices required for SenticGCN model training.

        Args:
            raw_data (list[str]): list of text, aspect word and polarity read from raw dataset file.

        Returns:
            Dict[str, Union[BatchEncoding, torch.Tensor]]: return a dictionary of dataset sub-type and their tensors.
        """
        all_data = []
        for i in range(0, len(raw_data), 3):
            # Process full text, aspect and polarity index
            text_left, _, text_right = [s.lower().strip() for s in raw_data[i].partition("$T$")]
            aspect = raw_data[i + 1].lower().strip()
            full_text = f"{text_left} {aspect} {text_right}"
            polarity = raw_data[i + 2].strip()

            # Process indices
            text_indices = self.tokenizer(full_text, return_tensors="pt")
            aspect_indices = self.tokenizer(aspect, return_tensors="pt")
            left_indices = self.tokenizer(text_left, return_tensors="pt")
            polarity = int(polarity) + 1
            polarity = torch.tensor(polarity)
            graph = generate_dependency_adj_matrix(full_text, aspect, self.senticnet, self.spacy_pipeline)
            graph = torch.tensor(polarity)

            all_data.append(
                {
                    "text_indices": text_indices.to(self.device),
                    "aspect_indices": aspect_indices.to(self.device),
                    "left_indices": left_indices.to(self.device),
                    "polarity": polarity.to(self.device),
                    "sdat_graph": graph.to(self.device),
                }
            )
        return all_data

    def _generate_senticgcnbert_dataset(self, raw_data: list[str]) -> Dict[str, torch.Tensor]:
        """
        Data preprocess method to generate all indices required for SenticGCNBert model training.

        Args:
            raw_data (list[str]): list of text, aspect word and polarity read from raw dataset file.

        Returns:
            Dict[str, torch.Tensor]: return a dictionary of dataset sub-type and their tensors.
        """
        all_data = []
        max_len = self.config.max_len
        for i in range(0, len(raw_data), 3):
            # Process full text, aspect and polarity index
            text_left, _, text_right = [s.lower().strip() for s in raw_data[i].partition("$T$")]
            aspect = raw_data[i + 1].lower().strip()
            polarity = raw_data[i + 2].strip()
            full_text = f"{text_left} {aspect} {text_right}"
            full_text_with_bert_tokens = f"[CLS] {full_text} [SEP] {aspect} [SEP]"

            # Process indices
            text_indices = self.tokenizer(full_text, return_tensors="pt")
            aspect_indices = self.tokenizer(aspect, return_tensors="pt")
            left_indices = self.tokenizer(text_left, return_tensors="pt")
            polarity = int(polarity) + 1
            polarity = torch.tensor(polarity)

            # Process bert related indices
            text_bert_indices = self.tokenizer(full_text_with_bert_tokens)
            text_len = np.sum(text_indices["input_ids"] != 0)
            aspect_len = np.sum(aspect_indices["input_ids"] != 0)

            # array of [0] for texts including [CLS] and [SEP] and [1] for aspect and ending [SEP]
            concat_segment_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
            concat_segment_indices = pad_and_truncate(concat_segment_indices, max_len)
            concat_segment_indices = BatchEncoding({"input_ids": concat_segment_indices})
            concat_segment_indices = concat_segment_indices.convert_to_tensors("pt")

            # Process graph
            graph = generate_dependency_adj_matrix(full_text, aspect, self.senticnet, self.spacy_pipeline)
            sdat_graph = np.pad(
                graph,
                (
                    (0, max_len - graph.shape[0]),
                    (0, max_len - graph.shape[0]),
                ),
                "constant",
            )
            sdat_graph = BatchEncoding({"graph": sdat_graph})
            sdat_graph = sdat_graph.convert_to_tensors("pt")

            all_data.append(
                {
                    "text_indices": text_indices.to(self.device),
                    "aspect_indices": aspect_indices.to(self.device),
                    "left_indices": left_indices.to(self.device),
                    "text_bert_indices": text_bert_indices.to(self.device),
                    "bert_segment_indices": concat_segment_indices.to(self.device),
                    "polarity": polarity.to(self.device),
                    "sdat_graph": sdat_graph.to(self.device),
                }
            )
        return all_data

    def generate_datasets(self) -> Tuple[SenticGCNDataset, SenticGCNDataset, SenticGCNDataset]:
        """
        Main wrapper method to generate datasets for both SenticGCN and SenticGCNBert based on config.

        Returns:
            Tuple[SenticGCNDataset, SenticGCNDataset, SenticGCNDataset]:
                return SenticGCNDataset instances for train/val/test data.
        """
        # Read raw data from dataset files
        raw_train_data = self._read_raw_dataset(self.config.dataset_train)
        raw_test_data = self._read_raw_dataset(self.config.dataset_test)

        # Generate dataset dictionary
        if self.config.model == "senticgcn":
            train_data = self._generate_senticgcn_dataset(raw_train_data)
            test_data = self._generate_senticgcn_dataset(raw_test_data)
        else:
            train_data = self._generate_senticgcnbert_dataset(raw_train_data)
            test_data = self._generate_senticgcnbert_dataset(raw_test_data)
        # Train/Val/Test split
        if self.config.valset_ratio > 0:
            valset_len = int(len(train_data) * self.config.valset_ratio)
            train_data, val_data = random_split(train_data, (len(train_data) - valset_len, valset_len))
        else:
            val_data = test_data
        return SenticGCNDataset(train_data), SenticGCNDataset(val_data), SenticGCNDataset(test_data)
