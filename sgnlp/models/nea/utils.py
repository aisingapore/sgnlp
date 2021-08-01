import argparse
import json
import os
import pathlib
from typing import Dict, List, Tuple, Union, Callable
import requests
import urllib

import numpy as np
import torch
from transformers import Trainer, TrainingArguments

from .config import NEAConfig
from .modeling import (
    NEARegModel,
    NEARegPoolingModel,
    NEABiRegModel,
    NEABiRegPoolingModel,
)
from .tokenization import NEATokenizer
from .data_class import NEAArguments


ASAP_RANGES = {
    0: (0, 60),
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60),
}


def init_model(
    config: NEAArguments,
) -> Union[NEARegModel, NEARegPoolingModel, NEABiRegPoolingModel, NEABiRegPoolingModel]:
    """Helper method to initalize model based on model type.

    Args:
        config (NEAArguments): NEAArguments config load from configuration file.

    Returns:
        Union[NEARegModel, NEARegPoolingModel, NEABiRegPoolingModel, NEABiRegPoolingModel]: [description]
    """
    model_type = config.model_type

    if model_type == "reg":
        config = NEAConfig()
        model = NEARegModel(config=config)
    elif model_type == "regp":
        config = NEAConfig()
        model = NEARegPoolingModel(config=config)
    elif model_type == "breg":
        config = NEAConfig(linear_input_dim=600)
        model = NEABiRegModel(config=config)
    elif model_type == "bregp":
        config = NEAConfig(linear_input_dim=600)
        model = NEABiRegPoolingModel(config=config)

    return model


def get_emb_matrix(vocab: Dict, emb_path: str) -> torch.Tensor:
    """Takes a embedding file and convert them into embedding matrix using the
    vocab dictionary. Eg. if the word 'the' is mapped to id 5 in the vocab, then
    the 5th row of the embedding matrix will represent the embedding vector
    of the word 'the'.

    Args:
        vocab (Dict): mapping of words to id used by tokenizer
        emb_path (str): path of embedding file

    Returns:
        torch.Tensor: torch tensor of embedding matrix
    """
    with open(emb_path, "r") as emb_file:
        file_lines = emb_file.readlines()

    emb_dict = {}
    first_line = file_lines[0].rstrip().split()
    if len(first_line) != 2:
        raise ValueError(
            "The first line in W2V embeddings must be the pair (vocab_size, emb_dim)"
        )

    emb_dim = int(first_line[1])
    emb_vocab_size = int(first_line[0])
    for line in file_lines[1:]:
        tokens = line.rstrip().split()
        word = tokens[0]
        vector = [float(token) for token in tokens[1:]]
        if len(vector) != emb_dim:
            raise ValueError("The number of dimensions does not match the header info")
        emb_dict[word] = vector

    if emb_vocab_size != len(emb_dict.keys()):
        raise ValueError("Vocab size does not match the header info")

    vocab_len = len(vocab.keys())
    emb_matrix = torch.zeros((vocab_len, emb_dim))
    for word, index in vocab.items():
        if word in emb_dict.keys():
            emb_matrix[index] = torch.Tensor(emb_dict[word])

    return emb_matrix


def pad_sequences_from_list(array: List[List[int]]) -> torch.Tensor:
    """Performs post padding on sequences to make sequences same length.
    Adds padding with id 0

    Args:
        array (List[List[int]]): List of list of token_ids

    Returns:
        torch.Tensor: Padded tensor of token ids
    """
    tensor_list = [torch.tensor(row) for row in array]
    padded_sequences = torch.nn.utils.rnn.pad_sequence(
        tensor_list, batch_first=True
    ).int()

    return padded_sequences


def get_model_friendly_scores(
    scores_array: List, prompt_id_array: List
) -> torch.Tensor:
    """Function was adapted from paper's code and refactored.
    Convert essay scores to a value between 0 and 1. Conversion is dependent on
    each instance's prompt_id

    Args:
        scores_array (List): List of score of each instance
        prompt_id_array (List): List of prompt_id of each instance

    Raises:
        ValueError: Shape of Scores Array is different from shape of Prompt Id array
        ValueError: Scores array values are not between 0 and 1

    Returns:
        torch.Tensor: tensor of scores between 0 and 1
    """
    scores_array = np.array(scores_array)
    prompt_id_array = np.array(prompt_id_array)
    try:
        assert scores_array.shape[0] == prompt_id_array.shape[0]
    except AssertionError:
        raise ValueError(
            f"Shape of Scores Array {scores_array.shape[0]} \
            is different from Prompt Id {prompt_id_array.shape[0]}"
        )
    dim = scores_array.shape[0]
    low = np.zeros(dim)
    high = np.zeros(dim)
    for i in range(dim):
        low[i], high[i] = ASAP_RANGES[prompt_id_array[i]]
    scores_array = (scores_array - low) / (high - low)
    try:
        assert np.all(scores_array >= 0) and np.all(scores_array <= 1)
    except AssertionError:
        raise ValueError("Scores array values are not between 0 and 1!")
    scores_array_tensor = torch.from_numpy(scores_array).float()
    return scores_array_tensor


def parse_args_and_load_config(
    config_path: str = "config/nea_config.json",
) -> NEAArguments:
    """Args parser helper method

    Args:
        config_path (str): path of config.json

    Returns:
        NEAArguments: NEAArguments instance
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=config_path)
    args = parser.parse_args()
    with open(args.config, "r") as cfg_file:
        cfg = json.load(cfg_file)
    nea_args = NEAArguments(**cfg)
    return nea_args


class NEADataset(torch.utils.data.Dataset):
    """Class to create torch Dataser instance which is the required data type
    for Transformer's Trainer

    Args:
        input_ids (torch.Tensor): input_ids of dataset from NEATokenizer
        labels (torch.Tensor, optional): labels are not required to compute
            output in the model's forward method but it is required for
            the computation of loss function when using Transformer's Trainer.
            Defaults to None.
    """

    def __init__(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        """Constructor method"""
        self.input_ids = input_ids
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict:
        """Get a dictionary of the selected instance for each batch

        Args:
            idx (int): idx to select instances for each batch

        Returns:
            Dict: dictionary containing input_ids of the selected instance
        """
        item = {}
        item["input_ids"] = self.input_ids[idx]
        if self.labels != None:
            item["labels"] = self.labels[idx]
        return item

    def __len__(self) -> int:
        """Returns length of dataset

        Returns:
            int: length of dataset
        """
        return len(self.input_ids)


class NEATrainer(Trainer):
    """Create inherited Trainer class to create loss function and optimizer
    specific to NEA model"""

    def create_optimizer(self) -> None:
        """Override Trainer's default optimizer to use custom optimizer"""
        if self.args.optimizer_type == "rmsprop":
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters(),
                lr=self.args.learning_rate,
                eps=self.args.optimizer_epsilon,
            )
        elif self.args.optimizer_type == "adagrad":
            self.optimizer = torch.optim.Adagrad(
                self.model.parameters(),
                lr=self.args.learning_rate,
                eps=self.args.optimizer_epsilon,
            )
        elif self.args.optimizer_type == "adamax":
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters(),
                lr=self.args.learning_rate,
                eps=self.args.optimizer_epsilon,
            )
        elif self.args.optimizer_type == "adadelta":
            self.optimizer = torch.optim.Adadelta(
                self.model.parameters(),
                lr=self.args.learning_rate,
                eps=self.args.optimizer_epsilon,
            )
        elif self.args.optimizer_type == "adamax":
            self.optimizer = torch.optim.Adamax(
                self.model.parameters(),
                lr=self.args.learning_rate,
                eps=self.args.optimizer_epsilon,
            )


class NEATrainingArguments(TrainingArguments):
    """Inherit TrainingArguments to add in additional parameters specific to NEA
    This class inherits all the parameters of TrainingArguments so all the parameters
    from TrainingArguments will be accepted here

    Args:
        loss_function (str, optional): mse/ mae. Defaults to "mse".
        optimizer_epsilon (float, optional): epsilon parameter of optimizer.  Defaults to 1e-6.
        optimizer_type (str, optional): optimizer type: rsmprop/ adam/ adagrad/ adadelta/ adamax.
            Defaults to "rmsprop".
    """

    def __init__(
        self,
        loss_function: str = "mse",
        optimizer_epsilon: float = 1e-6,
        optimizer_type: str = "rmsprop",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.optimizer_epsilon = optimizer_epsilon
        self.optimizer_type = optimizer_type
        self.loss_function = loss_function


def convert_to_dataset_friendly_scores(
    scores_array: np.ndarray, prompt_id_array: Union[int, np.ndarray]
) -> np.ndarray:
    """Function was adapted from paper's code and refactored.
    Convert values of between 0 and 1 to the essay scores

    Args:
        scores_array (np.ndarray): scores of between 0 and 1
        prompt_id_array (Union[int, np.ndarray]): prompt_id of each instance to convert to
                original essay score range

    Returns:
        np.ndarray: essay score of each instance
    """
    arg_type = type(prompt_id_array)
    assert arg_type in {int, np.ndarray}
    if arg_type is int:
        low, high = ASAP_RANGES[prompt_id_array]
        scores_array = scores_array * (high - low) + low
        assert np.all(scores_array >= low) and np.all(scores_array <= high)
    else:
        assert scores_array.shape[0] == prompt_id_array.shape[0]
        dim = scores_array.shape[0]
        low = np.zeros(dim)
        high = np.zeros(dim)
        for ii in range(dim):
            low[ii], high[ii] = ASAP_RANGES[prompt_id_array[ii]]
        scores_array = scores_array * (high - low) + low
    return scores_array


def confusion_matrix(
    rater_a: np.ndarray,
    rater_b: np.ndarray,
    min_rating: int = None,
    max_rating: int = None,
) -> np.ndarray:
    """Function was adapted from paper's code and refactored.
    Returns the confusion matrix between rater's ratings

    Args:
        rater_a (np.ndarray): essay score assessed by rater a
        rater_b (np.ndarray): essay score assessed by rater b
        min_rating (int): min essay score
        max_rating (int): maxx essay score

    Returns:
        np.ndarray: confusion matrix output
    """
    assert len(rater_a) == len(rater_b)
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)] for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(
    ratings: np.ndarray, min_rating: int = None, max_rating: int = None
) -> np.ndarray:
    """Function was adapted from paper's code and refactored.
    Returns the counts of each type of rating that a rater made

    Args:
        ratings (np.ndarray): rating
        min_rating (int, optional): min rating. Defaults to None.
        max_rating (int, optional): max rating. Defaults to None.

    Returns:
        np.ndarray: output
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(
    rater_a: np.ndarray,
    rater_b: np.ndarray,
    min_rating: int = None,
    max_rating: int = None,
) -> float:
    """Function was adapted from paper's code and refactored.
    Calculates the quadratic weighted kappa
    quadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.

    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.

    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.

    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating

    Args:
        rater_a (np.ndarray): essay score assessed by rater a
        rater_b (np.ndarray): essay score assessed by rater b
        min_rating (int): min essay score
        max_rating (int): maxx essay score

    Returns:
        float: quadratic weighted kappa score
    """
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert len(rater_a) == len(rater_b)
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b, min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = hist_rater_a[i] * hist_rater_b[j] / num_scored_items
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator


def qwk(
    rater_a: np.ndarray, rater_b: np.ndarray, min_rating: int, max_rating: int
) -> float:
    """Function was adapted from paper's code and refactored.
    Computes qwk given 2 essay score array

    Args:
        rater_a (np.ndarray): essay score assessed by rater a
        rater_b (np.ndarray): essay score assessed by rater b
        min_rating (int): min essay score
        max_rating (int): maxx essay score

    Returns:
        float: quadratic weighted kappa score
    """
    assert np.issubdtype(
        rater_a.dtype, np.integer
    ), "Integer array expected, got " + str(rater_a.dtype)
    assert np.issubdtype(
        rater_b.dtype, np.integer
    ), "Integer array expected, got " + str(rater_b.dtype)
    return quadratic_weighted_kappa(rater_a, rater_b, min_rating, max_rating)


def build_compute_metrics_fn(cfg: NEAArguments) -> Callable:
    """Build the compute metrics function for NEATrainer given the config

    Args:
        cfg (NEAArguments): NEAArguments config load from config file

    Returns:
        Callable: function to compute qwk metrics during evaluation
    """

    def compute_metrics_fn(output):
        low, high = ASAP_RANGES[cfg.preprocess_data_args["prompt_id"]]
        pred, labels = output
        pred_score = convert_to_dataset_friendly_scores(pred.squeeze(), 1)
        pred_score = np.rint(pred_score).astype("int32")

        labels_score = convert_to_dataset_friendly_scores(labels, 1)
        labels_score = np.rint(labels_score).astype("int32")

        qwk_metric = qwk(labels_score, pred_score, low, high)
        return {"qwk": qwk_metric}

    return compute_metrics_fn


def train_and_save_tokenizer(cfg: NEAArguments) -> NEATokenizer:
    """Train tokenizer on train data then save it so it can be used during eval

    Args:
        cfg (NEAArguments): NEAArguments config load from configuration file

    Returns:
        NEATokenizer: Trained NEATokenizer
    """
    train_path = cfg.tokenizer_args["vocab_train_file"]
    save_dir = cfg.tokenizer_args["save_folder"]

    nea_tokenizer = NEATokenizer(train_file=train_path, train_vocab=True)
    nea_tokenizer.save_pretrained(save_dir)
    nea_tokenizer = NEATokenizer.from_pretrained(save_dir)

    return nea_tokenizer


def read_dataset(
    file_path: str,
    prompt_id: int,
    maxlen: int,
    to_lower: bool,
    score_index: int = 6,
) -> Tuple[List[int], List[int], int, int]:
    """Read and process dataset file.

    Args:
        file_path (str): path to dataset file
        prompt_id (int): selected prompt_id
        maxlen (str): max length
        to_lower (bool): option to process text to lower case
        score_index (int): selected score_index

    Returns:
        Tuple[ List[int], List[int], List[int]]: return train, label data, prompt id
    """
    data_x, data_y, prompt_ids = [], [], []
    with open(file_path, "r", encoding="utf8") as input_file:
        count = 0
        for line in input_file:
            if count == 0:
                count = 1
                continue
            tokens = line.strip().split("\t")
            essay_set = int(tokens[1])
            content = tokens[2].strip()
            score = float(tokens[score_index])
            if essay_set == prompt_id or prompt_id <= 0:
                if to_lower:
                    content = content.lower()
                else:
                    content = content.split()
                if maxlen > 0 and len(content) > maxlen:
                    continue
                data_x.append(content)
                data_y.append(score)
                prompt_ids.append(essay_set)
    return data_x, data_y, prompt_ids


def load_train_dev_dataset(cfg: NEAArguments) -> Tuple[Tuple[List], Tuple[List]]:
    """Reads config and load train and dev dataset

    Args:
        cfg (NEAArguments): NEAArguments config from config.json

    Returns:
        Tuple[Tuple[List], Tuple[List]]: Train dataset and dev dataset
    """
    X_train, y_train, train_prompt_ids = read_dataset(
        cfg.preprocess_data_args["train_path"],
        cfg.preprocess_data_args["prompt_id"],
        cfg.preprocess_data_args["maxlen"],
        cfg.preprocess_data_args["to_lower"],
        cfg.preprocess_data_args["score_index"],
    )
    X_dev, y_dev, dev_prompt_ids = read_dataset(
        cfg.preprocess_data_args["dev_path"],
        cfg.preprocess_data_args["prompt_id"],
        cfg.preprocess_data_args["maxlen"],
        cfg.preprocess_data_args["to_lower"],
        cfg.preprocess_data_args["score_index"],
    )
    return (X_train, y_train, train_prompt_ids), (X_dev, y_dev, dev_prompt_ids)


def download_tokenizer_files_from_azure(cfg: NEAArguments) -> None:
    """Download all required files for tokenizer from Azure storage.

    Args:
        cfg (NEAArguments): NEAArguments config load from config file.
    """
    remote_folder_path = cfg.tokenizer_args["azure_path"]
    file_paths = [
        urllib.parse.urljoin(remote_folder_path, path)
        for path in cfg.tokenizer_args["files"]
    ]
    for fp in file_paths:
        download_url_file(fp, cfg.tokenizer_args["save_folder"])


def download_url_file(url: str, save_folder: str) -> None:
    """Helpder method to download url file.

    Args:
        url (str): url file address string.
        save_folder (str): local folder name to save downloaded files.
    """
    os.makedirs(save_folder, exist_ok=True)
    fn_start_pos = url.rfind("/") + 1
    file_name = url[fn_start_pos:]
    save_file_name = pathlib.Path(save_folder).joinpath(file_name)
    req = requests.get(url)
    if req.status_code == requests.codes.ok:
        with open(save_file_name, "wb") as f:
            for data in req:
                f.write(data)


def load_test_dataset(cfg: Dict) -> Tuple[Tuple[List]]:
    """Read config and load test dataset

    Args:
        cfg (Dict): config from config.json

    Returns:
        Tuple[Tuple[List]]: Test dataset
    """
    X_test, y_test, test_prompt_ids = read_dataset(
        cfg.preprocess_data_args["test_path"],
        cfg.preprocess_data_args["prompt_id"],
        cfg.preprocess_data_args["maxlen"],
        cfg.preprocess_data_args["to_lower"],
        cfg.preprocess_data_args["score_index"],
    )
    return (X_test, y_test, test_prompt_ids)


def process_results(metrics: Dict) -> str:
    """Filter to include only useful metrics from trainer.predict()'s output

    Args:
        metrics (Dict): Metrics from trainer.predict()

    Returns:
        str: str containing relevant eval metrics
    """
    results = {k: v for k, v in metrics.items() if k in ["eval_loss", "eval_qwk"]}
    return str(results)
