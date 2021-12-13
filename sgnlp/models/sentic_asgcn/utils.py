import argparse
import json
import math
import random
import pathlib

import numpy as np
import torch

from data_class import SenticASGCNTrainArgs


def parse_args_and_load_config(
    config_path: str = "config/sentic_asgcn_config.json",
) -> SenticASGCNTrainArgs:
    """Get config from config file using argparser

    Returns:
        SenticASGCNTrainArgs: SenticASGCNTrainArgs instance populated from config
    """
    parser = argparse.ArgumentParser(description="SenticASGCN Training")
    parser.add_argument("--config", type=str, default=config_path)
    args = parser.parse_args()

    cfg_path = pathlib.Path(__file__).parent / args.config
    with open(cfg_path, "r") as cfg_file:
        cfg = json.load(cfg_file)

    sentic_asgcn_args = SenticASGCNTrainArgs(**cfg)
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


class BucketIterator(object):
    def __init__(
        self, data, batch_size, sort_key="text_indices", shuffle=True, sort=True
    ):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        sorted_data = (
            sorted(data, key=lambda x: len(x[self.sort_key])) if self.sort else data
        )
        batches = [
            self.pad_data(sorted_data[i * batch_size : (i + 1) * batch_size])
            for i in range(num_batch)
        ]
        return batches

    def pad_data(self, batch_data):
        batch_text_indices = []
        batch_context_indices = []
        batch_aspect_indices = []
        batch_left_indices = []
        batch_polarity = []
        batch_dependency_graph = []
        batch_dependency_tree = []
        max_len = max([len(t[self.sort_key]) for t in batch_data])
        # [text_indices, context_indices, aspect_indices, left_indices, polarity, dependency_graph, dependency_tree]
        for item in batch_data:
            text_indices = item["text_indices"]
            context_indices = item["context_indices"]
            aspect_indices = item["aspect_indices"]
            left_indices = item["left_indices"]
            polarity = item["polarity"]
            dependency_graph = item["dependency_graph"]
            dependency_tree = item["dependency_tree"]

            text_padding = [0] * (max_len - len(text_indices))
            context_padding = [0] * (max_len - len(context_indices))
            aspect_padding = [0] * (max_len - len(aspect_indices))
            left_padding = [0] * (max_len - len(left_indices))

            batch_text_indices.append(text_indices + text_padding)
            batch_context_indices.append(context_indices + context_padding)
            batch_aspect_indices.append(aspect_indices + aspect_padding)
            batch_left_indices.append(left_indices + left_padding)
            batch_polarity.append(polarity)
            batch_dependency_graph.append(
                np.pad(
                    dependency_graph,
                    (
                        (0, max_len - len(text_indices)),
                        (0, max_len - len(text_indices)),
                    ),
                    "constant",
                )
            )
            batch_dependency_tree.append(
                np.pad(
                    dependency_tree,
                    (
                        (0, max_len - len(text_indices)),
                        (0, max_len - len(text_indices)),
                    ),
                    "constant",
                )
            )
            return {
                "text_indices": torch.tensor(batch_text_indices),
                "context_indices": torch.tensor(batch_context_indices),
                "aspect_indices": torch.tensor(batch_aspect_indices),
                "left_indices": torch.tensor(batch_left_indices),
                "polarity": torch.tensor(batch_polarity),
                "dependency_graph": torch.tensor(batch_dependency_graph),
                "dependency_tree": torch.tensor(batch_dependency_tree),
            }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
