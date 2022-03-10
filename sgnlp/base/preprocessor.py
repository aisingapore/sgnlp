from abc import abstractmethod
from typing import Tuple, Dict, Any, Sized, List

import torch


def default_merge(batches: List[Dict[str, list]]):
    keys = batches[0].keys()
    merged = {}

    for key in keys:
        merged[key] = []
        for batch in batches:
            merged[key].extend(batch[key])

    return merged


class PreprocessorBase:

    def __init__(self, batch_size=16):
        self.batch_size = batch_size

    def __call__(self, data: Dict[str, Sized], batch_merge_fn=None, *args, **kwargs) -> Dict[str, torch.Tensor]:
        # Check that all data in keys are of same length
        data_lengths = []
        for k, v in data.items():
            data_lengths.append(len(v))

        if len(set(data_lengths)) != 1:
            raise ValueError("Data values are not of same length.")

        data_length = data_lengths[0]

        start_idx = 0
        end_idx = start_idx + self.batch_size
        processed_batches = []
        while start_idx < data_length:
            batch = {k: v[start_idx:end_idx] for k, v in data.items()}
            processed_batch = self.preprocess(batch)
            processed_batches.append(processed_batch)
            start_idx += self.batch_size
            end_idx += self.batch_size

        if batch_merge_fn is None:
            batch_merge_fn = default_merge
        processed_data = batch_merge_fn(processed_batches)

        return processed_data

    @abstractmethod
    def preprocess(self, data: Dict[str, Any], *args, **kwargs) -> Tuple:
        raise NotImplementedError

    @abstractmethod
    def merge_batches(self):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError
