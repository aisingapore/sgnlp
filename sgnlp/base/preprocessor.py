from abc import abstractmethod
from typing import Tuple, Dict, Any, Sized, List

import torch


def simple_merge(batches: List[Dict[str, list]]):
    keys = batches[0].keys()
    merged = {}

    for key in keys:
        merged[key] = []
        for batch in batches:
            merged[key].extend(batch[key])

    return merged


def get_merge_function(
    return_as_tensor=True,
    padding=True,
    pad_id=0,
    pad_direction="right",
    max_padding_lengths: Dict[str, int] = None,
):
    """Returns a function that is used to merge inputs to :class:`~sgnlp.base.preprocessor.PreprocessorBase`.

    The merge functions returned from this function covers general use cases for NLP problems. If preprocessing needs
    are more complicated, it is recommended to write your own merge function.

    Args:
        return_as_tensor(:obj:`bool`, defaults to True): Whether to return as a torch tensor.
        padding(:obj:`bool`, defaults to True): Whether to pad the vectors.
        pad_id(:obj:`int`, defaults to 0): ID for padding
        pad_direction(:obj:`str`, defaults to "right"): Pad direction. Either "left" or "right".
        max_padding_lengths(:obj: Dict[str, int], defaults to None): Max padding length for each input field. Expected
            input format to be a dictionary with the keys being the same as the inputs and the values being the max
            padding length.
    """
    def merge_function(batches: List[Dict[str, list]]):
        keys = batches[0].keys()
        merged = {}

        for key in keys:
            merged[key] = []
            for batch in batches:
                merged[key].extend(batch[key])

        if padding:
            if max_padding_lengths is None:
                _max_padding_lengths = {}
                for key in keys:
                    _max_padding_lengths[key] = max(map(len, merged[key]))
            else:
                _max_padding_lengths = max_padding_lengths

            for key in keys:
                pad_length = (
                    _max_padding_lengths[key]
                    if isinstance(_max_padding_lengths, dict)
                    else _max_padding_lengths
                )
                for i, instance in enumerate(merged[key]):
                    # for i, instance in enumerate(field):
                    pad_values = max(pad_length - len(instance), 0) * [pad_id]
                    if pad_direction == "right":
                        new_instance = instance + pad_values
                    elif pad_direction == "left":
                        new_instance = pad_values + instance
                    else:
                        raise ValueError(
                            f"Unknown pad_direction: {pad_direction}. "
                            f"Set pad_direction to 'left' or 'right'."
                        )
                    new_instance = new_instance[
                        :pad_length
                    ]  # Truncate to max pad length
                    merged[key][i] = new_instance

        if return_as_tensor:
            for key in keys:
                merged[key] = torch.tensor(merged[key])

        return merged

    return merge_function


class PreprocessorBase:
    """This is the base class for preprocessors.

    Args:
        batch_size: Number of inputs processed in each iteration.
        batch_merge_fn: This defines the way the preprocessed batches are merged and returned. A custom function can be
            defined to change the merge behaviour.
    """

    def __init__(self, batch_size=16, batch_merge_fn=None):
        self.batch_size = batch_size
        self.batch_merge_fn = (
            get_merge_function() if batch_merge_fn is None else batch_merge_fn
        )

    def __call__(
        self, data: Dict[str, Sized], *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
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

        processed_data = self.batch_merge_fn(processed_batches)

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
