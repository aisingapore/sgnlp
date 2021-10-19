from math import inf
import os
import pathlib
import urllib
import requests


def download_tokenizer_files(base_url: str, local_path: str) -> None:
    """Download all required files for tokenizer from online storage.

    Args:
        base_url (str): Base url of storage location.
        local_path (str): path ot the folder on the local machine.
    """
    tokenizer_files = [
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]

    file_paths = [urllib.parse.urljoin(base_url, path) for path in tokenizer_files]
    for fp in file_paths:
        download_url_file(fp, local_path)


def download_url_file(url: str, save_folder: str) -> None:
    """Helper method to download url file.

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


class Buffer:
    def __init__(self, max_len):
        self.max_len = max_len
        self.elements = []

    def get_first_element(self):
        return self.elements.pop(0)

    def get_element(self, idx):
        return self.elements.pop(idx)

    def get_current_len(self):
        return len(self.elements)

    def __len__(self):
        return len(self.elements)

    def add_element(self, element):
        assert self.get_current_len() < self.max_len, "Exceeded max buffer length."
        self.elements.append(element)
        return

    def __repr__(self):
        return str(self.elements)


class Beam:
    def __init__(self, beam_size):
        self.beam_size = beam_size
        self.elements = []

    def add_element(self, score, indices):
        # Note that the score should be a float and the indices should be a list of integers
        assert (
            isinstance(score, float) and isinstance(score, int),
        ), "score should be a float or integer"
        assert isinstance(indices, list), "indices should be a list"
        assert all(
            [isinstance(x, int) for x in indices]
        ), "elements in indices should be integers"

        new_element = {"score": score, "indices": indices}

        # The number of elements should be at most equal to the beam size
        if len(self.elements) < self.beam_size:
            self._add_element(new_element)
        else:
            # If the current number of elements is equal to the beam_size,
            # we compare the lowest scoring element with the new element's score
            # and replace it if the new element's score is higher.
            # Otherwise no change is made
            if new_element["score"] > self.get_lowest_score():
                self._remove_last_element()
                self._add_element(new_element)
        return

    def add_elements(self, scores_lst, indices_lst):
        for score, element in zip(scores_lst, indices_lst):
            self.add_element(score, element)
        return

    def get_elements(self):
        return self.elements

    def get_best_element(self):
        return self.elements[0]

    def _order_elements(self):
        self.elements = sorted(self.elements, key=lambda x: x["score"], reverse=True)
        return

    def _add_element(self, element):
        assert len(self.elements) < self.beam_size, "Beam is full"
        assert isinstance(element, dict), "element should be a dictionary object"

        self.elements.append(element)
        # This step is necessary to ensure the last element is always the lowest score
        self._order_elements()
        return

    def _remove_last_element(self):
        self.elements.pop(-1)
        return

    def get_lowest_score(self):
        if len(self.elements) == 0:
            return -inf
        return self.elements[-1]["score"]
