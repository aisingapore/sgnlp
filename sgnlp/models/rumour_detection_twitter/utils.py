from datasets import load_dataset
import re
from torch.utils.data import DataLoader
import os
import pathlib
import urllib
import requests


def download_tokenizer_files_from_azure(azure_path: str, local_path: str) -> None:
    """Download all required files for tokenizer from Azure storage.

    Args:
        azure_path (str): url to the tokenizer files on the Azure blob.
        local_path (str): path ot the folder on the local machine.
    """
    tokenizer_files = [
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.pkl",
    ]

    file_paths = [urllib.parse.urljoin(azure_path, path) for path in tokenizer_files]
    for fp in file_paths:
        download_url_file(fp, local_path)


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


def load_transform_dataset(
    data_path,
    batch_size,
    tokenizer,
    split,
    max_tokens,
    max_posts,
    post_padding_idx,
    time_padding_idx,
):
    """
    data_path
        Path to the data json file
    batch_size : int
        Batch size
    tokenizer : RumourDetectionTokenizer
        Tokenizer object for tweets
    """

    # TODO make the cache_dir an argument

    dataset = load_dataset(
        "json",
        data_files=data_path,
        field=split,
        # cache_dir="/polyaxon-data/SG-NLP/rumour_refactor/data/",
    )

    dataset = dataset.map(
        lambda x: encode_dataset(
            x, tokenizer, max_tokens, max_posts, post_padding_idx, time_padding_idx
        ),
        batched=True,
    )

    dataset.set_format(
        type="torch",
        columns=[
            "tweet_token_ids",
            "time_delay_ids",
            "structure_ids",
            "token_attention_mask",
            "post_attention_mask",
            "label",
            "id_",
        ],
    )

    shuffle = True if split == "train" else False

    dataloader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=shuffle)

    return dataloader


def encode_dataset(
    examples, tokenizer, max_tokens, max_posts, post_padding_idx, time_padding_idx
):
    examples["id_"] = list(map(int, examples["id_"]))

    examples["tweets"] = list(
        map(lambda thread: list(map(clean_text, thread)), examples["tweets"])
    )

    examples["time_delay_ids"], post_attention_mask = pad_batched_sequences(
        examples["time_delay_ids"], max_posts, time_padding_idx
    )

    examples["structure_ids"] = list(
        map(
            lambda structure_ids: pad_structure(
                structure_ids, post_padding_idx, max_posts
            ),
            examples["structure_ids"],
        )
    )

    tweet_token_ids, token_attention_mask = tokenizer.tokenize_threads(
        examples["tweets"],
        max_length=max_tokens,
        max_posts=max_posts,
        truncation=True,
        padding="max_length",
    )

    output = {
        "tweet_token_ids": tweet_token_ids,
        "token_attention_mask": token_attention_mask,
        "post_attention_mask": post_attention_mask,
    }

    return output


def clean_text(text):

    """
    This function cleans the text in the following ways:
    1. Replace websites with URL
    1. Replace 's with <space>'s (eg, her's --> her 's)
    """

    text = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "URL",
        text,
    )  # Replace urls with special token
    text = text.replace("'s", "")
    text = text.replace("'", "")
    text = text.replace("n't", " n't")
    text = text.replace("@", "")
    text = text.replace("#", "")
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace("&amp;", "")
    text = text.replace("&gt;", "")
    text = text.replace('"', "")
    text = text.replace(".", "")
    text = text.replace(",", "")
    text = text.replace("(", "")
    text = text.replace(")", "")

    text = " ".join(text.split())

    return text.strip()


def pad_batched_sequences(batched_ids, max_length, padding_idx):
    padded_batch = list(
        map(lambda x: pad_sequence(x, max_length, padding_idx), batched_ids)
    )

    attention_masks = list(
        map(lambda x: get_attention_masks(x, padding_idx), padded_batch)
    )

    return padded_batch, attention_masks


def pad_sequence(input_ids, max_length, padding_idx):
    padded_sequence = input_ids[:max_length] + [padding_idx] * (
        max_length - len(input_ids)
    )
    return padded_sequence


def get_attention_masks(padded_sequence, padding_idx):
    attention_mask = list(map(lambda x: int(x != padding_idx), padded_sequence))
    return attention_mask


def pad_structure(structure_ids, post_padding_idx, max_posts):
    structure_ids = list(
        map(lambda x: pad_sequence(x, max_posts, post_padding_idx), structure_ids)
    )

    structure_ids = structure_ids[:max_posts] + [[post_padding_idx] * max_posts] * (
        max_posts - len(structure_ids)
    )

    return structure_ids


def load_datasets(train_args, config=None, tokenizer=None):
    if train_args["preprocessed"]:
        return list(
            map(
                lambda split: load_preprocessed_dataset(
                    train_args["data_path"], train_args["batch_size"], split
                ),
                ["train", "val", "test"],
            )
        )
    else:
        assert (
            tokenizer is not None
        ), "Please provide a tokenizer if the dataset must be preprocessed"

        assert (
            config is not None
        ), "Please provide a model config if the dataset must be preprocessed"

        return list(
            map(
                lambda split: load_transform_dataset(
                    ["data_path"],
                    ["batch_size"],
                    tokenizer,
                    split,
                    config.max_length,
                    config.max_tweets,
                    config.num_structure_index,
                    config.size,
                ),
                ["train", "val", "test"],
            )
        )


def load_preprocessed_dataset(data_path, batch_size, split):

    dataset = load_dataset(
        "json",
        data_files=data_path,
        field=split,
        cache_dir="/polyaxon-data/SG-NLP/rumour_refactor/data/",
    )

    dataset.set_format(
        type="torch",
        columns=[
            "tweet_token_ids",
            "time_delay_ids",
            "structure_ids",
            "token_attention_mask",
            "post_attention_mask",
            "label",
        ],
    )

    shuffle = True if split == "train" else False

    dataloader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=shuffle)

    return dataloader
