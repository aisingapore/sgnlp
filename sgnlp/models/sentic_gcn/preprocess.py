import logging
import pathlib
import shutil
import tempfile
import urllib.parse
from typing import Dict, List, Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding

from config import SenticGCNEmbeddingConfig, SenticGCNBertEmbeddingConfig
from modeling import SenticGCNEmbeddingModel, SenticGCNBertEmbeddingModel
from tokenization import SenticGCNTokenizer, SenticGCNBertTokenizer
from utils import download_tokenizer_files


logging.basicConfig(level=logging.DEBUG)


class SenticGCNPreprocessor:
    def __init__(
        self,
        tokenizer: Union[
            str, PreTrainedTokenizer
        ] = "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_tokenizer/",
        embedding_model: Union[
            str, PreTrainedModel
        ] = "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_embedding_model/",
        config_filename: str = "config.json",
        model_filename: str = "pytorch_model.bin",
        device: str = "cpu",
    ):
        # Set device
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu") if not device else torch.device(device)
        )

        try:
            # Init Tokenizer
            if isinstance(tokenizer, PreTrainedTokenizer):
                # Load from external instance
                tokenizer_ = tokenizer
            else:
                if tokenizer.startswith("https://") or tokenizer.startswith("http://"):
                    # Load from cloud
                    # Download tokenizer files to temp dir
                    with tempfile.TemporaryDirectory() as tmpdir:
                        temp_dir = pathlib.Path(tmpdir)
                    download_tokenizer_files(tokenizer, temp_dir)
                    tokenizer_ = SenticGCNTokenizer.from_pretrained(temp_dir)
                    shutil.rmtree(temp_dir, ignore_errors=True)
                else:
                    # Load from local directory or from HuggingFace model repository
                    tokenizer_ = SenticGCNTokenizer.from_pretrained(tokenizer)
            self.tokenizer = tokenizer_
        except Exception as e:
            logging.error(e)
            raise Exception(
                """
                Error initializing tokenizer! Please ensure that input tokenizer is either a PreTrainedTokenizer instance,
                an url to cloud storage folder, local folder or HuggingFace model name.
                """
            )

        try:
            # Init Embedding model
            if isinstance(embedding_model, PreTrainedModel):
                # Load from external instance
                embed_model = embedding_model
            else:
                if embedding_model.startswith("https://") or embedding_model.startswith("http://"):
                    # Load from cloud
                    config_url = urllib.parse.urljoin(embedding_model, config_filename)
                    model_url = urllib.parse.urljoin(embedding_model, model_filename)
                    embed_config = SenticGCNEmbeddingConfig.from_pretrained(config_url)
                    embed_model = SenticGCNEmbeddingModel.from_pretrained(model_url, config=embed_config)
                else:
                    # Load from local folder
                    embed_model_name = pathlib.Path(embedding_model)
                    if embed_model_name.is_dir():
                        config_path = embed_model_name.joinpath(config_filename)
                        model_path = embed_model_name.joinpath(model_filename)
                        embed_config = SenticGCNEmbeddingConfig.from_pretrained(config_path)
                        embed_model = SenticGCNEmbeddingModel.from_pretrained(model_path, config=embed_config)
                    else:
                        # Load from HuggingFace model repository
                        embed_config = SenticGCNEmbeddingConfig.from_pretrained(embedding_model)
                        embed_model = SenticGCNEmbeddingModel.from_pretrained(embedding_model, config=embed_config)
            self.embedding_model = embed_model
            self.embedding_model.to(self.device)
        except Exception as e:
            logging.error(e)
            raise Exception(
                """
                Error initializing embedding model! Please ensure that input tokenizer is either a PreTrainedModel instance,
                an url to cloud storage folder, local folder or HuggingFace model name.
                """
            )

    def __call__(self, data_batch: List[Dict[str, List[str]]]) -> BatchEncoding:
        pass  # TODO


class SenticGCNBertPreprocessor:
    def __init__(
        self,
        tokenizer: Union[str, PreTrainedTokenizer] = "bert-base-uncased",
        embedding_model: Union[str, PreTrainedModel] = "bert-base-uncased",
        config_filename: str = "config.json",
        model_filename: str = "pytorch_model.bin",
        device: str = "cpu",
    ):
        # Set device
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu") if not device else torch.device(device)
        )

        try:
            # Init Tokenizer
            if isinstance(tokenizer, PreTrainedTokenizer):
                # Load from external instance
                tokenizer_ = tokenizer
            else:
                if tokenizer.startswith("https://") or tokenizer.startswith("http://"):
                    # Load from cloud
                    # Download tokenizer files to temp dir
                    with tempfile.TemporaryDirectory() as tmpdir:
                        temp_dir = pathlib.Path(tmpdir)
                    download_tokenizer_files(tokenizer, temp_dir)
                    tokenizer_ = SenticGCNBertTokenizer.from_pretrained(temp_dir)
                    shutil.rmtree(temp_dir, ignore_errors=True)
                else:
                    # Load from local directory or from HuggingFace model repository
                    tokenizer_ = SenticGCNBertTokenizer.from_pretrained(tokenizer)
            self.tokenizer = tokenizer_
        except Exception as e:
            logging.error(e)
            raise Exception(
                """
                Error initializing tokenizer! Please ensure that input tokenizer is either a PreTrainedTokenizer instance,
                an url to cloud storage folder, local folder or HuggingFace model name.
                """
            )

        try:
            # Init Embedding model
            if isinstance(embedding_model, PreTrainedModel):
                # Load from external instance
                embed_model = embedding_model
            else:
                if embedding_model.startswith("https://") or embedding_model.startswith("http://"):
                    # Load from cloud
                    config_url = urllib.parse.urljoin(embedding_model, config_filename)
                    model_url = urllib.parse.urljoin(embedding_model, model_filename)
                    embed_config = SenticGCNBertEmbeddingConfig.from_pretrained(config_url)
                    embed_model = SenticGCNBertEmbeddingModel.from_pretrained(model_url, config=embed_config)
                else:
                    # Load from local folder
                    embed_model_name = pathlib.Path(embedding_model)
                    if embed_model_name.is_dir():
                        config_path = embed_model_name.joinpath(config_filename)
                        model_path = embed_model_name.joinpath(model_filename)
                        embed_config = SenticGCNBertEmbeddingConfig.from_pretrained(config_path)
                        embed_model = SenticGCNBertEmbeddingModel.from_pretrained(model_path, config=embed_config)
                    else:
                        # Load from HuggingFace model repository
                        embed_config = SenticGCNBertEmbeddingConfig.from_pretrained(embedding_model)
                        embed_model = SenticGCNBertEmbeddingModel.from_pretrained(embedding_model, config=embed_config)
            self.embedding_model = embed_model
            self.embedding_model.to(self.device)
        except Exception as e:
            logging.error(e)
            raise Exception(
                """
                Error initializing embedding model! Please ensure that input tokenizer is either a PreTrainedModel instance,
                an url to cloud storage folder, local folder or HuggingFace model name.
                """
            )

    def __call__(self, data_batch: List[Dict[str, List[str]]]) -> BatchEncoding:
        pass  # TODO
