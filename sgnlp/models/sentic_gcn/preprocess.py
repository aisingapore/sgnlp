import pathlib
import urllib.parse
from typing import Dict, List

import torch
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils.dummy_pt_objects import PreTrainedModel

from config import SenticGCNEmbeddingConfig, SenticGCNBertEmbeddingConfig
from modeling import SenticGCNEmbeddingModel, SenticGCNBertEmbeddingModel
from tokenization import SenticGCNTokenizer, SenticGCNBertTokenizer


class SenticGCNPreprocessor:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer = None,
        embedding_model: PreTrainedModel = None,
        tokenizer_name: str = "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_tokenizer/",
        embedding_model_name: str = "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_embedding_model/",
        config_filename: str = "config.json",
        model_filename: str = "pytorch_model.bin",
        device: str = "cpu",
    ):
        # Set device
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu") if not device else torch.device(device)
        )
        # Init Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = SenticGCNTokenizer.from_pretrained(tokenizer_name)
        # Init Embedding model
        if embedding_model is not None:
            # Load from external instance
            self.embedding_model = embedding_model
            self.embedding_model.to(self.device)
        else:
            if embedding_model_name.startswith("https://") or embedding_model_name.startswith("http://"):
                # Load from cloud
                config_url = urllib.parse.urljoin(embedding_model_name, config_filename)
                model_url = urllib.parse.urljoin(embedding_model_name, model_filename)
                embed_config = SenticGCNEmbeddingConfig.from_pretrained(config_url)
                embed_model = SenticGCNEmbeddingModel.from_pretrained(model_url, config=embed_config)
            else:
                # Load from local folder
                embed_model_name = pathlib.Path(embedding_model_name)
                if embed_model_name.is_dir():
                    config_path = embed_model_name.joinpath("config.json")
                    model_path = embed_model_name.joinpath("pytorch_model.bin")
                    embed_config = SenticGCNEmbeddingConfig.from_pretrained(config_path)
                    embed_model = SenticGCNEmbeddingModel.from_pretrained(model_path, config=embed_config)
                else:
                    # Load from HuggingFace model repository
                    embed_config = SenticGCNEmbeddingConfig.from_pretrained(embedding_model_name)
                    embed_model = SenticGCNEmbeddingModel.from_pretrained(embedding_model_name, config=embed_config)
            self.embedding_model = embed_model
            self.embedding_model.to(self.device)

    def __call__(self, data_batch: List[Dict[str, List[str]]]) -> BatchEncoding:
        pass  # TODO


class SenticGCNBertPreprocessor:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer = None,
        embedding_model: PreTrainedModel = None,
        tokenizer_name: str = "bert-base-uncased",
        embedding_model_name: str = "bert-base-uncased",
        config_filename: str = "config.json",
        model_filename: str = "pytorch_model.bin",
        device: str = "cpu",
    ):
        # Set device
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu") if not device else torch.device(device)
        )
        # Init Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = SenticGCNBertTokenizer.from_pretrained(tokenizer_name)
        # Init Embedding model
        if embedding_model is not None:
            # Load from external instance
            self.embedding_model = embedding_model
            self.embedding_model.to(self.device)
        else:
            if embedding_model_name.startswith("https://") or embedding_model_name.startswith("http://"):
                # Load from cloud
                config_url = urllib.parse.urljoin(embedding_model_name, config_filename)
                model_url = urllib.parse.urljoin(embedding_model_name, model_filename)
                embed_config = SenticGCNBertEmbeddingConfig.from_pretrained(config_url)
                embed_model = SenticGCNBertEmbeddingModel.from_pretrained(model_url, config=embed_config)
            else:
                # Load from local folder
                embed_model_name = pathlib.Path(embedding_model_name)
                if embed_model_name.is_dir():
                    config_path = embed_model_name.joinpath("config.json")
                    model_path = embed_model_name.joinpath("pytorch_model.bin")
                    embed_config = SenticGCNBertEmbeddingConfig.from_pretrained(config_path)
                    embed_model = SenticGCNBertEmbeddingModel.from_pretrained(model_path, config=embed_config)
                else:
                    # Load from HuggingFace model repository
                    embed_config = SenticGCNBertEmbeddingConfig.from_pretrained(embedding_model_name)
                    embed_model = SenticGCNBertEmbeddingModel.from_pretrained(
                        embedding_model_name, config=embed_config
                    )
            self.embedding_model = embed_model
            self.embedding_model.to(self.device)

    def __call__(self, data_batch: List[Dict[str, List[str]]]) -> BatchEncoding:
        pass  # TODO
