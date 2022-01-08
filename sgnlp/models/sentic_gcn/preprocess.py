import logging
import pathlib
import shutil
import tempfile
import urllib.parse
from collections import namedtuple
from typing import Dict, List, Union

import spacy
import torch
from transformers import PreTrainedTokenizer, PretrainedConfig, PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding

from config import SenticGCNEmbeddingConfig, SenticGCNBertEmbeddingConfig
from modeling import SenticGCNEmbeddingModel, SenticGCNBertEmbeddingModel
from tokenization import SenticGCNTokenizer, SenticGCNBertTokenizer
from utils import download_tokenizer_files


logging.basicConfig(level=logging.DEBUG)


SenticGCNData = namedtuple("SenticGCNData", ["full_text", "aspect", "left_text"])
SenticGCNBertData = namedtuple("SenticGCNBertData", ["full_text", "aspect", "left_text", "full_text_with_bert_tokens"])


class SenticGCNBasePreprocessor:
    def __init__(
        self,
        tokenizer: Union[str, PreTrainedTokenizer],
        embedding_model: Union[str, PreTrainedTokenizer],
        tokenizer_class: PreTrainedTokenizer,
        embedding_config_class: PretrainedConfig,
        embedding_model_class: PreTrainedModel,
        config_filename: str = "config.json",
        model_filename: str = "pytorch_model.bin",
        spacy_pipeline: str = "en_core_web_sm",
        device: str = "cpu",
    ) -> None:
        # Set device
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu") if not device else torch.device(device)
        )
        self.spacy_pipeline = spacy.load(spacy_pipeline)

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
                    tokenizer_ = tokenizer_class.from_pretrained(temp_dir)
                    shutil.rmtree(temp_dir, ignore_errors=True)
                else:
                    # Load from local directory or from HuggingFace model repository
                    tokenizer_ = tokenizer_class.from_pretrained(tokenizer)
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
                    embed_config = embedding_config_class.from_pretrained(config_url)
                    embed_model = embedding_model_class.from_pretrained(model_url, config=embed_config)
                else:
                    # Load from local folder
                    embed_model_name = pathlib.Path(embedding_model)
                    if embed_model_name.is_dir():
                        config_path = embed_model_name.joinpath(config_filename)
                        model_path = embed_model_name.joinpath(model_filename)
                        embed_config = embedding_config_class.from_pretrained(config_path)
                        embed_model = embedding_model_class.from_pretrained(model_path, config=embed_config)
                    else:
                        # Load from HuggingFace model repository
                        embed_config = embedding_config_class.from_pretrained(embedding_model)
                        embed_model = embedding_model_class.from_pretrained(embedding_model, config=embed_config)
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


class SenticGCNPreprocessor(SenticGCNBasePreprocessor):
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
        spacy_pipeline: str = "en_core_web_sm",
        device: str = "cpu",
    ):
        super().__init__(
            tokenizer=tokenizer,
            embedding_model=embedding_model,
            tokenizer_class=SenticGCNTokenizer,
            embedding_config_class=SenticGCNEmbeddingConfig,
            embedding_model_class=SenticGCNEmbeddingModel,
            config_filename=config_filename,
            model_filename=model_filename,
            spacy_pipeline=spacy_pipeline,
            device=device,
        )

    def __call__(self, data_batch: List[Dict[str, List[str]]]) -> BatchEncoding:
        pass  # TODO


class SenticGCNBertPreprocessor(SenticGCNBasePreprocessor):
    def __init__(
        self,
        tokenizer: Union[str, PreTrainedTokenizer] = "bert-base-uncased",
        embedding_model: Union[str, PreTrainedModel] = "bert-base-uncased",
        config_filename: str = "config.json",
        model_filename: str = "pytorch_model.bin",
        spacy_pipeline: str = "en_core_web_sm",
        device: str = "cpu",
    ):
        super().__init__(
            tokenizer=tokenizer,
            embedding_model=embedding_model,
            tokenizer_class=SenticGCNBertTokenizer,
            embedding_config_class=SenticGCNBertEmbeddingConfig,
            embedding_model_class=SenticGCNBertEmbeddingModel,
            config_filename=config_filename,
            model_filename=model_filename,
            spacy_pipeline=spacy_pipeline,
            device=device,
        )

    def _process_indices(self, data_batch: List[SenticGCNBertData]):
        pass

    def _process_inputs(self, data_batch: List[Dict[str, List[str]]]) -> List[SenticGCNBertData]:
        processed_inputs = []
        for batch in data_batch:
            full_text = batch["sentence"].lower().strip()
            for aspect in batch["aspect"]:
                aspect = aspect.lower().strip()
                aspect_idxs = [index for index in range(len(full_text)) if full_text.startswith(aspect, index)]
                for aspect_index in aspect_idxs:
                    left_text = full_text[:aspect_index].strip()
                    full_text_with_bert_tokens = f"[CLS] {full_text} [SEP] {aspect} [SEP]"
                    processed_inputs.append(
                        SenticGCNBertData(
                            full_text=full_text,
                            aspect=aspect,
                            left_text=left_text,
                            full_text_with_bert_tokens=full_text_with_bert_tokens,
                        )
                    )
        return processed_inputs

    def __call__(self, data_batch: List[Dict[str, List[str]]]) -> BatchEncoding:
        pass  # TODO
