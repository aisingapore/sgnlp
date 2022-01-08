import logging
import pathlib
import shutil
import tempfile
import urllib.parse
from collections import namedtuple
from typing import Dict, List, Union

import numpy as np
import spacy
import torch
from transformers import PreTrainedTokenizer, PretrainedConfig, PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding

from config import SenticGCNEmbeddingConfig, SenticGCNBertEmbeddingConfig
from modeling import SenticGCNEmbeddingModel, SenticGCNBertEmbeddingModel
from tokenization import SenticGCNTokenizer, SenticGCNBertTokenizer
from utils import (
    load_and_process_senticnet,
    download_tokenizer_files,
    pad_and_truncate,
    generate_dependency_adj_matrix,
)


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
        senticnet: str = "senticnet.pickle",
        device: str = "cpu",
    ) -> None:
        # Set device
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu") if not device else torch.device(device)
        )
        self.spacy_pipeline = spacy.load(spacy_pipeline)

        if senticnet.endswith(".pkl") or senticnet.endswith(".pickle"):
            self.senticnet = load_and_process_senticnet(saved_preprocessed_senticnet_file_path=senticnet)
        elif senticnet.endswith(".txt"):
            self.senticnet = load_and_process_senticnet(senticnet_file_path=senticnet)
        else:
            raise ValueError(
                f"""
                Invalid SenticNet file!
                For processed SenticNet dictionary, please provide pickle file location
                (i.e. file with .pkl or .pickle extension).
                For raw SenticNet-5.0 file, please provide text file path (i.e. file with .txt extension)
                """
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
        senticnet: str = "senticnet.pkl",
        max_len: int = 85,
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
            senticnet=senticnet,
            device=device,
        )
        self.max_len = max_len

    def _process_indices(self, data_batch: List[SenticGCNBertData]) -> List[torch.Tensor]:
        all_text_indices = []
        all_aspect_indices = []
        all_left_indices = []
        all_text_bert_indices = []
        all_bert_segment_indices = []
        all_sdat_graph = []
        for data in data_batch:
            text_indices = self.tokenizer(
                data.full_text,
                padding="max_length",
                truncation=True,
                add_special_tokens=False,
                return_tensors=None,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            aspect_indices = self.tokenizer(
                data.aspect,
                padding="max_length",
                truncation=True,
                add_special_tokens=False,
                return_tensors=None,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            left_indices = self.tokenizer(
                data.left_text,
                padding="max_length",
                truncation=True,
                add_special_tokens=False,
                return_tensors=None,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            text_bert_indices = self.tokenizer(
                data.full_text_with_bert_tokens,
                padding="max_length",
                truncation=True,
                add_special_tokens=False,
                return_tensors=None,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            text_len = np.sum(text_indices["input_ids"] != 0)
            aspect_len = np.sum(aspect_indices["input_ids"] != 0)
            concat_segment_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
            concat_segment_indices = pad_and_truncate(concat_segment_indices, self.max_len)

            graph = generate_dependency_adj_matrix(data.full_text, data.aspect, self.senticnet, self.spacy_pipeline)
            sdat_graph = np.pad(
                graph,
                (
                    (0, self.max_len - graph.shape[0]),
                    (0, self.max_len - graph.shape[0]),
                ),
                "constant",
            )

            all_text_indices.append(text_indices["input_ids"])
            all_aspect_indices.append(aspect_indices["input_ids"])
            all_left_indices.append(left_indices["input_ids"])
            all_text_bert_indices.append(text_bert_indices["input_ids"])
            all_bert_segment_indices.append(concat_segment_indices)
            all_sdat_graph.append(sdat_graph)

        all_text_bert_indices = torch.tensor(all_text_bert_indices).to(self.device)
        all_bert_segment_indices = torch.tensor(np.array(all_bert_segment_indices)).to(self.device)
        text_embeddings = self.embedding_model(all_text_bert_indices, token_type_ids=all_bert_segment_indices)[
            "last_hidden_state"
        ]
        return [
            torch.tensor(all_text_indices),
            torch.tensor(all_aspect_indices),
            torch.tensor(all_left_indices),
            text_embeddings,
            torch.tensor(all_sdat_graph),
        ]

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

    def __call__(self, data_batch: List[Dict[str, List[str]]]) -> List[torch.Tensor]:
        processed_inputs = self._process_inputs(data_batch)
        return self._process_indices(processed_inputs)
