import logging
import pathlib
import shutil
import string
import tempfile
import urllib.parse
from collections import namedtuple
from typing import Dict, List, Tuple, Union

import numpy as np
import spacy
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, PretrainedConfig, PreTrainedModel

from .config import SenticGCNEmbeddingConfig, SenticGCNBertEmbeddingConfig
from .modeling import SenticGCNEmbeddingModel, SenticGCNBertEmbeddingModel
from .tokenization import SenticGCNTokenizer, SenticGCNBertTokenizer
from .utils import (
    load_and_process_senticnet,
    download_tokenizer_files,
    download_url_file,
    pad_and_truncate,
    generate_dependency_adj_matrix,
)


logging.basicConfig(level=logging.DEBUG)


SenticGCNData = namedtuple(
    "SenticGCNData", ["full_text", "aspect", "left_text", "full_text_tokens", "aspect_token_indexes"]
)
SenticGCNBertData = namedtuple(
    "SenticGCNBertData",
    ["full_text", "aspect", "left_text", "full_text_with_bert_tokens", "full_text_tokens", "aspect_token_indexes"],
)


class SenticGCNBasePreprocessor:
    """
    Base preprocessor class provides initialization for spacy, senticnet, tokenizer and embedding model.
    Class is only meant to be inherited by derived preprocessor.
    """

    def __init__(
        self,
        tokenizer: Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast],
        embedding_model: Union[str, PreTrainedModel],
        tokenizer_class: PreTrainedTokenizer,
        embedding_config_class: PretrainedConfig,
        embedding_model_class: PreTrainedModel,
        config_filename: str = "config.json",
        model_filename: str = "pytorch_model.bin",
        spacy_pipeline: str = "en_core_web_sm",
        senticnet: Union[
            str, Dict[str, float]
        ] = "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticnet.pickle",
        device: str = "cpu",
    ) -> None:
        # Set device
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu") if not device else torch.device(device)
        )
        self.spacy_pipeline = spacy.load(spacy_pipeline)

        try:
            # Load senticnet
            if isinstance(senticnet, dict):
                senticnet_ = senticnet
            elif senticnet.startswith("https://") or senticnet.startswith("http://"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    temp_dir = pathlib.Path(tmpdir)
                download_url_file(senticnet, temp_dir)
                saved_path = temp_dir.joinpath("senticnet.pickle")
                senticnet_ = load_and_process_senticnet(saved_preprocessed_senticnet_file_path=saved_path)
                shutil.rmtree(temp_dir, ignore_errors=True)
            elif senticnet.endswith(".pkl") or senticnet.endswith(".pickle"):
                senticnet_ = load_and_process_senticnet(saved_preprocessed_senticnet_file_path=senticnet)
            elif senticnet.endswith(".txt"):
                senticnet_ = load_and_process_senticnet(senticnet_file_path=senticnet)
            else:
                raise ValueError(
                    """
                    Error initializing SenticNet!
                    For downloading from cloud storage, please provide url to pickle file location
                    (i.e. string url starting with https:// or http://).
                    For processed SenticNet dictionary, please provide pickle file location
                    (i.e. file with .pkl or .pickle extension).
                    For raw SenticNet-5.0 file, please provide text file path (i.e. file with .txt extension).
                    For externally created SenticNet dictionary, please provide a dictionary with words as key
                    and sentic score as values.
                    """
                )
            self.senticnet = senticnet_
        except Exception as e:
            logging.error(e)
            raise Exception(
                """
                    Error initializing SenticNet! Please ensure that input is either a dictionary, a str path to
                    a saved pickle file, an url to cloud storage or str path to the raw senticnet file.
                """
            )

        try:
            # Init Tokenizer
            if isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast):
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
    """
    Class for preprocessing sentence(s) and its aspect(s) to a batch of tensors for the SenticGCNBertModel
    to predict on.
    """

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
        senticnet: Union[
            str, Dict[str, float]
        ] = "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticnet.pickle",
        device: str = "cpu",
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            embedding_model=embedding_model,
            tokenizer_class=SenticGCNTokenizer,
            embedding_config_class=SenticGCNEmbeddingConfig,
            embedding_model_class=SenticGCNEmbeddingModel,
            config_filename=config_filename,
            model_filename=model_filename,
            spacy_pipeline=spacy_pipeline,
            senticnet=senticnet,
            device=device,
        )

    def _process_indices(self, data_batch: List[SenticGCNData]) -> List[torch.Tensor]:
        """
        Private helper method to generate all indices and embeddings from list of input data
        required for model input.

        Args:
            data_batch (List[SenticGCNData]): list of processed inputs as SenticGCNData

        Returns:
            List[torch.Tensor]: return a list of tensors for model input
        """
        all_text_indices = []
        all_aspect_indices = []
        all_left_indices = []
        all_sdat_graph = []
        all_data = []
        max_len = 0
        for data in data_batch:
            text_indices = self.tokenizer(
                data.full_text,
                return_tensors=None,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            aspect_indices = self.tokenizer(
                data.aspect,
                return_tensors=None,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            left_indices = self.tokenizer(
                data.left_text,
                return_tensors=None,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            graph = generate_dependency_adj_matrix(data.full_text, data.aspect, self.senticnet, self.spacy_pipeline)
            all_data.append(
                {
                    "text_indices": text_indices["input_ids"],
                    "aspect_indices": aspect_indices["input_ids"],
                    "left_indices": left_indices["input_ids"],
                    "sdat_graph": graph,
                }
            )
            if max_len < len(text_indices["input_ids"]):
                max_len = len(text_indices["input_ids"])

        for item in all_data:
            (text_indices, aspect_indices, left_indices, sdat_graph,) = (
                item["text_indices"],
                item["aspect_indices"],
                item["left_indices"],
                item["sdat_graph"],
            )

            text_padding = [0] * (max_len - len(text_indices))
            aspect_padding = [0] * (max_len - len(aspect_indices))
            left_padding = [0] * (max_len - len(left_indices))

            sdat_graph = np.pad(
                sdat_graph,
                ((0, max_len - len(text_indices)), (0, max_len - len(text_indices))),
                "constant",
            )

            all_text_indices.append(text_indices + text_padding)
            all_aspect_indices.append(aspect_indices + aspect_padding)
            all_left_indices.append(left_indices + left_padding)
            all_sdat_graph.append(sdat_graph)

        all_text_indices = torch.tensor(all_text_indices).to(self.device)
        text_embeddings = self.embedding_model(all_text_indices)

        return [
            all_text_indices,
            torch.tensor(all_aspect_indices).to(self.device),
            torch.tensor(all_left_indices).to(self.device),
            text_embeddings,
            torch.tensor(all_sdat_graph).to(self.device),
        ]

    def _process_inputs(self, data_batch: List[Dict[str, Union[str, List[str]]]]) -> List[SenticGCNData]:
        """
        Private helper method to process input data batch.
        Input entries are repeated for each input aspect.
        If input aspect have multiple occurance in the sentence, each occurance is process as an entry.

        Args:
            data_batch (List[Dict[str, Union[str, List[str]]]]): list of dictionaries with 2 keys, 'sentence' and 'aspect'.
                                            'sentence' value are strings and 'aspect' value is a list of accompanying aspect.

        Returns:
            List[SenticGCNData]: return list of processed inputs as SenticGCNData
        """
        processed_inputs = []
        for batch in data_batch:
            full_text = batch["sentence"].lower().strip()
            full_text_tokens = batch["sentence"].split()
            for aspect in batch["aspects"]:
                aspect = aspect.lower().strip()
                aspect_tokens = aspect.translate(str.maketrans("", "", string.punctuation)).split()
                aspect_indexes = []
                for idx in range(len(full_text_tokens)):
                    try:
                        if (
                            " ".join(full_text_tokens[idx : idx + len(aspect_tokens)])
                            .translate(str.maketrans("", "", string.punctuation))
                            .lower()
                            == aspect
                        ):
                            aspect_indexes.append(list(map(lambda x: idx + x, [*range(len(aspect_tokens))])))
                    except IndexError:
                        continue

                aspect_idxs = [index for index in range(len(full_text)) if full_text.startswith(aspect, index)]
                for aspect_index, aspect_token_indexes in zip(aspect_idxs, aspect_indexes):
                    left_text = full_text[:aspect_index].strip()
                    processed_inputs.append(
                        SenticGCNData(
                            full_text=full_text,
                            aspect=aspect,
                            left_text=left_text,
                            full_text_tokens=full_text_tokens,
                            aspect_token_indexes=aspect_token_indexes,
                        )
                    )
        return processed_inputs

    def __call__(
        self, data_batch: List[Dict[str, Union[str, List[str]]]]
    ) -> Tuple[List[SenticGCNData], List[torch.Tensor]]:
        """
        Method to generate list of input tensors from a list of sentences and their accompanying list of aspect.

        Args:
            data_batch (List[Dict[str, Union[str, List[str]]]]): list of dictionaries with 2 keys, 'sentence' and 'aspect'.
                                            'sentence' value are strings and 'aspect' value is a list of accompanying aspect.

        Returns:
            Tuple[List[SenticGCNData], List[torch.Tensor]]: return a list of ordered tensors for 'text_indices',
                'aspect_indices', 'left_indices', 'text_embeddings' and 'sdat_graph'.
        """
        processed_inputs = self._process_inputs(data_batch)
        return processed_inputs, self._process_indices(processed_inputs)


class SenticGCNBertPreprocessor(SenticGCNBasePreprocessor):
    """
    Class for preprocessing sentence(s) and its aspect(s) to a batch of tensors for the SenticGCNBertModel
    to predict on.
    """

    def __init__(
        self,
        tokenizer: Union[str, PreTrainedTokenizer] = "bert-base-uncased",
        embedding_model: Union[str, PreTrainedModel] = "bert-base-uncased",
        config_filename: str = "config.json",
        model_filename: str = "pytorch_model.bin",
        spacy_pipeline: str = "en_core_web_sm",
        senticnet: Union[
            str, Dict[str, float]
        ] = "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticnet.pickle",
        max_len: int = 85,
        device: str = "cpu",
    ) -> None:
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
        """
        Private helper method to generate all indices and embeddings from list of input data
        required for model input.

        Args:
            data_batch (List[SenticGCNBertData]): list of processed inputs as SenticGCNBertData

        Returns:
            List[torch.Tensor]: return a list of tensors for model input
        """
        all_text_indices = []
        all_aspect_indices = []
        all_left_indices = []
        all_text_bert_indices = []
        all_bert_segment_indices = []
        all_sdat_graph = []
        for data in data_batch:
            text_indices = self.tokenizer(
                data.full_text,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                add_special_tokens=False,
                return_tensors=None,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            aspect_indices = self.tokenizer(
                data.aspect,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                add_special_tokens=False,
                return_tensors=None,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            left_indices = self.tokenizer(
                data.left_text,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                add_special_tokens=False,
                return_tensors=None,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            text_bert_indices = self.tokenizer(
                data.full_text_with_bert_tokens,
                max_length=self.max_len,
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
            torch.tensor(all_text_indices).to(self.device),
            torch.tensor(all_aspect_indices).to(self.device),
            torch.tensor(all_left_indices).to(self.device),
            text_embeddings,
            torch.tensor(all_sdat_graph).to(self.device),
        ]

    def _process_inputs(self, data_batch: List[Dict[str, Union[str, List[str]]]]) -> List[SenticGCNBertData]:
        """
        Private helper method to process input data batch.
        Input entries are repeated for each input aspect.
        If input aspect have multiple occurance in the sentence, each occurance is process as an entry.

        Args:
            data_batch (List[Dict[str, Union[str, List[str]]]]): list of dictionaries with 2 keys, 'sentence' and 'aspect'.
                                            'sentence' value are strings and 'aspect' value is a list of accompanying aspect.

        Returns:
            List[SenticGCNBertData]: return list of processed inputs as SenticGCNBertData
        """
        processed_inputs = []
        for batch in data_batch:
            full_text = batch["sentence"].lower().strip()
            full_text_tokens = batch["sentence"].split()
            for aspect in batch["aspects"]:
                aspect = aspect.lower().strip()
                aspect_tokens = aspect.translate(str.maketrans("", "", string.punctuation)).split()
                aspect_indexes = []
                for idx in range(len(full_text_tokens)):
                    try:
                        if (
                            " ".join(full_text_tokens[idx : idx + len(aspect_tokens)])
                            .translate(str.maketrans("", "", string.punctuation))
                            .lower()
                            == aspect
                        ):
                            aspect_indexes.append(list(map(lambda x: idx + x, [*range(len(aspect_tokens))])))
                    except IndexError:
                        continue

                aspect_idxs = [index for index in range(len(full_text)) if full_text.startswith(aspect, index)]
                for aspect_index, aspect_token_indexes in zip(aspect_idxs, aspect_indexes):
                    left_text = full_text[:aspect_index].strip()
                    full_text_with_bert_tokens = f"[CLS] {full_text} [SEP] {aspect} [SEP]"
                    processed_inputs.append(
                        SenticGCNBertData(
                            full_text=full_text,
                            aspect=aspect,
                            left_text=left_text,
                            full_text_with_bert_tokens=full_text_with_bert_tokens,
                            full_text_tokens=full_text_tokens,
                            aspect_token_indexes=aspect_token_indexes,
                        )
                    )
        return processed_inputs

    def __call__(
        self, data_batch: List[Dict[str, Union[str, List[str]]]]
    ) -> Tuple[List[SenticGCNBertData], List[torch.Tensor]]:
        """
        Method to generate list of input tensors from a list of sentences and their accompanying list of aspect.

        Args:
            data_batch (List[Dict[str, Union[str, List[str]]]]): list of dictionaries with 2 keys, 'sentence' and 'aspect'.
                                            'sentence' value are strings and 'aspect' value is a list of accompanying aspect.

        Returns:
            Tuple[List[SenticGCNData], List[torch.Tensor]]: return a list of ordered tensors for 'text_indices',
                'aspect_indices', 'left_indices', 'text_embeddings' and 'sdat_graph'.
        """
        processed_inputs = self._process_inputs(data_batch)
        return processed_inputs, self._process_indices(processed_inputs)
