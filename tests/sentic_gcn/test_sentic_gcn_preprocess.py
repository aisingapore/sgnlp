import pathlib
import pytest
import unittest

import torch

from sgnlp.models.sentic_gcn.config import SenticGCNEmbeddingConfig, SenticGCNBertEmbeddingConfig
from sgnlp.models.sentic_gcn.modeling import SenticGCNEmbeddingModel, SenticGCNBertEmbeddingModel
from sgnlp.models.sentic_gcn.preprocess import (
    SenticGCNBasePreprocessor,
    SenticGCNPreprocessor,
    SenticGCNBertPreprocessor,
    SenticGCNData,
    SenticGCNBertData,
)
from sgnlp.models.sentic_gcn.tokenization import SenticGCNTokenizer, SenticGCNBertTokenizer


PARENT_DIR = str(pathlib.Path(__file__).parent)


class TestSenticGCNPreprocessorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.test_tokenizer = SenticGCNTokenizer(
            train_files=[PARENT_DIR + "/test_data/test_train.raw", PARENT_DIR + "/test_data/test_test.raw"],
            train_vocab=True,
        )
        test_embed_config = SenticGCNEmbeddingConfig()
        self.test_embed_model = SenticGCNEmbeddingModel(config=test_embed_config)
        self.test_inputs = [
            {"aspect": ["Soup"], "sentence": "Soup is tasty but soup is a little salty. Salty funkysoup."},  # 1, -1
            {
                "aspect": ["service"],
                "sentence": "Everyone that sat in the back outside agreed that it was the worst service we had ever received.",
            },  # -1
            {
                "aspect": ["location", "food"],
                "sentence": "it 's located in a strip mall near the beverly center , not the greatest location , but the food keeps me coming back for more .",
            },  # 0, 1
        ]
        self.test_senticnet = {"test": 1.0}

    @pytest.mark.slow
    def test_senticgcn_preprocessor(self):
        """
        Create preprocessor with all defaults input arguments
        """
        pre_proc = SenticGCNPreprocessor()
        self.assertTrue(issubclass(pre_proc.__class__, SenticGCNBasePreprocessor))
        self.assertEqual(pre_proc.tokenizer.__class__, SenticGCNTokenizer)
        self.assertEqual(pre_proc.embedding_model.__class__, SenticGCNEmbeddingModel)
        self.assertTrue(isinstance(pre_proc.senticnet, dict))

        processed_inputs, processed_indices = pre_proc(self.test_inputs)
        self.assertEqual(len(processed_inputs), 5)
        self.assertEqual(len(processed_indices), 5)

        for proc_input in processed_inputs:
            self.assertTrue(isinstance(proc_input, SenticGCNData))
            for key in ["full_text", "aspect", "left_text", "full_text_tokens", "aspect_token_index"]:
                self.assertTrue(hasattr(proc_input, key))

        for proc_index in processed_indices:
            self.assertTrue(isinstance(proc_index, torch.Tensor))
        self.assertEqual(processed_indices[0].shape, torch.Size([5, 27]))
        self.assertEqual(processed_indices[1].shape, torch.Size([5, 27]))
        self.assertEqual(processed_indices[2].shape, torch.Size([5, 27]))
        self.assertEqual(processed_indices[3].shape, torch.Size([5, 27, 300]))
        self.assertEqual(processed_indices[4].shape, torch.Size([5, 27, 27]))

    def test_senticgcn_preprocessor_from_external(self):
        """
        Create preprocessor with tokenizer, embedding model and senticnet from external instances
        """
        pre_proc = SenticGCNPreprocessor(
            tokenizer=self.test_tokenizer, embedding_model=self.test_embed_model, senticnet=self.test_senticnet
        )
        self.assertTrue(issubclass(pre_proc.__class__, SenticGCNBasePreprocessor))
        self.assertEqual(pre_proc.tokenizer.__class__, SenticGCNTokenizer)
        self.assertEqual(pre_proc.embedding_model.__class__, SenticGCNEmbeddingModel)
        self.assertTrue(isinstance(pre_proc.senticnet, dict))

        processed_inputs, processed_indices = pre_proc(self.test_inputs)
        self.assertEqual(len(processed_inputs), 5)
        self.assertEqual(len(processed_indices), 5)

    def test_senticgcn_preprocessor_from_file(self):
        """
        Create preprocessor with senticnet from pickle file
        """
        pre_proc = SenticGCNPreprocessor(
            tokenizer=self.test_tokenizer,
            embedding_model=self.test_embed_model,
            senticnet=PARENT_DIR + "/test_data/test_senticnet.pickle",
        )
        self.assertTrue(issubclass(pre_proc.__class__, SenticGCNBasePreprocessor))
        self.assertEqual(pre_proc.tokenizer.__class__, SenticGCNTokenizer)
        self.assertEqual(pre_proc.embedding_model.__class__, SenticGCNEmbeddingModel)
        self.assertTrue(isinstance(pre_proc.senticnet, dict))

        processed_inputs, processed_indices = pre_proc(self.test_inputs)
        self.assertEqual(len(processed_inputs), 5)
        self.assertEqual(len(processed_indices), 5)


class TestSenticGCNBertPreprocessorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.test_tokenizer = SenticGCNBertTokenizer.from_pretrained("bert-base-uncased")
        test_embed_config = SenticGCNBertEmbeddingConfig()
        self.test_embed_model = SenticGCNBertEmbeddingModel(config=test_embed_config)
        self.test_inputs = [
            {"aspect": ["Soup"], "sentence": "Soup is tasty but soup is a little salty. Salty funkysoup."},  # 1, -1
            {
                "aspect": ["service"],
                "sentence": "Everyone that sat in the back outside agreed that it was the worst service we had ever received.",
            },  # -1
            {
                "aspect": ["location", "food"],
                "sentence": "it 's located in a strip mall near the beverly center , not the greatest location , but the food keeps me coming back for more .",
            },  # 0, 1
        ]
        self.test_senticnet = {"test": 1.0}

    @pytest.mark.slow
    def test_senticgcnbert_preprocessor(self):
        """
        Create preprocessor with all defaults input arguments
        """
        pre_proc = SenticGCNBertPreprocessor()
        self.assertTrue(issubclass(pre_proc.__class__, SenticGCNBasePreprocessor))
        self.assertEqual(pre_proc.tokenizer.__class__, SenticGCNBertTokenizer)
        self.assertEqual(pre_proc.embedding_model.__class__, SenticGCNBertEmbeddingModel)
        self.assertTrue(isinstance(pre_proc.senticnet, dict))

        processed_inputs, processed_indices = pre_proc(self.test_inputs)
        self.assertEqual(len(processed_inputs), 5)
        self.assertEqual(len(processed_indices), 5)

        for proc_input in processed_inputs:
            self.assertTrue(isinstance(proc_input, SenticGCNBertData))
            for key in [
                "full_text",
                "aspect",
                "left_text",
                "full_text_with_bert_tokens",
                "full_text_tokens",
                "aspect_token_index",
            ]:
                self.assertTrue(hasattr(proc_input, key))

        for proc_index in processed_indices:
            self.assertTrue(isinstance(proc_index, torch.Tensor))
        self.assertEqual(processed_indices[0].shape, torch.Size([5, 85]))
        self.assertEqual(processed_indices[1].shape, torch.Size([5, 85]))
        self.assertEqual(processed_indices[2].shape, torch.Size([5, 85]))
        self.assertEqual(processed_indices[3].shape, torch.Size([5, 85, 768]))
        self.assertEqual(processed_indices[4].shape, torch.Size([5, 85, 85]))

    def test_senticgcnbert_preprocessor_from_external(self):
        """
        Create preprocessor with tokenizer, embedding model and senticnet from external instances
        """
        pre_proc = SenticGCNBertPreprocessor(
            tokenizer=self.test_tokenizer, embedding_model=self.test_embed_model, senticnet=self.test_senticnet
        )
        self.assertTrue(issubclass(pre_proc.__class__, SenticGCNBasePreprocessor))
        self.assertEqual(pre_proc.tokenizer.__class__, SenticGCNBertTokenizer)
        self.assertEqual(pre_proc.embedding_model.__class__, SenticGCNBertEmbeddingModel)
        self.assertTrue(isinstance(pre_proc.senticnet, dict))

        processed_inputs, processed_indices = pre_proc(self.test_inputs)
        self.assertEqual(len(processed_inputs), 5)
        self.assertEqual(len(processed_indices), 5)

    def test_senticgcnbert_preprocessor_from_file(self):
        """
        Create preprocessor with senticnet from pickle file
        """
        pre_proc = SenticGCNBertPreprocessor(
            tokenizer=self.test_tokenizer,
            embedding_model=self.test_embed_model,
            senticnet=PARENT_DIR + "/test_data/test_senticnet.pickle",
        )
        self.assertTrue(issubclass(pre_proc.__class__, SenticGCNBasePreprocessor))
        self.assertEqual(pre_proc.tokenizer.__class__, SenticGCNBertTokenizer)
        self.assertEqual(pre_proc.embedding_model.__class__, SenticGCNBertEmbeddingModel)
        self.assertTrue(isinstance(pre_proc.senticnet, dict))

        processed_inputs, processed_indices = pre_proc(self.test_inputs)
        self.assertEqual(len(processed_inputs), 5)
        self.assertEqual(len(processed_indices), 5)
