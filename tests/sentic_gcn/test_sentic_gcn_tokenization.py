import pathlib
import pytest
import unittest

from transformers import PreTrainedTokenizer
from transformers.file_utils import to_numpy

from sgnlp.models.sentic_gcn.tokenization import SenticGCNTokenizer, SenticGCNBertTokenizer


PARENT_DIR = str(pathlib.Path(__file__).parent)


class TestSenticGCNTokenizerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.test_train_files = [PARENT_DIR + "/test_data/test_train.raw", PARENT_DIR + "/test_data/test_test.raw"]
        self.test_vocab_file = PARENT_DIR + "/test_data/test_vocab.pkl"

    def test_senticgcn_tokenizer_from_vocab(self):
        tokenizer = SenticGCNTokenizer(vocab_file=self.test_vocab_file)
        self.assertTrue(issubclass(tokenizer.__class__, PreTrainedTokenizer))

        output = tokenizer("fee fi fo fum")
        self.assertEqual(output["input_ids"], [10, 20, 30, 40])

    def test_senticgcn_tokenizer_from_train_files(self):
        tokenizer = SenticGCNTokenizer(train_files=self.test_train_files, train_vocab=True)
        self.assertTrue(issubclass(tokenizer.__class__, PreTrainedTokenizer))

        output = tokenizer("night service center")
        self.assertEqual(output["input_ids"], [6, 24, 25])


class TestSenticGCNBertTokenizerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.pretrained_tokenizer_name = "bert-base-uncased"

    @pytest.mark.slow
    def test_senticgcnbert_tokenizer(self):
        tokenizer = SenticGCNBertTokenizer.from_pretrained(self.pretrained_tokenizer_name)
        self.assertTrue(issubclass(tokenizer.__class__, PreTrainedTokenizer))

        output = tokenizer("fee fi fo fum")
        self.assertEqual(output["input_ids"], [7408, 10882, 1042, 2080, 11865, 2213])

        output = tokenizer("fee fi fo fum", max_length=30, padding="max_length")
        self.assertEqual(len(output["input_ids"]), 30)

        output = tokenizer("", max_length=10, padding="max_length")
        self.assertEqual(len(output["input_ids"]), 10)
