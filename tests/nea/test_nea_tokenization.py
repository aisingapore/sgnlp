import collections
import pathlib
import pytest
import shutil
import unittest

from sgnlp.models.nea.data_class import NEAArguments
from sgnlp.models.nea.tokenization import NEA_NLTK_Tokenizer, NEATokenizer
from sgnlp.models.nea.utils import download_tokenizer_files_from_azure

PARENT_DIR = pathlib.Path(__file__).parent
TEST_STRING = "Dear @CAPS1 @CAPS2, I believe that using computers will benefit us in many ways"


class NEA_NLTK_TokenizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = NEA_NLTK_Tokenizer(do_lower_case=True)

    def test_nea_nltk_tokenizer(self):
        tokens = self.tokenizer.tokenize(TEST_STRING)
        self.assertEqual(len(tokens), 15)
        self.assertEqual(tokens[1], '@caps')
        self.assertEqual(tokens[2], '@caps')

    def test_nea_nltk_tokenizer_no_lower_case(self):
        self.tokenizer.do_lower_case = False
        tokens = self.tokenizer.tokenize(TEST_STRING)
        self.assertEqual(len(tokens), 15)
        self.assertEqual(tokens[1], '@CAPS')
        self.assertEqual(tokens[2], '@CAPS')


class NEATokenizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = NEATokenizer(
            train_file=str(PARENT_DIR / "test_data/train.tsv"), train_vocab=True,
            vocab_size=40)
        self.save_vocab_dir = PARENT_DIR / "test_output/vocab"

    def test_forwardpass(self):
        output = self.tokenizer(TEST_STRING)
        self.assertEqual(len(output), 3)
        self.assertEqual(len(output['input_ids']), 15)

    def test_load_vocab(self):
        vocab = NEATokenizer.load_vocabulary(str(PARENT_DIR / "test_data/test_vocab/vocab.txt"))
        self.assertIsInstance(vocab, collections.OrderedDict)
        self.assertEqual(len(vocab), 50)

    def test_save_vocab(self):
        vocab_file = self.tokenizer.save_vocabulary(self.save_vocab_dir)
        self.assertEqual(vocab_file[0], str(self.save_vocab_dir / 'vocab.txt'))
        self.assertTrue((self.save_vocab_dir / 'vocab.txt').exists())

    def tearDown(self) -> None:
        shutil.rmtree(self.save_vocab_dir, ignore_errors=True)


@pytest.mark.slow
class NEATokenizerIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = NEAArguments()

    def test_from_pretrained(self):
        download_tokenizer_files_from_azure(self.cfg)
        tokenizer = NEATokenizer.from_pretrained(self.cfg.tokenizer_args['save_folder'])
        self.assertTrue(pathlib.Path(self.cfg.tokenizer_args['save_folder']).exists())
        self.assertTrue(pathlib.Path(self.cfg.tokenizer_args['save_folder']).joinpath(
            self.cfg.tokenizer_args['files'][0]).exists())
        self.assertTrue(pathlib.Path(self.cfg.tokenizer_args['save_folder']).joinpath(
            self.cfg.tokenizer_args['files'][1]).exists())
        self.assertTrue(pathlib.Path(self.cfg.tokenizer_args['save_folder']).joinpath(
            self.cfg.tokenizer_args['files'][2]).exists())
        output = tokenizer(TEST_STRING)
        self.assertEqual(len(output), 3)
        self.assertEqual(len(output['input_ids']), 15)

    def tearDown(self) -> None:
        shutil.rmtree(pathlib.Path(self.cfg.tokenizer_args['save_folder']), ignore_errors=True)
