import pathlib
import pytest
import shutil
import unittest

from sgnlp.models.nea import (
    NEAArguments,
    NEAPreprocessor,
    NEATokenizer,
    download_tokenizer_files_from_azure,
)


class TestNEAPreprocessorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = NEAArguments()
        download_tokenizer_files_from_azure(self.cfg)
        self.test_string = [
            "Dear @CAPS1 @CAPS2, I believe that using computers will benefit us in many ways like talking and becoming\
            friends will others through websites like facebook and mysace.",
            "Dear, @CAPS1 @CAPS2 @CAPS3 More and more people use computers, but not everyone agrees that this benefits\
            society."]

    @pytest.mark.slow
    def test_preprocessor(self):
        preprocessor = NEAPreprocessor(tokenizer_name=self.cfg.tokenizer_args['save_folder'])
        tokens = preprocessor(self.test_string)

        self.assertTrue(len(tokens), 3)
        self.assertTrue(tokens['input_ids'].shape, tokens['attention_mask'].shape)
        self.assertTrue(tokens['input_ids'].shape, tokens['token_type_ids'].shape)
        self.assertTrue(tokens['input_ids'][0].shape, tokens['input_ids'][1].shape)
        self.assertTrue(tokens['input_ids'][0].shape, tokens['attention_mask'][0].shape)
        self.assertTrue(tokens['input_ids'][1].shape, tokens['attention_mask'][1].shape)

    @pytest.mark.slow
    def test_preprocessor_with_external_tokenizer(self):
        tokenizer = NEATokenizer.from_pretrained(self.cfg.tokenizer_args['save_folder'])
        preprocessor = NEAPreprocessor(tokenizer=tokenizer)
        tokens = preprocessor(self.test_string)

        self.assertTrue(len(tokens), 3)
        self.assertTrue(tokens['input_ids'].shape, tokens['attention_mask'].shape)
        self.assertTrue(tokens['input_ids'].shape, tokens['token_type_ids'].shape)
        self.assertTrue(tokens['input_ids'][0].shape, tokens['input_ids'][1].shape)
        self.assertTrue(tokens['input_ids'][0].shape, tokens['attention_mask'][0].shape)
        self.assertTrue(tokens['input_ids'][1].shape, tokens['attention_mask'][1].shape)

    def tearDown(self) -> None:
        shutil.rmtree(pathlib.Path(self.cfg.tokenizer_args['save_folder']), ignore_errors=True)
