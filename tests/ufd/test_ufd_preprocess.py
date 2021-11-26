import pytest
import unittest

from sgnlp.models.ufd import (
    UFDPreprocessor,
    UFDTokenizer
)


class TestUFDPreprocessorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.test_string = [
            'dieser film ist wirklich gut!',
            'Diese Fortsetzung ist nicht so gut wie die Vorgeschichte']

    @pytest.mark.slow
    def test_preprocessor(self):
        preprocessor = UFDPreprocessor()
        text_features = preprocessor(self.test_string)
        self.assertTrue(len(text_features), len(self.test_string))

    @pytest.mark.slow
    def test_preprocessor_with_external_tokenizer(self):
        tokenizer = UFDTokenizer.from_pretrained('xlm-roberta-large')
        preprocessor = UFDPreprocessor(tokenizer=tokenizer)
        text_features = preprocessor(self.test_string)

        self.assertTrue('data_batch' in text_features.keys())
        self.assertTrue(len(text_features['data_batch']), len(self.test_string))
