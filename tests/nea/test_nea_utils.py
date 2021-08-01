import shutil
import pathlib
import unittest
import pytest

import numpy as np
import torch

from sgnlp_models.models.nea import NEARegPoolingModel
from sgnlp_models.models.nea import NEAArguments
from sgnlp_models.models.nea.utils import (
    init_model,
    get_emb_matrix,
    pad_sequences_from_list,
    get_model_friendly_scores,
    convert_to_dataset_friendly_scores,
    qwk,
    train_and_save_tokenizer,
    read_dataset,
    process_results,
)

PARENT_DIR = pathlib.Path(__file__).parent


class NEAUtilsTest(unittest.TestCase):
    def test_init_model(self):
        config = {"model_type": "regp"}
        cfg = NEAArguments(**config)
        model = init_model(cfg)
        self.assertIsInstance(model, NEARegPoolingModel)

    def test_get_emb_matrix(self):
        vocab = {"the": 0, "and": 1, "in": 2}
        emb_path = str(PARENT_DIR / "test_data/embeddings.w2v.txt")
        full_emb_path = pathlib.Path(__file__).parent / emb_path
        output = get_emb_matrix(vocab, full_emb_path)
        output_shape = output.shape

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(len(vocab.keys()), output_shape[0])
        self.assertEqual(output_shape[1], 50)

    def test_pad_sequences_from_list(self):
        array = [[1, 1, 1, 1], [1], [1, 1]]
        output = pad_sequences_from_list(array)
        output_shape = output.shape
        expected_output = torch.Tensor([[1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 0, 0]]).int()

        self.assertEqual(len(array), output_shape[0])
        self.assertEqual(len(array[0]), output_shape[1])
        self.assertTrue(torch.allclose(expected_output, output, atol=1e-4))
        self.assertIsInstance(output, torch.Tensor)

    def test_get_model_friendly_scores(self):
        scores_array = [10, 11, 9]
        prompt_id_array = [1, 1, 1]
        output = get_model_friendly_scores(scores_array, prompt_id_array)

        self.assertEqual(len(output), len(scores_array))
        self.assertEqual(len(output[output > 1]), 0)
        self.assertEqual(len(output[output < 0]), 0)
        self.assertIsInstance(output, torch.Tensor)

    def test_convert_to_dataset_friendly_scores(self):
        scores_array = np.array([0.8, 0.9, 0.7])
        prompt_id_array = np.array([1, 1, 1])
        output = convert_to_dataset_friendly_scores(scores_array, prompt_id_array)

        self.assertEqual(len(output), len(scores_array))
        self.assertIsInstance(output, np.ndarray)

    def test_qwk(self):
        rater_a_array = np.array([5, 6, 7])
        rater_b_array = np.array([5, 6, 9])
        min_rating = 2
        max_rating = 12
        qwk_score = qwk(rater_a_array, rater_b_array, min_rating, max_rating)

        self.assertIsInstance(qwk_score, float)
        self.assertTrue(-1 < qwk_score < 1)

    def test_read_dataset(self):
        file_path = str(PARENT_DIR / "test_data/train.tsv")

        data_x, data_y, prompt_ids = read_dataset(file_path, 1, 0, True)

        self.assertIsInstance(data_x, list)
        self.assertIsInstance(data_y, list)
        self.assertIsInstance(prompt_ids, list)
        self.assertTrue(1069, len(data_x))
        self.assertTrue(1069, len(data_y))
        self.assertTrue(1069, len(prompt_ids))

    def test_process_results(self):
        metrics = {"eval_loss": 0.01, "eval_qwk": 0.9, "eval_memory": 100}
        output = process_results(metrics)

        self.assertIsInstance(output, str)
        self.assertTrue("eval_loss" in output)
        self.assertTrue("eval_qwk" in output)


class TraindAndSaveTokenizerTest(unittest.TestCase):
    def setUp(self):
        args = {
            "preprocess_data_args": {
                "train_path": str(PARENT_DIR / "test_data/train.tsv"),
                "prompt_id": 1,
                "maxlen": 0,
            },
            "tokenizer_args": {
                "vocab_train_file": str(PARENT_DIR / "test_data/train.tsv"),
                "save_folder": str(PARENT_DIR / "test_data/vocab/"),
            },
        }
        self.args = NEAArguments(**args)

    @pytest.mark.slow
    def test_train_and_save_tokenizer(self):
        train_and_save_tokenizer(self.args)
        nea_tokenizer_path = pathlib.Path(self.args.tokenizer_args["save_folder"])

        self.assertTrue(
            pathlib.Path(nea_tokenizer_path / "special_tokens_map.json").exists()
        )
        self.assertTrue(
            pathlib.Path(nea_tokenizer_path / "tokenizer_config.json").exists()
        )
        self.assertTrue(pathlib.Path(nea_tokenizer_path / "vocab.txt").exists())

    def tearDown(self):
        shutil.rmtree(self.args.tokenizer_args["save_folder"])
