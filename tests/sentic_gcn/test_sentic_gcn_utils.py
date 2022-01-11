import pathlib
import shutil
import tempfile
import unittest
import unittest.mock as mock

import numpy as np
import spacy

from sgnlp.models.sentic_gcn.data_class import SenticGCNTrainArgs
from sgnlp.models.sentic_gcn.utils import (
    SenticGCNDataset,
    SenticGCNDatasetGenerator,
    pad_and_truncate,
    load_and_process_senticnet,
    generate_dependency_adj_matrix,
)


PARENT_DIR = str(pathlib.Path(__file__).parent)


class TestPadandTruncateTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.test_input = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.max_len = 50

    def test_pad_and_truncate(self):
        output = pad_and_truncate(self.test_input, max_len=self.max_len)
        self.assertEqual(type(output), np.ndarray)
        self.assertEqual(len(output), self.max_len)


class TestLoadandProcessSenticNetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.test_file = pathlib.Path(PARENT_DIR).joinpath("test_data").joinpath("senticnet.txt")
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.temp_dir = tmp_dir
        self.test_save_file_path = pathlib.Path(self.temp_dir).joinpath("senticnet.pkl")

    def tearDown(self) -> None:
        shutil.rmtree(self.test_save_file_path, ignore_errors=True)

    def test_load_and_process_senticnet_from_file(self):
        senticnet = load_and_process_senticnet(senticnet_file_path=self.test_file)
        self.assertEqual(type(senticnet), dict)
        self.assertTrue("CONCEPT" not in senticnet.keys())
        self.assertEqual(len(senticnet), 12)
        self.assertTrue("abandoned_person" not in senticnet.keys())
        self.assertTrue("abandoned_quarry" not in senticnet.keys())
        self.assertEqual(senticnet["abase"], "-0.90")

    def test_load_and_process_senticnet_save_file(self):
        _ = load_and_process_senticnet(
            senticnet_file_path=self.test_file,
            save_preprocessed_senticnet=True,
            saved_preprocessed_senticnet_file_path=self.test_save_file_path,
        )
        self.assertTrue(self.test_save_file_path.exists())

    def test_load_and_process_senticnet_from_pickle_file(self):
        _ = load_and_process_senticnet(
            senticnet_file_path=self.test_file,
            save_preprocessed_senticnet=True,
            saved_preprocessed_senticnet_file_path=self.test_save_file_path,
        )
        senticnet = load_and_process_senticnet(
            save_preprocessed_senticnet=False, saved_preprocessed_senticnet_file_path=str(self.test_save_file_path)
        )
        self.assertEqual(type(senticnet), dict)
        self.assertEqual(len(senticnet), 12)


class TestGenerateDependencyAdjMatrixTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.test_file = pathlib.Path(PARENT_DIR).joinpath("test_data").joinpath("senticnet.txt")
        self.senticnet = load_and_process_senticnet(self.test_file)
        self.spacy_pipeline = spacy.load("en_core_web_sm")
        self.test_text = "Soup is tasty but soup is a little salty."
        self.test_aspect = "soup"

    def test_generate_dependency_adj_matrix(self):
        matrix = generate_dependency_adj_matrix(self.test_text, self.test_aspect, self.senticnet, self.spacy_pipeline)
        self.assertTrue(type(matrix), np.ndarray)
        self.assertEqual(matrix.shape, (9, 9))


class TestSenticGCNDatasetGeneratorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        cfg = {
            "senticnet_word_file_path": PARENT_DIR + "/test_data/senticnet.txt",
            "spacy_pipeline": "en_core_web_sm",
            "dataset_train": [PARENT_DIR + "/test_data/test_train.raw"],
            "dataset_test": [PARENT_DIR + "/test_data/test_test.raw"],
            "valset_ratio": 0,
            "model": "senticgcn",
        }
        self.cfg = SenticGCNTrainArgs(**cfg)

    def test_read_raw_dataset(self):
        with mock.patch("sgnlp.models.sentic_gcn.tokenization.SenticGCNTokenizer") as MockClass:
            fake_tokenizer = MockClass()
        dataset_gen = SenticGCNDatasetGenerator(self.cfg, fake_tokenizer)
        data = dataset_gen._read_raw_dataset("train")
        self.assertEqual(len(data), 15)

    def test_generate_senticgcn_dataset(self):
        with mock.patch("sgnlp.models.sentic_gcn.tokenization.SenticGCNTokenizer") as MockClass:
            fake_tokenizer = MockClass(return_value={"input_ids": [1.0, 2.0, 3.0, 4.0, 5.0]})
        dataset_gen = SenticGCNDatasetGenerator(self.cfg, fake_tokenizer)
        dataset = dataset_gen._read_raw_dataset(self.cfg.dataset_train)
        data = dataset_gen._generate_senticgcn_dataset(dataset)
        self.assertEqual(len(data), 5)
        for data_row in data:
            keys = data_row.keys()
            self.assertTrue("text_indices" in keys)
            self.assertTrue("aspect_indices" in keys)
            self.assertTrue("left_indices" in keys)
            self.assertTrue("polarity" in keys)
            self.assertTrue("sdat_graph" in keys)

    def test_generate_senticgcn_bert_dataset(self):
        with mock.patch("sgnlp.models.sentic_gcn.tokenization.SenticGCNBertTokenizer") as MockClass:
            fake_tokenizer = MockClass(return_value={"input_ids": [1.0, 2.0, 3.0, 4.0, 5.0]})
        dataset_gen = SenticGCNDatasetGenerator(self.cfg, fake_tokenizer)
        dataset = dataset_gen._read_raw_dataset(self.cfg.dataset_train)
        data = dataset_gen._generate_senticgcnbert_dataset(dataset)
        self.assertEqual(len(data), 5)
        for data_row in data:
            keys = data_row.keys()
            self.assertTrue("text_indices" in keys)
            self.assertTrue("aspect_indices" in keys)
            self.assertTrue("left_indices" in keys)
            self.assertTrue("text_bert_indices" in keys)
            self.assertTrue("bert_segment_indices" in keys)
            self.assertTrue("polarity" in keys)
            self.assertTrue("sdat_graph" in keys)

    def test_generate_dataset(self):
        for model_type in ["senticgcn", "senticgcnbert"]:
            self.cfg.model = model_type
            class_path = (
                "sgnlp.models.sentic_gcn.tokenization.SenticGCNTokenizer"
                if model_type == "senticgcn"
                else "sgnlp.models.sentic_gcn.tokenization.SenticGCNBertTokenizer"
            )
            with mock.patch(class_path) as MockClass:
                fake_tokenizer = MockClass(return_value={"input_ids": [1.0, 2.0, 3.0, 4.0, 5.0]})
            dataset_gen = SenticGCNDatasetGenerator(self.cfg, fake_tokenizer)
            train_data, val_data, test_data = dataset_gen.generate_datasets()
            self.assertEqual(type(train_data), SenticGCNDataset)
            self.assertEqual(type(val_data), SenticGCNDataset)
            self.assertEqual(type(test_data), SenticGCNDataset)
            self.assertEqual(len(train_data), 5)
            self.assertEqual(len(val_data), 5)
            self.assertEqual(len(test_data), 5)
