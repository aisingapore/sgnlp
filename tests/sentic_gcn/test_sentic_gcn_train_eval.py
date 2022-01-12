import pathlib
import pickle
import pytest
import shutil
import tempfile
import unittest

from sgnlp.models.sentic_gcn.data_class import SenticGCNTrainArgs
from sgnlp.models.sentic_gcn.train import SenticGCNTrainer, SenticGCNBertTrainer

PARENT_DIR = str(pathlib.Path(__file__).parent)


def find_result_file(path):
    for p in pathlib.Path(path).iterdir():
        if p.is_file() and p.suffix == ".pkl":
            yield p.resolve()


class TestSenticGCNTrainTestCase(unittest.TestCase):
    def setUp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.model_save_folder = pathlib.Path(tmpdir)
        with tempfile.TemporaryDirectory() as tmpdir:
            self.results_save_folder = pathlib.Path(tmpdir)

        cfg = {
            "senticnet_word_file_path": "",
            "save_preprocessed_senticnet": False,
            "saved_preprocessed_senticnet_file_path": PARENT_DIR + "/test_data/test_senticnet.pickle",
            "spacy_pipeline": "en_core_web_sm",
            "word_vec_file_path": "./glove/glove.840B.300d.txt",
            "dataset_train": [PARENT_DIR + "/test_data/test_train.raw"],
            "dataset_test": [PARENT_DIR + "/test_data/test_train.raw"],
            "valset_ratio": 0,
            "model": "senticgcn",
            "save_best_model": True,
            "save_model_path": str(self.model_save_folder),
            "tokenizer": "senticgcn",
            "train_tokenizer": True,
            "save_tokenizer": False,
            "save_tokenizer_path": "",
            "embedding_model": "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_embedding_model/",
            "build_embedding_model": False,
            "save_embedding_model": False,
            "save_embedding_model_path": "./embed_models/senticgcn_embed_semeval14_rest/",
            "save_results": True,
            "save_results_folder": str(self.results_save_folder),
            "initializer": "xavier_uniform_",
            "optimizer": "adam",
            "loss_function": "cross_entropy",
            "learning_rate": 0.001,
            "l2reg": 0.00001,
            "epochs": 100,
            "batch_size": 2,
            "log_step": 5,
            "embed_dim": 300,
            "hidden_dim": 300,
            "polarities_dim": 3,
            "dropout": 0.3,
            "seed": 776,
            "device": "cpu",
            "repeats": 2,
            "patience": 5,
            "max_len": 85,
        }
        self.cfg = SenticGCNTrainArgs(**cfg)

    def tearDown(self) -> None:
        shutil.rmtree(self.model_save_folder, ignore_errors=True)
        shutil.rmtree(self.results_save_folder, ignore_errors=True)

    @pytest.mark.slow
    def test_train(self):
        trainer = SenticGCNTrainer(self.cfg)
        trainer.train()

        result_file = list(find_result_file(self.results_save_folder))[0]

        with open(result_file, "rb") as f:
            results = pickle.load(f)

        self.assertTrue("Repeat_1" in results.keys())
        self.assertTrue("Repeat_2" in results.keys())
        self.assertTrue("test" in results.keys())
        for key, val in results.items():
            self.assertTrue("max_val_acc" in val.keys())
            self.assertTrue("max_val_f1" in val.keys())
            if key != "test":
                self.assertTrue("max_val_epoch" in val.keys())

        config_filepath = self.model_save_folder.joinpath("config.json")
        model_filepath = self.model_save_folder.joinpath("pytorch_model.bin")
        self.assertTrue(config_filepath.is_file())
        self.assertTrue(model_filepath.is_file())


class TestSenticGCNBertTrainTestCase(unittest.TestCase):
    def setUp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.model_save_folder = pathlib.Path(tmpdir)
        with tempfile.TemporaryDirectory() as tmpdir:
            self.results_save_folder = pathlib.Path(tmpdir)

        cfg = {
            "senticnet_word_file_path": "",
            "save_preprocessed_senticnet": False,
            "saved_preprocessed_senticnet_file_path": PARENT_DIR + "/test_data/test_senticnet.pickle",
            "spacy_pipeline": "en_core_web_sm",
            "word_vec_file_path": "./glove/glove.840B.300d.txt",
            "dataset_train": [PARENT_DIR + "/test_data/test_train.raw"],
            "dataset_test": [PARENT_DIR + "/test_data/test_train.raw"],
            "valset_ratio": 0,
            "model": "senticgcnbert",
            "save_best_model": True,
            "save_model_path": str(self.model_save_folder),
            "tokenizer": "bert-base-uncased",
            "embedding_model": "bert-base-uncased",
            "save_results": True,
            "save_results_folder": str(self.results_save_folder),
            "initializer": "xavier_uniform_",
            "optimizer": "adam",
            "loss_function": "cross_entropy",
            "learning_rate": 0.001,
            "l2reg": 0.00001,
            "epochs": 100,
            "batch_size": 2,
            "log_step": 5,
            "embed_dim": 300,
            "hidden_dim": 768,
            "polarities_dim": 3,
            "dropout": 0.3,
            "seed": 776,
            "device": "cpu",
            "repeats": 2,
            "patience": 5,
            "max_len": 85,
        }
        self.cfg = SenticGCNTrainArgs(**cfg)

    def tearDown(self) -> None:
        shutil.rmtree(self.model_save_folder, ignore_errors=True)
        shutil.rmtree(self.results_save_folder, ignore_errors=True)

    @pytest.mark.slow
    def test_train(self):
        trainer = SenticGCNBertTrainer(self.cfg)
        trainer.train()

        result_file = list(find_result_file(self.results_save_folder))[0]

        with open(result_file, "rb") as f:
            results = pickle.load(f)

        self.assertTrue("Repeat_1" in results.keys())
        self.assertTrue("Repeat_2" in results.keys())
        self.assertTrue("test" in results.keys())
        for key, val in results.items():
            self.assertTrue("max_val_acc" in val.keys())
            self.assertTrue("max_val_f1" in val.keys())
            if key != "test":
                self.assertTrue("max_val_epoch" in val.keys())

        config_filepath = self.model_save_folder.joinpath("config.json")
        model_filepath = self.model_save_folder.joinpath("pytorch_model.bin")
        self.assertTrue(config_filepath.is_file())
        self.assertTrue(model_filepath.is_file())
