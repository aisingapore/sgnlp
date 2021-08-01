import pathlib
import shutil
import unittest

from sgnlp_models.models.nea.preprocess_raw_dataset import preprocess
from sgnlp_models.models.nea.data_class import NEAArguments

PARENT_DIR = pathlib.Path(__file__).parent


class NEAPreprocessDataTest(unittest.TestCase):
    def setUp(self) -> None:
        cfg = {
            "preprocess_raw_dataset_args": {
                "data_folder": str(PARENT_DIR / "test_data/preprocess_raw_dataset/"),
                "input_file": str(
                    PARENT_DIR
                    / "test_data/preprocess_raw_dataset/training_set_rel3.tsv"
                ),
            },
        }
        self.cfg = NEAArguments(**cfg)
        # make 4 more folds required for preprocess()
        train_ids_fold0 = (
            pathlib.Path(self.cfg.preprocess_raw_dataset_args["data_folder"])
            / "fold_0/train_ids.txt"
        )
        dev_ids_fold0 = (
            pathlib.Path(self.cfg.preprocess_raw_dataset_args["data_folder"])
            / "fold_0/dev_ids.txt"
        )
        test_ids_fold0 = (
            pathlib.Path(self.cfg.preprocess_raw_dataset_args["data_folder"])
            / "fold_0/test_ids.txt"
        )

        for idx in range(1, 5):
            fold_num = "fold_" + str(idx)
            fold_dir = (
                pathlib.Path(self.cfg.preprocess_raw_dataset_args["data_folder"])
                / fold_num
            )
            fold_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(train_ids_fold0, fold_dir / "train_ids.txt")
            shutil.copy(dev_ids_fold0, fold_dir / "dev_ids.txt")
            shutil.copy(test_ids_fold0, fold_dir / "test_ids.txt")

    def test_preprocess_raw_dataset(self):
        preprocess(self.cfg)
        train_path = (
            pathlib.Path(self.cfg.preprocess_raw_dataset_args["data_folder"])
            / "fold_0/train.tsv"
        )
        dev_path = (
            pathlib.Path(self.cfg.preprocess_raw_dataset_args["data_folder"])
            / "fold_0/dev.tsv"
        )
        test_path = (
            pathlib.Path(self.cfg.preprocess_raw_dataset_args["data_folder"])
            / "fold_0/test.tsv"
        )
        with open(train_path, "r") as f:
            train_path_lines = len(f.readlines())
        with open(train_path, "r") as f:
            dev_path_lines = len(f.readlines())
        with open(train_path, "r") as f:
            test_path_lines = len(f.readlines())

        self.assertTrue(train_path.exists())
        self.assertTrue(dev_path.exists())
        self.assertTrue(test_path.exists())

        self.assertEqual(train_path_lines, 6)
        self.assertEqual(dev_path_lines, 6)
        self.assertEqual(test_path_lines, 6)

    def tearDown(self):
        # Remove train_tsv, dev_tsv and test_tsv from fold 0
        train_tsv_path_fold0 = (
            pathlib.Path(self.cfg.preprocess_raw_dataset_args["data_folder"])
            / "fold_0/train.tsv"
        )
        dev_tsv_path_fold0 = (
            pathlib.Path(self.cfg.preprocess_raw_dataset_args["data_folder"])
            / "fold_0/dev.tsv"
        )
        test_tsv_path_fold0 = (
            pathlib.Path(self.cfg.preprocess_raw_dataset_args["data_folder"])
            / "fold_0/test.tsv"
        )
        train_tsv_path_fold0.unlink()
        dev_tsv_path_fold0.unlink()
        test_tsv_path_fold0.unlink()

        # remove all other fold folders
        for fold_idx in range(1, 5):
            fold_num = "fold_" + str(fold_idx)
            fold_dir = (
                pathlib.Path(self.cfg.preprocess_raw_dataset_args["data_folder"])
                / fold_num
            )
            shutil.rmtree(fold_dir)
