import pathlib
import pytest
import shutil
import tempfile
import unittest

import numpy as np

from sgnlp.models.sentic_gcn.utils import pad_and_truncate, load_and_process_senticnet


PARENT_DIR = str(pathlib.Path(__file__).parent)


class TestPadandTruncate(unittest.TestCase):
    def setUp(self) -> None:
        self.test_input = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.max_len = 50

    def test_pad_and_truncate(self):
        output = pad_and_truncate(self.test_input, max_len=self.max_len)
        self.assertEqual(type(output), np.ndarray)
        self.assertEqual(len(output), self.max_len)


class TestLoadandProcessSenticNet(unittest.TestCase):
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
