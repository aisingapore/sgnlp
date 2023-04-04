import unittest

import torch

from sgnlp.models.rumour_stance.utils import set_device_and_seed


class TestInitEnvironment(unittest.TestCase):
    def setUp(self) -> None:
        self.env = set_device_and_seed()

    def test_call(self):
        self.assertIsInstance(self.env["local_rank"], int)
        self.assertIsInstance(self.env["device"], torch.device)
        self.assertIsInstance(self.env["n_gpu"], int)
