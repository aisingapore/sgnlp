import pathlib
import unittest

from sgnlp_models.models.nea.preprocess_embeddings import preprocess_embedding
from sgnlp_models.models.nea.data_class import NEAArguments

PARENT_DIR = pathlib.Path(__file__).parent


class ProcessEmbeddingTest(unittest.TestCase):
    def setUp(self) -> None:
        cfg = {
            "preprocess_embedding_args": {
                "raw_embedding_file": str(
                    PARENT_DIR / "test_data/preprocess_embeddings/raw_embeddings.txt"
                ),
                "preprocessed_embedding_file": str(
                    PARENT_DIR
                    / "test_data/preprocess_embeddings/processed_embeddings.w2v.txt"
                ),
            }
        }
        self.cfg = NEAArguments(**cfg)

    def test_function(self):
        preprocess_embedding(self.cfg)

        processed_emb_path = pathlib.Path(
            self.cfg.preprocess_embedding_args["preprocessed_embedding_file"]
        )
        self.assertTrue(processed_emb_path.exists())

    def tearDown(self) -> None:
        pathlib.Path(
            self.cfg.preprocess_embedding_args["preprocessed_embedding_file"]
        ).unlink()
