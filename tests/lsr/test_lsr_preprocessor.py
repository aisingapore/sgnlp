import unittest
import torch
import pickle
import pathlib
from transformers import cached_path

from sgnlp.models.lsr import LsrPreprocessor, LsrConfig

DIR = pathlib.Path(__file__).parent


class TestLsrPreprocessor(unittest.TestCase):
    def setUp(self):
        with open(DIR / 'test_data/sample_preprocessed_input.pickle', 'rb') as f:
            self.sample_preprocessed_input = pickle.load(f)

    def test_preprocessor(self):
        instance = {
            "vertexSet": [
                [{"name": "Lark Force", "pos": [0, 2], "sent_id": 0, "type": "ORG"}],
                [{"name": "Australian Army", "pos": [4, 6], "sent_id": 0, "type": "ORG"}],
                [{"pos": [9, 11], "type": "TIME", "sent_id": 0, "name": "March 1941"}],
                [{"name": "World War II", "pos": [12, 15], "sent_id": 0, "type": "MISC"}],
                [{"name": "New Britain", "pos": [18, 20], "sent_id": 0, "type": "LOC"}],
                [{"name": "New Ireland", "pos": [21, 23], "sent_id": 0, "type": "LOC"}],
                [{"name": "John Scanlan", "pos": [6, 8], "sent_id": 1, "type": "PER"}],
                [{"name": "Australia", "pos": [13, 14], "sent_id": 1, "type": "LOC"}],
                [{"name": "Rabaul", "pos": [17, 18], "sent_id": 1, "type": "LOC"}],
                [{"name": "Kavieng", "pos": [19, 20], "sent_id": 1, "type": "LOC"}],
                [{"pos": [22, 24], "type": "MISC", "sent_id": 1, "name": "SS Katoomba"}],
                [{"pos": [25, 27], "type": "MISC", "sent_id": 1, "name": "MV Neptuna"}],
                [{"name": "HMAT Zealandia", "pos": [28, 30], "sent_id": 1, "type": "MISC"}],
            ],
            "labels": [
                {"r": "P607", "h": 1, "t": 3, "evidence": [0]},
                {"r": "P17", "h": 1, "t": 7, "evidence": [0, 1]},
                {"r": "P241", "h": 6, "t": 1, "evidence": [0, 1]},
                {"r": "P607", "h": 6, "t": 3, "evidence": [0, 1]},
                {"r": "P27", "h": 6, "t": 7, "evidence": [0, 1]},
                {"r": "P1344", "h": 7, "t": 3, "evidence": [0, 1]},
                {"r": "P17", "h": 11, "t": 7, "evidence": [1]},
                {"r": "P17", "h": 12, "t": 7, "evidence": [0, 1]},
                {"r": "P137", "h": 0, "t": 1, "evidence": [0, 1]},
                {"r": "P571", "h": 0, "t": 2, "evidence": [0]},
                {"r": "P607", "h": 0, "t": 3, "evidence": [0]},
                {"r": "P17", "h": 0, "t": 7, "evidence": [0, 1]}
            ],
            "title": "Lark Force",
            "sents": [
                ["Lark", "Force", "was", "an", "Australian", "Army", "formation", "established", "in", "March", "1941",
                 "during", "World", "War", "II", "for", "service", "in", "New", "Britain", "and", "New", "Ireland",
                 "."],
                ["Under", "the", "command", "of", "Lieutenant", "Colonel", "John", "Scanlan", ",", "it", "was",
                 "raised", "in", "Australia", "and", "deployed", "to", "Rabaul", "and", "Kavieng", ",", "aboard", "SS",
                 "Katoomba", ",", "MV", "Neptuna", "and", "HMAT", "Zealandia", ",", "to", "defend", "their",
                 "strategically", "important", "harbours", "and", "airfields", "."],
            ]
        }

        rel2id_path = cached_path('https://sgnlp.blob.core.windows.net/models/lsr/rel2id.json')
        word2id_path = cached_path('https://sgnlp.blob.core.windows.net/models/lsr/word2id.json')
        ner2id_path = cached_path('https://sgnlp.blob.core.windows.net/models/lsr/ner2id.json')
        config = LsrConfig.from_pretrained("https://sgnlp.blob.core.windows.net/models/lsr/config.json")
        preprocessor = LsrPreprocessor(rel2id_path=rel2id_path, word2id_path=word2id_path, ner2id_path=ner2id_path,
                                       config=config, is_train=True)
        preprocessed_input = preprocessor([instance])

        for k, v in preprocessed_input.items():
            if isinstance(v, torch.Tensor):
                self.assertTrue(torch.equal(v, self.sample_preprocessed_input[k]))
            else:
                self.assertEqual(self.sample_preprocessed_input[k], v)
