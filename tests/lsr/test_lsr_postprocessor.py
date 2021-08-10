import unittest
import pickle
from transformers import cached_path

from sgnlp.models.lsr.postprocess import LsrPostprocessor


class TestLsrPostprocessor(unittest.TestCase):
    def setUp(self):
        with open('test_data/sample_model_prediction.pickle', 'rb') as f:
            self.sample_model_prediction = pickle.load(f)

    def test_postprocessor(self):
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

        expected_result = {
            'clusters': [[[0, 2]], [[4, 6]], [[9, 11]], [[12, 15]], [[18, 20]], [[21, 23]], [[30, 32]], [[37, 38]],
                         [[41, 42]], [[43, 44]], [[46, 48]], [[49, 51]], [[52, 54]]],
            'document': ['Lark', 'Force', 'was', 'an', 'Australian', 'Army', 'formation', 'established', 'in', 'March',
                         '1941', 'during', 'World', 'War', 'II', 'for', 'service', 'in', 'New', 'Britain', 'and', 'New',
                         'Ireland', '.', 'Under', 'the', 'command', 'of', 'Lieutenant', 'Colonel', 'John', 'Scanlan',
                         ',', 'it', 'was', 'raised', 'in', 'Australia', 'and', 'deployed', 'to', 'Rabaul', 'and',
                         'Kavieng', ',', 'aboard', 'SS', 'Katoomba', ',', 'MV', 'Neptuna', 'and', 'HMAT', 'Zealandia',
                         ',', 'to', 'defend', 'their', 'strategically', 'important', 'harbours', 'and', 'airfields',
                         '.'],
            'relations': [{'score': 0.6414930820465088, 'relation': 'operator', 'object_idx': 0, 'subject_idx': 1},
                          {'score': 0.9599064588546753, 'relation': 'inception', 'object_idx': 0, 'subject_idx': 2},
                          {'score': 0.8167697787284851, 'relation': 'conflict', 'object_idx': 0, 'subject_idx': 3},
                          {'score': 0.7238405346870422, 'relation': 'country', 'object_idx': 0, 'subject_idx': 7},
                          {'score': 0.8506803512573242, 'relation': 'conflict', 'object_idx': 1, 'subject_idx': 3},
                          {'score': 0.8363937735557556, 'relation': 'country', 'object_idx': 1, 'subject_idx': 7},
                          {'score': 0.5114890336990356, 'relation': 'military branch', 'object_idx': 6,
                           'subject_idx': 1},
                          {'score': 0.36037707328796387, 'relation': 'conflict', 'object_idx': 6, 'subject_idx': 3},
                          {'score': 0.5998827219009399, 'relation': 'country of citizenship', 'object_idx': 6,
                           'subject_idx': 7},
                          {'score': 0.7587713599205017, 'relation': 'participant of', 'object_idx': 7,
                           'subject_idx': 3},
                          {'score': 0.4836481213569641, 'relation': 'country', 'object_idx': 8, 'subject_idx': 7},
                          {'score': 0.5315709710121155, 'relation': 'country', 'object_idx': 9, 'subject_idx': 7},
                          {'score': 0.384608656167984, 'relation': 'country', 'object_idx': 10, 'subject_idx': 7},
                          {'score': 0.4059601426124573, 'relation': 'country', 'object_idx': 11, 'subject_idx': 7}]
        }

        rel2id_path = cached_path('https://sgnlp.blob.core.windows.net/models/lsr/rel2id.json')
        rel_info_path = cached_path('https://sgnlp.blob.core.windows.net/models/lsr/rel_info.json')

        postprocessor = LsrPostprocessor.from_file_paths(rel2id_path=rel2id_path, rel_info_path=rel_info_path)
        result = postprocessor(self.sample_model_prediction[0], instance)

        self.assertEqual(expected_result, result)
