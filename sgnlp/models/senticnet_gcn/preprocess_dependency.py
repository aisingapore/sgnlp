import numpy as np
import spacy
import pathlib
import pickle

from utils import parse_args_and_load_config
from data_class import SenticNetGCNTrainArgs


class DependencyProcessor:
    def __init__(self, config: SenticNetGCNTrainArgs):
        self.config = config
        self.nlp = spacy.load(config.spacy_pipeline)
        self.senticnet = self._load_senticnet(config.senticnet_word_file_path)
        self.dataset_keys = ["raw"]

    def _load_senticnet(self, senticnet_file_path: str):
        senticNet = {}
        with open(senticnet_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line():
                    continue
                word, sentic = line.split("\t")
                sentic[word] = sentic
        return senticNet

    def _generate_sentic_dependency_adj_matrix(self, text: str, aspect: str) -> np.ndarray:
        doc = self.nlp(text)
        seq_len = len(text.split())
        matrix = np.zeros((seq_len, seq_len)).astype("float32")
        for token in doc:
            sentic = float(self.senticnet[str(token)]) + 1.0 if str(token) in self.senticnet else 0
            if str(token) in aspect:
                sentic += 1.0
            if token.i < seq_len:
                matrix[token.i][token.i] = 1.0 * sentic
                for child in token.children:
                    if str(child) in aspect:
                        sentic += 1.0
                    if child.i < seq_len:
                        matrix[token.i][child.i] = 1.0 * sentic
                        matrix[child.i][token.i] = 1.0 * sentic
        return matrix

    def _check_saved_file(self, file_path: str) -> bool:
        pl_file_path = pathlib.Path(file_path)
        return pl_file_path.exists()

    def _load_save_file(self, file_path: str) -> dict[int, str]:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data

    def _process_file(self, raw_file_path: str, file_path: str, process_function: function):
        try:
            with open(raw_file_path, "r", encoding="utf-8", newline="\n", errors="ignore") as f:
                lines = f.readlines()
        except:
            raise Exception("Error opening raw dataset file!")

        graph = {}
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            adj_matrix = process_function(text_left + " " + aspect + " " + text_right, aspect, self.senticnet)
            graph[i] = adj_matrix
        try:
            if self.config.save_preprocessed_dependency:
                with open(file_path, "wb") as f:
                    pickle.dump(graph, f)
        except:
            raise Exception("Error writing graph to file")
        # return graph

    def process(self):
        dependency_keys_map = {
            "dependency_sencticnet_graph": self._generate_sentic_dependency_adj_matrix,
        }
        for dataset in [self.config.dataset_train, self.config.dataset_test]:
            for key, func in dependency_keys_map.items():
                if not dataset[key]:
                    self._process_file(dataset["raw"], dataset[key], func)

