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

    def _generate_dependency_adj_matrix(self, text: str) -> np.ndarray:
        words_list = text.split()
        seq_len = len(words_list)
        matrix = np.zeros((seq_len, seq_len)).astype("float32")
        for i in range(seq_len):
            word = words_list[i]
            sentic = float(self.senticnet[word]) + 1.0 if word in self.senticnet else 0.5
            for j in range(seq_len):
                matrix[i][j] += sentic
            for k in range(seq_len):
                matrix[k][i] += sentic
            matrix[i][i] = 1.0
        return matrix

    def _generate_sentic_graph(self, text: str, aspect: str) -> np.ndarray:
        words_list = text.split()
        seq_len = len(words_list)
        matrix = np.zeros((seq_len, seq_len)).astype("float32")
        for i in range(seq_len):
            word = words_list[i]
            sentic = float(self.senticnet[word]) + 1.0 if word in self.senticnet else 0
            if word in aspect:
                sentic += 1.0
            for j in range(seq_len):
                matrix[i][j] += sentic
                matrix[j][i] += sentic
        for i in range(seq_len):
            if matrix[i][i] == 0:
                matrix[i][i] = 1.0
        return matrix

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

    def process(self):
        pass


if __name__ == "__main__":
    cfg = parse_args_and_load_config()
    import pprint

    pprint.pprint(cfg.dataset_files)
