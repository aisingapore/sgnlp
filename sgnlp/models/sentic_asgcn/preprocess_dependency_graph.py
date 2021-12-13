import numpy as np
import spacy
import pickle
from spacy.tokens import Doc

from utils import parse_args_and_load_config


class WhiteSpaceTokenizer(object):
    """
    Simple white space tokenizer
    """

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text: str):
        words = text.split()
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


class DependencyGraphPreprocessor(object):
    """
    Preprocessor wrapper class for processing dependency graph.
    """

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.tokenizer = WhiteSpaceTokenizer(self.nlp.vocab)

    def __dependency_adj_matrix(self, text: str) -> np.ndarray:
        tokens = self.nlp(text)
        words = text.split()
        matrix = np.zeros((len(words), len(words))).astype("float32")

        for token in tokens:
            matrix[token.i][token.i] = 1
            for child in token.children:
                matrix[token.i][child.i] = 1
                matrix[child.i][token.i] = 1
        return matrix

    def process(self, filename: str):
        """
        Main processing method, takes in raw data file and convert to adj matrix.

        Args:
            filename (str): filename of raw dataset to process
        """
        with open(
            filename, "r", encoding="utf-8", newline="\n", errors="ignore"
        ) as fin:
            lines = fin.readlines()
        idx2graph = {}
        with open(f"{filename}.graph", "wb") as fout:
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [
                    s.lower().strip() for s in lines[i].partition("$T$")
                ]
                aspect = lines[i + 1].lower().strip()
                idx2graph[i] = self.__dependency_adj_matrix(
                    f"{text_left} {aspect} {text_right}"
                )
            pickle.dump(idx2graph, fout)


if __name__ == "__main__":
    dgp = DependencyGraphPreprocessor()
    cfg = parse_args_and_load_config()
    for data_path in cfg.dependency_graph_preprocess:
        dgp.process(data_path)
