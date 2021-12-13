import pathlib
import pickle
from typing import List

from transformers import PreTrainedTokenizer


class SenticASGCNTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file: str = None,
        train_files: List[str] = None,
        train_vocab: bool = False,
        do_lower_case: bool = True,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        **kwargs,
    ):
        super().__init__(
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs,
        )
        self.do_lower_case = do_lower_case
        if train_vocab:
            self.vocab = self.create_vocab(train_files)
        else:
            with open(vocab_file, "rb") as fin:
                self.vocab = pickle.load(fin)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

    @property
    def do_lower_case(self):
        return self.do_lower_case

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens(index, self.unk_token)

    @staticmethod
    def __read_text_file(file_names: List[str]) -> str:
        """
        Helper method to read contents of a list of text files.

        Args:
            file_names (List[str]): list of text files to read.

        Returns:
            str: return a concatenated string of text files contents.
        """
        text = ""
        for fname in file_names:
            with open(
                fname, "r", encoding="utf-8", newline="\n", errors="ignore"
            ) as fin:
                lines = fin.readlines()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [
                    s.lower().strip() for s in lines[i].partition("$T$")
                ]
                aspect = lines[i + 1].lower().strip()
                text += f"{text_left} {aspect} {text_right} "  # Left a space at the end
        return text

    def create_vocab(self, save_directory: str):
        text = self.__read_text_file()
        if self.do_lower_case:
            text = text.lower()
        vocab = {}
        vocab[self.pad_token] = 0
        vocab[self.unk_token] = 1
        offset = len(vocab.keys())

        words = text.split()
        for word in words:
            if word not in vocab:
                vocab[word] = offset
                offset += 1
        return vocab

    def _tokenize(self, text, **kwargs):
        if self.do_lower_case:
            text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.vocab[w] if w in self.vocab else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence
