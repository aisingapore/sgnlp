import collections
import logging
import nltk
import operator
import os
import pathlib
import re
from typing import List, Optional, Tuple

from transformers import PreTrainedTokenizer


logging.basicConfig(level=logging.DEBUG)


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}


class NEA_NLTK_Tokenizer(object):
    """Tokenizer for NEA.
       Performs word level tokenization via NLTK package followed by combining entity placeholders.
    """
    def __init__(self, do_lower_case: bool = True):
        self.do_lower_case = do_lower_case

    def tokenize(self, text: str) -> List[str]:
        """Main tokenize method

        Args:
            text (str): text string to tokenize

        Returns:
            List[str]: return a list of tokenized string
        """
        if self.do_lower_case:
            text = text.lower()
        tokens = nltk.word_tokenize(text)
        tokens = self._merge_tokens(tokens)
        return tokens

    def _merge_tokens(self, tokens: List[str]) -> List[str]:
        for index, token in enumerate(tokens):
            if token == '@' and (index + 1) < len(tokens):
                tokens[index + 1] = '@' + re.sub('[0-9]+.*', '', tokens[index + 1])
                tokens.pop(index)
        return tokens


class NEATokenizer(PreTrainedTokenizer):
    """
    The NEA Tokenizer class used WordLevel tokenization to generate tokens.

    Args:
        text (:obj:`str`):
            input text string to tokenize

    Example::
        # 1. From local vocab file
        vocab_file = 'vocab.txt'
        tokenizer = NEATokenizer(vocab_file=vocab_file)
        tokens = tokenizer("Hello world!")
        tokens["input_ids"]

        # 2. Train vocab from dataset file
        train_file = 'dataset.tsv'
        tokenizer = NEATokenizer(train_file=train_file, train_vocab=True)
        tokens = tokenizer("Hello world!")
        tokens["input_ids"]

        # 3. Download pretrained from Azure storage
        import sgnlp_models.models.nea import NEAArguments
        import sgnlp_models.models.nea.utils import download_tokenizer_files_from_azure
        cfg = NEAArguments()
        download_tokenizer_files_from_azure(cfg)
        tokenizer = NEATokenizer.from_pretrained(cfg.tokenizer_args["save_folder"])
        tokens = tokenizer("Hello world!")
        tokens["input_ids"]
    """

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
            self,
            vocab_file: str = None,
            train_file: str = None,
            train_vocab: bool = False,
            prompt_id: int = 1,
            maxlen: int = 0,
            vocab_size: int = 4000,
            do_lower_case: bool = True,
            unk_token: str = "<unk>",
            pad_token: str = "<pad>",
            num_token: str = "<num>",
            **kwargs):
        super().__init__(
            prompt_id=prompt_id,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            pad_token=pad_token,
            num_token=num_token,
            **kwargs)
        self.nea_tokenizer = NEA_NLTK_Tokenizer(do_lower_case)
        if train_vocab:
            self.vocab = self.create_vocab(train_file, prompt_id, maxlen, vocab_size)
        else:
            self.vocab = NEATokenizer.load_vocabulary(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

    @property
    def do_lower_case(self):
        return self.nea_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def _tokenize(self, text: str) -> List[str]:
        return self.nea_tokenizer.tokenize(text)

    def _convert_token_to_id(self, token: str):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int):
        return self.ids_to_tokens(index, self.unk_token)

    def create_vocab(self, file_path: str, prompt_id: int, maxlen: int, vocab_size: int):
        total_words, unique_words = 0, 0
        word_freqs = {}
        with open(file_path, 'r', encoding='utf-8') as input_file:
            next(input_file)
            for line in input_file:
                tokens = line.strip().split('\t')
                essay_set = int(tokens[1])
                content = tokens[2].strip()
                if essay_set == prompt_id or prompt_id <= 0:
                    if self.do_lower_case:
                        content = content.lower()
                    content = self.tokenize(content)
                    if maxlen > 0 and len(content) > maxlen:
                        continue
                    for word in content:
                        try:
                            word_freqs[word] += 1
                        except KeyError:
                            unique_words += 1
                            word_freqs[word] = 1
                        total_words += 1
        sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)
        if vocab_size <= 0:
            vocab_size = 0
            for word, freq in sorted_word_freqs:
                if freq > 1:
                    vocab_size += 1
        vocab = collections.OrderedDict()
        vocab['<pad>'] = 0
        vocab['<unk>'] = 1
        vocab['<num>'] = 2
        vocab_len = len(vocab.keys())
        offset = vocab_len
        for word, _ in sorted_word_freqs[:vocab_size - vocab_len]:
            vocab[word] = offset
            offset += 1
        return vocab

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        os.makedirs(save_directory, exist_ok=True)
        vocab_file = pathlib.Path(save_directory).joinpath(VOCAB_FILES_NAMES['vocab_file'])
        with open(vocab_file, 'w', encoding='utf-8') as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logging.warning(
                        f'Saving vocabulary to {vocab_file}: Vocabulary indices are not consecutive.'
                        'Please check vocabulary is not corrupted!')
                writer.write(token + '\n')
                index += 1
        return (str(vocab_file),)

    @staticmethod
    def load_vocabulary(vocab_file: str):
        vocab = collections.OrderedDict()
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip("\n")
            vocab[token] = index
        return vocab
