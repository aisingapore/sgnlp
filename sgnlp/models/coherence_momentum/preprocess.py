from typing import List

import torch
from transformers import XLNetTokenizer


class CoherenceMomentumPreprocessor:
    def __init__(self, model_size, max_len, tokenizer=None):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = XLNetTokenizer.from_pretrained(f"xlnet-{model_size}-cased")

        self.max_len = max_len

    def __call__(self, texts: List[str]):
        """

        Args:
            texts (List[str]): List of input texts

        Returns:
            Dict[str, str]: Returns a dictionary with the following key-values:
                "tokenized_texts": (torch.tensor) Tensors of tokenized ids of input texts
        """

        result = []
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            ids = self.pad_ids(ids)
            ids = self.tokenizer.build_inputs_with_special_tokens(ids)
            result.append(torch.tensor(ids))

        return {"tokenized_texts": torch.stack(result)}

    def pad_ids(self, ids):
        if len(ids) < self.max_len:
            padding_size = self.max_len - len(ids)
            padding = [
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
                for _ in range(padding_size)
            ]
            ids = ids + padding
        else:
            ids = ids[: self.max_len]

        return ids
