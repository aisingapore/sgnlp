import re
import torch
from typing import List

from nltk import word_tokenize, sent_tokenize


def prepare_sentences(text):
    # tokenize paragraph into sentences
    original_sentences = sent_tokenize(text)
    original_sentences = list(
        map(lambda x: " ".join(word_tokenize(x)), original_sentences)
    )

    output = []
    ctx = []

    for idx, src in enumerate(original_sentences):
        if idx == 0:
            output += [[src, [src]]]
        else:
            output += [[src, ctx]]
        if len(ctx) == 2:
            ctx = ctx[1:]
        ctx += [src]

    output = list(map(lambda x: [x[0], " ".join(x[1])], output))
    original_sentences = list(
        map(
            lambda sent: re.sub(r'\s([?.!,"](?:\s|$))', r"\1", sent), original_sentences
        )
    )
    return original_sentences, output


class CsgecPreprocessor:
    def __init__(self, src_tokenizer, ctx_tokenizer):
        self.src_tokenizer = src_tokenizer
        self.ctx_tokenizer = ctx_tokenizer

    def __call__(self, texts: List[str]):
        batch_src_ids = []
        batch_ctx_ids = []

        for text in texts:
            src_ids = []
            ctx_ids = []
            original_sentences, prepared_inputs = prepare_sentences(text)

            for src_text, ctx_text in prepared_inputs:
                src_ids.append(torch.LongTensor(self.src_tokenizer(src_text).input_ids))
                ctx_ids.append(torch.LongTensor(self.ctx_tokenizer(ctx_text).input_ids))

            batch_src_ids.append(src_ids)
            batch_ctx_ids.append(ctx_ids)

        return batch_src_ids, batch_ctx_ids
