import os
import string
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import TypedDict, List
from collections import Counter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab


def process_ctx_quac(ctx: str) -> str:
    """
    Process the context passage of QuAC for ease of use.
    Moves CANNOTANSWER token from end to front.
    """
    ctx = ctx[:-13]
    ctx = "CANNOTANSWER " + ctx
    return ctx


def process_para_json(paragraph_json):
    """
    Process a paragraph json
    """
    paragraph_json["context"] = process_ctx_quac(paragraph_json["context"])
    for qa in paragraph_json["qas"]:
        for answer in qa["prev_ans"]:
            if answer["text"] == "CANNOTANSWER":
                answer["answer_start"] = 0
            else:
                answer["answer_start"] += 13
    return paragraph_json


def concatenate_qa(prev_qns_text_list, prev_ans_text_list):
    """
    Concatenates two lists of questions and answers.
    """
    qa = ""
    for q, a in zip(prev_qns_text_list, prev_ans_text_list):
        qa += q + " | " + a + " || "
    return qa


def char_tokenize(word):
    return [char for char in word]


def tokens2vector(tokens, vocab):
    return [vocab[token] for token in tokens]


def flatten_list(x):
    return [item for sublist in x for item in sublist]


def pad_char_vector(x, min_word_padding_size):
    x = [[torch.tensor(char_tokens, dtype=torch.long) for char_tokens in text] for text in x]

    text_lens = [len(text) for text in x]
    word_lens = [len(word) for text in x for word in text]
    word_lens.append(min_word_padding_size)

    x = flatten_list(x)
    x.append(torch.zeros(min_word_padding_size))  # Adds a dummy tensor to ensure padding to min size
    x = pad_sequence(x, batch_first=True)
    x = x[:-1]  # Remove dummy vector

    y = torch.zeros((len(text_lens), max(text_lens), max(word_lens)),
                    dtype=torch.long)

    start_idx = 0
    for i, text_len in enumerate(text_lens):
        y[i, :text_len] = x[start_idx:start_idx + text_len]
        start_idx += text_len

    return y


CHAR_TOKENS_KEYS = ['char_context', 'char_qa', 'char_candidate']
WORD_TOKENS_KEYS = ['word_context', 'word_qa', 'word_candidate']


def lif_3way_ap_collate_fn(min_word_padding_size):
    def collate_fn(batch):
        new_batch = {}
        for key in WORD_TOKENS_KEYS:
            x = [torch.tensor(instance[key], dtype=torch.long) for instance in batch]
            new_batch[key] = pad_sequence(x, batch_first=True)

        for key in CHAR_TOKENS_KEYS:
            x = [instance[key] for instance in batch]
            new_batch[key] = pad_char_vector(x, min_word_padding_size)

        try:
            labels = torch.tensor([instance['label'] for instance in batch],
                                  dtype=torch.int)
            new_batch['label'] = labels
        except KeyError:
            pass

        return new_batch

    return collate_fn


class LIF3WayAPModelInputInstance(TypedDict):
    passage: str
    qa: str
    candidate: str


class QuestionsAndAnswersDict(TypedDict):
    question: str
    answer: str


class RawText(TypedDict):
    questions_and_answers: List[QuestionsAndAnswersDict]
    context: str
    candidate: str


class ProcessedText(TypedDict):
    context: str
    qa: str
    candidate: str
    label: int  # Optional


class TokenizedData(TypedDict):
    word_context: int
    word_qa: int
    word_candidate: int
    char_context: List[int]
    char_qa: List[int]
    char_candidate: List[int]
    label: int  # Optional


class LIF3WayAPPreprocessor:
    def __init__(self, min_word_padding_size, word_vocab=None, char_vocab=None, num_context_answers=3):
        self.tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.num_context_answers = num_context_answers
        self.collate_fn = lif_3way_ap_collate_fn(min_word_padding_size)

    def load_vocab(self, word_vocab_path, char_vocab_path):
        self.word_vocab = torch.load(word_vocab_path)
        self.char_vocab = torch.load(char_vocab_path)

    def save_vocab(self, output_dir):
        torch.save(self.word_vocab, os.path.join(output_dir, 'word_vocab.pt'))
        torch.save(self.char_vocab, os.path.join(output_dir, 'char_vocab.pt'))

    def process_dataset(self, dataset) -> List[ProcessedText]:
        """
        Function for processing LIF dataset to be ready to be tokenized.
        """
        batch_processed_text = []
        for article in dataset:
            for paragraph_json in article['paragraphs']:
                paragraph_json = process_para_json(paragraph_json)
                context = paragraph_json["context"]
                qas = paragraph_json['qas']
                for qa in qas:
                    prev_qns_text_list = [q.strip().replace("\n", "") for q in qa['prev_qs']]
                    start_idx = max(0, len(prev_qns_text_list) - self.num_context_answers)
                    prev_qns_text_list = prev_qns_text_list[start_idx:]
                    prev_ans_text_list = [a['text'] for a in qa['prev_ans']][start_idx:]
                    candidate = qa['candidate'].strip()
                    label = qa['label']

                    qa_text = concatenate_qa(prev_qns_text_list, prev_ans_text_list)

                    batch_processed_text.append({
                        'context': context,
                        'qa': qa_text,
                        'candidate': candidate,
                        'label': label
                    })

        return batch_processed_text

    def process_text_input(self, batch_data: List[RawText]) -> List[ProcessedText]:
        """
        Preprocesses raw text into to be ready to be tokenized.
        Appends CANNOTANSWER token to start of context. If answer is an empty string, replace with CANNOTANSWER token.
        Concatenates previous questions and answers to a single string.
        """
        batch_processed_text = []
        for instance in batch_data:
            truncated_qa = instance["questions_and_answers"][-self.num_context_answers:]
            prev_qs = [item["question"] for item in truncated_qa]
            prev_ans = [item["answer"] if item["answer"] != "" else "CANNOTANSWER" for item in truncated_qa]

            context = "CANNOTANSWER " + instance["context"]
            qa = concatenate_qa(prev_qns_text_list=prev_qs, prev_ans_text_list=prev_ans)
            candidate = instance["candidate"].strip()

            processed_text = {
                "context": context,
                "qa": qa,
                "candidate": candidate
            }
            batch_processed_text.append(processed_text)

        return batch_processed_text

    def tokenize(self, batch_data: List[ProcessedText], build_word_vocab=False) -> (List[TokenizedData], Counter):
        if build_word_vocab:
            word_counter = Counter()

        tokenized_data = []
        for instance in batch_data:
            tokenized_context = self.tokenizer(instance["context"])
            tokenized_qa = self.tokenizer(instance["qa"])
            tokenized_candidate = self.tokenizer(instance["candidate"])

            tokenized_instance = {
                'word_context': tokenized_context,
                'word_qa': tokenized_qa,
                'word_candidate': tokenized_candidate,
                'char_context': [char_tokenize(word) for word in tokenized_context],
                'char_qa': [char_tokenize(word) for word in tokenized_qa],
                'char_candidate': [char_tokenize(word) for word in tokenized_candidate],
            }

            if 'label' in instance.keys():
                tokenized_instance['label'] = instance['label']

            if build_word_vocab:
                word_counter.update(tokenized_context)
                word_counter.update(tokenized_qa)
                word_counter.update(tokenized_candidate)

            tokenized_data.append(tokenized_instance)

        if build_word_vocab:
            return tokenized_data, word_counter
        return tokenized_data, None

    def vectorize(self, tokenized_data: List[TokenizedData]):
        vectorized_data = []
        for instance in tokenized_data:
            new_instance = {}

            for c_key in CHAR_TOKENS_KEYS:
                new_instance[c_key] = [tokens2vector(tokens=char_tokens, vocab=self.char_vocab)
                                       for char_tokens in instance[c_key]]

            for w_key in WORD_TOKENS_KEYS:
                new_instance[w_key] = tokens2vector(tokens=instance[w_key], vocab=self.word_vocab)

            if 'label' in instance.keys():
                new_instance['label'] = instance['label']

            vectorized_data.append(new_instance)

        return vectorized_data

    def build_vocab(self, batch_data, max_word_vocab_size=None):
        """
        Builds vocab. Returns tokenized data as a byproduct.
        """
        # Char vocab
        char_counter = Counter()
        char_counter.update(string.printable)  # Initialize counter with printable chars
        self.char_vocab = Vocab(char_counter, min_freq=1,
                                specials=('<pad>', '<unk>'),
                                max_size=None)

        # Word vocab
        tokenized_data, word_counter = self.tokenize(batch_data, build_word_vocab=True)
        self.word_vocab = Vocab(word_counter, min_freq=1,
                                specials=('<pad>', '<unk>'),
                                max_size=max_word_vocab_size)
        return tokenized_data

    def __call__(self, batch_data: List[RawText]):
        """
        Processes RawText into tensors for input into model.
        """
        processed_batch_data = self.process_text_input(batch_data)
        tokenized_data, _ = self.tokenize(processed_batch_data)
        vectorized_data = self.vectorize(tokenized_data)
        tensors = self.collate_fn(vectorized_data)

        return tensors
