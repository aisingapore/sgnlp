import torch
import itertools


def join_document(json_instance):
    """Returns the document as a single string.

    json_instance: A JSON instance from DocRED dataset
    """
    joined_sentences = [' '.join(sent) for sent in json_instance['sents']]
    document = ' '.join(joined_sentences)
    return document


def h_t_idx_generator(length):
    """Generates idx for all possible head -> tail node combinations excluding self-reference"""
    for h_idx, t_idx in itertools.product(range(length), range(length)):
        if h_idx != t_idx:
            yield h_idx, t_idx


def idx2ht(idx, vertex_set_length):
    """Gets h_idx, t_idx from enumerated idx from h_t_idx_generator"""
    h_idx = idx // (vertex_set_length - 1)
    t_idx = idx % (vertex_set_length - 1)
    if t_idx >= h_idx:
        t_idx += 1
    return h_idx, t_idx


def get_default_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
