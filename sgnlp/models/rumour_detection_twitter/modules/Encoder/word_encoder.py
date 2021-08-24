import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


class WordEncoder(nn.Module):

    """

    Encodes the information into vectors

    There are 2 pieces of information that goes into the encoded information:
    1. Word Embedding
    2. Position Embedding

    This set of codes would encode the word embedding information

    """

    def __init__(self, config):
        # modified the original script to remove the need for the data loader
        super(WordEncoder, self).__init__()
        self.config = config
        self.vocab_size = config.max_vocab

        # <------------- Defining the word embedding dimensions ------------->
        self.embedding_dim = self.config.emb_dim

        # <------------- Loadings the pretrained embedding weights  ------------->
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.emb.weight.requires_grad = (
            self.config.train_word_emb
        )  # make embedding non trainable

    def load_pretrained_embedding(self, pretrained_embedding_path):
        pretrained_embedding = torch.from_numpy(np.load(pretrained_embedding_path))
        self.emb.weight.data.copy_(pretrained_embedding)

    def forward(self, token_ids):

        """
        Ref:
        https://pytorch.org/docs/stable/nn.html

        Does encoding for the input:
        1. WE encoding

        <--------- WE Embedding --------->
        Input:
            token_ids : LongTensor shape of (batch_size, num_posts, num_words)

        Output:
        encoded_we_features : Tensor with shape of (batch_size, num_posts, num_words, emb_dim)

        """

        encoded_we_features = self.emb(token_ids)

        return encoded_we_features
