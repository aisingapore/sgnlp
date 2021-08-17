import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.tokenization_utils_base import BatchEncoding


class EncoderRNN(nn.Module):
    def __init__(self, embedding_model, word_dim, hidden_size, device, rnn_layers=6, dropout=0.2):
        super(EncoderRNN, self).__init__()
        self.embedding_model = embedding_model.to(device)
        self.word_dim = word_dim
        self.hidden_size = hidden_size
        self.device = device
        self.rnn_layers = rnn_layers
        self.nnDropout = nn.Dropout(dropout)
        self.batchnorm_input = nn.BatchNorm1d(word_dim, affine=False, tracking_running_stats=False)
        self.gru = nn.GRU(
            word_dim,
            hidden_size,
            rnn_layers,
            batch_first=True,
            dropout=(0 if rnn_layers == 1 else dropout),
            bidirectional=True)

    def forward(self, input_embeddings: BatchEncoding):
        # batch norm
        embeddings = input_embeddings['data_batch'][0]
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.batchnorm_input(embeddings)
        embeddings = embeddings.permute(0, 2, 1)

        # apply dropout
        embeddings = self.nnDropout(embeddings)

        # TODO: Get input text length?

    def init_hidden(self, batch_size):
        h_0 = torch.zeros(2 * self.rnn_layers, batch_size, self.hidden_size)
        h_0 = h_0.to(self.device)

        return h_0
