from typing import List, Tuple

import torch
import torch.nn as nn
from transformers.tokenization_utils_base import BatchEncoding


class EncoderRNN(nn.Module):
    """
    EncoderRNN model to be used in the encoder of the RST Parser network.
    """

    def __init__(self, word_dim, hidden_size, rnn_layers=6, dropout=0.2):
        super(EncoderRNN, self).__init__()
        self.word_dim = word_dim
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.dropout_layer = nn.Dropout(dropout)
        self.batchnorm_input = nn.BatchNorm1d(
            word_dim, affine=False, track_running_stats=False
        )
        self.gru = nn.GRU(
            word_dim,
            hidden_size,
            rnn_layers,
            batch_first=True,
            dropout=(0 if rnn_layers == 1 else dropout),
            bidirectional=True,
        )

    def forward(
        self, input_embeddings: BatchEncoding, input_lengths: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for encoder RNN.

        Args:
            input_embeddings (BatchEncoding): input embeddings from the RST preprocessor.
            input_lengths (List[int]): list of input lengths.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: return outputs and hidden states of encoder.
        """
        # batch norm
        embeddings = input_embeddings["elmo_representations"][0]
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.batchnorm_input(embeddings)
        embeddings = embeddings.permute(0, 2, 1)

        # apply dropout
        embeddings = self.dropout_layer(embeddings)

        # added enforce_sorted=False because input_lengths are not sorted. enfore_sorted=True is only reuqired for
        # ONNX export. Reference: https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, input_lengths, batch_first=True, enforce_sorted=False
        )

        # initialize hidden state
        batch_size = embeddings.size(0)
        hidden_initial = self.init_hidden(batch_size)

        # feed-forward through GRU
        outputs, hidden = self.gru(packed, hidden_initial)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        # apply dropout
        outputs = outputs.contiguous()
        outputs = self.dropout_layer(outputs)

        # sum bidirectional GRU outputs
        outputs = outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size :]

        # obtain last hidden state of encoder
        hidden = hidden.contiguous()
        hidden = hidden[: self.rnn_layers, :, :] + hidden[self.rnn_layers :, :, :]

        return outputs, hidden

    def init_hidden(self, batch_size):
        h_0 = torch.zeros(2 * self.rnn_layers, batch_size, self.hidden_size)
        device = self.gru.all_weights[0][
            0
        ].device  # checks device that layer has been put on
        h_0 = h_0.to(device)

        return h_0
