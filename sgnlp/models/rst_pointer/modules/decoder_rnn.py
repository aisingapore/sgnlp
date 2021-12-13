from typing import Tuple

import torch
import torch.nn as nn


class DecoderRNN(nn.Module):
    """
    DecoderRNN model to be used in the decoder of the RST Parser network.
    """

    def __init__(self, input_size, hidden_size, rnn_layers=6, dropout=0.2):
        super(DecoderRNN, self).__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            rnn_layers,
            batch_first=True,
            dropout=(0 if rnn_layers == 1 else dropout),
        )

    def forward(
        self, input_hidden_states: torch.Tensor, last_hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for decoder RNN.

        Args:
            input_hidden_states (torch.Tensor): input hidden tensor from encoder RNN output.
            last_hidden (torch.Tensor): last hidden state from encoder RNN.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: return the output and final hidden state.
        """
        outputs, hidden = self.gru(input_hidden_states, last_hidden)
        return outputs, hidden
