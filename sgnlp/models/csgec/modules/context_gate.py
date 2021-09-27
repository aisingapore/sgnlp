from numpy import sqrt
import torch
import torch.nn as nn


class ContextGate(nn.Module):
    """
    Layer that calculates the values of the context gate to determine how much of the context to include in decoding the output token.
    """

    def __init__(self, hidden_dim):
        """
        hidden_dim : int
            Dimension of the hidden vector passed between the convolutional decoder layers.
        """

        super(ContextGate, self).__init__()
        self.decoder_state_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, Y, C):
        """
        Y : torch Tensor
            Conv1dDecoder output. Shape of (batch size, sequence length, hidden dim).

        C : torch Tensor
            Summarised representation of encoder states (ie, attention transformed context encoder outputs). Shape of (batch size, sequence length, hidden dim).
        """
        # print("inside gate Y", Y.shape)
        # print("inside gate C", C.shape)
        transformed_y = self.decoder_state_proj(Y)
        transformed_c = self.attn_proj(C)

        # print("after gate Y", Y.shape)
        # print("after gate C", C.shape)

        return torch.sigmoid((transformed_y + transformed_c))
