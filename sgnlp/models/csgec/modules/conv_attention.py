from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAttention(nn.Module):
    """
    Attention module to compare the encoder outputs with the target word embedding in the decoder.
    """

    def __init__(self, hidden_dim, token_embedding_dim, normalization_constant=0.5):
        """
        hidden_dim : int
            Decoder output dimension size.
        token_embedding_dim : int
            Token embedding dimension size.
        """
        super(ConvAttention, self).__init__()
        self.in_projection = nn.Linear(hidden_dim, token_embedding_dim)
        self.out_projection = nn.Linear(token_embedding_dim, hidden_dim)
        self.normalization_constant = normalization_constant

    def forward(self, Y, T, E, ES, encoder_padding_mask=None):
        """
        Y : torch Tensor
            ConvGLU output of the [BOS] until the (n-1)th tokens. Shape of (batch size, sequence length, hidden dim).
        T : torch Tensor
            Target token embedding of the [BOS] until the (n-1)th tokens. Shape of (batch size, sequence length, token embedding dim).
        E : torch Tensor
            Encoder output of all source/context tokens. Shape of (batch size, sequence length, token embedding dim).
        ES : torch Tensor
            Elementwise sum of the token embeddings and encoder outputs of all source/context tokens. Shape of (batch size, sequence length, token embedding dim).
        """

        Z = (self.in_projection(Y) + T) * sqrt(
            self.normalization_constant
        )  # b x n x embed dim
        E = E.transpose(1, 2)  # b x embed dim x |s|

        x = torch.matmul(
            Z, E
        )  # performs matrix multiplication for the corresponding matrices in Z and E for each batch element

        if encoder_padding_mask is not None:
            x = (
                x.float()
                .masked_fill(encoder_padding_mask.unsqueeze(1), float("-inf"))
                .type_as(x)
            )
        alpha = F.softmax(x, dim=2)  # b x n x |s|
        x = torch.matmul(alpha, ES)
        s = ES.size(1)

        # Scale the atteniton outputs (respecting potentially different lengths) (?)
        if encoder_padding_mask is None:
            x = x * (s * sqrt(1.0 / s))
        else:
            s = s - encoder_padding_mask.type_as(x).sum(
                dim=1, keepdim=True
            )  # exclude padding
            s = s.unsqueeze(-1)
            x = x * (s * s.rsqrt())
        C = self.out_projection(x)
        return C
