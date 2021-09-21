from math import sqrt
import torch.nn as nn
import torch.nn.functional as F

from .conv_glu import ConvGLU
from .positional_embedding import PositionalEmbedding


class ConvDecoder(nn.Module):
    """
    CNN based encoder. Inputs are padded on both sides before passing through a 1D CNN, a GLU activation function, a skip connection, an optional dropout layer and a fully connected linear layer.
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        max_seq_len,
        padding_idx,
        token_dropout,
        hidden_dim,
        kernel_size,
        dropout,
        num_conv_layers,
    ):
        """
        input_dim : int
            Encoder input (and output) embedding dimension size.
        kernel_size : int
            Kernel size / patch size. Number of tokens for each convolution.
        dropout : float
            Probability of setting each embedding dimension to 0 during training.
        """

        super(ConvDecoder, self).__init__()

        self.embed_tokens = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.embed_positions = PositionalEmbedding(
            num_embeddings=max_seq_len,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.dropout = dropout
        self.token_dropout = token_dropout
        self.padding_idx = padding_idx

        self.fc1 = nn.Linear(in_features=embedding_dim, out_features=hidden_dim)

        self.convolutions = nn.ModuleList(
            [
                ConvGLU(
                    hidden_dim,
                    kernel_size,
                    dropout,
                )
                for i in range(num_conv_layers)
            ]
        )

        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=embedding_dim)

    def forward(self, prev_output_tokens, incremental_state=None):
        """
        prev_output_tokens : torch LongTensor
            Indices of the previous tokens. Size of (batch_size, seq_length)
        incremental_state :
        """
        pos_embed = self.embed_positions(prev_output_tokens, incremental_state)
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        x = self.embed_tokens(prev_output_tokens, incremental_state)
        x += pos_embed

        target_embedding = x

        # x = F.dropout(x, p=self.dropout, training=self.training) # need to handle this
        x = self.fc1(x)

        for conv in self.convolutions:
            # Mask the padding tokens
            encoder_padding_mask = src_tokens.eq(self.padding_idx).unsqueeze(-1)
            x = x.masked_fill(encoder_padding_mask, 0)

            # Dropout before the conv layers
            # x = F.dropout(x, p=self.dropout, training=self.training)

            x = conv(x)

        x = self.fc2(x)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).unsqueeze(-1)
        x = x.masked_fill(encoder_padding_mask, 0)

        y = (x + input_embedding) * sqrt(0.5)

        return {
            "encoder_out": (x, y),
            "encoder_padding_mask": encoder_padding_mask,  # B x T
        }
