from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from .context_gate import ContextGate
from .conv_attention import ConvAttention
from .conv_glu import ConvGLUDecoder
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
        normalization_constant=0.5,
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
        self.num_conv_layers = num_conv_layers
        self.kernel_size = kernel_size

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
                ConvGLUDecoder(hidden_dim, kernel_size, dropout, self.padding_idx)
                for i in range(num_conv_layers)
            ]
        )

        self.aux_attention = nn.ModuleList(
            [ConvAttention(hidden_dim, embedding_dim) for i in range(num_conv_layers)]
        )

        self.enc_attention = nn.ModuleList(
            [ConvAttention(hidden_dim, embedding_dim) for i in range(num_conv_layers)]
        )

        self.aux_gates = nn.ModuleList(
            [ContextGate(hidden_dim) for i in range(num_conv_layers)]
        )

        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=embedding_dim)

        self.normalization_constant = normalization_constant

        self.fc3 = nn.Linear(in_features=embedding_dim, out_features=num_embeddings)

    def forward(
        self,
        prev_output_tokens,
        encoder_out_dict,
        auxencoder_out_dict,
        incremental_state=None,
    ):
        auxencoder_E = auxencoder_out_dict["encoder_out"][0]
        auxencoder_ES = auxencoder_out_dict["encoder_out"][1]
        auxencoder_padding_mask = auxencoder_out_dict["encoder_padding_mask"]

        if not torch.any(auxencoder_padding_mask):
            auxencoder_padding_mask = None

        encoder_E = encoder_out_dict["encoder_out"][0]
        encoder_ES = encoder_out_dict["encoder_out"][1]
        encoder_padding_mask = encoder_out_dict["encoder_padding_mask"]
        if not torch.any(encoder_padding_mask):
            encoder_padding_mask = None

        pos_embed = self.embed_positions(prev_output_tokens, incremental_state)
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        x = self._embed_tokens(prev_output_tokens, incremental_state)
        x += pos_embed
        target_embedding = x
        # print("target_embedding \n", target_embedding)
        # x = F.dropout(x, p=self.dropout, training=self.training) # need to handle this
        Y = self.fc1(x)
        # print("after fc1 \n", Y)

        for conv, aux_attention, enc_attention, aux_gate in zip(
            self.convolutions, self.aux_attention, self.enc_attention, self.aux_gates
        ):
            # Dropout before the conv layers
            # x = F.dropout(x, p=self.dropout, training=self.training)
            # print("Y", Y.shape)
            residual_Y = Y
            if (
                incremental_state is not None
                and len(incremental_state) >= self.num_conv_layers
            ):
                Y = torch.cat(
                    (incremental_state.get_first_element()[:, 1:, :], Y), dim=1
                )
                incremental_state.add_element(Y)
            else:
                Y = F.pad(
                    Y.transpose(1, 2), (self.kernel_size - Y.shape[1], 0), value=0
                ).transpose(1, 2)
                incremental_state.add_element(Y)

            # print("Y", Y.shape)
            Y = conv(Y)
            # print("Y after conv \n", Y, "\n")
            # print("Y shape after conv \n", Y.shape, "\n")
            acx = aux_attention(
                Y,
                target_embedding,
                auxencoder_E,
                auxencoder_ES,
                auxencoder_padding_mask,
            )

            # print("acx \n", acx, "\n")
            ctx = enc_attention(
                Y,
                target_embedding,
                encoder_E,
                encoder_ES,
                encoder_padding_mask,
            )
            # print("ctx \n", ctx, "\n")

            auxgt = aux_gate(Y, ctx)
            # print("auxgt \n", auxgt, "\n")
            # print("Y before last", Y.shape)

            Y = (Y + ctx) * sqrt(self.normalization_constant)
            Y = (Y + auxgt * acx) * sqrt(self.normalization_constant)
            Y = (Y + residual_Y) * sqrt(self.normalization_constant)
            # print("Y after iteration", Y.shape)
            # print("Y after each layer", Y)

        x = self.fc2(Y)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        # print("after fc3", x)
        return x

    def _embed_tokens(self, tokens, incremental_state):
        if incremental_state is not None:
            # keep only the last token for incremental forward pass
            tokens = tokens[:, -1:]
        return self.embed_tokens(tokens)
