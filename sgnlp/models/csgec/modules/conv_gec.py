from numpy import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .context_gate import ContextGate
from .conv_1d_decoder import Conv1dDecoder
from .conv_1d_encoder import Conv1dEncoder
from .conv_attention import ConvAttention
from .combined_embedding import CombinedEmbedding


class ConvGEC(nn.Module):
    """
    Main model class for the cross sentence grammatical error correction model. Implementation is based on the paper by (Chollampatt et al., 2019).
    """

    def __init__(self, config):
        """
        config
            Contains model parameters and configuration
        """

        super(ConvGEC, self).__init__()

        # souce embeddings and transformation
        self.source_embedding = CombinedEmbedding(
            config.source_vocab_size,
            config.src_max_seq_len,
            config.token_embedding_dim,
            config.padding_idx,
            config.dropout,
        )
        self.linear1 = nn.Linear(config.token_embedding_dim, config.hidden_dim)

        # source encoders
        self.encoders = nn.ModuleList(
            [
                Conv1dEncoder(
                    config.hidden_dim,
                    config.hidden_dim,
                    config.kernel_size,
                    config.dropout,
                )
                for i in range(config.num_encoders)
            ]
        )
        self.linear2 = nn.Linear(config.hidden_dim, config.token_embedding_dim)

        # target embeddings and transformation
        self.target_embedding = CombinedEmbedding(
            config.target_vocab_size,
            config.trg_max_seq_len,
            config.token_embedding_dim,
            config.padding_idx,
            config.dropout,
        )
        self.linear3 = nn.Linear(config.token_embedding_dim, config.hidden_dim)

        # decoders
        self.decoders = nn.ModuleList(
            [
                Conv1dDecoder(
                    config.hidden_dim,
                    config.kernel_size,
                    config.dropout,
                    config.token_embedding_dim,
                )
                for i in range(config.num_decoders)
            ]
        )

        # source attention layers
        self.attention_layers = nn.ModuleList(
            [
                ConvAttention(config.hidden_dim, config.token_embedding_dim)
                for i in range(config.num_decoders)
            ]
        )

        # check if context encoders needed
        self.encode_context = config.num_ctx_encoders > 0
        if self.encode_context:
            # context embeddings and transformation
            self.context_embedding = CombinedEmbedding(
                config.source_vocab_size,
                config.ctx_max_seq_len,
                config.token_embedding_dim,
                config.padding_idx,
                config.dropout,
            )
            self.linear_ctx_in = nn.Linear(
                config.token_embedding_dim, config.hidden_dim
            )

            # context encoders
            self.ctx_encoders = nn.ModuleList(
                [
                    Conv1dEncoder(
                        config.hidden_dim,
                        config.hidden_dim,
                        config.kernel_size,
                        config.dropout,
                    )
                    for i in range(config.num_ctx_encoders)
                ]
            )
            self.linear_ctx_out = nn.Linear(
                config.hidden_dim, config.token_embedding_dim
            )

            # context attention & gates
            self.ctx_attention_layers = nn.ModuleList(
                [
                    ConvAttention(config.hidden_dim, config.token_embedding_dim)
                    for i in range(config.num_decoders)
                ]
            )
            self.context_gates = nn.ModuleList(
                [ContextGate(config.hidden_dim) for i in range(config.num_decoders)]
            )

        # mapping the decoder output to the target vocab dim
        self.decoder_output_dropout = nn.Dropout(config.dropout)
        self.linear4 = nn.Linear(config.hidden_dim, config.token_embedding_dim)
        self.linear5 = nn.Linear(config.token_embedding_dim, config.target_vocab_size)

    def load_pretrained_embedding(
        self, pretrained_source_embedding_path, pretrained_target_embedding_path
    ):

        self.source_embedding.load_pretrained_embedding(
            pretrained_source_embedding_path
        )
        self.context_embedding.load_pretrained_embedding(
            pretrained_source_embedding_path
        )
        self.target_embedding.load_pretrained_embedding(
            pretrained_target_embedding_path
        )

    def forward(self, source_ids, context_ids, input_target_ids):
        """
        source_ids : torch LongTensor
            LongTensor containing the token indices of a batch of source sequences. Shape of (batch size, source sequence length).
        context_ids : torch LongTensor
            LongTensor containing the token indices of a batch of context sequences. Shape of (batch size, source sequence length).
        input_target_ids : torch LongTensor
            LongTensor containing the token indices of a batch of input target sequences. This should contain the token indices of the [BOS] token and 1st to (n-1)th tokens that have been previously predicted. Shape of (batch size, source sequence length).
        """
        S = self.source_embedding(source_ids)
        H = self.linear1(S)
        for encoder in self.encoders:
            H = encoder(H)
        E = self.linear2(H)
        ES = (S + E) * sqrt(0.5)

        if self.encode_context:
            S_ctx = self.context_embedding(context_ids)
            H_ctx = self.linear_ctx_in(S_ctx)
            for ctx_encoder in self.ctx_encoders:
                H_ctx = ctx_encoder(H_ctx)
            E_ctx = self.linear_ctx_out(H_ctx)
            ES_ctx = (S_ctx + E_ctx) * sqrt(0.5)

        T = self.target_embedding(input_target_ids)
        G = self.linear3(T)

        for decoder, attention_layer, ctx_attention_layer, context_gate in zip(
            self.decoders,
            self.attention_layers,
            self.ctx_attention_layers,
            self.context_gates,
        ):
            Y = decoder(G)
            C = attention_layer(Y, T, E, ES)
            YC = (Y + C) * sqrt(0.5)
            if self.encode_context:
                C_ctx = ctx_attention_layer(Y, T, E_ctx, ES_ctx)
                L = context_gate(Y, C)
                YC = (YC + torch.mul(L, C_ctx)) * sqrt(0.5)
            G = (YC + G) * sqrt(0.5)

        D = self.linear4(G)
        D = D.transpose(1, 2)
        D = self.decoder_output_dropout(D)
        D = D.transpose(1, 2)

        O = self.linear5(D)

        return O
