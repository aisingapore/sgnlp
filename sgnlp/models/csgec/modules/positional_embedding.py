import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Embedding):
    """
    Class for learned positional embeddings. Behaviour is similar to torch.nn.Embedding. Each position index will be assigned its own embedding. Note that this restricts the maximum length of the input sequence length.

    This is not the sinosoidal positional embedding introducted in the Transformer model architecture (Vaswani et al., 2017). These embeddings do not have any functional constraints on how they should behave. The embeddings are learned entirely from scratch.
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx, **kwargs):
        """
        max_seq_len : int
            Maximum number of tokens in the input sequence. This is used to create the total number of position embeddings.
        embedding_dim : int
            Number of dimensions for both the token and position embeddings.
        padding_idx : int
            Index for the padding token in your tokenizer. This index should be smaller than all the other tokens because embeddings with index lower than this are not updated during training.
        """

        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            **kwargs
        )

        # This requires that all other special tokens be indexed before the padding token
        self.max_seq_len = self.num_embeddings - self.padding_idx - 1

    def forward(self, input_ids, incremental_state=None):
        """
        input_ids : torch LongTensor
            LongTensor containing the token indices of a batch of input sequences. Shape of (batch size, sequence length).
        """
        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            position_ids = input_ids.data.new(1, 1).fill_(
                self.padding_idx + input_ids.size(1)
            )
        else:
            position_ids = self.make_positions(input_ids)

        return F.embedding(
            position_ids,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def make_positions(self, input_ids):
        tokens_mask = input_ids.ne(self.padding_idx)
        position_ids = (
            torch.cumsum(tokens_mask, dim=1) * tokens_mask
        ).long() + self.padding_idx
        return position_ids
