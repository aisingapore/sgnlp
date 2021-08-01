import numpy as np
import torch
import torch.nn as nn
from allennlp.nn.util import masked_softmax


class CharCNNEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_filters, kernel_size, padding_idx=0, stride=1, dropout=0.3):
        super(CharCNNEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                                      padding_idx=padding_idx)
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters,
                                kernel_size=kernel_size, stride=stride)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Assumes x is of shape (batch_size, max_num_words, max_num_chars).
        """

        input_shape = x.shape
        flattened_shape = [-1] + list(input_shape[2:])
        x = x.contiguous().view(*flattened_shape)  # (batch_size * max_num_words, max_num_chars)

        x = self.embedding(x)  # (batch_size * max_num_words, max_num_chars, embedding_dim)

        # transpose to correct shape that conv1d expects
        x = torch.transpose(x, 1, 2)  # (batch_size * max_num_words, embedding_dim, max_num_chars)

        x = self.conv1d(x)  # (batch_size * max_num_words, num_filters, num_strides)

        # Max pool over last dimension
        # Equivalent to nn.functional.max_pool1d(x, kernel_size=num_strides).squeeze()
        x = x.max(dim=2).values  # (batch_size * max_num_words, num_filters)

        output_shape = list(input_shape[:2]) + [-1]
        x = x.contiguous().view(*output_shape)  # (batch_size, max_num_words, num_filters)

        x = self.activation(x)
        x = self.dropout(x)

        return x


class WordEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=0, trainable=False):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                                      padding_idx=padding_idx)
        self.embedding.weight.requires_grad = trainable

    def forward(self, x):
        return self.embedding(x)

    def load_pretrained_embeddings(self, file_path, vocab):
        assert len(vocab) == self.embedding.num_embeddings

        # Process glove embeddings: File format is word followed by weights (space-separated)
        word_to_embedding = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                try:
                    values = line.split()
                    word = values[0]
                    word_embedding = np.asarray(values[1:], dtype=np.float32)
                    word_to_embedding[word] = word_embedding
                except ValueError as e:
                    # Some words have spaces in them, leading to an error in the above logic
                    # Skip these words
                    pass

        # Take arbitrary word
        emb_dim = word_to_embedding["a"].shape[0]
        weights_matrix = np.zeros((len(vocab), emb_dim))

        for word, idx in vocab.stoi.items():
            try:
                weights_matrix[idx] = word_to_embedding[word]
            except KeyError:
                weights_matrix[idx] = np.random.normal(scale=0.6, size=(emb_dim,))

        self.embedding.load_state_dict({'weight': torch.Tensor(weights_matrix)})


class Linear(torch.nn.Module):
    def __init__(self, dropout, input_dim, hidden_dims, num_layers=None, activation=None):
        super(Linear, self).__init__()
        if num_layers:
            assert isinstance(num_layers, int)
            if num_layers > 1:
                assert len(hidden_dims) == num_layers
            elif num_layers == 1:
                assert isinstance(hidden_dims, int)

        self.layers = []
        if isinstance(hidden_dims, list):
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                self.layers.append(nn.Linear(in_features=prev_dim, out_features=hidden_dim))
        elif isinstance(hidden_dims, int) and num_layers == 1:
            self.layers.append(nn.Linear(in_features=input_dim, out_features=hidden_dims))
        else:
            raise ValueError
        self.layers = nn.ModuleList(self.layers)

        self.dropout = nn.Dropout(dropout)
        if activation == "relu":
            self.activation = torch.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            self.activation = None

    def forward(self, x):
        for linear in self.layers:
            x = linear(x)
            if self.activation:
                x = self.activation(x)
            x = self.dropout(x)
        return x


class SeqAttnMat(torch.nn.Module):
    """
    Given sequences X and Y, calculate the attention matrix.
    """

    def __init__(self, projector_args: dict = None,
                 identity: bool = True) -> None:
        super(SeqAttnMat, self).__init__()
        if not identity:
            assert projector_args is not None
            self.linear = Linear(**projector_args)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2
        Output:
            scores: batch * len1 * len2
            alpha: batch * len1 * len2
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.contiguous().view(-1, x.size(2))).view(x.size())
            y_proj = self.linear(y.contiguous().view(-1, y.size(2))).view(y.size())
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))  # batch * len1 * len2

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())  # b * l1 * l2
        alpha = masked_softmax(scores, y_mask, dim=-1).view(-1, x.size(1), y.size(1))

        return scores, alpha


class GatedEncoding(torch.nn.Module):
    """
    Gating over a sequence:
    * o_i = sigmoid(Wx_i) * x_i for x_i in X.
    """

    def __init__(self, gate_args: dict):
        super(GatedEncoding, self).__init__()
        self.linear = Linear(**gate_args)

    def forward(self, x):
        """
        Args:
            x: batch * len * hdim
        Output:
            gated_x: batch * len * hdim
        """
        gate = self.linear(x.view(-1, x.size(2))).view(x.size())
        gate = torch.sigmoid(gate)
        gated_x = torch.mul(gate, x)
        return gated_x


class GatedMultifactorSelfAttnEnc(torch.nn.Module):
    """
    Gated multi-factor self attentive encoding over a sequence:

    """

    def __init__(self, projector_args: dict,
                 gate_args: dict,
                 num_factor: int = 4,
                 attn_pooling: str = 'max'):
        super(GatedMultifactorSelfAttnEnc, self).__init__()
        self.num_factor = num_factor
        if self.num_factor > 0:
            self.linear = Linear(**projector_args)
        else:
            self.linear = None
        self.linear_gate = Linear(**gate_args)
        self.attn_pooling = attn_pooling

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * hdim
            x_mask: batch * len
        Output:
            gated_multi_attentive_enc: batch * len * 2hdim
        """
        x = x.contiguous()
        if self.linear is not None:
            self_attn_multi = []
            y_multi = self.linear(x.view(-1, x.size(2)))
            y_multi = y_multi.view(x.size(0), x.size(1), x.size(2), self.num_factor)
            for fac in range(self.num_factor):
                y = y_multi.narrow(3, fac, 1).squeeze(-1)
                attn_fac = y.bmm(y.transpose(2, 1))
                attn_fac = attn_fac.unsqueeze(-1)
                self_attn_multi.append(attn_fac)
            self_attn_multi = torch.cat(self_attn_multi, -1)  # batch * len * len *  num_factor

            if self.attn_pooling == 'max':
                self_attn, _ = torch.max(self_attn_multi, 3)  # batch * len * len
            elif self.attn_pooling == 'min':
                self_attn, _ = torch.min(self_attn_multi, 3)
            else:
                self_attn = torch.mean(self_attn_multi, 3)
        else:
            self_attn = x.bmm(x.transpose(2, 1))  # batch * len * len

        mask = x_mask.reshape(x_mask.size(0), x_mask.size(1), 1) \
               * x_mask.reshape(x_mask.size(0), 1, x_mask.size(1))  # batch * len * len

        self_mask = torch.eye(x_mask.size(1), x_mask.size(1), device=x_mask.device)
        self_mask = self_mask.reshape(1, x_mask.size(1), x_mask.size(1))
        mask = mask * (1 - self_mask.long())

        # Normalize with softmax
        alpha = masked_softmax(self_attn, mask, dim=-1)  # batch * len * len

        # multifactor attentive enc
        multi_attn_enc = alpha.bmm(x)  # batch * len * hdim

        # merge with original x
        gate_input = [x, multi_attn_enc]
        joint_ctx_input = torch.cat(gate_input, 2)

        # gating
        gate_joint_ctx_self_match = self.linear_gate(joint_ctx_input.view(-1, joint_ctx_input.size(2))).view(
            joint_ctx_input.size())
        gate_joint_ctx_self_match = torch.sigmoid(gate_joint_ctx_self_match)

        gated_multi_attentive_enc = torch.mul(gate_joint_ctx_self_match, joint_ctx_input)

        return gated_multi_attentive_enc
