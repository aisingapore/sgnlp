import torch
import torch.nn as nn


class DynamicLSTM(nn.Module):
    """
    A dynamic LSTM class which can hold variable length sequence
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0,
        bidirectional: bool = False,
        only_use_last_hidden_state: bool = False,
        rnn_type: str = "LSTM",
    ) -> None:
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type
        self.__init_rnn()

    def __init_rnn(self) -> None:
        """
        Helper method to initalized RNN type
        """
        input_args = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "bias": self.bias,
            "batch_first": self.batch_first,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
        }
        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(**input_args)
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(**input_args)
        elif self.rnn_type == "RNN":
            self.rnn = nn.RNN(**input_args)

    def forward(self, x: torch.Tensor, x_len: torch.Tensor, h0: torch.Tensor = None) -> torch.Tensor:
        # Sort
        x_sort_idx = torch.argsort(-x_len)
        x_unsort_idx = torch.argsort(x_sort_idx).long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx.long()]

        # Pack
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len.cpu(), batch_first=self.batch_first)

        if self.rnn_type == "LSTM":
            out_pack, (ht, ct) = self.rnn(x_emb_p, None) if h0 is None else self.rnn(x_emb_p, (h0, h0))
        else:
            out_pack, ht = self.rnn(x_emb_p, None) if h0 is None else self.rnn(x_emb_p, h0)
            ct = None

        # Unsort
        # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = torch.transpose(ht, 0, 1)[x_unsort_idx]
        ht = torch.transpose(ht, 0, 1)

        if self.only_use_last_hidden_state:
            return ht
        else:
            # Unpack: out
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0]
            out = out[x_unsort_idx]

            # Unsort: out c
            if self.rnn_type == "LSTM":
                # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
                ct = torch.transpose(ct, 0, 1)[x_unsort_idx]
                ct = torch.transpose(ct, 0, 1)
            return out, (ht, ct)
