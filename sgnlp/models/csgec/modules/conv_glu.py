from numpy import sqrt
import torch.nn as nn
import torch.nn.functional as F


class ConvGLU(nn.Module):
    """
    CNN based encoder. Inputs are padded on both sides before passing through a 1D CNN, a GLU activation function, a skip connection, an optional dropout layer and a fully connected linear layer.
    """

    def __init__(self, input_dim, kernel_size, dropout):
        """
        input_dim : int
            Encoder input (and output) embedding dimension size.
        kernel_size : int
            Kernel size / patch size. Number of tokens for each convolution.
        dropout : float
            Probability of setting each embedding dimension to 0 during training.
        """

        super(ConvGLU, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim * 2,  # note that this is multiplied by 2 for the GLU
            kernel_size=kernel_size,
            padding=int((kernel_size - 1) / 2),
        )
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, H):
        """
        H : torch Tensor
            Output from the previous encoder layer. Shape of (batch size, sequence length, hidden dim / number of "channels").
        """

        residual_H = H
        H = H.transpose(1, 2)
        H = self.conv(H)
        H = H.transpose(1, 2)
        H = F.glu(H)
        H = (H + residual_H) * sqrt(0.5)

        return H


class ConvGLUDecoder(nn.Module):
    """
    CNN based encoder. Inputs are padded on both sides before passing through a 1D CNN, a GLU activation function, a skip connection, an optional dropout layer and a fully connected linear layer.
    """

    def __init__(self, input_dim, kernel_size, dropout, padding_idx):
        """
        input_dim : int
            Encoder input (and output) embedding dimension size.
        kernel_size : int
            Kernel size / patch size. Number of tokens for each convolution.
        dropout : float
            Probability of setting each embedding dimension to 0 during training.
        """

        super(ConvGLUDecoder, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim * 2,  # note that this is multiplied by 2 for the GLU
            kernel_size=kernel_size,
            padding=0,
        )
        self.padding_idx = padding_idx
        self.kernel_size = kernel_size

    def forward(self, H):
        """
        H : torch Tensor
            Output from the previous encoder layer. Shape of (batch size, sequence length, hidden dim / number of "channels").
        """
        # print("H", H.shape)
        H = H.transpose(1, 2)
        H = F.pad(
            H, (self.kernel_size - H.shape[2], 0), value=0
        )  # TODO Check the padding idx
        H = self.conv(H)
        H = H.transpose(1, 2)
        H = F.glu(H)

        return H
