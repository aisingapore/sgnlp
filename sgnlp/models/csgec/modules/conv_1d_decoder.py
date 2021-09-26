import torch.nn as nn
import torch.nn.functional as F


class Conv1dDecoder(nn.Module):
    """
    CNN based decoder. Inputs are padded on the left side before passing through a 1D CNN, a GLU activation function, an optional dropout layer and a fully connected linear layer.
    """
    def __init__(self, input_dim, kernel_size, dropout, output_dim):
        """
        input_dim : int
            Decoder input embedding dimension size.
        kernel_size : int
            Kernel size / patch size. Number of tokens for each convolution.
        dropout : float
            Probability of setting each embedding dimension to 0 during training.
        output_dim : int
            Decoder output embedding dimension size.
        """
        super(Conv1dDecoder, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim*2, # note that this is multiplied by 2 for the GLU
            kernel_size=kernel_size
        )
        self.dropout = nn.Dropout2d(dropout)
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, G):
        """
        G : torch Tensor
            Output from the previous decoder layer. Shape of (batch size, sequence length, hidden dim).
        """
        G = G.transpose(1,2)
        G = F.pad(G, (self.kernel_size-1, 0))
        G = self.conv(G)
        G = G.transpose(1,2)
        Y = F.glu(G)
        Y = Y.transpose(1,2)
        Y = self.dropout(Y)
        Y = Y.transpose(1,2)
        return Y