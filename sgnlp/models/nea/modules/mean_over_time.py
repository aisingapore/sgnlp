import torch
from torch import nn


class MeanOverTime(nn.Module):
    """Class to initialise MeanOverTime layer as used in the original research code"""

    def __init__(self):

        super().__init__()

    def init_mask(self, mask: torch.Tensor) -> None:
        """Initialises the mask tensor from input data, which is needed for
        forward pass method. Mask tensor contains the number of non padding
        tokens for each instance.

        Args:
            mask (torch.Tensor): mask from input data.
        """
        self.mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs mean over time computation on the input tensor

        Args:
            x (torch.Tensor): Input tensor. shape:(batch_size * seq_len * rec_dim)

        Returns:
            torch.Tensor: Mean of tensors across the sequence dimension.
                shape:(batch_size * rec_dim)
        """
        x = x.sum(axis=1) / self.mask
        return x
