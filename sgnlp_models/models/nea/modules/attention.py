from typing import Union

import torch
from torch import nn


class Attention(nn.Module):
    """Class to initialise Attention layer as used in the original research code

    Args:
        op (str, optional): Either attsum or attmean. Defaults to "attsum".
        activation (Union[None, str], optional): If None . Defaults to "tanh".
        init_stdev (float, optional): [description]. Defaults to 0.01.
    """

    def __init__(
        self,
        op: str = "attsum",
        activation: Union[None, str] = "tanh",
        init_stdev: float = 0.01,
    ):
        super().__init__()
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev

    def _initialise_layer(self, input_shape: torch.Tensor) -> None:
        """Initialise attention weights based on shape of input

        Args:
            input_shape (torch.Tensor): Shape of input tensor
        """
        self.att_v = (torch.rand(input_shape[2]) * self.init_stdev).float()
        self.att_W = (
            torch.rand(input_shape[2], input_shape[2]) * self.init_stdev
        ).float()

    def init_mask(self, mask: torch.Tensor) -> None:
        """Initialises the mask tensor from input data, which is needed for
        forward pass method. Mask tensor contains the number of non padding
        tokens for each instance.

        Args:
            mask (torch.Tensor): mask from input data.
        """
        self.mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass computation for attention layer, following the
        implementation in the original code.

        Args:
            x (torch.Tensor): input tensor to the layer

        Returns:
            torch.Tensor: output tensor of attention layer
        """
        self._initialise_layer(x.shape)
        y = torch.matmul(x, self.att_W)

        if not self.activation:
            weights = torch.tensordot(self.att_v, y, [[0], [2]])
        elif self.activation == "tanh":
            weights = torch.tensordot(self.att_v, torch.tanh(y), [[0], [2]])
        weights = torch.nn.functional.softmax(weights, dim=1)

        out = x * weights.unsqueeze(1).repeat(1, x.shape[2], 1).permute(0, 2, 1)

        if self.op == "attsum":
            out = out.sum(axis=1)
        elif self.op == "attmean":
            out = out.sum(axis=1) / self.mask

        return out.float()
