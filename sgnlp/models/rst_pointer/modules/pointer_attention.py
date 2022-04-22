from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerAtten(nn.Module):
    """
    Pointer attention model to be used in RST parser network model.
    """

    def __init__(self, atten_model: str, hidden_size: int):
        super(PointerAtten, self).__init__()
        self.atten_model = atten_model
        self.weight1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.weight2 = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self, encoder_outputs: torch.Tensor, curr_decoder_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Pointer Attention model.

        Args:
            encoder_outputs (torch.Tensor): output tensor from encoder RNN model.
            curr_decoder_outputs (torch.Tensor): output tensor from decoder RNN model.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: return softmax and log softmax tensors of pointer attention model.
        """
        if self.atten_model == "Biaffine":
            EW1_temp = self.weight1(encoder_outputs)
            EW1 = torch.matmul(EW1_temp, curr_decoder_outputs).unsqueeze(1)
            EW2 = self.weight2(encoder_outputs)
            bi_affine = EW1 + EW2
            bi_affine = bi_affine.permute(1, 0)

            atten_weights = F.softmax(bi_affine, 0)
            log_atten_weights = F.log_softmax(bi_affine, 0)

        elif self.atten_model == "Dotproduct":
            dot_prod = torch.matmul(encoder_outputs, curr_decoder_outputs).unsqueeze(0)
            atten_weights = F.softmax(dot_prod, 1)
            log_atten_weights = F.log_softmax(dot_prod, 1)

        return atten_weights, log_atten_weights
