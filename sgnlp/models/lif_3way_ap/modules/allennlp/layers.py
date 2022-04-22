import logging
from typing import *

import torch
from allennlp.common.from_params import FromParams
from allennlp.modules import FeedForward
from allennlp.nn.util import masked_softmax

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SeqAttnMat(torch.nn.Module, FromParams):
    """
    Given sequences X and Y, calculate the attention matrix.
    """

    def __init__(
        self, projector: Optional[FeedForward] = None, identity: bool = True
    ) -> None:
        super(SeqAttnMat, self).__init__()
        if not identity:
            assert projector is not None
            self.linear = projector  # put relu activation
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
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))  # batch * len1 * len2

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())  # b * l1 * l2
        alpha = masked_softmax(scores, y_mask, dim=-1).view(-1, x.size(1), y.size(1))
        # scores = scores * y_mask.float()

        return scores, alpha


class GatedEncoding(torch.nn.Module, FromParams):
    """
    Gating over a sequence:
    * o_i = sigmoid(Wx_i) * x_i for x_i in X.
    """

    def __init__(self, gate: FeedForward):
        super(GatedEncoding, self).__init__()
        self.linear = gate  # put linear activation # nn.Linear(input_size, input_size)

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


class GatedMultifactorSelfAttnEnc(torch.nn.Module, FromParams):
    """
    Gated multi-factor self attentive encoding over a sequence:

    """

    def __init__(
        self,
        projector: Optional[FeedForward] = None,
        gate: Optional[FeedForward] = None,
        num_factor: int = 4,
        attn_pooling: str = "max",
    ):
        super(GatedMultifactorSelfAttnEnc, self).__init__()
        self.num_factor = num_factor
        if self.num_factor > 0:
            self.linear = projector
        else:
            self.linear = None
        assert gate is not None
        self.linear_gate = gate
        self.attn_pooling = attn_pooling

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * hdim
            x_mask: batch * len
        Output:
            gated_multi_attentive_enc: batch * len * 2hdim
        """
        if self.linear is not None:
            self_attn_multi = []
            y_multi = self.linear(x.view(-1, x.size(2))).view(
                x.size(0), x.size(1), x.size(2) * self.num_factor
            )
            # y_multi = torch.relu(y_multi)
            y_multi = y_multi.view(x.size(0), x.size(1), x.size(2), self.num_factor)
            for fac in range(self.num_factor):
                y = y_multi.narrow(3, fac, 1).squeeze(-1)
                attn_fac = y.bmm(y.transpose(2, 1))
                attn_fac = attn_fac.unsqueeze(-1)
                self_attn_multi.append(attn_fac)
            self_attn_multi = torch.cat(
                self_attn_multi, -1
            )  # batch * len * len *  num_factor

            if self.attn_pooling == "max":
                self_attn, _ = torch.max(self_attn_multi, 3)  # batch * len * len
            elif self.attn_pooling == "min":
                self_attn, _ = torch.min(self_attn_multi, 3)
            else:
                self_attn = torch.mean(self_attn_multi, 3)
        else:
            self_attn = x.bmm(x.transpose(2, 1))  # batch * len * len

        mask = x_mask.reshape(x_mask.size(0), x_mask.size(1), 1) * x_mask.reshape(
            x_mask.size(0), 1, x_mask.size(1)
        )  # batch * len * len

        self_mask = torch.eye(x_mask.size(1), x_mask.size(1), device=x_mask.device)
        self_mask = self_mask.reshape(1, x_mask.size(1), x_mask.size(1))
        mask = mask * (1 - self_mask.long())

        # Normalize with softmax
        alpha = masked_softmax(self_attn, mask, dim=-1)  # batch * len * len

        # multifactor attentive enc
        multi_attn_enc = alpha.bmm(x)  # batch * len * hdim

        # merge with original x
        gate_input = [x]
        gate_input.append(multi_attn_enc)
        joint_ctx_input = torch.cat(gate_input, 2)

        # gating
        gate_joint_ctx_self_match = self.linear_gate(
            joint_ctx_input.view(-1, joint_ctx_input.size(2))
        ).view(joint_ctx_input.size())
        gate_joint_ctx_self_match = torch.sigmoid(gate_joint_ctx_self_match)

        gated_multi_attentive_enc = torch.mul(
            gate_joint_ctx_self_match, joint_ctx_input
        )

        return gated_multi_attentive_enc


class AttnPooling(torch.nn.Module, FromParams):
    """
    Attentive Pooling/aggregate a sequence based on learned attention scores

    """

    def __init__(
        self, projector: FeedForward, intermediate_projector: FeedForward = None
    ) -> None:
        super(AttnPooling, self).__init__()
        self._projector = projector
        self._int_proj = intermediate_projector

    def forward(
        self, xinit: torch.FloatTensor, xmask: torch.Tensor
    ) -> torch.FloatTensor:
        """
        Args:
        :param xinit: B * T * H
        :param xmask: B * T
        :return: B * H
        """
        if self._int_proj is not None:
            x = self._int_proj(xinit)
            x = x * xmask.unsqueeze(-1)
        else:
            x = xinit
        attn = self._projector(x)  # B * T * 1
        attn = attn.squeeze(-1)  # B * T
        attn = masked_softmax(attn, xmask, dim=-1)
        pooled = attn.unsqueeze(1).bmm(xinit).squeeze(1)  # B * H
        return pooled
