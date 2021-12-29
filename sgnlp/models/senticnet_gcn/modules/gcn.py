import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    """
    Simple GCN Layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features: torch.Tensor, out_features: torch.Tensor, bias=True) -> None:
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, text: torch.Tensor, adj: torch.Tensor):
        text = text.to(torch.float32)
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        return output + self.bias if self.bias is not None else output
