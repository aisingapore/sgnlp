from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelClassifier(nn.Module):
    """
    Label Classifier model to be used in RST parser network model.
    """

    def __init__(
        self,
        input_size,
        classifier_hidden_size,
        classes_label=39,
        bias=True,
        dropout=0.5,
    ):
        super(LabelClassifier, self).__init__()
        self.classifier_hidden_size = classifier_hidden_size
        self.labelspace_left = nn.Linear(input_size, classifier_hidden_size, bias=False)
        self.labelspace_right = nn.Linear(
            input_size, classifier_hidden_size, bias=False
        )
        self.weight_left = nn.Linear(classifier_hidden_size, classes_label, bias=False)
        self.weight_right = nn.Linear(classifier_hidden_size, classes_label, bias=False)
        self.nnDropout = nn.Dropout(dropout)

        self.weight_bilateral = nn.Bilinear(
            classifier_hidden_size, classifier_hidden_size, classes_label, bias=bias
        )

    def forward(
        self, input_left: torch.Tensor, input_right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the label classifier.

        Args:
            input_left (torch.Tensor): encoder RNN output
            input_right (torch.Tensor): encoder RNN output

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: return softmax and log softmax tensors of label classifier model.
        """
        labelspace_left = F.elu(self.labelspace_left(input_left))
        labelspace_right = F.elu(self.labelspace_right(input_right))

        union = torch.cat((labelspace_left, labelspace_right), 1)
        union = self.nnDropout(union)
        labelspace_left = union[:, : self.classifier_hidden_size]
        labelspace_right = union[:, self.classifier_hidden_size :]
        output = (
            self.weight_bilateral(labelspace_left, labelspace_right)
            + self.weight_left(labelspace_left)
            + self.weight_right(labelspace_right)
        )

        relation_weights = F.softmax(output, 1)
        log_relation_weights = F.log_softmax(output, 1)

        return relation_weights, log_relation_weights
