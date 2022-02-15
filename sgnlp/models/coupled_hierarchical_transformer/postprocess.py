import numpy as np
import torch
import torch.nn.functional as F

from sgnlp.models.coupled_hierarchical_transformer.modeling import DualBertModelOutput


class DualBertPostprocessor:
    def __init__(self, rumour_labels=["FR", "TR", "UR"], stance_labels=["PAD", "B-DENY", "B-SUPPORT", "B-QUERY", "B-COMMENT"]):
        self.rumor_labels = rumour_labels
        self.stance_labels = stance_labels

    def __call__(self, model_outputs: [DualBertModelOutput], stance_label_mask):

        rumor_labels = []
        for rumor_logits in model_outputs.rumour_logits:
            rumour_label_idx = np.argmax(rumor_logits.detach().cpu().numpy())
            rumor_labels.append(self.rumor_labels[rumour_label_idx])

        stance_labels = []
        stance_label_idx = torch.argmax(F.log_softmax(model_outputs.stance_logits, dim=2), dim=2)
        stance_label_mask = stance_label_mask.to("cpu").numpy()
        for i, mask in enumerate(stance_label_mask):
            temp_2 = []
            for j, m in enumerate(mask):
                if m:
                    temp_2.append(self.stance_labels[stance_label_idx[i][j]])
                else:
                    break
            stance_labels.append(temp_2)

        return {
            "rumor_labels": rumor_labels,
            "stance_labels": stance_labels
        }
