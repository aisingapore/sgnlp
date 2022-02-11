import numpy as np
import torch
import torch.nn.functional as F

from sgnlp.models.coupled_hierarchical_transformer.modeling import DualBertModelOutput


class DualBertPostprocessor:
    def __init__(self, rumour_labels=["FR", "TR", "UR"], stance_labels=["PAD", "B-DENY", "B-SUPPORT", "B-QUERY", "B-COMMENT"]):
        self.rumor_labels = rumour_labels
        self.stance_labels = stance_labels

    def __call__(self, model_output: DualBertModelOutput, stance_label_mask):
        rumour_label_idx = np.argmax(model_output.rumour_logits.detach().cpu().numpy())
        rumour_label = self.rumor_labels[rumour_label_idx]

        stance_label_idx = torch.argmax(F.log_softmax(model_output.stance_logits, dim=2), dim=2)
        # stance_label = self.stance_labels[stance_label_idx]

        stance_preds = []
        stance_label_mask = stance_label_mask.to("cpu").numpy()
        for i, mask in enumerate(stance_label_mask):
            temp_2 = []
            for j, m in enumerate(mask):
                if m:
                    temp_2.append(self.stance_labels[stance_label_idx[i][j]])
                else:
                    break
            stance_preds.append(temp_2)


        return {
            "rumour_label": rumour_label,
            "stance_label": stance_preds
        }
