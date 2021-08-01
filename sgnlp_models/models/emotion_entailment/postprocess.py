from typing import List

import numpy as np
from transformers.modeling_outputs import SequenceClassifierOutput


class RecconEmotionEntailmentPostprocessor:
    """Class to postprocess RecconEmotionEntailmentModel output to predicted labels"""

    def __call__(self, raw_pred: SequenceClassifierOutput) -> List[int]:
        """Convert raw prediction (logits) to predicted label.

        Args:
            raw_pred (SequenceClassifierOutput): output of RecconEmotionEntailmentModel

        Returns:
            List[int]: list of predicted label
        """
        raw_pred = raw_pred["logits"].detach().numpy()
        pred = list(np.argmax(raw_pred, axis=1))
        pred = [int(prediction) for prediction in pred]

        return pred
