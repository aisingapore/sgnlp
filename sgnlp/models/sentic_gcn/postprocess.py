from typing import Dict, List, Union

import torch.nn.functional as F

from .preprocess import SenticGCNData, SenticGCNBertData
from .modeling import SenticGCNModelOutput, SenticGCNBertModelOutput


class SenticGCNBasePostprocessor:
    """
    Base postprocessor class providing common post processing functions.
    """

    def __init__(self, return_full_text: bool = False, return_aspects_text: bool = False) -> None:
        self.return_full_text = return_full_text
        self.return_aspects_text = return_aspects_text

    def __call__(
        self,
        processed_inputs: List[Union[SenticGCNData, SenticGCNBertData]],
        model_outputs: Union[SenticGCNModelOutput, SenticGCNBertModelOutput],
    ) -> List[Dict[str, Union[List[str], List[int], float]]]:
        # Get predictions
        probabilities = F.softmax(model_outputs.logits, dim=-1).detach().numpy()
        predictions = [probabilities.argmax(axis=-1)[idx] - 1 for idx in range(len(probabilities))]
        # Process output
        outputs = []
        for processed_input, prediction in zip(processed_inputs, predictions):
            exists = False
            # Check to see if the full_text_tokens already exists
            # If found, append the aspect_token_index, prediction and optionally aspect texts.
            for idx, proc_output in enumerate(outputs):
                if proc_output["sentence"] == processed_input.full_text_tokens:
                    exists = True
                    outputs[idx]["aspects"].append(processed_input.aspect_token_indexes)
                    outputs[idx]["labels"].append(int(prediction))
                    if self.return_aspects_text:
                        outputs[idx]["aspects_text"].append(processed_input.aspect)
                    break
            if exists:
                continue
            processed_dict = {}
            processed_dict["sentence"] = processed_input.full_text_tokens
            processed_dict["aspects"] = [processed_input.aspect_token_indexes]
            processed_dict["labels"] = [int(prediction)]
            if self.return_full_text:
                processed_dict["full_text"] = processed_input.full_text
            if self.return_aspects_text:
                processed_dict["aspects_text"] = [processed_input.aspect]
            outputs.append(processed_dict)
        return outputs


class SenticGCNPostprocessor(SenticGCNBasePostprocessor):
    """
    Class to initialise the Postprocessor for SenticGCNModel.
    Class to postprocess SenticGCNModel output to get a list of input text tokens,
    aspect token index and prediction labels.

    Args:
        return_full_text (bool): Flag to indicate if the full text should be included in the output.
        return_aspects_text (bool): Flag to indicate if the list of aspects text should be included in the output.
    """

    def __init__(self, return_full_text: bool = False, return_aspects_text: bool = False) -> None:
        super().__init__(return_full_text=return_full_text, return_aspects_text=return_aspects_text)


class SenticGCNBertPostprocessor(SenticGCNBasePostprocessor):
    """
    Class to initialise the Postprocessor for SenticGCNBertModel.
    Class to postprocess SenticGCNBertModel output to get a list of input text tokens,
    aspect token index and prediction labels.

    Args:
        return_full_text (bool): Flag to indicate if the full text should be included in the output.
        return_aspects_text (bool): Flag to indicate if the list of aspects text should be included in the output.
    """

    def __init__(self, return_full_text: bool = False, return_aspects_text: bool = False) -> None:
        super().__init__(return_full_text=return_full_text, return_aspects_text=return_aspects_text)
