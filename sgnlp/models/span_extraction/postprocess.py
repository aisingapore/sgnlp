from typing import List, Dict, Tuple, Union

from transformers.data.processors.squad import SquadFeatures, SquadExample
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from .utils import (
    RawResult,
    get_best_predictions,
)


class RecconSpanExtractionPostprocessor:
    """Class to initialise the Postprocessor for RecconSpanExtraction model.
    Class to postprocess RecconSpanExtractionModel raw output to get the causal span
    and probabilities

    Args:
        threshold (float, optional): probability threshold value to extract causal span.
                                        Defaults to 0.7.
    """

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold

    def __call__(
        self,
        raw_pred: QuestionAnsweringModelOutput,
        evidences: List[Dict[str, Union[int, str]]],
        examples: List[SquadExample],
        features: List[SquadFeatures],
    ) -> Tuple[List[List[str]], List[List[int]], List[List[Union[int, float]]]]:
        """Convert raw prediction (logits) to:
        1. list of list of spans
        2. List of list of integer to indicate if corresponding span is causal
        3. List of list of float to indicate probability of corresponding
        span being causal

        Args:
            raw_pred (QuestionAnsweringModelOutput): output of RecconSpanExtractionModel
            evidences (List[Dict[str, Union[int, str]]]): List of evidence utterances
            examples (List[SquadExample]): List of SquadExample instance - output of RecconSpanExtractionPreprocessor
            features ( List[SquadFeatures]): List of SquadFeatures instance - output of RecconSpanExtractionPreprocessor

        Returns:
            Tuple[List[List[str]], List[List[int]], List[List[Union[int, float]]]]: 1. List of list of spans
                                                                                    2. List of list of integer to
                                                                                        indicate if corresponding span is causal
                                                                                    3. List of list of int/float to indicate
                                                                                        probability of corresponding span being causal.
                                                                                        -1 indicates span is not causal
        """
        all_results = self._process_raw_pred(raw_pred)
        answers = get_best_predictions(
            all_examples=examples,
            all_features=features,
            all_results=all_results,
            n_best_size=20,
            max_answer_length=200,
            do_lower_case=False,
            verbose_logging=False,
            version_2_with_negative=True,
            null_score_diff_threshold=False,
        )
        context, evidence_span, probability = self._process_answers(answers, evidences)

        return context, evidence_span, probability

    def _process_raw_pred(
        self, raw_pred: QuestionAnsweringModelOutput
    ) -> List[RawResult]:
        """Process raw output into a list of RawResult which can be used with
        get_best_prediction()

        Args:
            raw_pred (QuestionAnsweringModelOutput): output of RecconSpanExtractionModel

        Returns:
            List[RawResult]: list of RawResult which can be used with get_best_predictions()
        """
        all_results = []
        for i in range(len(raw_pred["start_logits"])):
            result = RawResult(
                unique_id=i + 1000000000,
                start_logits=raw_pred["start_logits"][i].tolist(),
                end_logits=raw_pred["end_logits"][i].tolist(),
            )
            all_results.append(result)
        return all_results

    def _process_answers(
        self, answers: List[Dict[str, any]], evidences: List[Dict[str, Union[int, str]]]
    ) -> Tuple[List[List[str]], List[List[int]], List[List[float]]]:
        """Post process prediction generated from get_best_predictions()

        Args:
            answers (List[Dict[str, any]]): List of predictions from get_best_predictions()
            evidences (List[Dict[str, Union[int, str]]]): List of evidence utterance

        Returns:
            Tuple[List[List[str]], List[List[int]], List[List[float]]]: Return processed lists of
                        the full context, evidence_span and probabilities score.
        """
        context = []
        evidence_span = []
        probability = []

        for ans, evid in zip(answers, evidences):
            span = ans["answer"][0]
            prob = ans["probability"][0]
            lower_than_threshold = prob < self.threshold
            span_equals_evidence = evid["evidence"] == span
            if not span or lower_than_threshold:
                ctx = [evid["evidence"]]
                span = [0]
                prob = [-1]
            elif not span_equals_evidence:
                ctx, span, prob = self._process_span(evid["evidence"], span, prob)
            else:
                ctx = [evid["evidence"]]
                span = [1]
                prob = [prob]
            context.append(ctx)
            evidence_span.append(span)
            probability.append(prob)
        return context, evidence_span, probability

    def _process_span(
        self, evidence: str, span: str, prob: float
    ) -> Tuple[List[str], List[str], List[float]]:
        """Helper function to split evidence string if evidence_span is a sub-string of the evidence string.

        Args:
            evidence (str): the full evidence string
            span (str): evidence span which is a substring of the full evidence string
            prob (float): probability score for the evidence span

        Returns:
            Tuple[List[str], List[str], List[float]]: lists containing the splitted evidence spans and probability
        """
        evidence_split = evidence.split(span)
        if evidence.startswith(span):
            ctx_list = [span, evidence_split[1]]
            evidence_span = [1, 0]
            prob_list = [prob, -1]
        elif evidence.endswith(span):
            ctx_list = [evidence_split[0], span]
            evidence_span = [0, 1]
            prob_list = [-1, prob]
        else:
            ctx_list = [evidence_split[0], span, evidence_split[1]]
            evidence_span = [0, 1, 0]
            prob_list = [-1, prob, -1]
        return ctx_list, evidence_span, prob_list
