import math
import json
import pathlib
import argparse
import collections
from functools import partial
from typing import List, Dict, Tuple


import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch.functional import Tensor
from transformers.models.bert.tokenization_bert import BasicTokenizer
from transformers.data.processors.squad import (
    SquadExample,
    SquadFeatures,
    squad_convert_example_to_features,
    squad_convert_example_to_features_init,
)

from .tokenization import RecconSpanExtractionTokenizer
from .evaluate_squad import compute_f1
from .data_class import RecconSpanExtractionArguments


def parse_args_and_load_config(
    config_path: str = "config/span_extraction_config.json",
) -> RecconSpanExtractionArguments:
    """Get config from config file using argparser

    Returns:
        RecconSpanExtractionArguments: RecconSpanExtractionArguments instance
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=config_path)
    args = parser.parse_args()

    cfg_path = pathlib.Path(__file__).parent / args.config
    with open(cfg_path, "r") as cfg_file:
        cfg = json.load(cfg_file)

    span_extraction_args = RecconSpanExtractionArguments(**cfg)
    return span_extraction_args


def get_all_evidence_utterance_from_conversation(
    emotion: str, conversation_history: List[str]
) -> Dict[str, List[str]]:
    """Iterate through a conversation history to let each utterance be the evidence
    utterance. The last utterance is treated as the target utterance. Ouput dictionary is
    in a format which can be used with RecconSpanExtractionPreprocessor

    Args:
        emotion (str): Emotion of the target utterance
        conversation_history (List[str]): List of utterance in a conversation. The
                                        last utterance is used as the target utterance.

    Returns:
        Dict[str, List[str]]: Dictionary in a format that can be used with RecconSpanExtractionPreprocessor
            The dictionary looks like this:
            {'emotion': ['happiness'],
            'target_utterance': ['......'],
            'evidence_utterance': ['......'],
            'conversation_history': ['......']}
    """
    conversation_history_text = " ".join(conversation_history)
    target_utterance = conversation_history[-1]

    output = {
        "emotion": [],
        "target_utterance": [],
        "evidence_utterance": [],
        "conversation_history": [],
    }

    for evidence_utterance in conversation_history:
        output["emotion"].append(emotion)
        output["target_utterance"].append(target_utterance)
        output["evidence_utterance"].append(evidence_utterance)
        output["conversation_history"].append(conversation_history_text)

    return output


class RecconSpanExtractionData(torch.utils.data.Dataset):
    """Class to create torch Dataset instance, which is the required data type
    for Transformer's Trainer

    Args:
        dataset (TensorDataset): TensorDataset object
        for_predict (bool, optional): Option to set for predict. Defaults to False.
    """

    def __init__(self, dataset: TensorDataset, for_predict: bool = False) -> None:
        self.dataset = dataset
        self.for_predict = for_predict

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a dictionary of the selected instance for each batch

        Args:
            idx (int): idx to select instances for each batch

        Returns:
            (Dict): dictionary containing input_ids, attention_mask and token_type_ids
                    of the selected instance
        """
        item = {}
        if self.for_predict:
            (input_ids, attention_mask, token_type_ids, _, _, _) = self.dataset[idx]
            item["input_ids"] = input_ids
            item["attention_mask"] = attention_mask
            item["token_type_ids"] = token_type_ids
        else:
            (
                input_ids,
                attention_mask,
                token_type_ids,
                start_positions,
                end_positions,
                _,
                _,
                _,
            ) = self.dataset[idx]
            item["input_ids"] = input_ids
            item["attention_mask"] = attention_mask
            item["token_type_ids"] = token_type_ids
            item["start_positions"] = start_positions
            item["end_positions"] = end_positions

        return item

    def __len__(self) -> int:
        """Returns length of dataset

        Returns:
            int: length of the dataset attribute
        """
        return len(self.dataset)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        unique_id,
        example_index,
        doc_span_index,
        tokens,
        token_to_orig_map,
        token_is_max_context,
        input_ids,
        input_mask,
        segment_ids,
        cls_index,
        p_mask,
        paragraph_len,
        start_position=None,
        end_position=None,
        is_impossible=None,
    ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def truncate_to_max_length(
    feature: List[SquadFeatures], max_length: int
) -> List[SquadFeatures]:
    """Truncate length of SquadFeatures' attributes

    Args:
        feature (List[SquadFeatures]): list of SquadFeatures
        max_length (int): set maximum length of tokens

    Returns:
        List[SquadFeatures]: list of truncated SquadFeatures
    """
    feature[0].input_ids = feature[0].input_ids[:max_length]
    feature[0].p_mask = feature[0].p_mask[:max_length]
    feature[0].token_type_ids = feature[0].token_type_ids[:max_length]
    feature[0].tokens = feature[0].tokens[:max_length]
    feature[0].attention_mask = feature[0].attention_mask[:max_length]
    return feature


def squad_convert_examples_to_features(
    examples: List[SquadExample],
    tokenizer: RecconSpanExtractionTokenizer,
    max_seq_length: int,
    doc_stride: int,
    max_query_length: int,
    is_training: bool,
    padding_strategy: str = "max_length",
    tqdm_enabled: bool = True,
) -> Tuple[List[SquadFeatures], TensorDataset]:
    """[summary]

    Args:
        examples (List[SquadExample]): list of SquadExample
        tokenizer (RecconSpanExtractionTokenizer): RecconSpanExtractionTokenizer from sgnlp
        max_seq_length (int): set max_seq_length
        doc_stride (int): set doc_stride
        max_query_length (int): set max_query_length
        is_training (bool): set is_training
        padding_strategy (str, optional): set padding_strategy. Defaults to "max_length".
        tqdm_enabled (bool, optional): set tqdm_enabled. Defaults to True.

    Returns:
        Tuple[List[SquadFeatures], TensorDataset]: Contains list of SquadFeatures and TensorDataset
    """
    features = []
    squad_convert_example_to_features_init(tokenizer)
    annotate_ = partial(
        squad_convert_example_to_features,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        padding_strategy=padding_strategy,
        is_training=is_training,
    )
    features = [
        truncate_to_max_length(annotate_(example), max_seq_length)
        for example in tqdm(examples, disable=not tqdm_enabled)
    ]

    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(
        features,
        total=len(features),
        desc="add example index and unique id",
        disable=not tqdm_enabled,
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long
    )
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long
    )
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    all_is_impossible = torch.tensor(
        [f.is_impossible for f in features], dtype=torch.float
    )

    if not is_training:
        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            all_feature_index,
            all_cls_index,
            all_p_mask,
        )
    else:
        all_start_positions = torch.tensor(
            [f.start_position for f in features], dtype=torch.long
        )
        all_end_positions = torch.tensor(
            [f.end_position for f in features], dtype=torch.long
        )
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            all_start_positions,
            all_end_positions,
            all_cls_index,
            all_p_mask,
            all_is_impossible,
        )

    return features, dataset


def get_examples(
    examples_to_process: List[Dict[str, torch.Tensor]], is_training: bool = True
) -> List[SquadExample]:
    """Converts list of dict of train data to list of SquadExample

    Args:
        examples_to_process (List[Dict]): list of train data
        is_training (bool, optional): option to set is_training. Defaults to True.

    Raises:
        TypeError: examples_to_process should be a list of examples.

    Returns:
        List[SquadExample]: list of SquadExample
    """
    if not isinstance(examples_to_process, list):
        raise TypeError("Input should be a list of examples.")

    examples = []
    for paragraph in examples_to_process:
        context_text = paragraph["context"]
        for qa in paragraph["qas"]:
            qas_id = qa["id"]
            question_text = qa["question"]
            start_position_character = None
            answer_text = None
            answers = []

            if "is_impossible" in qa:
                is_impossible = qa["is_impossible"]
            else:
                is_impossible = False

            if not is_impossible:
                if is_training:
                    answer = qa["answers"][0]
                    answer_text = answer["text"]
                    start_position_character = answer["answer_start"]
                else:
                    answers = qa["answers"]

            example = SquadExample(
                qas_id=qas_id,
                question_text=question_text,
                context_text=context_text,
                answer_text=answer_text,
                start_position_character=start_position_character,
                title=None,
                is_impossible=is_impossible,
                answers=answers,
            )
            examples.append(example)
    return examples


def load_examples(
    examples: List[Dict[str, torch.Tensor]],
    tokenizer: RecconSpanExtractionTokenizer,
    max_seq_length: int = 512,
    doc_stride: int = 512,
    max_query_length: int = 512,
    evaluate: bool = False,
    output_examples: bool = False,
) -> TensorDataset:
    """Convert list of examples to TensorDataset

    Args:
        examples (List[Dict[str, torch.Tensor]]): train data
        tokenizer (RecconSpanExtractionTokenizer): RecconSpanExtractionTokenizer from sgnlp
        max_seq_length (int, optional): set max_seq_length. Defaults to 512.
        doc_stride (int, optional): set max_seq_length. Defaults to 512.
        max_query_length (int, optional): set max_seq_length. Defaults to 512.
        evaluate (bool, optional): option to use for evaluation. Defaults to False.
        output_examples (bool, optional): option to output examples. Defaults to False.

    Returns:
        TensorDataset: train data converted to TensorDataset
    """
    examples = get_examples(examples, is_training=not evaluate)

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=not evaluate,
        tqdm_enabled=True,
    )

    if output_examples:
        return dataset, examples, features
    return dataset


def calculate_results(truth, predictions, **kwargs):
    truth_dict = {}
    questions_dict = {}
    for item in truth:
        for answer in item["qas"]:
            if answer["answers"]:
                truth_dict[answer["id"]] = answer["answers"][0]["text"]
            else:
                truth_dict[answer["id"]] = ""
            questions_dict[answer["id"]] = answer["question"]

    correct = 0
    incorrect = 0
    similar = 0
    correct_text = {}
    incorrect_text = {}
    similar_text = {}
    predicted_answers = []
    true_answers = []

    for q_id, answer in truth_dict.items():
        predicted_answers.append(predictions[q_id])
        true_answers.append(answer)
        if predictions[q_id].strip() == answer.strip():
            correct += 1
            correct_text[q_id] = answer
        elif (
            predictions[q_id].strip() in answer.strip()
            or answer.strip() in predictions[q_id].strip()
        ):
            similar += 1
            similar_text[q_id] = {
                "truth": answer,
                "predicted": predictions[q_id],
                "question": questions_dict[q_id],
            }
        else:
            incorrect += 1
            incorrect_text[q_id] = {
                "truth": answer,
                "predicted": predictions[q_id],
                "question": questions_dict[q_id],
            }

    extra_metrics = {}
    for metric, func in kwargs.items():
        extra_metrics[metric] = func(true_answers, predicted_answers)

    result = {
        "correct": correct,
        "similar": similar,
        "incorrect": incorrect,
        **extra_metrics,
    }

    texts = {
        "correct_text": correct_text,
        "similar_text": similar_text,
        "incorrect_text": incorrect_text,
    }

    return result, texts


RawResult = collections.namedtuple(
    "RawResult", ["unique_id", "start_logits", "end_logits"]
)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def get_best_predictions(
    all_examples,
    all_features,
    all_results,
    n_best_size,
    max_answer_length,
    do_lower_case,
    verbose_logging,
    version_2_with_negative,
    null_score_diff_threshold,
):

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"],
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        )
                    )
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True,
        )

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"]
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            if pred.start_index > 0:  # this is a non-null prediction
                feature = features[pred.feature_index]
                tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start : (orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(
                    tok_text, orig_text, do_lower_case, verbose_logging
                )
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                )
            )
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="", start_logit=null_start_logit, end_logit=null_end_logit
                    )
                )

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(
                    0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0)
                )

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = (
                score_null
                - best_non_null_entry.start_logit
                - (best_non_null_entry.end_logit)
            )
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    all_best = [
        {
            "id": id,
            "answer": [answer["text"] for answer in answers],
            "probability": [answer["probability"] for answer in answers],
        }
        for id, answers in all_nbest_json.items()
    ]
    return all_best


def write_predictions(
    all_examples,
    all_features,
    all_results,
    n_best_size,
    max_answer_length,
    do_lower_case,
    output_prediction_file,
    output_nbest_file,
    output_null_log_odds_file,
    verbose_logging,
    version_2_with_negative,
    null_score_diff_threshold,
):
    """Write final predictions to the json file and log-odds of null if needed."""
    # logger.info("Writing predictions to: %s" % (output_prediction_file))
    # logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"],
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        )
                    )
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True,
        )

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"]
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            if pred.start_index > 0:  # this is a non-null prediction
                feature = features[pred.feature_index]
                tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start : (orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(
                    tok_text, orig_text, do_lower_case, verbose_logging
                )
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                )
            )
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="", start_logit=null_start_logit, end_logit=null_end_logit
                    )
                )

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(
                    0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0)
                )

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = (
                score_null
                - best_non_null_entry.start_logit
                - (best_non_null_entry.end_logit)
            )
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions, all_nbest_json, scores_diff_json


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position : (orig_end_position + 1)]
    return output_text


def lcs(S, T):
    m = len(S)
    n = len(T)
    counter = [[0] * (n + 1) for x in range(m + 1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i + 1][j + 1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(S[i - c + 1 : i + 1])
                elif c == longest:
                    lcs_set.add(S[i - c + 1 : i + 1])

    return lcs_set


def evaluate_results(text):
    partial_match_scores = []
    lcs_all = []
    impos1, impos2, impos3, impos4 = 0, 0, 0, 0
    pos1, pos2, pos3 = 0, 0, 0
    fscores, squad_fscores = [], []

    for i, key in enumerate(["correct_text", "similar_text", "incorrect_text"]):
        for item in text[key]:
            if i == 0:
                if "impossible" in item and text[key][item]["predicted"] == "":
                    impos1 += 1
                elif "span" in item:
                    pos1 += 1
                    fscores.append(1)
                    squad_fscores.append(1)

            elif i == 1:
                if "impossible" in item:
                    impos2 += 1
                elif "span" in item:

                    z = text[key][item]
                    if z["predicted"] != "":
                        longest_match = list(lcs(z["truth"], z["predicted"]))[0]
                        lcs_all.append(longest_match)
                        partial_match_scores.append(
                            round(
                                len(longest_match.split()) / len(z["truth"].split()), 4
                            )
                        )
                        pos2 += 1
                        r = len(longest_match.split()) / len(z["truth"].split())
                        p = len(longest_match.split()) / len(z["predicted"].split())
                        f = 2 * p * r / (p + r)
                        fscores.append(f)
                        squad_fscores.append(compute_f1(z["truth"], z["predicted"]))
                    else:
                        pos3 += 1
                        impos4 += 1
                        fscores.append(0)
                        squad_fscores.append(0)

            if i == 2:
                if "impossible" in item:
                    impos3 += 1
                elif "span" in item:
                    if z["predicted"] == "":
                        impos4 += 1
                    pos3 += 1
                    fscores.append(0)
                    squad_fscores.append(0)

    total_pos = pos1 + pos2 + pos3
    imr = impos2 / (impos2 + impos3)
    imp = impos2 / (impos2 + impos4)
    imf = 2 * imp * imr / (imp + imr)

    p1 = "Postive Samples:"
    p2 = "Exact Match: {}/{} = {}%".format(
        pos1, total_pos, round(100 * pos1 / total_pos, 2)
    )
    p3 = "Partial Match: {}/{} = {}%".format(
        pos2, total_pos, round(100 * pos2 / total_pos, 2)
    )
    p4a = "LCS F1 Score = {}%".format(round(100 * np.mean(fscores), 2))
    p4b = "SQuAD F1 Score = {}%".format(round(100 * np.mean(squad_fscores), 2))
    p5 = "No Match: {}/{} = {}%".format(
        pos3, total_pos, round(100 * pos3 / total_pos, 2)
    )
    p6 = "\nNegative Samples"
    p7 = "Inv F1 Score = {}%".format(round(100 * imf, 2))
    # p7a = 'Inv Recall: {}/{} = {}%'.format(impos2, impos2+impos3, round(100*imr, 2))
    # p7b = 'Inv Precision: {}/{} = {}%'.format(impos2, impos2+impos4, round(100*imp, 2))

    p = "\n".join([p1, p2, p3, p4a, p4b, p5, p6, p7])
    return p
