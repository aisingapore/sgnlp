import argparse
import json
import re
from typing import Union

from .data_class import RstPointerParserTrainArgs, RstPointerSegmenterTrainArgs


def parse_args_and_load_config() -> Union[
    RstPointerParserTrainArgs, RstPointerSegmenterTrainArgs
]:
    """Helper method to parse input arguments

    Returns:
        Union[RstPointerParserTrainArgs, RstPointerSegmenterTrainArgs]: returns the corresponding TrainArgs object
        depending on the input arguments.
    """
    parser = argparse.ArgumentParser(description="RST Training")
    parser.add_argument(
        "--train_type",
        type=str,
        choices=["segmenter", "parser"],
        required=True,
        help="Select which model to train.",
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to config file."
    )
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = json.load(f)
    if args.train_type == "parser":
        data_class = RstPointerParserTrainArgs(**config)
    elif args.train_type == "segmenter":
        data_class = RstPointerSegmenterTrainArgs(**config)
    else:
        raise ValueError(f"Invalid train_type: {args.train_type}")
    return data_class


# This transforms the relation label in the discourse tree files to one of 18 rhetorical classes
# Check Section 4.1 of Carlson and D. Marcu. 2001. Discourse Tagging Reference Manual. for more details
relation_to_rhetorical_class_map = {
    "Analogy": "Comparison",
    "Cause-Result": "Cause",
    "Comment-Topic": "Topic-Comment",
    "Comparison": "Comparison",
    "Consequence": "Cause",
    "Contrast": "Contrast",
    "Disjunction": "Joint",
    "Evaluation": "Evaluation",
    "Interpretation": "Evaluation",
    "Inverted-Sequence": "Temporal",
    "List": "Joint",
    "Otherwise": "Condition",
    "Problem-Solution": "Topic-Comment",
    "Proportion": "Comparison",
    "Question-Answer": "Topic-Comment",
    "Reason": "Explanation",
    "Same-Unit": "Same-Unit",
    "Sequence": "Temporal",
    "Statement-Response": "Topic-Comment",
    "Temporal-Same-Time": "Temporal",
    "Topic-Comment": "Topic-Comment",
    "analogy": "Comparison",
    "analogy-e": "Comparison",
    "antithesis": "Contrast",
    "antithesis-e": "Contrast",
    "attribution": "Attribution",
    "attribution-e": "Attribution",
    "attribution-n": "Attribution",
    "background": "Background",
    "background-e": "Background",
    "cause": "Cause",
    "circumstance": "Background",
    "circumstance-e": "Background",
    "comment": "Evaluation",
    "comment-e": "Evaluation",
    "comparison": "Comparison",
    "comparison-e": "Comparison",
    "concession": "Contrast",
    "concession-e": "Contrast",
    "conclusion": "Evaluation",
    "condition": "Condition",
    "condition-e": "Condition",
    "consequence-n": "Cause",
    "consequence-n-e": "Cause",
    "consequence-s": "Cause",
    "consequence-s-e": "Cause",
    "contingency": "Condition",
    "definition": "Elaboration",
    "definition-e": "Elaboration",
    "elaboration-additional": "Elaboration",
    "elaboration-additional-e": "Elaboration",
    "elaboration-general-specific": "Elaboration",
    "elaboration-general-specific-e": "Elaboration",
    "elaboration-object-attribute": "Elaboration",
    "elaboration-object-attribute-e": "Elaboration",
    "elaboration-part-whole": "Elaboration",
    "elaboration-part-whole-e": "Elaboration",
    "elaboration-process-step-e": "Elaboration",
    "elaboration-set-member": "Elaboration",
    "elaboration-set-member-e": "Elaboration",
    "enablement": "Enablement",
    "enablement-e": "Enablement",
    "evaluation-n": "Evaluation",
    "evaluation-s": "Evaluation",
    "evaluation-s-e": "Evaluation",
    "evidence": "Explanation",
    "evidence-e": "Explanation",
    "example": "Elaboration",
    "example-e": "Elaboration",
    "explanation-argumentative": "Explanation",
    "explanation-argumentative-e": "Explanation",
    "hypothetical": "Condition",
    "interpretation-n": "Evaluation",
    "interpretation-s": "Evaluation",
    "interpretation-s-e": "Evaluation",
    "manner": "Manner-Means",
    "manner-e": "Manner-Means",
    "means": "Manner-Means",
    "means-e": "Manner-Means",
    "otherwise": "Condition",
    "preference": "Comparison",
    "preference-e": "Comparison",
    "purpose": "Enablement",
    "purpose-e": "Enablement",
    "question-answer-n": "Topic-Comment",
    "question-answer-s": "Topic-Comment",
    "reason": "Explanation",
    "reason-e": "Explanation",
    "restatement": "Summary",
    "restatement-e": "Summary",
    "result": "Cause",
    "result-e": "Cause",
    "rhetorical-question": "Topic-Comment",
    "span": "span",
    "temporal-after": "Temporal",
    "temporal-after-e": "Temporal",
    "temporal-before": "Temporal",
    "temporal-before-e": "Temporal",
    "temporal-same-time": "Temporal",
    "temporal-same-time-e": "Temporal",
}

# This is the list of classes used for training/testing in relation labelling
relation_table = [
    "Attribution_NS",
    "Attribution_SN",
    "Background_NS",
    "Background_SN",
    "Cause_NN",
    "Cause_NS",
    "Cause_SN",
    "Comparison_NN",
    "Comparison_NS",
    "Comparison_SN",
    "Condition_NN",
    "Condition_NS",
    "Condition_SN",
    "Contrast_NN",
    "Contrast_NS",
    "Contrast_SN",
    "Elaboration_NS",
    "Elaboration_SN",
    "Enablement_NS",
    "Enablement_SN",
    "Evaluation_NN",
    "Evaluation_NS",
    "Evaluation_SN",
    "Explanation_NN",
    "Explanation_NS",
    "Explanation_SN",
    "Joint_NN",
    "Manner-Means_NS",
    "Manner-Means_SN",
    "Same-Unit_NN",
    "Summary_NS",
    "Summary_SN",
    "Temporal_NN",
    "Temporal_NS",
    "Temporal_SN",
    "TextualOrganization_NN",
    "Topic-Comment_NN",
    "Topic-Comment_NS",
    "Topic-Comment_SN",
]


def get_relation_and_nucleus(label_index):
    relation = relation_table[label_index]
    temp = re.split(r"_", relation)
    sub1 = temp[0]
    sub2 = temp[1]

    if sub2 == "NN":
        nuclearity_left = "Nucleus"
        nuclearity_right = "Nucleus"
        relation_left = sub1
        relation_right = sub1

    elif sub2 == "NS":
        nuclearity_left = "Nucleus"
        nuclearity_right = "Satellite"
        relation_left = "span"
        relation_right = sub1

    elif sub2 == "SN":
        nuclearity_left = "Satellite"
        nuclearity_right = "Nucleus"
        relation_left = sub1
        relation_right = "span"

    return nuclearity_left, nuclearity_right, relation_left, relation_right
