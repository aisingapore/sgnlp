import os
import re
import copy
import nltk
import pickle
import argparse

from .utils import relation_to_rhetorical_class_map, relation_table
from .modules.type import DiscourseTreeNode, DiscourseTreeSplit, FileFormat


def read_sentences(filepath: str, file_format: FileFormat):
    if file_format == FileFormat.WSJ:
        return read_sentences_for_wsj_format(filepath)
    elif file_format == FileFormat.FILE:
        return read_sentences_for_file_format(filepath)
    else:
        raise ValueError(f"Invalid FileFormat provided: {file_format}")


def read_sentences_for_wsj_format(filepath: str):
    with open(filepath, "r") as f:
        lines = f.readlines()
    # return lines that are not empty
    return [line.strip() for line in lines if line.strip()]


def read_sentences_for_file_format(filepath: str):
    sentences = []
    current_sentence_idx = -1
    with open(filepath, "r") as f:
        lines = f.readlines()
        for line in lines:
            # lines that start with double space marks a new sentence
            if line.startswith("  "):
                current_sentence_idx += 1
                sentences.append(line.strip())
            else:
                sentences[current_sentence_idx] += " " + line.strip()
    return sentences


def read_discourse_tree(filepath):
    with open(filepath, "r") as f:
        raw_discourse_tree = f.readlines()

    discourse_tree_nodes = []

    for line in raw_discourse_tree:
        line = line.strip()

        # ignore empty lines and lines without more than 1 unit of text
        if len(line.split()) > 1:
            discourse_tree_nodes.append(parse_discourse_tree_node(line))

    return discourse_tree_nodes


def read_edus(filepath):
    with open(filepath, "r") as f:
        edus = f.readlines()

    return [edu.strip() for edu in edus]


def parse_discourse_tree(discourse_tree_nodes):
    """
    Parses the tree and returns 3 lists containing
        the splits in a pre-order traversal form (i.e. Root->left recursive->right recursive),
        the span's parent's representation (i.e. parent_index),
        the span's left-sibling's representation (i.e. sibling_index).
    The parent and left-sibling representations are the edu span end index.
        e.g. if the parent/left-sibling spans from 2:4, the representative index is 4
    """
    discourse_tree_splits = []
    parent_index = []
    sibling_index = []
    node_iter = iter(discourse_tree_nodes[1:])  # ignore root node

    parse_discourse_tree_helper(
        node_iter,
        discourse_tree_splits,
        parent_index,
        sibling_index,
        current_span=discourse_tree_nodes[0].span,
        parent_span=None,
        left_sibling_span=None,
    )

    decoder_input_index = [split.right.span[1] for split in discourse_tree_splits]

    return discourse_tree_splits, parent_index, sibling_index, decoder_input_index


def parse_discourse_tree_helper(
    node_iter,
    discourse_tree_splits,
    parent_index,
    sibling_index,
    current_span,
    parent_span,
    left_sibling_span,
):
    try:
        # get left node
        left_node = next(node_iter)

        discourse_tree_splits.append(DiscourseTreeSplit(left=left_node, right=None))
        current_idx = len(discourse_tree_splits) - 1

        if parent_span:
            parent_index.append(parent_span[1])
        else:
            parent_index.append(0)

        if left_sibling_span:
            sibling_index.append(left_sibling_span[1])
        else:
            sibling_index.append(99)

        # expand left
        if span_length(left_node.span) > 1:
            parse_discourse_tree_helper(
                node_iter,
                discourse_tree_splits,
                parent_index,
                sibling_index,
                current_span=left_node.span,
                parent_span=current_span,
                left_sibling_span=None,
            )

        # get right node
        right_node = next(node_iter)
        discourse_tree_splits[current_idx].right = right_node

        # expand right
        if span_length(right_node.span) > 1:
            parse_discourse_tree_helper(
                node_iter,
                discourse_tree_splits,
                parent_index,
                sibling_index,
                current_span=right_node.span,
                parent_span=current_span,
                left_sibling_span=left_node.span,
            )

    except StopIteration:
        return


def parse_discourse_tree_node(discourse_tree_node_raw):
    doc = re.sub("[()]", "", discourse_tree_node_raw).split()

    ns_type = doc[0]
    node_type = doc[1]
    if ns_type == "Root":
        span_start = int(doc[2])
        span_end = int(doc[3])
        label = None
        text = None
    elif node_type == "span":
        span_start = int(doc[2])
        span_end = int(doc[3])
        label = doc[5]
        text = None
    elif node_type == "leaf":
        span_start = int(doc[2])
        span_end = span_start
        label = doc[4]
        text = re.search("_!.*_!", discourse_tree_node_raw)
        text = text.group(0).replace("_!", "")
    else:
        raise ValueError(f"Unknown node type found: {node_type}")

    return DiscourseTreeNode(
        span=(span_start, span_end), ns_type=ns_type, label=label, text=text
    )


def get_splits_order_label(discourse_tree_splits):
    """
    This is used for training the parser. It is in the order of the splits in a pre-order traversal form,
    and represented using the index of the EDU immediately to the left of the split.
    This is called Parsing_Label in the original code.
    """
    return [split.left.span[1] for split in discourse_tree_splits]


def get_edu_spans(sentences, edus):
    edu_spans = []
    edu_idx = 0
    for sentence in sentences:
        formed_sentence = ""
        start_idx = edu_idx + 1
        while formed_sentence != sentence:
            if formed_sentence:
                formed_sentence = " ".join([formed_sentence, edus[edu_idx]])
            else:
                formed_sentence = edus[edu_idx]
            edu_idx += 1
        edu_spans.append((start_idx, edu_idx))
    return edu_spans


def get_sentence_edu_spans_from_discourse_tree_nodes(discourse_tree_nodes):
    full_span = discourse_tree_nodes[0].span
    sentence_edu_spans = []
    sentence_start_found = False
    sentence_start_span = 0
    ambiguous_breaks = []
    possible_sentence_spans = []

    for discourse_tree_node in discourse_tree_nodes:
        possible_sentence_spans.append(discourse_tree_node.span)
        if is_leaf(discourse_tree_node):
            if not sentence_start_found:
                sentence_start_span = discourse_tree_node.span[0]
                sentence_start_found = True

            if contains_end_of_sentence(discourse_tree_node):
                sentence_end_span = discourse_tree_node.span[0]

                if ambiguous_breaks:
                    # NOTE! This implementation only checks for a single one ambiguous break in a sentence.
                    # Resolve ambiguous breaks
                    for ambiguous_break in ambiguous_breaks:
                        if (
                            sentence_start_span,
                            sentence_end_span,
                        ) in possible_sentence_spans:
                            sentence_edu_spans.append(
                                (sentence_start_span, sentence_end_span)
                            )
                        else:
                            sentence_edu_spans.append(
                                (sentence_start_span, ambiguous_break)
                            )
                            sentence_edu_spans.append(
                                (ambiguous_break + 1, sentence_end_span)
                            )
                    ambiguous_breaks = []
                else:
                    sentence_edu_spans.append((sentence_start_span, sentence_end_span))
                sentence_start_found = False

            elif contains_ambiguous_end_of_sentence(discourse_tree_node):
                ambiguous_breaks.append(discourse_tree_node.span[0])

    # Validate sentence_edu_spans matches with full_span
    if not contains_full_span(sentence_edu_spans, full_span):
        raise ValueError("Sentence edu spans found does not match with full span!")

    return sentence_edu_spans


def contains_full_span(sentence_edu_spans, full_span):
    # Check first edu_span begins at the same point as full_span
    if not sentence_edu_spans[0][0] == full_span[0]:
        return False

    # Set span end as 0
    span_end = sentence_edu_spans[0][0] - 1
    for sentence_edu_span in sentence_edu_spans:
        if sentence_edu_span[0] == span_end + 1:
            span_end = sentence_edu_span[1]
        else:
            return False
    return True


def contains_ambiguous_end_of_sentence(node: DiscourseTreeNode):
    edu = node.text
    return edu.endswith("--")


def contains_end_of_sentence(node: DiscourseTreeNode):
    edu = node.text
    return (
        edu.endswith(".")
        or edu.endswith("<P>")
        or edu.endswith('."')
        or node.label == "TextualOrganization"
    )


def is_leaf(discourse_tree_node):
    return discourse_tree_node.span[1] == discourse_tree_node.span[0]


def get_discourse_nodes_slice(discourse_nodes, edu_span):
    for idx, discourse_node in enumerate(discourse_nodes):
        if discourse_node.span == edu_span:
            start_idx = idx
            break

    end_idx = start_idx
    for discourse_node in discourse_nodes[start_idx:]:
        if span_within(edu_span, discourse_node.span):
            end_idx += 1

    return discourse_nodes[start_idx:end_idx]


def normalize_nodes_slice(discourse_tree_splits):
    start_span = discourse_tree_splits[0].span[0]
    splits_copy = copy.deepcopy(discourse_tree_splits)
    for splits in splits_copy:
        splits.span = (splits.span[0] - start_span, splits.span[1] - start_span)

    return splits_copy


def span_length(span):
    return span[1] - span[0] + 1


def span_within(outer_span, inner_span):
    return outer_span[0] <= inner_span[0] and outer_span[1] >= inner_span[1]


def get_relation_label_from_split(discourse_tree_split: DiscourseTreeSplit):
    left_ns_type_char = discourse_tree_split.left.ns_type[0]
    right_ns_type_char = discourse_tree_split.right.ns_type[0]
    suffix = "_" + left_ns_type_char + right_ns_type_char

    left_rhetorical_class = discourse_tree_split.left.label

    if left_rhetorical_class != "span":
        return left_rhetorical_class + suffix
    else:
        right_rhetorical_class = discourse_tree_split.right.label
        return right_rhetorical_class + suffix


def get_relation_label_index(relation_label):
    return relation_table.index(relation_label)


def get_tokenized_sentence_and_edu_breaks(sentence_edus):
    tokenized_sentence = []
    edu_breaks = []
    for edu in sentence_edus:
        tokenized_edu = nltk.word_tokenize(edu)
        tokenized_sentence += tokenized_edu
        if len(edu_breaks) == 0:
            edu_breaks.append(len(tokenized_edu) - 1)
        else:
            edu_breaks.append(edu_breaks[-1] + len(tokenized_edu))

    return tokenized_sentence, edu_breaks


def transform_discourse_tree_splits_relation_label(
    discourse_tree_splits: DiscourseTreeSplit,
):
    splits_copy = copy.deepcopy(discourse_tree_splits)
    for discourse_tree_split in splits_copy:
        discourse_tree_split.left.label = relation_to_rhetorical_class_map[
            discourse_tree_split.left.label
        ]
        discourse_tree_split.right.label = relation_to_rhetorical_class_map[
            discourse_tree_split.right.label
        ]

    return splits_copy


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw RST-DT data files and save in formats needed for model training"
    )

    parser.add_argument(
        "--raw_data_dir", type=str, help="Directory of RST-DT data", required=True
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory for saving preprocessed files",
        required=True,
    )

    args = parser.parse_args()

    raw_data_dir = args.raw_data_dir
    save_dir = args.save_dir

    wsj_format_regex = "^wsj.*out$"
    file_format_regex = "^file[0-9]$"

    filepaths = os.listdir(raw_data_dir)

    base_filepaths = [
        os.path.join(raw_data_dir, filepath)
        for filepath in filepaths
        if re.search(wsj_format_regex, filepath)
        or re.search(file_format_regex, filepath)
    ]

    data = {
        "tokenized_sentences": [],
        "edu_breaks": [],
        "discourse_tree_splits": [],  # aka Gold Discourse Tree Structure
        "splits_order": [],  # aka Parsing_Label
        "relation_index": [],
        "decoder_input_index": [],
        "parent_index": [],
        "sibling_index": [],
    }

    for base_filepath in base_filepaths:
        dis_filepath = base_filepath + ".dis"
        edus_filepath = base_filepath + ".edus"

        # read files
        file_basename = os.path.basename(base_filepath)
        if re.search(wsj_format_regex, file_basename):
            file_format = FileFormat.WSJ
        elif re.search(file_format_regex, file_basename):
            file_format = FileFormat.FILE
        else:
            raise ValueError(f"Unrecognized file format for filepath: {file_basename}")

        file_discourse_tree = read_discourse_tree(dis_filepath)
        file_edus = read_edus(edus_filepath)

        file_sentence_edu_spans = get_sentence_edu_spans_from_discourse_tree_nodes(
            file_discourse_tree
        )

        for sentence_edu_span in file_sentence_edu_spans:
            # Each sentence_edu_span is equivalent to a sentence
            try:
                discourse_nodes_slice = get_discourse_nodes_slice(
                    file_discourse_tree, sentence_edu_span
                )
            except UnboundLocalError:
                # No well formed discourse tree for sentence, skip it
                continue

            discourse_nodes_slice = normalize_nodes_slice(discourse_nodes_slice)
            (
                discourse_tree_splits,
                parent_index,
                sibling_index,
                decoder_input_index,
            ) = parse_discourse_tree(discourse_nodes_slice)

            # converts relation label to coarse form (39 classes)
            discourse_tree_splits = transform_discourse_tree_splits_relation_label(
                discourse_tree_splits
            )

            sentence_relation_labels = [
                get_relation_label_from_split(split) for split in discourse_tree_splits
            ]
            sentence_relation_label_index = [
                get_relation_label_index(label) for label in sentence_relation_labels
            ]

            sentence_edus = file_edus[sentence_edu_span[0] - 1 : sentence_edu_span[1]]
            (
                tokenized_sentence,
                sentence_edu_breaks,
            ) = get_tokenized_sentence_and_edu_breaks(sentence_edus)

            splits_order_label = get_splits_order_label(discourse_tree_splits)

            data["tokenized_sentences"].append(tokenized_sentence)
            data["edu_breaks"].append(sentence_edu_breaks)
            data["discourse_tree_splits"].append(discourse_tree_splits)
            data["splits_order"].append(splits_order_label)
            data["relation_index"].append(sentence_relation_label_index)
            data["decoder_input_index"].append(decoder_input_index)
            data["parent_index"].append(parent_index)
            data["sibling_index"].append(sibling_index)

    # Save data
    os.makedirs(save_dir, exist_ok=True)
    for key, value in data.items():
        pickle.dump(value, open(os.path.join(save_dir, key + ".pickle"), "wb"))


if __name__ == "__main__":
    main()
