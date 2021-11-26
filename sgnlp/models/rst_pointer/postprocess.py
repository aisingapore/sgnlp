import logging
from typing import List


class RstPostprocessor:
    def __init__(self, detokenizer=None):
        if detokenizer is not None:
            self.detokenizer = detokenizer
        else:
            try:
                import nltk

                self.detokenizer = nltk.tokenize.treebank.TreebankWordDetokenizer()
            except ModuleNotFoundError:
                logging.warning(
                    "Could not import nltk.treebank.TreebankWordDetokenizer. "
                    "Please install nltk to use this postprocessor."
                )
                logging.warning('To use "nltk", please install with "pip install nltk"')

    def __call__(
        self,
        sentences: List[str],
        tokenized_sentences: List[List[str]],
        end_boundaries,
        discourse_tree_splits,
    ):

        trees = []
        for sentence, tokenized_sentence, end_boundary, discourse_tree_split in zip(
            sentences, tokenized_sentences, end_boundaries, discourse_tree_splits
        ):

            edus = []
            current_start = 0
            for edu_break in end_boundary:
                edus.append(
                    self.detokenizer.detokenize(
                        tokenized_sentence[current_start : edu_break + 1]
                    )
                )
                current_start = edu_break + 1

            hierplane_tree = (
                self._transform_discourse_tree_splits_to_hierplane_tree_format(
                    discourse_tree_split, sentence, edus
                )
            )
            trees.append(hierplane_tree)

        return trees

    def _transform_discourse_tree_splits_to_hierplane_tree_format(
        self, discourse_tree_splits, original_sentence, edus
    ):
        hierplane_tree = {
            "root": {"attributes": ["root"], "word": original_sentence},
            "text": original_sentence,
        }

        if len(discourse_tree_splits) == 0:
            return hierplane_tree

        current_node = hierplane_tree["root"]
        splits_iterator = iter(discourse_tree_splits)
        self._hierplane_tree_builder_helper(current_node, splits_iterator, edus)

        return hierplane_tree

    def _hierplane_tree_builder_helper(self, current_node, splits_iterator, edus):
        try:
            current_split = next(splits_iterator)

            left_node = {
                "attributes": [current_split.left.label],
                "link": current_split.left.ns_type,
                "word": self.detokenizer.detokenize(
                    edus[current_split.left.span[0] : current_split.left.span[1] + 1]
                ),
            }

            # If it is not a leaf node, expand
            if (current_split.left.span[1] - current_split.left.span[0]) != 0:
                self._hierplane_tree_builder_helper(left_node, splits_iterator, edus)

            right_node = {
                "attributes": [current_split.right.label],
                "link": current_split.right.ns_type,
                "word": self.detokenizer.detokenize(
                    edus[current_split.right.span[0] : current_split.right.span[1] + 1]
                ),
            }

            # If it is not a leaf node, expand
            if (current_split.right.span[1] - current_split.right.span[0]) != 0:
                self._hierplane_tree_builder_helper(right_node, splits_iterator, edus)

            current_node["children"] = [left_node, right_node]

        except StopIteration:
            return
