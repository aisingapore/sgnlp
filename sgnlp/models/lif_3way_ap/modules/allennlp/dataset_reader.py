import json
import logging
from typing import Any, Dict, List

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from overrides import overrides

from .util import process_para_json

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("lif_3way_ap_dataset_reader")
class Lif3WayApDatasetReader(DatasetReader):
    """
    Dataset Reader for 3-way Attentive Pooling Network

    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False,
        num_context_answers: int = 3,
    ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._num_context_answers = num_context_answers

    @overrides
    def _read(self, file_path: str):
        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json["data"]
        logger.info("Reading the dataset")
        for article in dataset:
            for paragraph_json in article["paragraphs"]:
                paragraph_json = process_para_json(paragraph_json)
                paragraph = paragraph_json["context"]
                tokenized_paragraph = self._tokenizer.tokenize(paragraph)
                qas = paragraph_json["qas"]
                for qa in qas:
                    metadata = {}
                    prev_q_text_list = [
                        q.strip().replace("\n", "") for q in qa["prev_qs"]
                    ]
                    start_idx = max(
                        0, len(prev_q_text_list) - self._num_context_answers
                    )
                    prev_q_text_list = prev_q_text_list[start_idx:]
                    prev_ans_text_list = [a["text"] for a in qa["prev_ans"]]
                    prev_ans_text_list = prev_ans_text_list[start_idx:]
                    metadata["prev_qs"] = prev_q_text_list
                    metadata["prev_ans"] = prev_ans_text_list
                    prev_span_starts_list = [
                        [a["answer_start"]] for a in qa["prev_ans"][start_idx:]
                    ]
                    prev_span_ends_list = [
                        [start[0] + len(answer)]
                        for start, answer in zip(
                            prev_span_starts_list, prev_ans_text_list
                        )
                    ]

                    assert len(prev_span_starts_list) == len(prev_span_ends_list)
                    candidate = qa["candidate"].strip()
                    answer = qa["label"]
                    metadata["candidate"] = candidate
                    metadata["label"] = answer
                    instance = self.text_to_instance(
                        prev_q_text_list,
                        prev_ans_text_list,
                        paragraph,
                        candidate,
                        prev_span_starts_list,
                        prev_span_ends_list,
                        tokenized_paragraph,
                        answer,
                        metadata,
                    )
                    yield instance

    @overrides
    def text_to_instance(
        self,  # type: ignore
        prev_q_text_list: List[str],
        prev_ans_text_list: List[str],
        passage_text: str,
        candidate: str,
        prev_start_span_list: List[List[int]] = None,
        prev_end_span_list: List[List[int]] = None,
        passage_tokens: List[Token] = None,
        answer: int = None,
        additional_metadata: Dict[str, Any] = None,
    ) -> Instance:
        """

        :param prev_q_text_list:
        :param prev_ans_text_list:
        :param passage_text:
        :param candidate:
        :param prev_start_span_list:
        :param prev_end_span_list:
        :param passage_tokens:
        :param answer:
        :param additional_metadata:
        :return:
        """
        fields: Dict[str, Field] = dict()
        fields["passage"] = TextField(passage_tokens, self._token_indexers)
        candidate_tokens = self._tokenizer.tokenize(candidate)
        fields["candidate"] = TextField(candidate_tokens, self._token_indexers)

        assert len(prev_q_text_list) == len(prev_ans_text_list)
        all_q_a_text = ""
        for q, a in zip(prev_q_text_list, prev_ans_text_list):
            all_q_a_text += q + " | " + a + " || "

        all_qa_tokens = self._tokenizer.tokenize(all_q_a_text)
        fields["all_qa"] = TextField(all_qa_tokens, self._token_indexers)

        source_tokens = (
            passage_tokens
            + [Token(">")]
            + all_qa_tokens
            + [Token(">")]
            + candidate_tokens
        )
        fields["combined_source"] = TextField(source_tokens, self._token_indexers)

        if answer is not None:
            fields["label"] = LabelField(answer, skip_indexing=True)

        metadata = additional_metadata or {}
        metadata.update(
            {
                "passage_tokens": passage_tokens,
                "candidate_tokens": candidate_tokens,
                "all_qa_tokens": all_qa_tokens,
            }
        )
        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)
