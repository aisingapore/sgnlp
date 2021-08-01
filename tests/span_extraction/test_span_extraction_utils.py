import pytest
import unittest
from torch.utils.data.dataset import TensorDataset

from transformers.data.processors.squad import (
    SquadExample,
    SquadFeatures,
)

from sgnlp_models.models.span_extraction import RecconSpanExtractionTokenizer
from sgnlp_models.models.span_extraction.utils import (
    RecconSpanExtractionData,
    InputFeatures,
    get_examples,
    load_examples,
    squad_convert_examples_to_features,
    truncate_to_max_length,
)


class SpanExtractionUtilsTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = RecconSpanExtractionTokenizer
        self.tokenizer_name = "mrm8488/spanbert-finetuned-squadv2"
        self.text = [
            {
                "context": "Our company's wei-ya is tomorrow night ! It's your first Chinese New Year in Taiwan--you must be excited !",
                "qas": [
                    {
                        "id": "dailydialog_tr_1097_utt_1_true_cause_utt_1_span_0",
                        "is_impossible": False,
                        "question": "The target utterance is Our company's wei-ya is tomorrow night ! It's your first Chinese New Year in Taiwan--you must be excited ! The evidence utterance is Our company's wei-ya is tomorrow night ! It's your first Chinese New Year in Taiwan--you must be excited ! What is the causal span from context that is relevant to the target utterance's emotion happiness ?",
                        "answers": [
                            {
                                "text": "Our company's wei-ya is tomorrow night ! It's your first Chinese New Year in Taiwan",
                                "answer_start": 0,
                            }
                        ],
                    }
                ],
            }
        ]

    def test_data(self):
        data = RecconSpanExtractionData(None, None)
        attributes = ["dataset", "for_predict", "__getitem__", "__len__"]
        for attr in attributes:
            self.assertTrue(hasattr(data, attr))

    def test_inputfeatures(self):
        inputfeatures = InputFeatures(
            None, None, None, None, None, None, None, None, None, None, None, None
        )
        attributes = [
            "unique_id",
            "example_index",
            "doc_span_index",
            "tokens",
            "token_to_orig_map",
            "token_is_max_context",
            "input_ids",
            "input_mask",
            "segment_ids",
            "cls_index",
            "p_mask",
            "paragraph_len",
            "start_position",
            "end_position",
            "is_impossible",
        ]
        for attr in attributes:
            self.assertTrue(hasattr(inputfeatures, attr))

    def test_truncate_to_max_length(self):
        length = 50
        squadfeatures = SquadFeatures(
            input_ids=[0] * length,
            attention_mask=[0] * length,
            token_type_ids=[0] * length,
            cls_index=0,
            p_mask=[0] * length,
            example_index=0,
            unique_id=0,
            paragraph_len=3,
            token_is_max_context={},
            tokens=[],
            token_to_orig_map={},
            start_position=0,
            end_position=0,
            is_impossible=False,
        )
        max_length = 20

        output = truncate_to_max_length([squadfeatures], max_length)

        self.assertIsInstance(output, list)
        self.assertIsInstance(output[0], SquadFeatures)

        self.assertTrue(len(output[0].input_ids) <= max_length)
        self.assertTrue(len(output[0].p_mask) <= max_length)
        self.assertTrue(len(output[0].token_type_ids) <= max_length)
        self.assertTrue(len(output[0].tokens) <= max_length)
        self.assertTrue(len(output[0].attention_mask) <= max_length)

    def test_get_examples(self):
        output = get_examples(self.text)
        self.assertIsInstance(output, list)
        self.assertIsInstance(output[0], SquadExample)

    @pytest.mark.slow
    def test_squad_convert_examples_to_features(self):
        examples = SquadExample(
            qas_id=0,
            question_text="question_text",
            context_text="context_text",
            answer_text="answer_text",
            start_position_character="",
            title=None,
            is_impossible=True,
            answers="answers",
        )
        features, dataset = squad_convert_examples_to_features(
            examples=[examples],
            tokenizer=self.tokenizer.from_pretrained(self.tokenizer_name),
            max_seq_length=20,
            doc_stride=20,
            max_query_length=20,
            is_training=True,
            tqdm_enabled=True,
        )

        self.assertIsInstance(features, list)
        self.assertIsInstance(features[0], SquadFeatures)

        self.assertIsInstance(dataset, TensorDataset)

    @pytest.mark.slow
    def test_load_examples(self):
        output = load_examples(
            self.text, self.tokenizer.from_pretrained(self.tokenizer_name)
        )

        self.assertIsInstance(output, TensorDataset)
