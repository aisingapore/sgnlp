import pytest
import unittest

from sgnlp.models.span_extraction import (
    RecconSpanExtractionPreprocessor,
    RecconSpanExtractionConfig,
    RecconSpanExtractionModel,
    RecconSpanExtractionPostprocessor,
)


class SpanExtractionTestPostprocessor(unittest.TestCase):
    def setUp(self) -> None:
        inputs = {
            "emotion": ["happiness", "sadness"],
            "target_utterance": ["this is target 1", "this is target 2"],
            "evidence_utterance": ["this is evidence 1", "this is evidence 2"],
            "conversation_history": [
                "this is conversation history 1",
                "this is conversation history 2",
            ],
        }
        preprocessor = RecconSpanExtractionPreprocessor()
        preprocessed_input, evidence, examples, features = preprocessor(inputs)
        config = RecconSpanExtractionConfig.from_pretrained(
            "https://sgnlp.blob.core.windows.net/models/reccon_span_extraction/config.json"
        )
        model = RecconSpanExtractionModel.from_pretrained(
            "https://sgnlp.blob.core.windows.net/models/reccon_span_extraction/pytorch_model.bin",
            config=config,
        )
        self.batch_size = len(inputs["emotion"])
        self.test_input = model(**preprocessed_input)
        self.evidence = evidence
        self.examples = examples
        self.features = features

    @pytest.mark.slow
    def test_postprocessor(self):
        postprocessor = RecconSpanExtractionPostprocessor()
        context, evidence_span, probability = postprocessor(
            self.test_input, self.evidence, self.examples, self.features
        )

        self.assertIsInstance(context, list)
        self.assertEqual(len(context), self.batch_size)
        self.assertIsInstance(context[0], list)
        self.assertIsInstance(context[0][0], str)

        self.assertIsInstance(evidence_span, list)
        self.assertEqual(len(evidence_span), self.batch_size)
        self.assertIsInstance(evidence_span[0], list)
        self.assertIsInstance(evidence_span[0][0], int)

        self.assertIsInstance(probability, list)
        self.assertEqual(len(probability), self.batch_size)
        self.assertIsInstance(probability[0], list)
