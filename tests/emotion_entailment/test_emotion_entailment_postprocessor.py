import pytest
import unittest

from sgnlp_models.models.emotion_entailment import (
    RecconEmotionEntailmentPreprocessor,
    RecconEmotionEntailmentConfig,
    RecconEmotionEntailmentModel,
    RecconEmotionEntailmentPostprocessor,
)


class EmotionEntailmentTestPostprocessor(unittest.TestCase):
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
        preprocessor = RecconEmotionEntailmentPreprocessor()
        preprocessed_input = preprocessor(inputs)
        config = RecconEmotionEntailmentConfig.from_pretrained(
            "https://sgnlp.blob.core.windows.net/models/reccon_emotion_entailment/config.json"
        )
        model = RecconEmotionEntailmentModel.from_pretrained(
            "https://sgnlp.blob.core.windows.net/models/reccon_emotion_entailment/pytorch_model.bin",
            config=config,
        )
        self.batch_size = len(inputs["emotion"])
        self.test_input = model(**preprocessed_input)

    @pytest.mark.slow
    def test_postprocessor(self):
        postprocessor = RecconEmotionEntailmentPostprocessor()
        output = postprocessor(self.test_input)

        self.assertIsInstance(output, list)
        self.assertEqual(len(output), self.batch_size)
        self.assertIsInstance(output[0], int)
