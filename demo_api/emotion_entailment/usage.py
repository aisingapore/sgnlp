from sgnlp.models.emotion_entailment import (
    RecconEmotionEntailmentConfig,
    RecconEmotionEntailmentTokenizer,
    RecconEmotionEntailmentModel,
    RecconEmotionEntailmentPreprocessor,
    RecconEmotionEntailmentPostprocessor,
)

# Load model
config = RecconEmotionEntailmentConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/reccon_emotion_entailment/config.json"
)
tokenizer = RecconEmotionEntailmentTokenizer.from_pretrained("roberta-base")
model = RecconEmotionEntailmentModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/reccon_emotion_entailment/pytorch_model.bin",
    config=config,
)
preprocessor = RecconEmotionEntailmentPreprocessor(tokenizer)
postprocessor = RecconEmotionEntailmentPostprocessor()

# Model predict
input_batch = {
    "emotion": ["happiness", "happiness", "happiness", "happiness"],
    "target_utterance": [
        "Thank you very much .",
        "Thank you very much .",
        "Thank you very much .",
        "Thank you very much .",
    ],
    "evidence_utterance": [
        "It's very thoughtful of you to invite me to your wedding .",
        "How can I forget my old friend ?",
        "My best wishes to you and the bride !",
        "Thank you very much .",
    ],
    "conversation_history": [
        "It's very thoughtful of you to invite me to your wedding . How can I forget my old friend ? My best wishes to you and the bride ! Thank you very much .",
        "It's very thoughtful of you to invite me to your wedding . How can I forget my old friend ? My best wishes to you and the bride ! Thank you very much .",
        "It's very thoughtful of you to invite me to your wedding . How can I forget my old friend ? My best wishes to you and the bride ! Thank you very much .",
        "It's very thoughtful of you to invite me to your wedding . How can I forget my old friend ? My best wishes to you and the bride ! Thank you very much .",
    ],
}

tensor_dict = preprocessor(input_batch)
raw_output = model(**tensor_dict)
output = postprocessor(raw_output)
