import json
from flask import request, jsonify

from demo_api.common import create_api
from sgnlp.models.emotion_entailment import (
    RecconEmotionEntailmentConfig,
    RecconEmotionEntailmentTokenizer,
    RecconEmotionEntailmentModel,
    RecconEmotionEntailmentPreprocessor,
    RecconEmotionEntailmentPostprocessor,
)
from sgnlp.models.emotion_entailment.utils import (
    get_all_evidence_utterance_from_conversation,
)

app = create_api(app_name=__name__, model_card_path="model_card/emotion_entailment.json")

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


@app.route("/predict", methods=["POST"])
def predict():
    """Iterate through each evidence utt in context to perform RECCON emotion entailment.
    The last utterance in context is used as the target utterance.

    Inputs:
        A json with 'context' and 'emotion' keys
    Example:
        {"context": [
            "Why don 't you watch where you 're going ?",
            "Me?",
            "You 're the one who pulled out in front of me !",
            "There was plenty of room for me to pull out .",
            "You didn 't have to stay in the lane you were in .",
            "Hey , listen . I had every right to stay in the lane I was in .",
            "You were supposed to wait until I passed to pull out .",
            "And anyhow , you didn 't give me any time to change lanes.",
            "All of a sudden--BANG--there you are right in front of me ."
        ],
        "emotion": "anger"}

    Returns:
        json: return a json with utterances, causal_idx and emotion keys
    Example:
        {
        "utterances": [
            "Why don 't you watch where you 're going ?",
            "Me?",
            "You 're the one who pulled out in front of me !",
            "There was plenty of room for me to pull out .",
            "You didn 't have to stay in the lane you were in .",
            "Hey , listen . I had every right to stay in the lane I was in .",
            "You were supposed to wait until I passed to pull out .",
            "And anyhow , you didn 't give me any time to change lanes.",
            "All of a sudden--BANG--there you are right in front of me ."
        ],
        "causal_idx": [0, 0, 1, 0, 0, 0, 1, 0, 1],
        "emotion": 'anger'
        }
    """

    req_body = request.get_json()
    context = req_body["context"]
    emotion = req_body["emotion"]

    input_batch = get_all_evidence_utterance_from_conversation(
        emotion=emotion, conversation_history=context
    )
    tensor_dict = preprocessor(input_batch)
    raw_output = model(**tensor_dict)
    output = postprocessor(raw_output)

    return jsonify(utterances=context, causal_idx=output, emotion=emotion)


if __name__ == "__main__":
    app.run()
