import json

from flask import request, jsonify

from demo_api.common import create_api
from sgnlp.models.span_extraction import (
    RecconSpanExtractionConfig,
    RecconSpanExtractionModel,
    RecconSpanExtractionTokenizer,
    RecconSpanExtractionPreprocessor,
    RecconSpanExtractionPostprocessor,
)
from sgnlp.models.span_extraction.utils import (
    get_all_evidence_utterance_from_conversation,
)

app = create_api(app_name=__name__, model_card_path="model_card/span_extraction.json")

config = RecconSpanExtractionConfig.from_pretrained(
    "https://sgnlp.blob.core.windows.net/models/reccon_span_extraction/config.json"
)
tokenizer = RecconSpanExtractionTokenizer.from_pretrained(
    "mrm8488/spanbert-finetuned-squadv2"
)
model = RecconSpanExtractionModel.from_pretrained(
    "https://sgnlp.blob.core.windows.net/models/reccon_span_extraction/pytorch_model.bin",
    config=config,
)
preprocessor = RecconSpanExtractionPreprocessor(tokenizer)
postprocessor = RecconSpanExtractionPostprocessor()


@app.route("/predict", methods=["POST"])
def predict():
    """Iterate through each evidence utt in context to perform RECCON span extraction.
    The last utterance in context is used as the target utterance.

    Inputs:
        A json with 'context' and 'emotion' keys
    Example:
        {"context": [
            "Linda ? Is that you ? I haven't seen you in ages !",
            "Hi George ! It's good to see you !"
        ],
        "emotion": "surprise"}

    Returns:
        json: return a json which consists of the emotion, evidence span index,
                the probability and utterances broken up into spans
    Example:
        {"emotion": "surprise",
        "evidence_span": [[0,1],
                        [0, 1]],
        "probability": [[-1, 0.943615029866203],
                        [-1, 0.8712913786944898]],
        "utterances": [["Linda ? Is that you ? ",
                        "I haven't seen you in ages !"],
                        ["Hi George ! ",
                        "It's good to see you !"]]}
    """
    req_body = request.get_json()
    conversation_history = req_body["context"]
    emotion = req_body["emotion"]

    input_batch = get_all_evidence_utterance_from_conversation(
        emotion=emotion, conversation_history=conversation_history
    )
    tensor_dict, evidences, examples, features = preprocessor(input_batch)
    raw_output = model(**tensor_dict)
    context, evidence_span, probability = postprocessor(
        raw_output, evidences, examples, features
    )

    return jsonify(
        {
            "utterances": context,
            "evidence_span": evidence_span,
            "probability": probability,
            "emotion": req_body["emotion"],
        }
    )


if __name__ == "__main__":
    app.run()
