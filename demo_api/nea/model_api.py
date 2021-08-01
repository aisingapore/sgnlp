import pathlib
import json

from flask import Flask, jsonify, request

from sgnlp_models.models.nea import NEAConfig, NEARegPoolingModel, NEATokenizer, NEAPreprocessor
from sgnlp_models.models.nea.utils import convert_to_dataset_friendly_scores

app = Flask(__name__)

nea_config = NEAConfig.from_pretrained('https://sgnlp.blob.core.windows.net/models/nea/config.json')
nea_model = NEARegPoolingModel.from_pretrained('https://sgnlp.blob.core.windows.net/models/nea/pytorch_model.bin',
                                               config=nea_config)
nea_tokenizer = NEATokenizer.from_pretrained('nea_tokenizer')
nea_preprocessor = NEAPreprocessor(tokenizer=nea_tokenizer)


@app.route("/model-card", methods=["GET"])
def get_model_card():
    """GET method for model card

    Returns:
        json: return the model card in json format
    """
    model_card_path = pathlib.Path(__file__).parent / "model_card/nea.json"
    with open(model_card_path) as f:
        model_card = json.load(f)
    return jsonify(**model_card)


@app.route("/predict", methods=["POST"])
def predict():
    """POST method to run inference against NEA model.

    Returns:
        JSON: return the friendly score for prompt_id 1.
    """
    req = request.get_json()
    text = req['essay']
    tokens = nea_preprocessor([text])
    score = nea_model(**tokens)
    friendly_score = int(convert_to_dataset_friendly_scores(score.logits.detach().numpy(), 1))
    return jsonify({"predictions": [friendly_score]})


if __name__ == "__main__":
    app.run(host="0.0.0.0")
