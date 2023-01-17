from flask import jsonify, request

from demo_api.common import create_api
from sgnlp.models.nea import (
    NEAConfig,
    NEARegPoolingModel,
    NEATokenizer,
    NEAPreprocessor,
)
from sgnlp.models.nea.utils import convert_to_dataset_friendly_scores


app = create_api(app_name=__name__, model_card_path="model_card/nea.json")

nea_config = NEAConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp-models/models/nea/config.json"
)
nea_model = NEARegPoolingModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp-models/models/nea/pytorch_model.bin",
    config=nea_config,
)
nea_tokenizer = NEATokenizer.from_pretrained("nea_tokenizer")
nea_preprocessor = NEAPreprocessor(tokenizer=nea_tokenizer)


@app.route("/predict", methods=["POST"])
def predict():
    """POST method to run inference against NEA model.

    Returns:
        JSON: return the friendly score for prompt_id 1.
    """
    req = request.get_json()
    text = req["essay"]
    tokens = nea_preprocessor([text])
    score = nea_model(**tokens)
    friendly_score = int(
        convert_to_dataset_friendly_scores(score.logits.detach().numpy(), 1)
    )
    return jsonify({"predictions": [friendly_score]})


if __name__ == "__main__":
    app.run()
