from flask import jsonify, request

from demo_api.common import create_api
from sgnlp.models.rumour_stance import (
    RumourVerificationConfig,
    RumourVerificationModel,
    RumourVerificationPostprocessor,
    RumourVerificationPreprocessor,
    RumourVerificationTokenizer,
)

app = create_api(
    app_name=__name__, model_card_path="model_card/rumour_verification.json"
)

config = RumourVerificationConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/rumour_stance/rumour_verification/config.json"
)

model = RumourVerificationModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/rumour_stance/rumour_verification/pytorch_model.bin",
    config=config,
)

tokenizer = RumourVerificationTokenizer.from_pretrained(
    "bert-base-uncased",
    do_lower_case=False,
)

preprocessor = RumourVerificationPreprocessor(tokenizer=tokenizer)

postprocessor = RumourVerificationPostprocessor()


@app.route("/predict", methods=["POST"])
def predict():
    inputs = request.get_json()

    processed_inputs = preprocessor(inputs)

    outputs = model(*processed_inputs)

    return jsonify(postprocessor(outputs))


if __name__ == "__main__":
    app.run()
