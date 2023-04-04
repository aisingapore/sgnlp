from flask import jsonify, request

from demo_api.common import create_api
from sgnlp.models.rumour_stance import (
    StanceClassificationConfig,
    StanceClassificationModel,
    StanceClassificationPostprocessor,
    StanceClassificationPreprocessor,
    StanceClassificationTokenizer,
)

app = create_api(
    app_name=__name__, model_card_path="model_card/stance_classification.json"
)

config = StanceClassificationConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/rumour_stance/stance_classification/config.json"
)

model = StanceClassificationModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/rumour_stance/stance_classification/pytorch_model.bin",
    config=config,
)

tokenizer = StanceClassificationTokenizer.from_pretrained(
    "bert-base-uncased",
    do_lower_case=False,
)

preprocessor = StanceClassificationPreprocessor(tokenizer=tokenizer)

postprocessor = StanceClassificationPostprocessor()


@app.route("/predict", methods=["POST"])
def predict():
    inputs = request.get_json()

    processed_inputs = preprocessor(inputs)

    outputs = model(*processed_inputs)

    return jsonify(postprocessor(inputs, outputs))


if __name__ == "__main__":
    app.run()
