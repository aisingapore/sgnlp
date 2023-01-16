from flask import request, jsonify

from demo_api.common import create_api
from sgnlp.models.sentic_gcn import (
    SenticGCNBertModel,
    SenticGCNBertConfig,
    SenticGCNBertPreprocessor,
    SenticGCNBertPostprocessor,
)

app = create_api(app_name=__name__, model_card_path="model_card/sentic_gcn.json")

preprocessor = SenticGCNBertPreprocessor(
    senticnet="https://storage.googleapis.com/sgnlp-models/models/sentic_gcn/senticnet.pickle", device="cpu"
)

postprocessor = SenticGCNBertPostprocessor()

# Load model
config = SenticGCNBertConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp-models/models/sentic_gcn/senticgcn_bert/config.json"
)

model = SenticGCNBertModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp-models/models/sentic_gcn/senticgcn_bert/pytorch_model.bin", config=config
)

app.logger.info("Preprocessing pipeline and model initialization complete.")


@app.route("/predict", methods=["POST"])
def predict():
    req_body = request.get_json()

    # Preprocessing
    processed_inputs, processed_indices = preprocessor([req_body])
    outputs = model(processed_indices)

    # Postprocessing
    post_outputs = postprocessor(processed_inputs=processed_inputs, model_outputs=outputs)

    return jsonify(post_outputs[0])


if __name__ == "__main__":
    app.run()
