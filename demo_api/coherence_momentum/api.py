from flask import request

from demo_api.common import create_api
from sgnlp.models.coherence_momentum import (
    CoherenceMomentumModel,
    CoherenceMomentumConfig,
    CoherenceMomentumPreprocessor
)

app = create_api(app_name=__name__, model_card_path="model_card/coherence_momentum.json")

# Load processors and models
config = CoherenceMomentumConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/coherence_momentum/config.json"
)
model = CoherenceMomentumModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/coherence_momentum/pytorch_model.bin",
    config=config
)

preprocessor = CoherenceMomentumPreprocessor(config.model_size, config.max_len)

app.logger.info("Model initialization complete")


@app.route("/predict", methods=["POST"])
def predict():
    req_body = request.get_json()

    text1 = req_body["text1"]
    text2 = req_body["text2"]

    text1_tensor = preprocessor([text1])
    text2_tensor = preprocessor([text2])

    text1_score = model.get_main_score(text1_tensor["tokenized_texts"]).item()
    text2_score = model.get_main_score(text2_tensor["tokenized_texts"]).item()

    return {
        "text1_score": text1_score,
        "text2_score": text2_score
    }


if __name__ == "__main__":
    app.run()
