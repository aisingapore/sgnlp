import logging
from flask import request

from demo_api.common import create_api
from sgnlp.models.lif_3way_ap import Lif3WayApModel
from sgnlp.models.lif_3way_ap.modules.allennlp.model import Lif3WayApAllenNlpModel
from sgnlp.models.lif_3way_ap.modules.allennlp.predictor import Lif3WayApPredictor
from sgnlp.models.lif_3way_ap.modules.allennlp.dataset_reader import (
    Lif3WayApDatasetReader,
)

app = create_api(app_name=__name__, model_card_path="model_card/lif_3way_ap.json")

gunicorn_logger = logging.getLogger("gunicorn.error")
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

# Load model
model = Lif3WayApModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/lif_3way_ap/model.tar.gz",
    predictor_name="lif_3way_ap_predictor",
)

app.logger.info("Initialization complete.")


@app.route("/predict", methods=["POST"])
def predict():
    req_body = request.get_json()
    output = model.predict_batch_json([req_body])
    return {"probability": output["label_probs"].item()}


if __name__ == "__main__":
    app.run()
