from flask import request
from flask import request
from transformers import cached_path

from demo_api.common import create_api
from sgnlp.models.sentic_gcn import SenticGCNModel

app = create_api(app_name=__name__, model_card_path="model_card/sentic_gcn.json")

# Download files from azure blob storage
#rel2id_path = cached_path("https://storage.googleapis.com/sgnlp/models/lsr/rel2id.json")


# Load model

config =

model = 


app.logger.info("Preprocessing pipeline and model initialization complete.")

@app.route("/predict", methods=["POST"])
def predict():
    req_body = request.get_json()
    document = req_body["document"]
    
    # Perform preprocessing from the imported pipeline
    

if __name__ == "__main__":
    app.run()