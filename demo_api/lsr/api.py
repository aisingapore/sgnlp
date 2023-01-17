from flask import request
from transformers import cached_path

from demo_api.common import create_api
from sgnlp.models.lsr import LsrModel, LsrConfig, LsrPreprocessor, LsrPostprocessor
from text_input_to_docred_pipeline import TextInputToDocredPipeline

app = create_api(app_name=__name__, model_card_path="model_card/lsr.json")

# Download files from azure blob storage
rel2id_path = cached_path("https://storage.googleapis.com/sgnlp-models/models/lsr/rel2id.json")
word2id_path = cached_path(
    "https://storage.googleapis.com/sgnlp-models/models/lsr/word2id.json"
)
ner2id_path = cached_path("https://storage.googleapis.com/sgnlp-models/models/lsr/ner2id.json")
rel_info_path = cached_path(
    "https://storage.googleapis.com/sgnlp-models/models/lsr/rel_info.json"
)

# Optimal threshold value found during training
PRED_THRESHOLD = 0.324

# Load processors
text2docred_pipeline = TextInputToDocredPipeline()
preprocessor = LsrPreprocessor(
    rel2id_path=rel2id_path, word2id_path=word2id_path, ner2id_path=ner2id_path
)
postprocessor = LsrPostprocessor.from_file_paths(
    rel2id_path=rel2id_path, rel_info_path=rel_info_path, pred_threshold=PRED_THRESHOLD
)

# Load model
config = LsrConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp-models/models/lsr/v2/config.json"
)
model = LsrModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp-models/models/lsr/v2/pytorch_model.bin",
    config=config,
)
model.eval()

app.logger.info("Preprocessing pipeline and model initialization complete.")


@app.route("/predict", methods=["POST"])
def predict():
    req_body = request.get_json()
    document = req_body["document"]

    docred_doc = text2docred_pipeline.preprocess(document)

    # If no entities in found
    if len(docred_doc["vertexSet"]) == 0:
        # Skip predict if no entities are found in the document
        return {"clusters": [], "document": document, "relations": []}
    else:
        tensor_doc = preprocessor([docred_doc])
        output = model(**tensor_doc)
        return postprocessor(output.prediction, [docred_doc])[0]


if __name__ == "__main__":
    app.run()
