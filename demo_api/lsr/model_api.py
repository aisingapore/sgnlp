import json
import logging
from flask import Flask, request, jsonify
from transformers import cached_path

from sgnlp_models.models.lsr import LsrModel, LsrConfig, LsrPreprocessor, LsrPostprocessor
from text_input_to_docred_pipeline import TextInputToDocredPipeline

app = Flask(__name__)

gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

# Download files from azure blob storage
rel2id_path = cached_path('https://sgnlp.blob.core.windows.net/models/lsr/rel2id.json')
word2id_path = cached_path('https://sgnlp.blob.core.windows.net/models/lsr/word2id.json')
ner2id_path = cached_path('https://sgnlp.blob.core.windows.net/models/lsr/ner2id.json')
rel_info_path = cached_path('https://sgnlp.blob.core.windows.net/models/lsr/rel_info.json')

# Optimal threshold value found during training
PRED_THRESHOLD = 0.324

# Load processors
text2docred_pipeline = TextInputToDocredPipeline()
preprocessor = LsrPreprocessor(rel2id_path=rel2id_path, word2id_path=word2id_path, ner2id_path=ner2id_path)
postprocessor = LsrPostprocessor.from_file_paths(rel2id_path=rel2id_path, rel_info_path=rel_info_path,
                                                 pred_threshold=PRED_THRESHOLD)

# Load model
config = LsrConfig.from_pretrained('https://sgnlp.blob.core.windows.net/models/lsr/config.json')
model = LsrModel.from_pretrained('https://sgnlp.blob.core.windows.net/models/lsr/pytorch_model.bin', config=config)
model.eval()

app.logger.info('Preprocessing pipeline and model initialization complete.')


@app.route('/predict', methods=['POST'])
def predict():
    req_body = request.get_json()
    document = req_body['document']

    docred_doc = text2docred_pipeline.preprocess(document)

    # If no entities in found
    if len(docred_doc['vertexSet']) == 0:
        # Skip predict if no entities are found in the document
        return {
            "clusters": [],
            "document": document,
            "relations": []
        }
    else:
        tensor_doc = preprocessor([docred_doc])
        output = model(**tensor_doc)
        return postprocessor(output.prediction[0], docred_doc)


model_card_path = "model_card/lsr.json"


@app.route("/model-card", methods=["GET"])
def get_model_card():
    """GET method for model card

    Returns:
        json: return model card in json format
    """
    with open(model_card_path) as f:
        model_card = json.load(f)
    return jsonify(**model_card)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
