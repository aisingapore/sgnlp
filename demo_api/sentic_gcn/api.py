import re
from transformers import cached_path

import torch.nn.functional as F

from flask import request, jsonify

from demo_api.common import create_api
from sgnlp.models.sentic_gcn import (
    SenticGCNBertModel,
    SenticGCNBertConfig,
    SenticGCNBertTokenizer,
    SenticGCNBertPreprocessor
    )

from sgnlp.models.sentic_gcn.postprocess import SenticGCNBertPostprocessor

from flask import request
import os


app = create_api(app_name=__name__, model_card_path="model_card/sentic_gcn.json")

# path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'senticnet5.pickle')
preprocessor = SenticGCNBertPreprocessor(senticnet='https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticnet.pickle', device='cpu')

# Load model
config = SenticGCNBertConfig.from_pretrained("https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_bert/config.json") # Input JSON file

model = SenticGCNBertModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_bert/pytorch_model.bin",
    config=config
)

app.logger.info("Preprocessing pipeline and model initialization complete.")

@app.route("/predict", methods=["POST"])
def predict():
    req_body = request.get_json()
    print(req_body)
    aspect = req_body["aspect"]
    sentence = req_body["sentence"]
    
    print('aspect: ',aspect)
    print('sentence: ',sentence)
    
    inputs = list()
    inputs.append(req_body)
    
    # Perform preprocessing from the imported pipeline
    processed_inputs, processed_indices = preprocessor(inputs)
    outputs = model(processed_indices)
    t_probs = F.softmax(outputs.logits)
    t_probs = t_probs.detach().numpy()

    infer_label = [t_probs.argmax(axis=-1)[idx] -1 for idx in range(len(t_probs))]
    
    # Postprocessing
    postprocessor = SenticGCNBertPostprocessor()
    post_outputs = postprocessor(processed_inputs=processed_inputs, model_outputs=outputs)
    
    print('post_outputs: ',post_outputs)
    return jsonify(post_outputs) # to fix the output
    

if __name__ == "__main__":
    # app.run()
    app.run(host="0.0.0.0", debug=True, port=8000)