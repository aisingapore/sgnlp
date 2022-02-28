from flask import request
from transformers import BertTokenizer

from demo_api.common import create_api
from sgnlp.models.dual_bert import (
    DualBert,
    DualBertConfig,
    DualBertPreprocessor,
    DualBertPostprocessor
)

app = create_api(app_name=__name__, model_card_path="model_card/rst_pointer.json")

# Load processors and models
config = DualBertConfig.from_pretrained("https://storage.googleapis.com/sgnlp/models/dual_bert/config.json")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
preprocessor = DualBertPreprocessor(config, tokenizer)
model = DualBert.from_pretrained("https://storage.googleapis.com/sgnlp/models/dual_bert/pytorch_model.bin",
                                 config=config)
postprocessor = DualBertPostprocessor()

model.eval()

app.logger.info("Model initialization complete")

@app.route("/predict", methods=["POST"])
def predict():
    req_body = request.get_json()
    examples = req_body["posts"]

    examples = [examples]  # Treat it as a batch size of 1

    model_inputs = preprocessor(examples)
    model_output = model(**model_inputs)
    outputs = postprocessor(model_output, model_inputs["stance_label_mask"])


    return {"rumor_labels": outputs["rumor_labels"][0],
            "stance_labels": outputs["stance_labels"][0],
            "text": examples[0]}


if __name__ == "__main__":
    app.run()
