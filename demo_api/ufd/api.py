import json
from typing import List

import torch
import torch.nn.functional as F
from flask import request, jsonify

from demo_api.common import create_api
from sgnlp.models.ufd import (
    UFDEmbeddingConfig,
    UFDEmbeddingModel,
    UFDTokenizer,
    UFDModelBuilder,
    UFDPreprocessor,
)

app = create_api(app_name=__name__, model_card_path="model_card/ufd.json")

# Constants
DEVICE = torch.device("cpu")

model_builder = UFDModelBuilder()
models = model_builder.build_model_group()
SOURCE_DOMAIN = model_builder.source_domains  # ['books', 'dvd', 'music']
TARGET_DOMAIN = model_builder.target_domains  # ['books', 'dvd', 'music']
MODELS_GROUP = model_builder.models_group

# This is a mapping table which maps the source domain to all available target domains
# i.e. Source Domain = 'books' -> Target Domains = ['dvd', 'music']
sourcedomain2targetdomains = {
    SOURCE_DOMAIN[0]: [TARGET_DOMAIN[1], TARGET_DOMAIN[2]],
    SOURCE_DOMAIN[1]: [TARGET_DOMAIN[0], TARGET_DOMAIN[2]],
    SOURCE_DOMAIN[2]: [TARGET_DOMAIN[0], TARGET_DOMAIN[1]],
}

# Xlr model
embedding_model_name = "xlm-roberta-large"
xlmr_model_config = UFDEmbeddingConfig.from_pretrained(embedding_model_name)
xlmr_model = UFDEmbeddingModel.from_pretrained(embedding_model_name).to(DEVICE)
xlmr_tokenizer = UFDTokenizer.from_pretrained(embedding_model_name)

preprocessor = UFDPreprocessor(tokenizer=xlmr_tokenizer)

app.logger.info("Initialization complete.")


@app.route("/predict", methods=["POST"])
def predict():
    """POST method to run inference against cross domains models with best performance saved for respective
    target language and target domain

    Returns:
        json: return a json with cross domain as key and probability and sentiment as sub keys for each cross domains
              models
    """
    req_body = request.get_json()
    t_lang = req_body["target_language"]
    t_dom = req_body["target_domain"]
    text = req_body["text"]

    model_keys = get_model_group_keys(t_lang, t_dom)

    results = {}
    for model_name in model_keys:
        s_dom = model_name.split("_")[0]

        text_features = preprocessor([text])
        output = models[model_name](**text_features)

        logits_probabilities = F.softmax(output.logits, dim=1)
        max_output = torch.max(logits_probabilities, axis=1)
        probabilities = max_output.values
        sentiments = max_output.indices
        results[s_dom] = {}
        results[s_dom]["target_language"] = t_lang
        results[s_dom]["target_domain"] = t_dom
        results[s_dom]["probability"] = probabilities.item()
        results[s_dom]["sentiment"] = sentiments.item()
    return jsonify(results)


def get_model_group_keys(t_lang: str, t_dom: str) -> List[str]:
    """Helper method to generate model keys grouping

    Args:
        t_lang (str): input target language
        t_dom (str): input target domain

    Returns:
        List[str]: return list of model groupings
    """
    model_grps = []
    for s_dom in sourcedomain2targetdomains[t_dom]:
        model_grps.append(f"{s_dom}_{t_lang}_{t_dom}")
    return model_grps


if __name__ == "__main__":
    app.run()
