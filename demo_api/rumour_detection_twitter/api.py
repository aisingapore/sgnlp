import pathlib
import json

from flask import jsonify, request
import torch
from torch.nn.functional import softmax

from demo_api.common import create_api
from create_inputs import generate_structure
from sgnlp.models.rumour_detection_twitter import (
    RumourDetectionTwitterConfig,
    RumourDetectionTwitterModel,
    RumourDetectionTwitterTokenizer,
    download_tokenizer_files_from_azure,
)


app = create_api(app_name=__name__, model_card_path="model_card/rumour.json")

config = RumourDetectionTwitterConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/rumour_detection_twitter/config.json"
)
model = RumourDetectionTwitterModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/rumour_detection_twitter/pytorch_model.bin",
    config=config,
)
download_tokenizer_files_from_azure(
    "https://storage.googleapis.com/sgnlp/models/rumour_detection_twitter/",
    "rumour_tokenizer",
)
tokenizer = RumourDetectionTwitterTokenizer.from_pretrained("rumour_tokenizer")

id_to_string = {
    0: "a false rumour",
    1: "a true rumour",
    2: "an unverified rumour",
    3: "a non-rumour",
}


@app.route("/predict", methods=["POST"])
def predict():
    """POST method to run inference against rumour detection model.

    Returns:
        JSON: return the probability distribution across the possible classes.
    """

    # Generate the inputs in the correct formats
    tweet_lst = request.get_json()["tweets"]
    thread_len = len(tweet_lst)
    token_ids, token_attention_mask = tokenizer.tokenize_threads(
        [tweet_lst],
        max_length=config.max_length,
        max_posts=config.max_tweets,
        truncation=True,
        padding="max_length",
    )

    time_delay_ids, structure_ids, post_attention_mask = generate_structure(
        thread_len=thread_len, max_posts=config.max_tweets
    )

    token_ids = torch.LongTensor(token_ids)
    token_attention_mask = torch.Tensor(token_attention_mask)
    time_delay_ids = torch.LongTensor(time_delay_ids)
    post_attention_mask = torch.Tensor(post_attention_mask)
    structure_ids = torch.LongTensor(structure_ids)

    # Get the raw logits of predictions. Note that the model assumes the input exists as a batch. The returned outputs will be for a batch too.
    logits = model(
        token_ids=token_ids,
        time_delay_ids=time_delay_ids,
        structure_ids=structure_ids,
        token_attention_mask=token_attention_mask,
        post_attention_mask=post_attention_mask,
    ).logits

    # Convert the outputs into the format the frontend accepts
    probabilities = softmax(logits, dim=1)
    predicted_y = torch.argmax(logits, dim=1)[0]
    predicted_y = id_to_string[int(predicted_y)]
    predicted_prob = round(float(torch.max(probabilities)) * 100, 1)

    return jsonify({"predicted_y": predicted_y, "predicted_prob": predicted_prob})


if __name__ == "__main__":
    app.run()
