from flask import request

from demo_api.common import create_api
from sgnlp.models.csgec import (
    CsgConfig,
    CsgModel,
    CsgTokenizer,
    CsgecPreprocessor,
    CsgecPostprocessor,
    download_tokenizer_files,
)

app = create_api(app_name=__name__, model_card_path="model_card/csgec.json")

config = CsgConfig.from_pretrained("https://storage.googleapis.com/sgnlp/models/csgec/config.json")
model = CsgModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/csgec/pytorch_model.bin",
    config=config,
)
download_tokenizer_files(
    "https://storage.googleapis.com/sgnlp/models/csgec/src_tokenizer/",
    "csgec_src_tokenizer",
)
download_tokenizer_files(
    "https://storage.googleapis.com/sgnlp/models/csgec/ctx_tokenizer/",
    "csgec_ctx_tokenizer",
)
download_tokenizer_files(
    "https://storage.googleapis.com/sgnlp/models/csgec/tgt_tokenizer/",
    "csgec_tgt_tokenizer",
)
src_tokenizer = CsgTokenizer.from_pretrained("csgec_src_tokenizer")
ctx_tokenizer = CsgTokenizer.from_pretrained("csgec_ctx_tokenizer")
tgt_tokenizer = CsgTokenizer.from_pretrained("csgec_tgt_tokenizer")

preprocessor = CsgecPreprocessor(src_tokenizer=src_tokenizer, ctx_tokenizer=ctx_tokenizer)
postprocessor = CsgecPostprocessor(tgt_tokenizer=tgt_tokenizer)

app.logger.info('Model initialization complete.')


@app.route("/predict", methods=["POST"])
def predict():
    req_body = request.get_json()
    text = req_body["text"]

    batch_source_ids, batch_context_ids = preprocessor([text])
    predicted_ids = model.decode(batch_source_ids, batch_context_ids)
    predicted_texts = postprocessor(predicted_ids)

    output = {
        "original_text": text,
        "corrected_text": predicted_texts[0]
    }

    return output


if __name__ == "__main__":
    app.run()
