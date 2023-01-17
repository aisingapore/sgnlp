from flask import request

from demo_api.common import create_api
from sgnlp.models.rst_pointer import (
    RstPointerParserConfig,
    RstPointerParserModel,
    RstPointerSegmenterConfig,
    RstPointerSegmenterModel,
    RstPreprocessor,
    RstPostprocessor,
)

app = create_api(app_name=__name__, model_card_path="model_card/rst_pointer.json")

# Load processors and models
preprocessor = RstPreprocessor()
postprocessor = RstPostprocessor()

segmenter_config = RstPointerSegmenterConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp-models/models/rst_pointer/segmenter/config.json"
)
segmenter = RstPointerSegmenterModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp-models/models/rst_pointer/segmenter/pytorch_model.bin",
    config=segmenter_config,
)
segmenter.eval()

parser_config = RstPointerParserConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp-models/models/rst_pointer/parser/config.json"
)
parser = RstPointerParserModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp-models/models/rst_pointer/parser/pytorch_model.bin",
    config=parser_config,
)
parser.eval()

app.logger.info("Model initialization complete")


@app.route("/predict", methods=["POST"])
def predict():
    req_body = request.get_json()
    sentence = req_body["sentence"]

    sentence = [sentence]  # Treat it as a batch size of 1
    tokenized_sentence_ids, tokenized_sentence, length = preprocessor(sentence)

    segmenter_output = segmenter(tokenized_sentence_ids, length)
    end_boundaries = segmenter_output.end_boundaries

    parser_output = parser(tokenized_sentence_ids, end_boundaries, length)

    hierplane_tree = postprocessor(
        sentences=sentence,
        tokenized_sentences=tokenized_sentence,
        end_boundaries=end_boundaries,
        discourse_tree_splits=parser_output.splits,
    )

    return {"tree": hierplane_tree[0]}


if __name__ == "__main__":
    app.run()
