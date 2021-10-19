import logging
from flask import request
from transformers import cached_path

from demo_api.common import create_api
from sgnlp.models.lif_3way_ap import LIF3WayAPModel, LIF3WayAPConfig, LIF3WayAPPreprocessor


app = create_api(app_name=__name__, model_card_path="model_card/lif_3way_ap.json")

gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

# Load model
config = LIF3WayAPConfig.from_pretrained('https://storage.googleapis.com/sgnlp/models/lif_3way_ap/config.json')
model = LIF3WayAPModel.from_pretrained('https://storage.googleapis.com/sgnlp/models/lif_3way_ap/pytorch_model.bin',
                                       config=config)
model.eval()

# Load preprocessor
word_vocab_path = cached_path('https://storage.googleapis.com/sgnlp/models/lif_3way_ap/word_vocab.pt')
char_vocab_path = cached_path('https://storage.googleapis.com/sgnlp/models/lif_3way_ap/char_vocab.pt')

preprocessor = LIF3WayAPPreprocessor(min_word_padding_size=config.char_embedding_args["kernel_size"])
preprocessor.load_vocab(word_vocab_path, char_vocab_path)

app.logger.info('Initialization complete.')


@app.route('/predict', methods=['POST'])
def predict():
    req_body = request.get_json()
    tensor_dict = preprocessor([req_body])
    output = model(**tensor_dict)
    return {"probability": output["label_probs"].item()}


if __name__ == '__main__':
    app.run()
