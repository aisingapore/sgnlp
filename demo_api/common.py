import logging
import json
from flask import Flask, jsonify, make_response


def create_api(app_name, model_card_path):
    app = Flask(app_name)

    # setup gunicorn logging
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

    # Common routes
    @app.route("/model-card", methods=["GET"])
    def get_model_card():
        """GET method for model card

        Returns:
            json: model card in json format
        """
        with open(model_card_path) as f:
            model_card = json.load(f)
        return jsonify(**model_card)

    # Kubernetes health check endpoint
    @app.route("/healthz", methods=["GET"])
    def healthz():
        return make_response(jsonify({"healthy": True}), 200)

    return app
