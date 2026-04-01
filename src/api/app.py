"""Flask API serving predictions and evaluation data."""
from flask import Flask, jsonify
from flask_cors import CORS
from loguru import logger

from src.api.routes.predictions import predictions_bp
from src.api.routes.evaluation import evaluation_bp
from src.api.routes.features import features_bp


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    app.register_blueprint(predictions_bp, url_prefix="/api/predictions")
    app.register_blueprint(evaluation_bp, url_prefix="/api/evaluation")
    app.register_blueprint(features_bp, url_prefix="/api/features")

    @app.route("/api/health")
    def health():
        return jsonify({"status": "healthy"})

    logger.info("Flask API initialized")
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
