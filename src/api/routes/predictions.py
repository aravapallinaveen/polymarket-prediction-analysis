"""Prediction endpoints."""
from flask import Blueprint, jsonify, request
from src.pipeline.rds_manager import RDSManager

predictions_bp = Blueprint("predictions", __name__)
rds = RDSManager()


@predictions_bp.route("/latest", methods=["GET"])
def get_latest_predictions():
    """Get latest predictions for all active markets."""
    limit = request.args.get("limit", 50, type=int)
    sql = """
        SELECT p.market_id, m.question, m.category,
               p.model_name, p.predicted_probability,
               p.prediction_date, p.brier_score
        FROM predictions p
        JOIN markets m ON p.market_id = m.market_id
        WHERE p.prediction_date = (
            SELECT MAX(prediction_date) FROM predictions
        )
        ORDER BY p.predicted_probability DESC
        LIMIT :limit
    """
    df = rds.query(sql, {"limit": limit})
    return jsonify(df.to_dict(orient="records"))


@predictions_bp.route("/market/<market_id>", methods=["GET"])
def get_market_predictions(market_id: str):
    """Get prediction history for a specific market."""
    sql = """
        SELECT p.*, m.question, m.category, m.resolution_outcome
        FROM predictions p
        JOIN markets m ON p.market_id = m.market_id
        WHERE p.market_id = :market_id
        ORDER BY p.prediction_date
    """
    df = rds.query(sql, {"market_id": market_id})
    return jsonify(df.to_dict(orient="records"))


@predictions_bp.route("/compare", methods=["GET"])
def compare_models():
    """Compare predictions across models for the same markets."""
    sql = """
        SELECT p.market_id, m.question, p.model_name,
               p.predicted_probability, p.actual_outcome, p.brier_score
        FROM predictions p
        JOIN markets m ON p.market_id = m.market_id
        WHERE p.actual_outcome IS NOT NULL
        ORDER BY p.market_id, p.model_name
    """
    df = rds.query(sql)
    return jsonify(df.to_dict(orient="records"))
