"""Evaluation endpoints for dashboard consumption."""
from flask import Blueprint, jsonify, request
import numpy as np

from src.pipeline.rds_manager import RDSManager
from src.evaluation.brier_score import BrierScoreAnalyzer
from src.evaluation.bias_analysis import BiasAnalyzer
from src.models.calibration import CalibrationAnalyzer

evaluation_bp = Blueprint("evaluation", __name__)
rds = RDSManager()
brier = BrierScoreAnalyzer()
bias = BiasAnalyzer()
calibration = CalibrationAnalyzer()


@evaluation_bp.route("/brier-scores", methods=["GET"])
def get_brier_scores():
    """Get Brier Score summary with decomposition."""
    model_name = request.args.get("model", "xgboost")
    sql = """
        SELECT predicted_probability, actual_outcome
        FROM predictions
        WHERE model_name = :model AND actual_outcome IS NOT NULL
    """
    df = rds.query(sql, {"model": model_name})

    if df.empty:
        return jsonify({"error": "No resolved predictions found"}), 404

    y_true = df["actual_outcome"].values.astype(float)
    y_prob = df["predicted_probability"].values.astype(float)

    decomposition = brier.decompose(y_true, y_prob)
    skill = brier.skill_score(y_true, y_prob)

    return jsonify({**decomposition, "skill_score": skill})


@evaluation_bp.route("/calibration", methods=["GET"])
def get_calibration_data():
    """Get calibration curve data."""
    model_name = request.args.get("model", "xgboost")
    n_bins = request.args.get("bins", 10, type=int)

    sql = """
        SELECT predicted_probability, actual_outcome
        FROM predictions
        WHERE model_name = :model AND actual_outcome IS NOT NULL
    """
    df = rds.query(sql, {"model": model_name})

    if df.empty:
        return jsonify({"error": "No data"}), 404

    y_true = df["actual_outcome"].values.astype(float)
    y_prob = df["predicted_probability"].values.astype(float)

    curve = calibration.compute_calibration_curve(y_true, y_prob, n_bins=n_bins)
    return jsonify(curve)


@evaluation_bp.route("/bias/favorite-longshot", methods=["GET"])
def get_favorite_longshot_bias():
    """Get favorite-longshot bias analysis."""
    sql = """
        SELECT predicted_probability, actual_outcome
        FROM predictions
        WHERE actual_outcome IS NOT NULL
    """
    df = rds.query(sql)

    if df.empty:
        return jsonify({"error": "No data"}), 404

    y_true = df["actual_outcome"].values.astype(float)
    y_prob = df["predicted_probability"].values.astype(float)

    result = bias.favorite_longshot_bias(y_true, y_prob)
    return jsonify(result.to_dict(orient="records"))


@evaluation_bp.route("/bias/category", methods=["GET"])
def get_category_bias():
    """Get bias analysis by market category."""
    sql = """
        SELECT p.predicted_probability, p.actual_outcome, m.category
        FROM predictions p
        JOIN markets m ON p.market_id = m.market_id
        WHERE p.actual_outcome IS NOT NULL AND m.category IS NOT NULL
    """
    df = rds.query(sql)

    if df.empty:
        return jsonify({"error": "No data"}), 404

    result = bias.category_bias(
        df,
        prob_col="predicted_probability",
        outcome_col="actual_outcome",
        category_col="category",
    )
    return jsonify(result.to_dict(orient="records"))


@evaluation_bp.route("/time-horizon", methods=["GET"])
def get_time_horizon_analysis():
    """Get Brier Score by time horizon."""
    sql = """
        SELECT p.predicted_probability, p.actual_outcome,
               EXTRACT(DAY FROM m.end_date - p.prediction_date) as days_to_resolution
        FROM predictions p
        JOIN markets m ON p.market_id = m.market_id
        WHERE p.actual_outcome IS NOT NULL
    """
    df = rds.query(sql)

    if df.empty:
        return jsonify({"error": "No data"}), 404

    result = brier.by_time_horizon(
        df,
        prob_col="predicted_probability",
        outcome_col="actual_outcome",
        time_col="days_to_resolution",
    )
    return jsonify(result.to_dict(orient="records"))
