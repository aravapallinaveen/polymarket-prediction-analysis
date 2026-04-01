"""Feature inspection endpoints."""
from flask import Blueprint, jsonify, request
from src.pipeline.rds_manager import RDSManager

features_bp = Blueprint("features", __name__)
rds = RDSManager()


@features_bp.route("/importance", methods=["GET"])
def get_feature_importance():
    """Get stored feature importance rankings."""
    sql = """
        SELECT feature_name, importance_score, importance_rank
        FROM model_metadata
        WHERE model_name = :model
        ORDER BY importance_rank
    """
    model = request.args.get("model", "xgboost")
    df = rds.query(sql, {"model": model})
    return jsonify(df.to_dict(orient="records"))


@features_bp.route("/market/<market_id>", methods=["GET"])
def get_market_features(market_id: str):
    """Get latest feature snapshot for a market."""
    sql = """
        SELECT features, snapshot_date
        FROM feature_store
        WHERE market_id = :market_id
        ORDER BY snapshot_date DESC
        LIMIT 1
    """
    df = rds.query(sql, {"market_id": market_id})
    if df.empty:
        return jsonify({"error": "No features found"}), 404
    return jsonify(df.iloc[0].to_dict())


@features_bp.route("/catalog", methods=["GET"])
def get_feature_catalog():
    """Get the full feature catalog with descriptions."""
    catalog = {
        "market_features": {
            "current_price": "Latest market price",
            "price_momentum_7d": "7-day price change",
            "price_momentum_14d": "14-day price change",
            "price_momentum_30d": "30-day price change",
            "volatility_7d": "7-day return volatility",
            "volatility_14d": "14-day return volatility",
            "volatility_30d": "30-day return volatility",
            "price_percentile_30d": "Price position in 30-day range",
            "price_percentile_60d": "Price position in 60-day range",
            "ma_7_30_crossover": "MA(7) minus MA(30)",
            "price_above_ma30": "Binary: price above 30-day MA",
            "trade_count_24h": "Number of trades",
            "volume_total": "Total trade volume",
            "volume_mean": "Average trade size",
            "vwap": "Volume-weighted average price",
            "volume_zscore": "Volume z-score vs. historical",
            "spread": "Bid-ask spread",
            "bid_depth": "Top-5 bid depth",
            "ask_depth": "Top-5 ask depth",
            "bid_ask_imbalance": "Order book imbalance ratio",
            "price_distance_from_50": "Distance from 50% probability",
            "is_extreme_price": "Binary: price < 10% or > 90%",
        },
        "sentiment_features": {
            "sentiment_mean_vader": "Mean VADER compound score",
            "sentiment_std_vader": "VADER standard deviation",
            "sentiment_mean_bert": "Mean BERT score",
            "sentiment_weighted_vader": "Engagement-weighted sentiment",
            "sentiment_momentum": "Recent vs. older sentiment shift",
            "sentiment_slope": "Linear trend in sentiment",
            "mention_count": "Total mentions across sources",
            "mention_velocity": "Mentions per unique day",
            "sentiment_dispersion": "Disagreement measure",
            "positive_ratio": "Fraction of positive mentions",
            "sentiment_mean_reddit": "Reddit-specific sentiment",
            "sentiment_mean_twitter": "Twitter-specific sentiment",
        },
        "trend_features": {
            "trend_current": "Latest Google Trends value",
            "trend_mean": "Average search interest",
            "trend_max": "Peak search interest",
            "trend_slope_7d": "7-day interest slope",
            "trend_slope_30d": "30-day interest slope",
            "trend_acceleration": "Change in trend slope",
            "trend_peak_ratio": "Current / peak interest",
            "trend_breakout_flag": "Binary: >2 std above mean",
            "trend_relative_volume": "Recent vs. 30-day average interest",
        },
        "interaction_features": {
            "sentiment_x_volume": "Sentiment x volume z-score",
            "trend_x_momentum": "Trend slope x price momentum",
            "sentiment_price_divergence": "Sentiment vs. price direction",
            "divergence_flag": "Binary divergence signal",
            "smart_money_indicator": "Volume x low dispersion",
            "attention_adjusted_price": "Price adjusted by search interest",
            "consensus_strength": "Agreement x stability",
            "contrarian_signal": "High bullish sentiment + low price",
        },
    }
    return jsonify(catalog)
