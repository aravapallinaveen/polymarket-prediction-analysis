"""Train XGBoost and Logistic Regression models on feature store data."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.xgboost_model import XGBoostPredictor
from src.models.logistic_model import LogisticPredictor
from src.models.calibration import CalibrationAnalyzer
from src.evaluation.brier_score import BrierScoreAnalyzer
from src.evaluation.hypothesis_tests import HypothesisTests
from src.config.settings import config


def generate_synthetic_training_data(n_samples: int = 2000, n_features: int = 53) -> tuple:
    """
    Generate realistic synthetic data for model training demonstration.
    In production, this is replaced by FeatureStore.load_training_dataset().
    """
    np.random.seed(config.model.random_state)

    feature_names = [
        # Market features (22)
        "current_price", "price_momentum_7d", "price_momentum_14d", "price_momentum_30d",
        "volatility_7d", "volatility_14d", "volatility_30d",
        "price_percentile_30d", "price_percentile_60d",
        "ma_7_30_crossover", "price_above_ma30",
        "trade_count_24h", "volume_total", "volume_mean", "vwap", "volume_zscore",
        "spread", "bid_depth", "ask_depth", "bid_ask_imbalance",
        "price_distance_from_50", "is_extreme_price",
        # Sentiment features (14)
        "sentiment_mean_vader", "sentiment_std_vader", "sentiment_mean_bert",
        "sentiment_weighted_vader", "sentiment_momentum", "sentiment_slope",
        "mention_count", "mention_velocity", "sentiment_dispersion", "positive_ratio",
        "sentiment_mean_reddit", "sentiment_mean_twitter",
        "mention_count_reddit", "mention_count_twitter",
        # Trend features (9)
        "trend_current", "trend_mean", "trend_max",
        "trend_slope_7d", "trend_slope_30d", "trend_acceleration",
        "trend_peak_ratio", "trend_breakout_flag", "trend_relative_volume",
        # Interaction features (8)
        "sentiment_x_volume", "trend_x_momentum", "sentiment_price_divergence",
        "divergence_flag", "smart_money_indicator", "attention_adjusted_price",
        "consensus_strength", "contrarian_signal",
    ]

    X = pd.DataFrame(np.random.randn(n_samples, len(feature_names)), columns=feature_names)

    # Make features realistic
    X["current_price"] = np.random.uniform(0.05, 0.95, n_samples)
    X["volatility_7d"] = np.abs(X["volatility_7d"]) * 0.05
    X["volatility_14d"] = np.abs(X["volatility_14d"]) * 0.04
    X["volatility_30d"] = np.abs(X["volatility_30d"]) * 0.03
    X["price_above_ma30"] = np.random.binomial(1, 0.55, n_samples)
    X["is_extreme_price"] = (X["current_price"] < 0.1).astype(int) | (X["current_price"] > 0.9).astype(int)
    X["positive_ratio"] = np.clip(X["positive_ratio"] * 0.3 + 0.5, 0, 1)
    X["trend_breakout_flag"] = np.random.binomial(1, 0.1, n_samples)
    X["divergence_flag"] = np.random.binomial(1, 0.15, n_samples)
    X["contrarian_signal"] = np.random.binomial(1, 0.08, n_samples)
    X["mention_count"] = np.abs(X["mention_count"]) * 50
    X["volume_total"] = np.abs(X["volume_total"]) * 10000
    X["trade_count_24h"] = np.abs(X["trade_count_24h"]) * 100
    X["trend_current"] = np.random.uniform(0, 100, n_samples)

    # Generate realistic outcome: correlated with price + some noise
    logit = (
        1.5 * (X["current_price"] - 0.5)
        + 0.5 * X["price_momentum_7d"]
        + 0.3 * X["sentiment_mean_vader"]
        + 0.2 * X["trend_slope_7d"]
        - 0.4 * X["volatility_7d"]
        + np.random.normal(0, 0.5, n_samples)
    )
    prob = 1 / (1 + np.exp(-logit))
    y = pd.Series(np.random.binomial(1, prob), name="outcome")

    return X, y


def main():
    logger.info("=" * 60)
    logger.info("POLYMARKET MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    # --- Generate / load data ---
    logger.info("Loading training data...")
    X, y = generate_synthetic_training_data(n_samples=2000)
    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.model.test_size,
        random_state=config.model.random_state,
        stratify=y,
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # --- Train XGBoost with Optuna tuning ---
    logger.info("\n--- Training XGBoost (with Optuna tuning) ---")
    xgb_model = XGBoostPredictor()
    xgb_model.tune(X_train, y_train, n_trials=30)
    xgb_model.train(X_train, y_train)
    xgb_metrics = xgb_model.evaluate(X_test, y_test)

    # --- Train Logistic Regression ---
    logger.info("\n--- Training Logistic Regression ---")
    lr_model = LogisticPredictor()
    lr_model.train(X_train, y_train)
    lr_metrics = lr_model.evaluate(X_test, y_test)

    # --- Calibration ---
    logger.info("\n--- Calibration Analysis ---")
    calibrator = CalibrationAnalyzer()

    xgb_proba = xgb_model.predict_proba(X_test)
    lr_proba = lr_model.predict_proba(X_test)

    xgb_cal = calibrator.compute_calibration_curve(y_test.values, xgb_proba)
    lr_cal = calibrator.compute_calibration_curve(y_test.values, lr_proba)

    logger.info(f"XGBoost ECE: {xgb_cal['ece']:.4f}, MCE: {xgb_cal['mce']:.4f}")
    logger.info(f"Logistic ECE: {lr_cal['ece']:.4f}, MCE: {lr_cal['mce']:.4f}")

    # --- Brier Score Analysis ---
    logger.info("\n--- Brier Score Analysis ---")
    brier = BrierScoreAnalyzer()

    xgb_brier = brier.compute(y_test.values, xgb_proba)
    lr_brier = brier.compute(y_test.values, lr_proba)
    baseline_brier = brier.compute_baseline(y_test.values)
    xgb_skill = brier.skill_score(y_test.values, xgb_proba)

    xgb_decomp = brier.decompose(y_test.values, xgb_proba)

    logger.info(f"Baseline Brier Score: {baseline_brier:.4f}")
    logger.info(f"XGBoost Brier Score:  {xgb_brier:.4f} (skill: {xgb_skill:.4f})")
    logger.info(f"Logistic Brier Score: {lr_brier:.4f}")
    logger.info(f"Brier Improvement over baseline: {((baseline_brier - xgb_brier) / baseline_brier * 100):.1f}%")
    logger.info(f"Brier Improvement over logistic: {((lr_brier - xgb_brier) / lr_brier * 100):.1f}%")

    logger.info(f"\nMurphy Decomposition:")
    logger.info(f"  Reliability: {xgb_decomp['reliability']:.4f}")
    logger.info(f"  Resolution:  {xgb_decomp['resolution']:.4f}")
    logger.info(f"  Uncertainty: {xgb_decomp['uncertainty']:.4f}")

    # --- Hypothesis Testing ---
    logger.info("\n--- Hypothesis Testing ---")
    hyp = HypothesisTests()

    xgb_preds = xgb_model.predict(X_test)
    lr_preds = lr_model.predict(X_test)

    mcnemar = hyp.mcnemar_test(y_test.values, xgb_preds, lr_preds)
    logger.info(f"McNemar's test: {mcnemar['conclusion']}")

    delong = hyp.delong_test(y_test.values, xgb_proba, lr_proba)
    logger.info(f"DeLong AUC test: XGBoost={delong['auc_a']:.4f} vs Logistic={delong['auc_b']:.4f}, p={delong['p_value']:.4f}")

    paired = hyp.paired_brier_test(y_test.values.astype(float), xgb_proba, lr_proba)
    logger.info(f"Paired Brier test: better={paired['better_model']}, p={paired['p_value']:.4f}")

    # --- Feature Importance ---
    logger.info("\n--- Top 10 Features (XGBoost gain) ---")
    importance = xgb_model.get_feature_importance()
    for _, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']:35s} {row['importance']:.4f}")

    # --- Save Models ---
    logger.info("\n--- Saving Models ---")
    xgb_model.save("models/xgboost_v1.json")
    lr_model.save("models/logistic_v1.pkl")

    # --- Final Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<30} {'XGBoost':>12} {'Logistic':>12}")
    logger.info("-" * 56)
    logger.info(f"{'Accuracy':<30} {xgb_metrics['accuracy']:>12.4f} {lr_metrics['accuracy']:>12.4f}")
    logger.info(f"{'Brier Score':<30} {xgb_metrics['brier_score']:>12.4f} {lr_metrics['brier_score']:>12.4f}")
    logger.info(f"{'Log Loss':<30} {xgb_metrics['log_loss']:>12.4f} {lr_metrics['log_loss']:>12.4f}")
    logger.info(f"{'ROC AUC':<30} {xgb_metrics['roc_auc']:>12.4f} {lr_metrics['roc_auc']:>12.4f}")
    logger.info(f"{'ECE':<30} {xgb_cal['ece']:>12.4f} {lr_cal['ece']:>12.4f}")
    logger.info("-" * 56)
    logger.info(f"Brier Skill Score (vs baseline): {xgb_skill:.4f}")
    logger.info(f"Models saved to models/")
    logger.info("=" * 60)

    # Save metrics as JSON for dashboard
    results = {
        "xgboost": {**xgb_metrics, "ece": xgb_cal["ece"], "mce": xgb_cal["mce"]},
        "logistic": {**lr_metrics, "ece": lr_cal["ece"], "mce": lr_cal["mce"]},
        "brier_decomposition": xgb_decomp,
        "brier_skill_score": xgb_skill,
        "hypothesis_tests": {
            "mcnemar": mcnemar,
            "delong": delong,
            "paired_brier": paired,
        },
    }
    Path("models/training_results.json").write_text(json.dumps(results, indent=2, default=str))
    logger.info("Results saved to models/training_results.json")


if __name__ == "__main__":
    main()
