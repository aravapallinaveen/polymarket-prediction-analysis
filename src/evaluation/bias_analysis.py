"""Bias analysis across market categories and prediction ranges."""
import numpy as np
import pandas as pd
from loguru import logger


class BiasAnalyzer:
    """Analyze systematic biases in predictions."""

    @staticmethod
    def favorite_longshot_bias(
        y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> pd.DataFrame:
        """
        Detect favorite-longshot bias.

        Prediction markets tend to overvalue longshots (low probability events)
        and undervalue favorites (high probability events).

        Returns DataFrame with actual vs predicted frequencies per bin.
        """
        bins = np.linspace(0, 1, n_bins + 1)
        records = []

        for i in range(n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
            if mask.sum() == 0:
                continue

            actual = y_true[mask].mean()
            predicted = y_prob[mask].mean()

            records.append({
                "bin_lower": float(bins[i]),
                "bin_upper": float(bins[i + 1]),
                "bin_center": float((bins[i] + bins[i + 1]) / 2),
                "predicted_prob": float(predicted),
                "actual_freq": float(actual),
                "bias": float(predicted - actual),
                "bias_direction": (
                    "overconfident" if predicted > actual else "underconfident"
                ),
                "n_samples": int(mask.sum()),
            })

        df = pd.DataFrame(records)
        logger.info(f"Favorite-longshot bias analysis: {len(df)} bins")
        return df

    @staticmethod
    def category_bias(
        predictions_df: pd.DataFrame,
        prob_col: str = "predicted_probability",
        outcome_col: str = "outcome",
        category_col: str = "category",
    ) -> pd.DataFrame:
        """Compute bias metrics grouped by market category."""
        results = []

        for cat, group in predictions_df.groupby(category_col):
            if len(group) < 10:
                continue

            y_true = group[outcome_col].values
            y_prob = group[prob_col].values
            bs = np.mean((y_prob - y_true) ** 2)
            mean_bias = y_prob.mean() - y_true.mean()

            results.append({
                "category": cat,
                "brier_score": float(bs),
                "mean_bias": float(mean_bias),
                "mean_prediction": float(y_prob.mean()),
                "base_rate": float(y_true.mean()),
                "n_samples": len(group),
            })

        return pd.DataFrame(results).sort_values("brier_score")

    @staticmethod
    def temporal_bias(
        predictions_df: pd.DataFrame,
        prob_col: str = "predicted_probability",
        outcome_col: str = "outcome",
        date_col: str = "prediction_date",
        freq: str = "M",
    ) -> pd.DataFrame:
        """Compute bias over time (monthly or weekly)."""
        predictions_df = predictions_df.copy()
        predictions_df[date_col] = pd.to_datetime(predictions_df[date_col])
        predictions_df["period"] = predictions_df[date_col].dt.to_period(freq)

        results = []
        for period, group in predictions_df.groupby("period"):
            y_true = group[outcome_col].values
            y_prob = group[prob_col].values
            bs = np.mean((y_prob - y_true) ** 2)

            results.append({
                "period": str(period),
                "brier_score": float(bs),
                "mean_bias": float(y_prob.mean() - y_true.mean()),
                "n_samples": len(group),
            })

        return pd.DataFrame(results)
