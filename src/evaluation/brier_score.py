"""Brier Score computation and decomposition."""
import numpy as np
import pandas as pd
from loguru import logger


class BrierScoreAnalyzer:
    """Compute and decompose Brier Scores."""

    @staticmethod
    def compute(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Standard Brier Score: mean squared error of probabilities."""
        return float(np.mean((y_prob - y_true) ** 2))

    @staticmethod
    def compute_baseline(y_true: np.ndarray) -> float:
        """Brier Score of a naive model that always predicts the base rate."""
        base_rate = y_true.mean()
        return float(np.mean((base_rate - y_true) ** 2))

    @staticmethod
    def skill_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Brier Skill Score: improvement over climatological baseline.
        BSS = 1 - BS / BS_ref
        Range: (-inf, 1]. 1 = perfect, 0 = no skill, <0 = worse than baseline.
        """
        bs = np.mean((y_prob - y_true) ** 2)
        bs_ref = np.mean((y_true.mean() - y_true) ** 2)
        return float(1 - bs / bs_ref) if bs_ref > 0 else 0.0

    @staticmethod
    def decompose(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
        """
        Murphy decomposition of Brier Score into reliability, resolution, uncertainty.

        BS = Reliability - Resolution + Uncertainty

        - Reliability (lower is better): measures calibration
        - Resolution (higher is better): measures how well predictions separate events
        - Uncertainty: base rate uncertainty (constant for a dataset)
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        n = len(y_true)
        base_rate = y_true.mean()
        uncertainty = base_rate * (1 - base_rate)

        reliability = 0.0
        resolution = 0.0

        for k in range(n_bins):
            mask = bin_indices == k
            n_k = mask.sum()
            if n_k == 0:
                continue

            o_k = y_true[mask].mean()
            f_k = y_prob[mask].mean()

            reliability += n_k * (f_k - o_k) ** 2
            resolution += n_k * (o_k - base_rate) ** 2

        reliability /= n
        resolution /= n

        return {
            "brier_score": float(np.mean((y_prob - y_true) ** 2)),
            "reliability": float(reliability),
            "resolution": float(resolution),
            "uncertainty": float(uncertainty),
            "check": float(reliability - resolution + uncertainty),
        }

    @staticmethod
    def by_time_horizon(
        df: pd.DataFrame,
        prob_col: str = "predicted_probability",
        outcome_col: str = "outcome",
        time_col: str = "days_to_resolution",
        horizons: list = None,
    ) -> pd.DataFrame:
        """
        Compute Brier Score across different time horizons.

        Args:
            df: DataFrame with predictions, outcomes, and time-to-resolution
            horizons: List of (min_days, max_days, label) tuples

        Returns DataFrame with brier_score per horizon.
        """
        if horizons is None:
            horizons = [
                (0, 7, "0-7 days"),
                (7, 30, "7-30 days"),
                (30, 90, "30-90 days"),
                (90, 365, "90-365 days"),
                (365, float("inf"), "365+ days"),
            ]

        results = []
        for min_d, max_d, label in horizons:
            mask = (df[time_col] >= min_d) & (df[time_col] < max_d)
            subset = df[mask]

            if len(subset) < 10:
                continue

            bs = np.mean(
                (subset[prob_col].values - subset[outcome_col].values) ** 2
            )
            results.append({
                "horizon": label,
                "brier_score": float(bs),
                "n_samples": len(subset),
                "mean_prediction": float(subset[prob_col].mean()),
                "base_rate": float(subset[outcome_col].mean()),
            })

        return pd.DataFrame(results)
