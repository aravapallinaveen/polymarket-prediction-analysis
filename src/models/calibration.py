"""Model calibration (Platt scaling and isotonic regression)."""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from loguru import logger


class CalibrationAnalyzer:
    """Calibrate model probabilities and analyze calibration quality."""

    @staticmethod
    def calibrate_model(model, X, y, method: str = "isotonic", cv: int = 5):
        """
        Apply post-hoc calibration using Platt scaling or isotonic regression.

        Args:
            model: Fitted sklearn-compatible model
            X: Feature matrix
            y: True labels
            method: 'sigmoid' (Platt) or 'isotonic'
            cv: Number of cross-validation folds

        Returns calibrated model
        """
        calibrated = CalibratedClassifierCV(model, method=method, cv=cv)
        calibrated.fit(X, y)
        logger.info(f"Model calibrated with {method} method")
        return calibrated

    @staticmethod
    def compute_calibration_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
        strategy: str = "uniform",
    ) -> dict:
        """
        Compute calibration curve data.

        Returns dict with:
            - fraction_of_positives: actual probability per bin
            - mean_predicted_value: mean predicted probability per bin
            - bin_counts: number of samples per bin
            - ece: Expected Calibration Error
            - mce: Maximum Calibration Error
        """
        fraction_positives, mean_predicted = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy=strategy
        )

        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        bin_counts = np.bincount(bin_indices, minlength=n_bins)

        # Expected Calibration Error
        weights = bin_counts / len(y_prob)
        ece = np.sum(
            weights[: len(fraction_positives)]
            * np.abs(fraction_positives - mean_predicted)
        )

        # Maximum Calibration Error
        mce = np.max(np.abs(fraction_positives - mean_predicted))

        return {
            "fraction_of_positives": fraction_positives.tolist(),
            "mean_predicted_value": mean_predicted.tolist(),
            "bin_counts": bin_counts.tolist(),
            "ece": float(ece),
            "mce": float(mce),
            "n_bins": n_bins,
        }

    @staticmethod
    def reliability_diagram_data(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Generate data for a reliability diagram.

        Returns DataFrame with columns:
            bin_center, actual_frequency, predicted_mean,
            count, calibration_error
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        records = []
        for i in range(n_bins):
            mask = bin_indices == i
            count = mask.sum()
            if count > 0:
                actual = y_true[mask].mean()
                predicted = y_prob[mask].mean()
            else:
                actual = np.nan
                predicted = np.nan

            records.append({
                "bin_center": bin_centers[i],
                "actual_frequency": actual,
                "predicted_mean": predicted,
                "count": count,
                "calibration_error": abs(actual - predicted) if count > 0 else np.nan,
            })

        return pd.DataFrame(records)
