"""Google Trends-derived features."""
import numpy as np
import pandas as pd
from loguru import logger


class TrendFeatureEngineer:
    """Compute features from Google Trends data."""

    @staticmethod
    def compute_all(trends_df: pd.DataFrame, keyword: str = None) -> dict:
        """
        Compute trend features from Google Trends interest-over-time data.

        Args:
            trends_df: DataFrame indexed by date with interest values (0-100)
            keyword: The primary keyword column to use (or first column)

        Returns dict of feature_name -> value (8 features)
        """
        features = {}

        if trends_df.empty:
            return features

        col = keyword if keyword and keyword in trends_df.columns else trends_df.columns[0]
        series = trends_df[col].astype(float)

        # --- Current interest level (3) ---
        features["trend_current"] = series.iloc[-1]
        features["trend_mean"] = series.mean()
        features["trend_max"] = series.max()

        # --- Trend slope (2) ---
        for window in [7, 30]:
            if len(series) >= window:
                recent = series.iloc[-window:]
                x = np.arange(len(recent))
                slope = np.polyfit(x, recent.values, 1)[0]
                features[f"trend_slope_{window}d"] = slope
            else:
                features[f"trend_slope_{window}d"] = np.nan

        # --- Trend acceleration (1) ---
        if len(series) >= 14:
            slope_recent = np.polyfit(np.arange(7), series.iloc[-7:].values, 1)[0]
            slope_prior = np.polyfit(np.arange(7), series.iloc[-14:-7].values, 1)[0]
            features["trend_acceleration"] = slope_recent - slope_prior
        else:
            features["trend_acceleration"] = np.nan

        # --- Peak ratio (1) ---
        peak = series.max()
        features["trend_peak_ratio"] = series.iloc[-1] / peak if peak > 0 else 0

        # --- Breakout detection (1) ---
        if len(series) >= 30:
            mean_30 = series.iloc[-30:].mean()
            std_30 = series.iloc[-30:].std()
            features["trend_breakout_flag"] = int(
                series.iloc[-1] > mean_30 + 2 * std_30 if std_30 > 0 else 0
            )
        else:
            features["trend_breakout_flag"] = 0

        # --- Relative search volume (1) ---
        if len(series) >= 30:
            features["trend_relative_volume"] = (
                series.iloc[-7:].mean() / series.iloc[-30:].mean()
                if series.iloc[-30:].mean() > 0 else 1.0
            )
        else:
            features["trend_relative_volume"] = np.nan

        logger.debug(f"Computed {len(features)} trend features")
        return features
