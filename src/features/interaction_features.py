"""Cross-source interaction features."""
import numpy as np
from loguru import logger


class InteractionFeatureEngineer:
    """Compute interaction features that combine signals across data sources."""

    @staticmethod
    def compute_all(
        market_features: dict,
        sentiment_features: dict,
        trend_features: dict,
    ) -> dict:
        """
        Compute interaction features by combining market, sentiment, and trend signals.

        Returns dict of feature_name -> value (8 features)
        """
        features = {}

        def get(d, key, default=np.nan):
            return d.get(key, default)

        # --- Sentiment x Volume interaction (1) ---
        sent_mean = get(sentiment_features, "sentiment_mean_vader")
        vol_z = get(market_features, "volume_zscore")
        if not np.isnan(sent_mean) and not np.isnan(vol_z):
            features["sentiment_x_volume"] = sent_mean * vol_z
        else:
            features["sentiment_x_volume"] = np.nan

        # --- Trend x Price momentum interaction (1) ---
        trend_slope = get(trend_features, "trend_slope_7d")
        price_mom = get(market_features, "price_momentum_7d")
        if not np.isnan(trend_slope) and not np.isnan(price_mom):
            features["trend_x_momentum"] = trend_slope * price_mom
        else:
            features["trend_x_momentum"] = np.nan

        # --- Sentiment-price divergence (2) ---
        sent_mom = get(sentiment_features, "sentiment_momentum")
        if not np.isnan(sent_mom) and not np.isnan(price_mom):
            features["sentiment_price_divergence"] = sent_mom - price_mom
            features["divergence_flag"] = int(
                (sent_mom > 0 and price_mom < -0.05)
                or (sent_mom < 0 and price_mom > 0.05)
            )
        else:
            features["sentiment_price_divergence"] = np.nan
            features["divergence_flag"] = 0

        # --- Smart money indicator (1) ---
        vol_total = get(market_features, "volume_total", 0)
        dispersion = get(sentiment_features, "sentiment_dispersion")
        if vol_total > 0 and not np.isnan(dispersion):
            features["smart_money_indicator"] = (
                np.log1p(vol_total) * (1 - min(dispersion, 1.0))
            )
        else:
            features["smart_money_indicator"] = np.nan

        # --- Attention-adjusted price (1) ---
        trend_current = get(trend_features, "trend_current")
        current_price = get(market_features, "current_price")
        if not np.isnan(trend_current) and not np.isnan(current_price):
            trend_norm = trend_current / 100.0
            features["attention_adjusted_price"] = current_price * (1 + trend_norm) / 2
        else:
            features["attention_adjusted_price"] = np.nan

        # --- Consensus strength (1) ---
        pos_ratio = get(sentiment_features, "positive_ratio", 0.5)
        volatility = get(market_features, "volatility_7d")
        if not any(
            np.isnan(x) for x in [pos_ratio, volatility]
            if isinstance(x, float)
        ):
            consensus = abs(pos_ratio - 0.5) * 2  # 0=split, 1=unanimous
            features["consensus_strength"] = consensus * max(0, 1 - volatility)
        else:
            features["consensus_strength"] = np.nan

        # --- Contrarian signal (1) ---
        features["contrarian_signal"] = int(
            get(sentiment_features, "positive_ratio", 0.5) > 0.8
            and get(market_features, "current_price", 0.5) < 0.6
        )

        logger.debug(f"Computed {len(features)} interaction features")
        return features
