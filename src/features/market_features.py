"""Market-derived features from price history and order book data."""
import numpy as np
import pandas as pd
from loguru import logger


class MarketFeatureEngineer:
    """Compute market structure features from price and trade data."""

    @staticmethod
    def compute_all(
        prices: pd.DataFrame,
        trades: pd.DataFrame = None,
        order_book: dict = None,
    ) -> dict:
        """
        Compute all market features for a single market snapshot.

        Args:
            prices: DataFrame with columns [timestamp, price] sorted by timestamp
            trades: Optional DataFrame with columns [timestamp, price, size]
            order_book: Optional dict with bids, asks, spread

        Returns dict of feature_name -> value (20 features)
        """
        features = {}

        if prices.empty:
            return features

        current_price = prices["price"].iloc[-1]
        features["current_price"] = current_price

        # --- Price momentum features (3) ---
        for window in [7, 14, 30]:
            col = f"price_momentum_{window}d"
            if len(prices) >= window:
                past_price = prices["price"].iloc[-window]
                features[col] = current_price - past_price
            else:
                features[col] = np.nan

        # --- Volatility features (3) ---
        returns = prices["price"].pct_change().dropna()
        for window in [7, 14, 30]:
            col = f"volatility_{window}d"
            if len(returns) >= window:
                features[col] = returns.iloc[-window:].std()
            else:
                features[col] = np.nan

        # --- Price position relative to range (2) ---
        for window in [30, 60]:
            if len(prices) >= window:
                recent = prices["price"].iloc[-window:]
                pmin, pmax = recent.min(), recent.max()
                range_val = pmax - pmin
                features[f"price_percentile_{window}d"] = (
                    (current_price - pmin) / range_val if range_val > 0 else 0.5
                )
            else:
                features[f"price_percentile_{window}d"] = np.nan

        # --- Moving average crossovers (2) ---
        if len(prices) >= 30:
            ma_7 = prices["price"].iloc[-7:].mean()
            ma_30 = prices["price"].iloc[-30:].mean()
            features["ma_7_30_crossover"] = ma_7 - ma_30
            features["price_above_ma30"] = int(current_price > ma_30)
        else:
            features["ma_7_30_crossover"] = np.nan
            features["price_above_ma30"] = np.nan

        # --- Volume features (5) ---
        if trades is not None and not trades.empty:
            features["trade_count_24h"] = len(trades)
            features["volume_total"] = trades["size"].sum()
            features["volume_mean"] = trades["size"].mean()

            features["vwap"] = (
                (trades["price"] * trades["size"]).sum() / trades["size"].sum()
            )

            daily_volumes = trades.set_index("timestamp")["size"].resample("1D").sum()
            if len(daily_volumes) > 7:
                vol_mean = daily_volumes.mean()
                vol_std = daily_volumes.std()
                latest_vol = daily_volumes.iloc[-1]
                features["volume_zscore"] = (
                    (latest_vol - vol_mean) / vol_std if vol_std > 0 else 0
                )
            else:
                features["volume_zscore"] = np.nan
        else:
            features.update({
                "trade_count_24h": 0, "volume_total": 0, "volume_mean": 0,
                "vwap": np.nan, "volume_zscore": np.nan
            })

        # --- Order book features (4) ---
        if order_book:
            features["spread"] = order_book.get("spread", np.nan)
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            features["bid_depth"] = sum(float(b.get("size", 0)) for b in bids[:5])
            features["ask_depth"] = sum(float(a.get("size", 0)) for a in asks[:5])
            total_depth = features["bid_depth"] + features["ask_depth"]
            features["bid_ask_imbalance"] = (
                (features["bid_depth"] - features["ask_depth"]) / total_depth
                if total_depth > 0 else 0
            )
        else:
            features.update({
                "spread": np.nan, "bid_depth": np.nan,
                "ask_depth": np.nan, "bid_ask_imbalance": np.nan
            })

        # --- Price extremity features (2) ---
        features["price_distance_from_50"] = abs(current_price - 0.5)
        features["is_extreme_price"] = int(current_price < 0.1 or current_price > 0.9)

        logger.debug(f"Computed {len(features)} market features")
        return features
