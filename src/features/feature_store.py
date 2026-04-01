"""Unified feature store that orchestrates all feature engineers."""
import json
from datetime import date
from typing import Optional

import pandas as pd
from loguru import logger

from src.features.market_features import MarketFeatureEngineer
from src.features.sentiment_features import SentimentFeatureEngineer
from src.features.trend_features import TrendFeatureEngineer
from src.features.interaction_features import InteractionFeatureEngineer
from src.pipeline.rds_manager import RDSManager


class FeatureStore:
    """Orchestrates feature computation and persistence."""

    def __init__(self, rds: Optional[RDSManager] = None):
        self.rds = rds or RDSManager()
        self.market_eng = MarketFeatureEngineer()
        self.sentiment_eng = SentimentFeatureEngineer()
        self.trend_eng = TrendFeatureEngineer()
        self.interaction_eng = InteractionFeatureEngineer()

    def compute_features(
        self,
        market_id: str,
        prices: pd.DataFrame,
        trades: pd.DataFrame,
        sentiment: pd.DataFrame,
        trends: pd.DataFrame,
        order_book: dict = None,
    ) -> dict:
        """Compute all 48 features for a single market."""
        market_feats = self.market_eng.compute_all(prices, trades, order_book)
        sentiment_feats = self.sentiment_eng.compute_all(sentiment)
        trend_feats = self.trend_eng.compute_all(trends)
        interaction_feats = self.interaction_eng.compute_all(
            market_feats, sentiment_feats, trend_feats
        )

        all_features = {
            **market_feats,
            **sentiment_feats,
            **trend_feats,
            **interaction_feats,
        }

        logger.info(
            f"Market {market_id}: {len(all_features)} total features computed"
        )
        return all_features

    def save_snapshot(self, market_id: str, features: dict, snapshot_date: date = None):
        """Persist a feature snapshot to the feature store."""
        snapshot_date = snapshot_date or date.today()

        df = pd.DataFrame([{
            "market_id": market_id,
            "snapshot_date": snapshot_date,
            "features": json.dumps(features, default=str),
        }])

        self.rds.upsert_dataframe(
            df, "feature_store", conflict_columns=["market_id", "snapshot_date"]
        )

    def load_training_dataset(self) -> pd.DataFrame:
        """
        Load all feature snapshots joined with ground truth outcomes.

        Returns a flat DataFrame where each row is a market-date observation
        with all features as columns plus an 'outcome' column (1/0).
        """
        sql = """
            SELECT
                f.market_id,
                f.snapshot_date,
                f.features,
                m.resolution_outcome,
                CASE
                    WHEN m.resolution_outcome = 'Yes' THEN 1
                    WHEN m.resolution_outcome = 'No' THEN 0
                    ELSE NULL
                END as outcome
            FROM feature_store f
            JOIN markets m ON f.market_id = m.market_id
            WHERE m.resolved = TRUE
            ORDER BY f.snapshot_date
        """
        df = self.rds.query(sql)

        if df.empty:
            return df

        feature_records = df["features"].apply(json.loads)
        features_df = pd.DataFrame(feature_records.tolist())

        result = pd.concat([
            df[["market_id", "snapshot_date", "outcome"]].reset_index(drop=True),
            features_df.reset_index(drop=True),
        ], axis=1)

        result = result.dropna(subset=["outcome"])
        result["outcome"] = result["outcome"].astype(int)

        logger.info(
            f"Training dataset: {len(result)} rows, "
            f"{len(features_df.columns)} features"
        )
        return result
