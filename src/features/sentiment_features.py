"""Sentiment-derived features from social media and NLP scores."""
import numpy as np
import pandas as pd
from loguru import logger


class SentimentFeatureEngineer:
    """Compute sentiment features from NLP scores."""

    @staticmethod
    def compute_all(sentiment_df: pd.DataFrame) -> dict:
        """
        Compute sentiment features from a DataFrame of scored texts.

        Expected columns: timestamp, vader_compound, textblob_polarity,
                          bert_score, source, score (upvotes)

        Returns dict of feature_name -> value (12 features)
        """
        features = {}

        if sentiment_df.empty:
            return features

        # --- Aggregate sentiment (4) ---
        features["sentiment_mean_vader"] = sentiment_df["vader_compound"].mean()
        features["sentiment_std_vader"] = sentiment_df["vader_compound"].std()
        features["sentiment_mean_bert"] = sentiment_df["bert_score"].mean()

        if "score" in sentiment_df.columns:
            weights = sentiment_df["score"].clip(lower=1)
            features["sentiment_weighted_vader"] = np.average(
                sentiment_df["vader_compound"], weights=weights
            )
        else:
            features["sentiment_weighted_vader"] = features["sentiment_mean_vader"]

        # --- Sentiment momentum (2) ---
        if "timestamp" in sentiment_df.columns and len(sentiment_df) > 10:
            sent_ts = sentiment_df.set_index("timestamp").sort_index()

            midpoint = len(sent_ts) // 2
            recent = sent_ts["vader_compound"].iloc[midpoint:].mean()
            older = sent_ts["vader_compound"].iloc[:midpoint].mean()
            features["sentiment_momentum"] = recent - older

            x = np.arange(len(sent_ts))
            y = sent_ts["vader_compound"].values
            if len(x) > 2:
                slope = np.polyfit(x, y, 1)[0]
                features["sentiment_slope"] = slope
            else:
                features["sentiment_slope"] = 0.0
        else:
            features["sentiment_momentum"] = np.nan
            features["sentiment_slope"] = np.nan

        # --- Volume of mentions (2) ---
        features["mention_count"] = len(sentiment_df)
        features["mention_velocity"] = (
            len(sentiment_df) / max(1, sentiment_df["timestamp"].nunique())
            if "timestamp" in sentiment_df.columns else len(sentiment_df)
        )

        # --- Sentiment dispersion (1) ---
        features["sentiment_dispersion"] = sentiment_df["vader_compound"].std()

        # --- Ratio of positive vs negative (1) ---
        pos_count = (sentiment_df["vader_compound"] > 0.05).sum()
        neg_count = (sentiment_df["vader_compound"] < -0.05).sum()
        total = pos_count + neg_count
        features["positive_ratio"] = pos_count / total if total > 0 else 0.5

        # --- Source breakdown (2) ---
        if "source" in sentiment_df.columns:
            for src in ["reddit", "twitter"]:
                subset = sentiment_df[sentiment_df["source"] == src]
                features[f"sentiment_mean_{src}"] = (
                    subset["vader_compound"].mean() if len(subset) > 0 else np.nan
                )
                features[f"mention_count_{src}"] = len(subset)

        logger.debug(f"Computed {len(features)} sentiment features")
        return features
