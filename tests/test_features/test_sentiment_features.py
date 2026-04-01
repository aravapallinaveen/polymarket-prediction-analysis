"""Tests for sentiment feature engineering."""
import numpy as np
import pandas as pd
import pytest

from src.features.sentiment_features import SentimentFeatureEngineer


@pytest.fixture
def sentiment_data():
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=40, freq="D"),
        "vader_compound": np.random.uniform(-1, 1, 40),
        "bert_score": np.random.uniform(0, 1, 40),
        "source": np.random.choice(["reddit", "twitter"], 40),
        "score": np.random.randint(1, 500, 40),
    })


def test_compute_all_returns_expected_features(sentiment_data):
    features = SentimentFeatureEngineer.compute_all(sentiment_data)

    assert "sentiment_mean_vader" in features
    assert "sentiment_mean_bert" in features
    assert "mention_count" in features
    assert "positive_ratio" in features
    assert len(features) >= 12


def test_empty_dataframe():
    features = SentimentFeatureEngineer.compute_all(pd.DataFrame())
    assert features == {}


def test_positive_ratio_bounds(sentiment_data):
    features = SentimentFeatureEngineer.compute_all(sentiment_data)
    assert 0 <= features["positive_ratio"] <= 1


def test_all_positive_sentiment():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="D"),
        "vader_compound": np.full(20, 0.8),
        "bert_score": np.full(20, 0.9),
        "source": ["reddit"] * 20,
        "score": [100] * 20,
    })
    features = SentimentFeatureEngineer.compute_all(df)
    assert features["positive_ratio"] == 1.0
    assert features["sentiment_mean_vader"] > 0.5
