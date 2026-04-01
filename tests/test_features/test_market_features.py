"""Tests for market feature engineering."""
import numpy as np
import pandas as pd
import pytest

from src.features.market_features import MarketFeatureEngineer


@pytest.fixture
def price_data():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    prices = np.linspace(0.3, 0.7, 60) + np.random.normal(0, 0.02, 60)
    return pd.DataFrame({"timestamp": dates, "price": np.clip(prices, 0, 1)})


@pytest.fixture
def trade_data():
    dates = pd.date_range("2024-02-01", periods=30, freq="D")
    return pd.DataFrame({
        "timestamp": dates,
        "price": np.random.uniform(0.4, 0.6, 30),
        "size": np.random.uniform(50, 500, 30),
    })


def test_compute_all_returns_expected_features(price_data, trade_data):
    features = MarketFeatureEngineer.compute_all(price_data, trade_data)

    assert "current_price" in features
    assert "price_momentum_7d" in features
    assert "volatility_30d" in features
    assert "volume_zscore" in features
    assert "vwap" in features
    assert len(features) >= 20


def test_prices_within_expected_range(price_data):
    features = MarketFeatureEngineer.compute_all(price_data)

    assert 0 <= features["current_price"] <= 1
    assert 0 <= features["price_distance_from_50"] <= 0.5


def test_empty_prices_returns_empty_dict():
    features = MarketFeatureEngineer.compute_all(
        pd.DataFrame(columns=["timestamp", "price"])
    )
    assert features == {}


def test_momentum_direction():
    """If prices are trending up, momentum should be positive."""
    uptrend = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=60, freq="D"),
        "price": np.linspace(0.3, 0.8, 60),
    })
    features = MarketFeatureEngineer.compute_all(uptrend)

    assert features["price_momentum_7d"] > 0
    assert features["price_momentum_30d"] > 0
