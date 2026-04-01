"""Tests for XGBoost model."""
import numpy as np
import pandas as pd
import pytest

from src.models.xgboost_model import XGBoostPredictor


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(200, 10), columns=[f"f_{i}" for i in range(10)])
    y = pd.Series(np.random.binomial(1, 0.5, 200))
    return X, y


def test_train_and_predict(sample_data):
    X, y = sample_data
    model = XGBoostPredictor()
    model.train(X, y, params={"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1})

    proba = model.predict_proba(X)
    assert len(proba) == len(X)
    assert all(0 <= p <= 1 for p in proba)

    preds = model.predict(X)
    assert set(preds).issubset({0, 1})


def test_evaluate_returns_all_metrics(sample_data):
    X, y = sample_data
    model = XGBoostPredictor()
    model.train(X, y, params={"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1})

    metrics = model.evaluate(X, y)
    assert "accuracy" in metrics
    assert "brier_score" in metrics
    assert "log_loss" in metrics
    assert "roc_auc" in metrics
    assert 0 <= metrics["accuracy"] <= 1


def test_feature_importance(sample_data):
    X, y = sample_data
    model = XGBoostPredictor()
    model.train(X, y, params={"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1})

    importance = model.get_feature_importance()
    assert "feature" in importance.columns
    assert "importance" in importance.columns
    assert len(importance) > 0


def test_save_and_load(sample_data, tmp_path):
    X, y = sample_data
    model = XGBoostPredictor()
    model.train(X, y, params={"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1})

    path = str(tmp_path / "model.json")
    model.save(path)

    loaded = XGBoostPredictor()
    loaded.load(path)

    original_preds = model.predict_proba(X)
    loaded_preds = loaded.predict_proba(X)
    np.testing.assert_array_almost_equal(original_preds, loaded_preds)
