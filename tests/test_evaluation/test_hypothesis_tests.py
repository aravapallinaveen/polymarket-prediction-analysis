"""Tests for hypothesis testing module."""
import numpy as np
import pytest

from src.evaluation.hypothesis_tests import HypothesisTests


def test_mcnemar_identical_models():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])
    result = HypothesisTests.mcnemar_test(y_true, y_pred, y_pred)
    assert not result["significant"]
    assert result["p_value"] == 1.0


def test_mcnemar_different_models():
    np.random.seed(42)
    n = 200
    y_true = np.random.binomial(1, 0.5, n)
    y_pred_a = y_true.copy()  # Perfect model
    y_pred_b = np.random.binomial(1, 0.5, n)  # Random model

    result = HypothesisTests.mcnemar_test(y_true, y_pred_a, y_pred_b)
    assert result["significant"]
    assert "Model A" in result["conclusion"]


def test_paired_brier_returns_correct_structure():
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.5, 100).astype(float)
    y_prob_a = np.clip(y_true + np.random.normal(0, 0.1, 100), 0, 1)
    y_prob_b = np.clip(y_true + np.random.normal(0, 0.3, 100), 0, 1)

    result = HypothesisTests.paired_brier_test(y_true, y_prob_a, y_prob_b)
    assert "mean_brier_a" in result
    assert "mean_brier_b" in result
    assert "p_value" in result
    assert result["better_model"] in ("A", "B")
