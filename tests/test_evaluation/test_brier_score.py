"""Tests for Brier Score computation."""
import numpy as np
import pytest

from src.evaluation.brier_score import BrierScoreAnalyzer


@pytest.fixture
def analyzer():
    return BrierScoreAnalyzer()


def test_perfect_predictions(analyzer):
    y_true = np.array([1, 0, 1, 0, 1])
    y_prob = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    assert analyzer.compute(y_true, y_prob) == 0.0


def test_worst_predictions(analyzer):
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.0, 1.0, 0.0, 1.0])
    assert analyzer.compute(y_true, y_prob) == 1.0


def test_skill_score_perfect(analyzer):
    y_true = np.array([1, 0, 1, 0, 1])
    y_prob = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    assert analyzer.skill_score(y_true, y_prob) == 1.0


def test_skill_score_baseline(analyzer):
    """A model that predicts base rate should have skill score ~0."""
    y_true = np.array([1, 1, 0, 0, 0])
    base_rate = y_true.mean()
    y_prob = np.full_like(y_true, base_rate, dtype=float)
    assert abs(analyzer.skill_score(y_true, y_prob)) < 0.01


def test_decomposition_sums_correctly(analyzer):
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.6, 500)
    y_prob = np.clip(y_true + np.random.normal(0, 0.2, 500), 0, 1)

    decomp = analyzer.decompose(y_true, y_prob)
    computed_bs = decomp["reliability"] - decomp["resolution"] + decomp["uncertainty"]
    assert abs(computed_bs - decomp["brier_score"]) < 0.01


def test_baseline_equals_base_rate_variance(analyzer):
    y_true = np.array([1, 1, 0, 0, 0])  # base rate = 0.4
    baseline = analyzer.compute_baseline(y_true)
    expected = 0.4 * 0.6  # p * (1-p) = 0.24
    assert abs(baseline - expected) < 0.01
