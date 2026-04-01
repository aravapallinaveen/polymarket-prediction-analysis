"""Tests for DataValidator."""
import pandas as pd
import pytest

from src.ingestion.data_validator import DataValidator


@pytest.fixture
def validator():
    return DataValidator()


def test_valid_market_data(validator):
    df = pd.DataFrame({
        "id": ["1", "2"],
        "question": ["Q1?", "Q2?"],
        "outcomes": ['["Yes","No"]', '["Yes","No"]'],
        "active": [True, True],
    })
    result = validator.validate_market_data(df)
    assert result.is_valid
    assert result.total_rows == 2


def test_missing_columns(validator):
    df = pd.DataFrame({"id": ["1"], "question": ["Q?"]})
    result = validator.validate_market_data(df)
    assert not result.is_valid
    assert any("Missing" in e for e in result.errors)


def test_duplicate_ids(validator):
    df = pd.DataFrame({
        "id": ["1", "1"],
        "question": ["Q?", "Q?"],
        "outcomes": ['["Yes","No"]', '["Yes","No"]'],
        "active": [True, True],
    })
    result = validator.validate_market_data(df)
    assert not result.is_valid
    assert result.duplicates == 1


def test_valid_price_data(validator):
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5),
        "price": [0.1, 0.5, 0.7, 0.3, 0.9],
    })
    result = validator.validate_price_data(df)
    assert result.is_valid


def test_out_of_range_prices(validator):
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3),
        "price": [0.5, 1.5, -0.1],
    })
    result = validator.validate_price_data(df)
    assert not result.is_valid
    assert result.out_of_range == 2
