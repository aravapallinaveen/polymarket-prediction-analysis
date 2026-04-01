"""Data validation for ingested records."""
from dataclasses import dataclass, field

import pandas as pd
from loguru import logger


@dataclass
class ValidationResult:
    is_valid: bool
    total_rows: int
    null_counts: dict
    duplicates: int
    out_of_range: int
    errors: list


class DataValidator:
    """Validates ingested data for quality issues."""

    def validate_market_data(self, df: pd.DataFrame) -> ValidationResult:
        errors = []
        null_counts = df.isnull().sum().to_dict()

        required = ["id", "question", "outcomes", "active"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")

        duplicates = df.duplicated(subset=["id"]).sum() if "id" in df.columns else 0
        if duplicates > 0:
            errors.append(f"Found {duplicates} duplicate market IDs")

        return ValidationResult(
            is_valid=len(errors) == 0,
            total_rows=len(df),
            null_counts=null_counts,
            duplicates=duplicates,
            out_of_range=0,
            errors=errors,
        )

    def validate_price_data(self, df: pd.DataFrame) -> ValidationResult:
        errors = []
        out_of_range = 0

        if "price" in df.columns:
            invalid_prices = ((df["price"] < 0) | (df["price"] > 1)).sum()
            if invalid_prices > 0:
                out_of_range = int(invalid_prices)
                errors.append(f"{invalid_prices} prices outside [0, 1]")

        if "timestamp" in df.columns:
            null_ts = df["timestamp"].isnull().sum()
            if null_ts > 0:
                errors.append(f"{null_ts} null timestamps")

        return ValidationResult(
            is_valid=len(errors) == 0,
            total_rows=len(df),
            null_counts=df.isnull().sum().to_dict(),
            duplicates=df.duplicated().sum(),
            out_of_range=out_of_range,
            errors=errors,
        )

    def validate_sentiment_data(self, df: pd.DataFrame) -> ValidationResult:
        errors = []
        out_of_range = 0

        if "vader_compound" in df.columns:
            invalid = ((df["vader_compound"] < -1) | (df["vader_compound"] > 1)).sum()
            if invalid > 0:
                out_of_range += int(invalid)
                errors.append(f"{invalid} VADER scores outside [-1, 1]")

        return ValidationResult(
            is_valid=len(errors) == 0,
            total_rows=len(df),
            null_counts=df.isnull().sum().to_dict(),
            duplicates=df.duplicated().sum(),
            out_of_range=out_of_range,
            errors=errors,
        )
