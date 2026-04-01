"""Logistic Regression baseline model."""
import numpy as np
import pandas as pd
import joblib
from loguru import logger
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.config.settings import config


class LogisticPredictor:
    """Logistic Regression with built-in cross-validation for regularization."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train logistic regression with CV-tuned regularization."""
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)

        self.model = LogisticRegressionCV(
            Cs=20,
            cv=config.model.cv_folds,
            scoring="neg_brier_score",
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            random_state=config.model.random_state,
            class_weight="balanced",
        )
        self.model.fit(X_scaled, y)
        logger.info(f"Logistic Regression trained. Best C={self.model.C_[0]:.4f}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        y_prob = self.predict_proba(X)
        y_pred = self.predict(X)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "brier_score": brier_score_loss(y, y_prob),
            "log_loss": log_loss(y, y_prob),
            "roc_auc": roc_auc_score(y, y_prob),
        }
        logger.info(f"Logistic evaluation: {metrics}")
        return metrics

    def get_coefficients(self) -> pd.DataFrame:
        """Get feature coefficients."""
        return pd.DataFrame({
            "feature": self.feature_names,
            "coefficient": self.model.coef_[0],
            "abs_coefficient": np.abs(self.model.coef_[0]),
        }).sort_values("abs_coefficient", ascending=False)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
        }, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
