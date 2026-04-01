"""XGBoost model for binary outcome prediction."""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from src.config.settings import config


class XGBoostPredictor:
    """XGBoost model with Optuna hyperparameter tuning."""

    def __init__(self):
        self.model: Optional[xgb.XGBClassifier] = None
        self.best_params: dict = {}
        self.feature_names: list = []

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = None,
    ) -> dict:
        """
        Run Optuna hyperparameter search with cross-validation.
        Returns the best parameter dict.
        """
        n_trials = n_trials or config.model.xgb_n_trials

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 3.0),
            }

            cv = StratifiedKFold(
                n_splits=config.model.cv_folds,
                shuffle=True,
                random_state=config.model.random_state,
            )

            scores = []
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = xgb.XGBClassifier(
                    **params,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=config.model.random_state,
                    verbosity=0,
                )
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )

                y_prob = model.predict_proba(X_val)[:, 1]
                brier = brier_score_loss(y_val, y_prob)
                scores.append(brier)

            return np.mean(scores)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=config.model.random_state),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params = study.best_params
        logger.info(f"Best Brier Score: {study.best_value:.4f}")
        logger.info(f"Best params: {json.dumps(self.best_params, indent=2)}")
        return self.best_params

    def train(self, X: pd.DataFrame, y: pd.Series, params: dict = None):
        """Train the final model with best parameters."""
        params = params or self.best_params
        self.feature_names = list(X.columns)

        self.model = xgb.XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=config.model.random_state,
        )
        self.model.fit(X, y, verbose=False)
        logger.info(f"XGBoost trained on {len(X)} samples, {len(self.feature_names)} features")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of positive class."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary predictions."""
        return self.model.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Compute all evaluation metrics."""
        y_prob = self.predict_proba(X)
        y_pred = self.predict(X)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "brier_score": brier_score_loss(y, y_prob),
            "log_loss": log_loss(y, y_prob),
            "roc_auc": roc_auc_score(y, y_prob),
        }
        logger.info(f"XGBoost evaluation: {json.dumps(metrics, indent=2)}")
        return metrics

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Get feature importances sorted by importance."""
        importances = self.model.get_booster().get_score(
            importance_type=importance_type
        )
        df = pd.DataFrame(
            [(k, v) for k, v in importances.items()],
            columns=["feature", "importance"],
        ).sort_values("importance", ascending=False)
        return df

    def save(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)
        meta = {
            "best_params": self.best_params,
            "feature_names": self.feature_names,
        }
        Path(path).with_suffix(".meta.json").write_text(json.dumps(meta))
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk."""
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)
        meta = json.loads(Path(path).with_suffix(".meta.json").read_text())
        self.best_params = meta["best_params"]
        self.feature_names = meta["feature_names"]
        logger.info(f"Model loaded from {path}")
