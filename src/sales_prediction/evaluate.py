"""Métriques d'évaluation et validation croisée."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_validate


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def cross_validate_model(
    model: BaseEstimator, X: pd.DataFrame, y: pd.Series, config: dict[str, Any]
) -> dict[str, float]:
    """Validation croisée répétée : moyenne et écart-type du R² et du RMSE."""
    cv_cfg = config["cv"]
    cv = RepeatedKFold(
        n_splits=cv_cfg["n_splits"],
        n_repeats=cv_cfg["n_repeats"],
        random_state=config["split"]["random_state"],
    )
    scores = cross_validate(
        clone(model), X, y, cv=cv, scoring=("r2", "neg_root_mean_squared_error")
    )
    return {
        "r2_mean": float(scores["test_r2"].mean()),
        "r2_std": float(scores["test_r2"].std()),
        "rmse_mean": float(-scores["test_neg_root_mean_squared_error"].mean()),
        "rmse_std": float(scores["test_neg_root_mean_squared_error"].std()),
    }
