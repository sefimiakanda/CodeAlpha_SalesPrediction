"""Ingénierie de variables et construction du pipeline scikit-learn.

Le pipeline reçoit TOUJOURS les 3 budgets bruts (TV, Radio, Newspaper) : la sélection de variables
et les termes d'interaction sont faits à l'intérieur, donc l'API ne peut pas les oublier.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def add_engineered_features(
    X: pd.DataFrame,
    selected: list[str],
    interaction: bool = True,
    quadratic: list[str] | None = None,
) -> pd.DataFrame:
    """Garde les variables sélectionnées et ajoute interactions (A x B) et carrés (A^2)."""
    out = X[selected].copy()
    if interaction:
        for a, b in combinations(selected, 2):
            out[f"{a}_x_{b}"] = X[a] * X[b]
    for col in quadratic or []:
        out[f"{col}_sq"] = X[col] ** 2
    return out


def build_pipeline(config: dict[str, Any]) -> Pipeline:
    """Pipeline complet : ingénierie de variables -> régression linéaire."""
    feat = config["features"]
    transformer = FunctionTransformer(
        add_engineered_features,
        kw_args={
            "selected": list(feat["selected"]),
            "interaction": bool(feat.get("interaction", False)),
            "quadratic": list(feat.get("quadratic", [])),
        },
        validate=False,
    )
    regressor = LinearRegression(fit_intercept=config["model"]["fit_intercept"])
    return Pipeline([("features", transformer), ("regressor", regressor)])
