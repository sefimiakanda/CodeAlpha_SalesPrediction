"""Analyse d'impact publicitaire : statistiques, effets marginaux, scénarios et optimisation.

Toutes les fonctions de simulation acceptent n'importe quel modèle exposant `predict(DataFrame)`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------- statistiques (OLS) ----------
def ols_summary(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, dict[str, float]]:
    """Régression linéaire par moindres carrés avec inférence statistique (sans statsmodels).

    Renvoie un tableau (coefficient, erreur standard, t, p-value, IC 95 %) et des statistiques
    globales (R², R² ajusté, statistique F, p-value de F).
    """
    A = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    n, k = A.shape
    beta, *_ = np.linalg.lstsq(A, y.to_numpy(dtype=float), rcond=None)
    residuals = y.to_numpy(dtype=float) - A @ beta
    dof = n - k
    sigma2 = float(residuals @ residuals) / dof  # variance résiduelle
    std_err = np.sqrt(np.diag(sigma2 * np.linalg.inv(A.T @ A)))
    t_values = beta / std_err
    p_values = 2 * stats.t.sf(np.abs(t_values), dof)
    margin = stats.t.ppf(0.975, dof) * std_err

    table = pd.DataFrame(
        {
            "coef": beta,
            "std_err": std_err,
            "t": t_values,
            "p_value": p_values,
            "ci_low": beta - margin,
            "ci_high": beta + margin,
        },
        index=["intercept", *X.columns],
    )
    ss_res = float(residuals @ residuals)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    f_stat = ((ss_tot - ss_res) / (k - 1)) / (ss_res / dof)
    summary = {
        "n": float(n),
        "r2": r2,
        "adj_r2": 1 - (1 - r2) * (n - 1) / dof,
        "f_stat": float(f_stat),
        "f_p_value": float(stats.f.sf(f_stat, k - 1, dof)),
    }
    return table, summary


def variance_inflation_factors(X: pd.DataFrame) -> pd.Series:
    """VIF de chaque variable : > 5 (ou 10) signale une forte multicolinéarité."""
    values = {}
    for col in X.columns:
        others = X.drop(columns=col).to_numpy(dtype=float)
        A = np.column_stack([np.ones(len(X)), others])
        target = X[col].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(A, target, rcond=None)
        resid = target - A @ beta
        r2 = 1 - float(resid @ resid) / float(((target - target.mean()) ** 2).sum())
        values[col] = 1 / (1 - r2)
    return pd.Series(values, name="VIF")


# ---------------------------------------------------------------- simulation ------------------
def _frame(budgets: Mapping[str, float] | Sequence[Mapping[str, float]]) -> pd.DataFrame:
    if isinstance(budgets, Mapping):
        return pd.DataFrame([dict(budgets)])
    return pd.DataFrame([dict(b) for b in budgets])


def predict_sales(model: Any, budgets: Mapping[str, float]) -> float:
    """Ventes prédites pour un budget {canal: montant}."""
    return float(model.predict(_frame(budgets))[0])


def marginal_effects(
    model: Any, budget: Mapping[str, float], step: float = 1.0
) -> dict[str, float]:
    """Ventes supplémentaires pour +`step` (milliers de $) sur chaque canal, les autres fixes."""
    base = predict_sales(model, budget)
    effects = {}
    for channel in budget:
        bumped = dict(budget)
        bumped[channel] = bumped[channel] + step
        effects[channel] = (predict_sales(model, bumped) - base) / step
    return effects


def what_if(
    model: Any, base: Mapping[str, float], pct_changes: Mapping[str, float]
) -> dict[str, float]:
    """Impact d'une variation en % des budgets (ex. {"TV": 10} = +10 % sur la TV)."""
    changed = {k: v * (1 + pct_changes.get(k, 0.0) / 100) for k, v in base.items()}
    before, after = predict_sales(model, base), predict_sales(model, changed)
    return {
        "sales_before": before,
        "sales_after": after,
        "delta": after - before,
        "delta_pct": (after - before) / before * 100 if before else float("nan"),
    }


def out_of_range(
    budget: Mapping[str, float], ranges: Mapping[str, Mapping[str, float]]
) -> list[str]:
    """Messages d'avertissement pour les budgets hors de la plage observée à l'entraînement."""
    warnings = []
    for channel, value in budget.items():
        low, high = ranges[channel]["min"], ranges[channel]["max"]
        if value < low or value > high:
            warnings.append(
                f"{channel} = {value:g} est hors de la plage observée à l'entraînement "
                f"({low:g} à {high:g}) : la prévision est une extrapolation, à interpréter avec prudence."
            )
    return warnings


def optimize_budget(
    model: Any,
    total: float,
    channels: Sequence[str],
    upper_bounds: Mapping[str, float],
    step: float = 1.0,
) -> dict[str, Any]:
    """Meilleure répartition d'un budget total entre 3 canaux (recherche exhaustive sur une grille).

    Les bornes supérieures sont les maxima observés à l'entraînement : on évite d'extrapoler.
    """
    if len(channels) != 3:
        raise ValueError("optimize_budget gère exactement 3 canaux")
    a, b, c = channels
    if total <= 0 or total > sum(upper_bounds[ch] for ch in channels):
        raise ValueError(
            f"Budget total hors limites : doit être entre 0 et {sum(upper_bounds.values()):g}"
        )
    grid_a = np.arange(0, min(total, upper_bounds[a]) + 1e-9, step)
    grid_b = np.arange(0, min(total, upper_bounds[b]) + 1e-9, step)
    A, B = np.meshgrid(grid_a, grid_b, indexing="ij")
    C = total - A - B  # le 3e canal reçoit le reste
    mask = (C >= -1e-9) & (upper_bounds[c] + 1e-9 >= C)
    if not mask.any():
        raise ValueError("Aucune répartition réalisable avec ces bornes")
    candidates = pd.DataFrame({a: A[mask], b: B[mask], c: np.clip(C[mask], 0, None)})
    predictions = model.predict(candidates)
    best = int(np.argmax(predictions))
    return {
        "allocation": {ch: float(candidates.iloc[best][ch]) for ch in channels},
        "predicted_sales": float(predictions[best]),
        "n_candidates": int(mask.sum()),
    }
