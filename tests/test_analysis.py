import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from sales_prediction import analysis


class FakeModel:
    """Modèle linéaire connu : ventes = 3*TV + 1*Radio + 0*Newspaper."""

    def predict(self, X):
        return 3 * X["TV"].to_numpy() + 1 * X["Radio"].to_numpy()


def _synthetic():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {"a": rng.normal(size=300), "b": rng.normal(size=300), "noise": rng.normal(size=300)}
    )
    y = 2.0 * X["a"] - 1.0 * X["b"] + 4.0 + rng.normal(scale=0.5, size=300)
    return X, y


def test_ols_matches_sklearn_coefficients():
    X, y = _synthetic()
    table, summary = analysis.ols_summary(X, y)
    reference = LinearRegression().fit(X, y)
    assert np.allclose(table.loc[list(X.columns), "coef"], reference.coef_)
    assert abs(table.loc["intercept", "coef"] - reference.intercept_) < 1e-9
    assert abs(summary["r2"] - reference.score(X, y)) < 1e-9


def test_ols_flags_irrelevant_feature():
    X, y = _synthetic()
    table, _ = analysis.ols_summary(X, y)
    assert table.loc["a", "p_value"] < 0.001 and table.loc["b", "p_value"] < 0.001
    assert table.loc["noise", "p_value"] > 0.05
    assert (table["ci_low"] < table["coef"]).all() and (table["coef"] < table["ci_high"]).all()


def test_vif_is_low_for_independent_features():
    X, _ = _synthetic()
    assert (analysis.variance_inflation_factors(X) < 1.2).all()


def test_marginal_effects_of_linear_model():
    effects = analysis.marginal_effects(FakeModel(), {"TV": 10, "Radio": 5, "Newspaper": 3})
    assert abs(effects["TV"] - 3) < 1e-9 and abs(effects["Radio"] - 1) < 1e-9
    assert abs(effects["Newspaper"]) < 1e-9


def test_what_if_percentage_change():
    result = analysis.what_if(FakeModel(), {"TV": 100, "Radio": 10, "Newspaper": 0}, {"TV": 10})
    assert abs(result["delta"] - 30) < 1e-9 and result["sales_after"] > result["sales_before"]


def test_out_of_range_warnings():
    ranges = {"TV": {"min": 0, "max": 300}, "Radio": {"min": 0, "max": 50}}
    assert analysis.out_of_range({"TV": 100, "Radio": 10}, ranges) == []
    assert len(analysis.out_of_range({"TV": 400, "Radio": 10}, ranges)) == 1


def test_optimizer_respects_total_and_bounds():
    upper = {"TV": 100.0, "Radio": 50.0, "Newspaper": 50.0}
    result = analysis.optimize_budget(FakeModel(), 120, ["TV", "Radio", "Newspaper"], upper, step=5)
    allocation = result["allocation"]
    assert abs(sum(allocation.values()) - 120) < 1e-6
    assert all(allocation[ch] <= upper[ch] + 1e-6 for ch in upper)
    assert allocation["TV"] == 100.0  # TV rapporte le plus : elle est servie jusqu'à sa borne


def test_optimizer_rejects_impossible_total():
    upper = {"TV": 10.0, "Radio": 10.0, "Newspaper": 10.0}
    try:
        analysis.optimize_budget(FakeModel(), 500, ["TV", "Radio", "Newspaper"], upper)
    except ValueError:
        return
    raise AssertionError("ValueError attendue")
