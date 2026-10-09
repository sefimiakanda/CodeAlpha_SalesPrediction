import pytest

from sales_prediction.predict import ModelService

BUDGET = {"TV": 150.0, "Radio": 25.0, "Newspaper": 30.0}


def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ModelService.load(tmp_path)


def test_prediction_is_plausible(model_service):
    assert 10 < model_service.predict_one(BUDGET) < 20


def test_newspaper_has_no_effect(model_service):
    assert model_service.predict_one(BUDGET) == model_service.predict_one(
        {**BUDGET, "Newspaper": 0.0}
    )
    assert model_service.effects(BUDGET)["Newspaper"] == 0.0


def test_radio_has_higher_marginal_return_than_tv(model_service):
    effects = model_service.effects(BUDGET)
    assert effects["Radio"] > effects["TV"] > 0


def test_extrapolation_warning(model_service):
    assert model_service.warnings(BUDGET) == []
    assert model_service.warnings({**BUDGET, "TV": 500.0})


def test_optimizer_beats_current_split(model_service):
    best = model_service.optimize(sum(BUDGET.values()))
    assert best["predicted_sales"] > model_service.predict_one(BUDGET)
    assert abs(sum(best["allocation"].values()) - sum(BUDGET.values())) < 1e-6
    assert best["allocation"]["Newspaper"] < 1.0  # la presse n'apporte rien : budget réalloué


def test_predict_many(model_service):
    assert len(model_service.predict_many([BUDGET, {**BUDGET, "TV": 200.0}])) == 2
