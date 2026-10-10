"""Tests de l'API : le modèle est injecté, aucun fichier global n'est utilisé."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sales_prediction.api.main import create_app

BUDGET = {"tv": 150, "radio": 25, "newspaper": 30}


@pytest.fixture
def client(model_service):
    with TestClient(create_app(model_service=model_service)) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_home_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200 and "text/html" in response.headers["content-type"]


def test_static_assets_are_served(client):
    css, js = client.get("/static/styles.css"), client.get("/static/app.js")
    assert css.status_code == 200 and "text/css" in css.headers["content-type"]
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    page = client.get("/").text
    assert "/static/styles.css" in page and "/static/app.js" in page


def test_predict(client):
    response = client.post("/predict", json=BUDGET, headers={"X-Request-ID": "abc"})
    body = response.json()
    assert response.status_code == 200
    assert 10 < body["predicted_sales"] < 20
    assert set(body["marginal_effects"]) == {"TV", "Radio", "Newspaper"}
    assert body["marginal_effects"]["Newspaper"] == 0
    assert response.headers["X-Request-ID"] == "abc"


def test_newspaper_is_optional(client):
    assert client.post("/predict", json={"tv": 100, "radio": 10}).status_code == 200


def test_extrapolation_returns_warning(client):
    body = client.post("/predict", json={**BUDGET, "tv": 600}).json()
    assert len(body["warnings"]) == 1


@pytest.mark.parametrize("override", [{"tv": -1}, {"radio": "beaucoup"}, {"tv": 5000}, {"unknown": 1}])
def test_invalid_input_returns_422(client, override):
    assert client.post("/predict", json={**BUDGET, **override}).status_code == 422


def test_optimize_improves_on_current_split(client):
    response = client.post("/optimize", json={"total_budget": 205, "current": BUDGET})
    body = response.json()
    assert response.status_code == 200
    assert abs(sum(body["allocation"].values()) - 205) < 0.05
    assert body["uplift"] > 0 and body["uplift_pct"] > 0


def test_optimize_without_current(client):
    body = client.post("/optimize", json={"total_budget": 100}).json()
    assert body["current_predicted_sales"] is None


def test_optimize_rejects_impossible_budget(client):
    assert client.post("/optimize", json={"total_budget": 2500}).status_code == 422
    assert client.post("/optimize", json={"total_budget": 0}).status_code == 422


def test_model_info(client):
    body = client.get("/model/info").json()
    assert "metrics" in body and "coefficients" in body
