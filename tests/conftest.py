"""Fixtures partagées : le modèle de test est entraîné une seule fois par session."""

from __future__ import annotations

from pathlib import Path

import pytest

from sales_prediction.config import load_config
from sales_prediction.predict import ModelService
from sales_prediction.train import train


@pytest.fixture(scope="session")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def model_dir(config, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("model")
    train(config, model_dir=out, reports_dir=out / "reports", save_processed_data=False)
    return out


@pytest.fixture(scope="session")
def model_service(model_dir) -> ModelService:
    return ModelService.load(model_dir)
