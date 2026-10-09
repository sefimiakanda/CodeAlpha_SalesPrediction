import json

import pytest

from sales_prediction.config import METADATA_FILENAME, MODEL_FILENAME
from sales_prediction.train import ModelQualityError, train


def test_artifacts_are_written(model_dir):
    assert (model_dir / MODEL_FILENAME).exists()
    assert (model_dir / METADATA_FILENAME).exists()
    assert (model_dir / "reports" / "metrics.json").exists()


def test_metadata_is_complete(model_dir):
    meta = json.loads((model_dir / METADATA_FILENAME).read_text())
    assert meta["channels"] == ["TV", "Radio", "Newspaper"]
    assert meta["engineered_columns"] == ["TV", "Radio", "TV_x_Radio", "TV_sq"]
    assert set(meta["feature_ranges"]) == {"TV", "Radio", "Newspaper"}
    assert meta["metrics"]["test"]["r2"] > 0.95
    assert len(meta["data_sha256"]) == 64


def test_tv_has_diminishing_returns_and_radio_synergy(model_dir):
    coefs = json.loads((model_dir / METADATA_FILENAME).read_text())["coefficients"]
    assert coefs["TV_sq"] < 0  # rendements décroissants de la TV
    assert coefs["TV_x_Radio"] > 0  # synergie TV x Radio


def test_quality_gate_blocks_bad_model(config, tmp_path):
    strict = {**config, "quality_gate": {"min_r2": 0.9999}}
    with pytest.raises(ModelQualityError):
        train(strict, model_dir=tmp_path, reports_dir=tmp_path, save_processed_data=False)
    assert not (tmp_path / MODEL_FILENAME).exists()


def test_training_is_reproducible(config, tmp_path):
    first = train(config, tmp_path / "a", tmp_path / "ra", save_processed_data=False)
    second = train(config, tmp_path / "b", tmp_path / "rb", save_processed_data=False)
    assert first["metrics"] == second["metrics"]
