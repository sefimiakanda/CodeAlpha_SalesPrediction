import pandas as pd
import pytest

from sales_prediction.config import resolve_path
from sales_prediction.data import clean, file_sha256, load_clean, load_raw, split_xy, validate


def test_raw_file_contains_index_column(config):
    assert "Unnamed: 0" in load_raw(config).columns


def test_clean_drops_index_and_keeps_contract(config):
    df = load_clean(config)
    assert list(df.columns) == ["TV", "Radio", "Newspaper", "Sales"]
    assert len(df) == 200


def test_clean_removes_duplicates(config):
    df = load_clean(config)
    doubled = pd.concat([df, df.head(3)], ignore_index=True)
    assert len(clean(doubled, config)) == len(df)


def test_validate_rejects_missing_column(config):
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        validate(load_clean(config).drop(columns="Radio"), config)


def test_validate_rejects_nan(config):
    df = load_clean(config)
    df.loc[0, "TV"] = None
    with pytest.raises(ValueError, match="manquantes"):
        validate(df, config)


def test_validate_rejects_negative_spend(config):
    df = load_clean(config)
    df.loc[0, "Radio"] = -1.0
    with pytest.raises(ValueError, match="négatives"):
        validate(df, config)


def test_validate_rejects_non_numeric(config):
    df = load_clean(config).astype({"TV": "object"})
    df.loc[0, "TV"] = "abc"
    with pytest.raises(ValueError, match="numérique"):
        validate(df, config)


def test_split_xy_shapes(config):
    X, y = split_xy(load_clean(config), config)
    assert list(X.columns) == config["data"]["channels"]
    assert y.name == "Sales" and len(X) == len(y)


def test_file_sha256_is_stable(config):
    path = resolve_path(config["data"]["raw_path"])
    assert file_sha256(path) == file_sha256(path) and len(file_sha256(path)) == 64
