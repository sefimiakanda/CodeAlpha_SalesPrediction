import pandas as pd

from sales_prediction.data import load_clean, split_xy
from sales_prediction.features import add_engineered_features, build_pipeline


def test_engineered_columns_follow_config():
    X = pd.DataFrame({"TV": [10.0, 20.0], "Radio": [1.0, 2.0], "Newspaper": [5.0, 6.0]})
    out = add_engineered_features(X, ["TV", "Radio"], interaction=True, quadratic=["TV"])
    assert list(out.columns) == ["TV", "Radio", "TV_x_Radio", "TV_sq"]
    assert out["TV_x_Radio"].tolist() == [10.0, 40.0]
    assert out["TV_sq"].tolist() == [100.0, 400.0]


def test_unselected_feature_is_dropped():
    X = pd.DataFrame({"TV": [1.0], "Radio": [1.0], "Newspaper": [9.0]})
    assert "Newspaper" not in add_engineered_features(X, ["TV", "Radio"]).columns


def test_pipeline_ignores_newspaper(config):
    X, y = split_xy(load_clean(config), config)
    model = build_pipeline(config).fit(X, y)
    changed = X.assign(Newspaper=X["Newspaper"] + 1000)
    assert (model.predict(X) == model.predict(changed)).all()


def test_pipeline_predicts_one_value_per_row(config):
    X, y = split_xy(load_clean(config), config)
    assert build_pipeline(config).fit(X, y).predict(X.head(7)).shape == (7,)
