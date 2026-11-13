import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ad_optimizer.common.schemas import ConversionAssumptions
from ad_optimizer.data.synthetic_generator import generate_internos_proxy
from ad_optimizer.inventory.evaluation import interval_coverage, regression_metrics
from ad_optimizer.inventory.features import (
    FORBIDDEN_FEATURES,
    NaiveBaseline,
    add_engineered,
    available_features,
    group_split,
    temporal_split,
)
from ad_optimizer.inventory.predict import estimate_inventory


@pytest.fixture(scope="module")
def df():
    d = generate_internos_proxy(40, seed=1)
    return add_engineered(d)


def test_temporal_split_is_chronological_and_disjoint(df):
    s = temporal_split(df, 0.15, 0.15)
    d = pd.to_datetime(df["date"])
    assert d[s == "train"].max() <= d[s == "val"].min() <= d[s == "val"].max() <= d[s == "test"].min()
    assert set(s.unique()) == {"train", "val", "test"}


def test_group_split_never_shares_a_media(df):
    s = group_split(df, 0.2, 0.2, seed=0)
    groups = {k: set(df.loc[s == k, "media_id"]) for k in ("train", "val", "test")}
    assert not (groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"])


def test_no_identifier_used_as_feature(df):
    num, cat = available_features(df)
    assert not (set(num + cat) & FORBIDDEN_FEATURES)
    assert "requests" not in num


def test_engineered_features(df):
    assert df["is_weekend"].isin([0, 1]).all()
    assert np.isfinite(df["ratio_day_week"]).all()


def test_naive_baseline_returns_last_value(df):
    assert np.allclose(NaiveBaseline().fit(df).predict(df), df["requests_one_day"])


def test_metrics_known_values():
    m = regression_metrics([10, 20, 0], [12, 18, 1])
    assert m["mae"] == pytest.approx((2 + 2 + 1) / 3)
    assert m["wape"] == pytest.approx(5 / 30)
    assert m["mape_pos"] == pytest.approx((0.2 + 0.1) / 2)   # la valeur nulle est exclue du MAPE
    assert interval_coverage([1, 5], [0, 0], [2, 4]) == 0.5


def test_conversion_formula_and_bounds():
    pred = pd.DataFrame({"y_pred": [100.0], "y_low": [80.0], "y_high": [130.0]})
    out = estimate_inventory(pred, ConversionAssumptions(ad_opportunities_per_view=2.0, eligibility_rate=0.5))
    assert out["theoretical_ad_opportunities"].iloc[0] == 200
    assert out["eligible_impressions"].iloc[0] == 100
    assert out["eligible_impressions_high"].iloc[0] == 130
    with pytest.raises(ValidationError):
        ConversionAssumptions(ad_opportunities_per_view=2, eligibility_rate=1.5)
    with pytest.raises(ValidationError):
        ConversionAssumptions(ad_opportunities_per_view=0, eligibility_rate=0.5)
