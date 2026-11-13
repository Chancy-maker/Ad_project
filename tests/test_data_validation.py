import numpy as np
import pandas as pd
import pytest

from ad_optimizer.data.preprocessing import (
    clean_internos,
    ensure_ids_and_date,
    normalize_columns,
)
from ad_optimizer.data.synthetic_generator import generate_internos_proxy, holiday_features
from ad_optimizer.data.validation import DataValidationError, raise_if_invalid, validate_dataframe


def test_validation_detects_missing_columns_dupes_negatives():
    df = pd.DataFrame({"a": [1, 1, -2], "b": [1, 1, np.nan]})
    rep = validate_dataframe(df, required=["a", "zzz"], non_negative=["a"])
    assert "zzz" in rep.missing_columns and not rep.ok
    assert rep.negative_counts == {"a": 1}
    assert rep.null_counts == {"b": 1}
    with pytest.raises(DataValidationError):
        raise_if_invalid(rep)


def test_empty_dataset_is_an_error():
    assert not validate_dataframe(pd.DataFrame(), required=[]).ok


def test_normalize_columns_matches_internos_schema():
    df = pd.DataFrame(columns=["Requests one week", "Day of week", "Video type", "Requests"])
    assert list(normalize_columns(df).columns) == ["requests_one_week", "day_of_week", "video_type", "requests"]


def test_clean_internos_drops_duplicates_and_null_target():
    n = 40
    df = pd.DataFrame({"requests_one_day": np.arange(n, dtype=float), "requests": np.arange(n, dtype=float),
                       "video_type": ["a"] * n})
    df.loc[3, "requests"] = np.nan
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)     # un doublon exact
    out, rep, actions = clean_internos(df)
    assert len(out) == n - 1 and actions["rows_dropped"] == 2


def test_clean_internos_blocks_when_target_mostly_missing():
    df = pd.DataFrame({"requests_one_day": [1, 2, 3, 4], "requests": [np.nan, np.nan, 1.0, 2.0],
                       "video_type": ["a"] * 4})
    with pytest.raises(DataValidationError):
        clean_internos(df)


def test_ensure_ids_and_date_flags_derived_columns():
    df = pd.DataFrame({"day_of_year": [1, 2, 3], "video_type": ["x", "y", "x"], "requests": [1, 2, 3]})
    out, notes = ensure_ids_and_date(df)
    assert "date" in out and "media_id" in out and len(notes) == 2


def test_proxy_is_deterministic_and_schema_complete():
    a = generate_internos_proxy(15, seed=3)
    b = generate_internos_proxy(15, seed=3)
    c = generate_internos_proxy(15, seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)
    for col in ("requests", "requests_one_day", "requests_one_week", "video_type", "days_published"):
        assert col in a.columns
    assert (a["requests"] >= 0).all()


def test_holiday_features():
    hf = holiday_features(pd.Series(pd.to_datetime(["2017-12-25", "2017-12-20"])))
    assert hf["is_holiday"].tolist() == [1, 0]
    assert hf["next_holidays"].tolist() == [0, 5]


# ---- fichiers Internos déjà découpés et normalisés (train / validation / test)
def _write_presplit(dirpath, n=60, seed=0):
    rng = np.random.default_rng(seed)
    cols = ("rep_one_week rep_two_week rep_one_day rep_two_day day_week_sin day_week_cos day_month_sin day_month_cos "
            "day_year_sin day_year_cos current_month_sin current_month_cos current_comments comments_in_section "
            "recent_publications publications_in_section duration current_votes votes_in_section days_published "
            "holidays_left next_holiday video_type_0 video_type_1 video_type_2 video_type_3 reproductions").split()
    for name in ("train", "validation", "test"):
        d = pd.DataFrame(rng.random((n, len(cols))), columns=cols)
        ang = rng.random(n) * 2 * np.pi
        d["day_year_sin"], d["day_year_cos"] = np.sin(ang), np.cos(ang)
        for i, c in enumerate(["video_type_0", "video_type_1", "video_type_2", "video_type_3"]):
            d[c] = np.where(rng.random(n) < 0.3, 0.2 * (i + 1), -0.2 * (4 - i))
        d["reproductions"] = rng.integers(0, 50, n) / 49.0
        d.to_csv(dirpath / f"{name}.csv", sep=" ", index=False)


def test_presplit_files_are_detected_and_loaded(tmp_path):
    from ad_optimizer.data.preprocessing import find_presplit_files, load_internos_presplit
    assert find_presplit_files(tmp_path) is None
    _write_presplit(tmp_path)
    files = find_presplit_files(tmp_path)
    assert files is not None and set(files) == {"train", "val", "test"}
    df, notes = load_internos_presplit(files)
    assert set(df["split"]) == {"train", "val", "test"} and len(df) == 180
    for c in ("requests", "requests_one_day", "requests_two_days", "vt_0", "video_type", "date"):
        assert c in df.columns
    assert set(df[["vt_0", "vt_1", "vt_2", "vt_3"]].stack().unique()) <= {0, 1}
    assert notes and df["date"].between("2017-01-01", "2017-12-31").all()


def test_target_scale_inference():
    from ad_optimizer.data.preprocessing import infer_target_scale
    counts = pd.Series([0, 1, 2, 3, 5, 100, 400])
    scale, _ = infer_target_scale(counts / 400.0)
    assert scale == pytest.approx(400.0)
    scale, msg = infer_target_scale(pd.Series(np.random.default_rng(0).random(500) ** 3))
    assert scale is None and "non déductible" in msg


def test_presplit_split_is_kept_and_pipeline_runs(tmp_path):
    from ad_optimizer.common.config import Paths, load_config
    from ad_optimizer.common.model_registry import load_active
    from ad_optimizer.data.preprocessing import prepare_inventory_dataset
    from ad_optimizer.inventory import train as inv
    from ad_optimizer.inventory.predict import predict_requests
    paths = Paths(tmp_path).ensure()
    _write_presplit(paths.data_raw, n=300)
    cfg = load_config("inventory", paths, {"hyperparameter_search": False, "importance_sample": 100,
                                           "models": ["naive", "lightgbm"]})
    df, mf = prepare_inventory_dataset(paths, cfg)
    assert mf["predefined_split"] and not mf["has_media_id"] and mf["is_real"]
    res = inv.train_inventory(df, mf, paths, cfg, progress=lambda m: None)
    assert res["metadata"]["split"]["type"] == "predefined"
    assert res["metadata"]["split"]["n_train"] == 300 and res["metadata"]["split"]["n_test"] == 300
    bundle, _ = load_active("inventory", paths)
    assert len(predict_requests(df.head(20), bundle)) == 20
    assert bundle["model"] is not None and "vt_0" in bundle["features"]["num"] and bundle["features"]["cat"] == []
