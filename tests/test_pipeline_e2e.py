"""Bout en bout : préparation -> entraînement -> sauvegarde -> rechargement -> prédiction -> export CSV."""
import numpy as np
import pandas as pd
import pytest

from ad_optimizer.common.model_registry import (
    ModelNotFoundError,
    active_metadata,
    list_versions,
    load_active,
    read_experiments,
)
from ad_optimizer.data.preprocessing import prepare_inventory_dataset
from ad_optimizer.inventory import train as inv
from ad_optimizer.inventory.predict import estimate_inventory, predict_requests
from ad_optimizer.under_delivery import train as ud
from ad_optimizer.under_delivery.predict import local_explanation, score_snapshots


def test_load_without_model_raises_clear_error(tmp_path):
    from ad_optimizer.common.config import Paths
    with pytest.raises(ModelNotFoundError):
        load_active("inventory", Paths(tmp_path).ensure())


@pytest.fixture(scope="module")
def inventory_run(small_paths, small_inv_cfg):
    df, manifest = prepare_inventory_dataset(small_paths, small_inv_cfg)
    res = inv.train_inventory(df, manifest, small_paths, small_inv_cfg, seed=7, progress=lambda m: None)
    return df, res


def test_inventory_train_save_reload_predict(small_paths, inventory_run):
    df, res = inventory_run
    bundle, meta = load_active("inventory", small_paths)
    assert meta["version"] == res["version"] and meta["split"]["type"] == "temporal"
    assert "naive" in meta["metrics"] and meta["selected_model"] in meta["metrics"]
    pred = predict_requests(df.head(200), bundle)
    assert len(pred) == 200 and (pred["y_pred"] >= 0).all()
    assert (pred["y_low"] <= pred["y_pred"] + 1e-9).all() and (pred["y_high"] >= pred["y_pred"] - 1e-9).all()
    inv_est = estimate_inventory(pred, {"ad_opportunities_per_view": 2.0, "eligibility_rate": 0.8})
    np.testing.assert_allclose(inv_est["eligible_impressions"], pred["y_pred"] * 1.6)
    csv = inv_est.to_csv(index=False)
    assert "eligible_impressions" in csv.splitlines()[0]


def test_inventory_selection_never_uses_test(inventory_run):
    _, res = inventory_run
    m = res["metadata"]["metrics"]
    best_val = min(m, key=lambda k: m[k]["val"]["wape"])
    assert res["metadata"]["selected_model"] == best_val
    assert res["metadata"]["split"]["train_end"] < res["metadata"]["split"]["test_start"]


def test_inventory_prediction_is_reproducible(small_paths, small_inv_cfg, inventory_run):
    df, _ = inventory_run
    r1 = inv.train_inventory(df, None, small_paths, small_inv_cfg, seed=7, progress=lambda m: None)
    p1 = r1["test_predictions"]["y_pred"].to_numpy()
    r2 = inv.train_inventory(df, None, small_paths, small_inv_cfg, seed=7, progress=lambda m: None)
    np.testing.assert_allclose(p1, r2["test_predictions"]["y_pred"].to_numpy())
    assert len(list_versions("inventory", small_paths)) >= 3


def test_under_delivery_end_to_end(small_paths, small_ud_cfg):
    manifest = ud.prepare_data(small_paths, small_ud_cfg, seed=5)
    assert 0.05 < manifest["prevalence_under_delivery"] < 0.8
    snaps = pd.read_parquet(small_paths.data_synthetic / "snapshots.parquet")
    res = ud.train_under_delivery(snaps, manifest, small_paths, small_ud_cfg, seed=5, progress=lambda m: None)
    bundle, meta = load_active("under_delivery", small_paths)
    assert {"dummy", "pacing_rule", "logistic_regression", "lightgbm"} <= set(meta["metrics"])
    # le modèle sélectionné doit faire mieux que le classifieur majoritaire en PR-AUC (validation)
    assert meta["metrics"][meta["selected_model"]]["val"]["pr_auc"] > meta["metrics"]["dummy"]["val"]["pr_auc"]
    scored = score_snapshots(snaps.head(50), bundle)
    assert scored["proba"].between(0, 1).all() and set(scored["risk_class"]) <= {"faible", "modéré", "élevé"}
    exp = local_explanation(bundle, scored.iloc[[0]])
    assert len(exp) > 0 and {"variable", "contribution"} <= set(exp.columns)
    assert meta["holdout_batch"]["n"] > 0
    assert res["version"] == active_metadata("under_delivery", small_paths)["version"]


def test_under_delivery_data_reproducible(small_paths, small_ud_cfg):
    a = ud.prepare_data(small_paths, small_ud_cfg, seed=21)["data_version"]
    b = ud.prepare_data(small_paths, small_ud_cfg, seed=21)["data_version"]
    c = ud.prepare_data(small_paths, small_ud_cfg, seed=22)["data_version"]
    assert a == b and a != c


def test_experiments_are_logged(small_paths, inventory_run):
    kinds = {e["kind"] for e in read_experiments(small_paths)}
    assert "inventory" in kinds
