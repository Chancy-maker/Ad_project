"""Entraînement du module B : classification du risque de sous-impression."""
from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..common.config import Paths, default_paths, load_config
from ..common.logging import get_logger
from ..common.model_registry import log_experiment, save_model
from ..data.preprocessing import data_version
from ..data.synthetic_generator import generate_campaigns
from ..inventory.features import make_preprocessor
from .aggregation import prepare_delivery_profiles
from .evaluation import (
    calibration_table,
    classification_metrics,
    select_threshold,
    subgroup_metrics,
    threshold_sweep,
)
from .snapshot_builder import (
    CAT_FEATURES,
    FORBIDDEN_FEATURES,
    NUM_FEATURES,
    TARGET,
    build_snapshots,
)

log = get_logger()
ALL_MODELS = ["dummy", "pacing_rule", "logistic_regression", "random_forest", "hist_gb", "lightgbm"]

GRIDS = {
    "hist_gb": [dict(learning_rate=0.1, max_iter=150, max_leaf_nodes=15),
                dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0),
                dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=31, min_samples_leaf=40)],
    "lightgbm": [dict(n_estimators=200, learning_rate=0.05, num_leaves=15),
                 dict(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=40),
                 dict(n_estimators=250, learning_rate=0.05, num_leaves=7, reg_lambda=1.0)],
}


class PacingRuleBaseline(BaseEstimator, ClassifierMixin):
    """Baseline métier sans apprentissage : risque élevé si le rythme récent < rythme nécessaire."""

    def __init__(self, column: str = "recent_to_required_rate_ratio", k: float = 4.0):
        self.column = column
        self.k = k

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        r = np.clip(np.asarray(X[self.column], float), 0, RATIO_MAX)
        p1 = 1 / (1 + np.exp(self.k * (r - 1.0)))
        return np.c_[1 - p1, p1]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


RATIO_MAX = 10.0


def build_model(name: str, num, cat, seed: int, params: dict | None = None):
    params = params or {}
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "pacing_rule":
        return PacingRuleBaseline()
    if name == "logistic_regression":
        return Pipeline([("pre", make_preprocessor(num, cat, True)),
                         ("model", LogisticRegression(max_iter=2000, C=1.0, random_state=seed))])
    if name == "random_forest":
        p = dict(n_estimators=250, min_samples_leaf=5, n_jobs=-1, random_state=seed)
        return Pipeline([("pre", make_preprocessor(num, cat, False)),
                         ("model", RandomForestClassifier(**{**p, **params}))])
    if name == "hist_gb":
        p = dict(learning_rate=0.1, max_iter=150, max_leaf_nodes=15, random_state=seed)
        return Pipeline([("pre", make_preprocessor(num, cat, False)),
                         ("model", HistGradientBoostingClassifier(**{**p, **params}))])
    if name == "lightgbm":
        p = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, random_state=seed, verbose=-1,
                 n_jobs=2)
        return Pipeline([("pre", make_preprocessor(num, cat, False)),
                         ("model", LGBMClassifier(**{**p, **params}))])
    raise ValueError(f"Modèle inconnu : {name}")


def group_split_ids(campaign_ids: np.ndarray, val_frac: float, test_frac: float, seed: int) -> dict:
    """Split train/val/test PAR campagne (jamais par ligne de snapshot)."""
    ids = np.unique(campaign_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_test, n_val = int(round(len(ids) * test_frac)), int(round(len(ids) * val_frac))
    return {"test": set(ids[:n_test]), "val": set(ids[n_test:n_test + n_val]),
            "train": set(ids[n_test + n_val:])}


def assert_no_leakage(features: list[str]) -> None:
    bad = set(features) & FORBIDDEN_FEATURES
    if bad:
        raise AssertionError(f"Fuite de données : variables interdites en entrée : {sorted(bad)}")


# --------------------------------------------------------------------------- données synthétiques
def prepare_data(paths: Paths | None = None, cfg: dict | None = None, n_campaigns: int | None = None,
                 seed: int | None = None) -> dict:
    """Génère campagnes / livraisons / snapshots (déterministe) et écrit data/synthetic/*."""
    paths = (paths or default_paths()).ensure()
    cfg = cfg or load_config("under_delivery", paths)
    seed = int(seed if seed is not None else cfg.get("random_seed", 42))
    g = cfg["generator"]
    n = int(n_campaigns or g["n_campaigns"])
    profiles, info = prepare_delivery_profiles(paths)
    camps, daily = generate_campaigns(n, seed, profiles, g["median_daily_inventory"], g["start_date"],
                                      g["tolerance"])
    snaps = build_snapshots(camps, daily, cfg["snapshots"], g["tolerance"])
    camps.to_parquet(paths.data_synthetic / "synthetic_campaigns.parquet", index=False)
    daily.to_parquet(paths.data_synthetic / "synthetic_campaign_delivery_daily.parquet", index=False)
    snaps.to_parquet(paths.data_synthetic / "snapshots.parquet", index=False)
    manifest = {
        "n_campaigns": n, "n_daily_rows": len(daily), "n_snapshots": len(snaps), "seed": seed,
        "tolerance": g["tolerance"], "prevalence_under_delivery": float(snaps[TARGET].mean()),
        "criteo": info, "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_version": data_version(snaps.drop(columns=["snapshot_date"])),
        "median_daily_inventory": g["median_daily_inventory"],
    }
    (paths.data_synthetic / "under_delivery_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Campagnes synthétiques : %s campagnes, %s snapshots, prévalence %.1f%%",
             n, len(snaps), 100 * manifest["prevalence_under_delivery"])
    return manifest


# --------------------------------------------------------------------------- entraînement
def train_under_delivery(snapshots: pd.DataFrame | None = None, manifest: dict | None = None,
                         paths: Paths | None = None, cfg: dict | None = None,
                         models: list[str] | None = None, seed: int | None = None,
                         progress=None) -> dict:
    paths = (paths or default_paths()).ensure()
    cfg = cfg or load_config("under_delivery", paths)
    seed = int(seed if seed is not None else cfg.get("random_seed", 42))
    models = models or cfg.get("models", ALL_MODELS)
    say = progress or (lambda m: log.info(m))
    if snapshots is None:
        f = paths.data_synthetic / "snapshots.parquet"
        if not f.exists():
            raise FileNotFoundError("Snapshots absents : générez d'abord les données synthétiques.")
        snapshots = pd.read_parquet(f)
        mf = paths.data_synthetic / "under_delivery_manifest.json"
        if manifest is None and mf.exists():
            manifest = json.loads(mf.read_text(encoding="utf-8"))
    manifest = manifest or {}

    num, cat = list(NUM_FEATURES), list(CAT_FEATURES)
    assert_no_leakage(num + cat)
    ids = group_split_ids(snapshots["campaign_id"].to_numpy(), cfg["split"]["val_frac"],
                                      cfg["split"]["test_frac"], seed)
    masks = {k: snapshots["campaign_id"].isin(v).to_numpy() for k, v in ids.items()}
    # Garde-fou : aucune campagne dans deux jeux
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])
    X, y = snapshots[num + cat], snapshots[TARGET].astype(int).to_numpy()
    tr, va, te = masks["train"], masks["val"], masks["test"]
    say(f"Split par campagne : train={tr.sum()}, val={va.sum()}, test={te.sum()} snapshots")

    th = cfg["threshold"]
    fitted, results = {}, {}
    for name in models:
        say(f"Entraînement : {name}")
        t0 = time.time()
        grid = GRIDS.get(name, [None]) if cfg.get("hyperparameter_search", True) else [None]
        best = None
        for params in grid:
            est = build_model(name, num, cat, seed, params).fit(X[tr], y[tr])
            pv = est.predict_proba(X[va])[:, 1]
            score = classification_metrics(y[va], pv, 0.5)["pr_auc"]
            if best is None or score > best[0]:
                best = (score, est, params, pv)
        _, est, params, pv = best
        thr = select_threshold(y[va], pv, th["strategy"], th["recall_target"]) if name != "dummy" else 0.5
        pt = est.predict_proba(X[te])[:, 1]
        results[name] = {"val": classification_metrics(y[va], pv, thr),
                         "test": classification_metrics(y[te], pt, thr), "threshold": thr,
                         "params": params or {}, "fit_seconds": round(time.time() - t0, 1)}
        fitted[name] = est
        t = results[name]["test"]
        say(f"  {name}: PR-AUC test={t['pr_auc']:.3f} recall={t['recall']:.2f} precision={t['precision']:.2f}")

    selected = max(results, key=lambda k: np.nan_to_num(results[k]["val"]["pr_auc"], nan=-1))
    best_est, thr = fitted[selected], results[selected]["threshold"]

    # Référence pour l'explication locale (médiane / mode du TRAIN)
    reference = {c: float(snapshots.loc[tr, c].median()) for c in num}
    reference.update({c: snapshots.loc[tr, c].mode().iloc[0] for c in cat})

    # Validation sur un lot de campagnes généré séparément (autre graine)
    ho = cfg["holdout"]
    g = cfg["generator"]
    profiles = (pd.read_parquet(paths.data_processed / "criteo_profiles.parquet")
                if (paths.data_processed / "criteo_profiles.parquet").exists() else None)
    hc, hd = generate_campaigns(ho["n_campaigns"], seed + ho["seed_offset"], profiles,
                                g["median_daily_inventory"], g["start_date"], g["tolerance"], id_prefix="H")
    hs = build_snapshots(hc, hd, cfg["snapshots"], g["tolerance"])
    p_h = best_est.predict_proba(hs[num + cat])[:, 1]
    holdout_metrics = classification_metrics(hs[TARGET], p_h, thr)

    # Analyses sur le test (diagnostic uniquement, rien n'en est déduit pour le choix)
    pt = best_est.predict_proba(X[te])[:, 1]
    test_df = snapshots.loc[te, ["campaign_id", "snapshot_pct", "campaign_type", "scenario"]].copy()
    test_df["y_true"], test_df["proba"] = y[te], pt
    by_pct = subgroup_metrics(test_df, "snapshot_pct", thr)
    by_type = subgroup_metrics(test_df, "campaign_type", thr)
    by_scen = subgroup_metrics(test_df, "scenario", thr)
    sweep = threshold_sweep(y[te], pt)
    calib = calibration_table(y[te], pt)

    say("Importance des variables (permutation, validation)")
    pi = permutation_importance(best_est, X[va], y[va], scoring="average_precision", n_repeats=5,
                                random_state=seed, n_jobs=1)
    importance = (pd.DataFrame({"feature": num + cat, "importance": pi.importances_mean,
                                "std": pi.importances_std})
                  .sort_values("importance", ascending=False).reset_index(drop=True))
    for nm, tbl in (("under_delivery_by_snapshot", by_pct), ("under_delivery_by_campaign_type", by_type),
                    ("under_delivery_by_scenario", by_scen), ("under_delivery_threshold_sweep", sweep),
                    ("under_delivery_calibration", calib), ("under_delivery_feature_importance", importance)):
        tbl.to_csv(paths.reports_metrics / f"{nm}.csv", index=False)

    metadata = {
        "trained_at": datetime.now().isoformat(timespec="seconds"), "seed": seed,
        "selected_model": selected, "selection_metric": "PR-AUC (validation)",
        "threshold": thr, "threshold_strategy": th,
        "split": {"type": "group_by_campaign", "n_train_snapshots": int(tr.sum()),
                  "n_val_snapshots": int(va.sum()), "n_test_snapshots": int(te.sum()),
                  "n_train_campaigns": len(ids["train"]), "n_val_campaigns": len(ids["val"]),
                  "n_test_campaigns": len(ids["test"]),
                  "test_campaign_ids": sorted(ids["test"]), "val_campaign_ids": sorted(ids["val"])},
        "metrics": results, "holdout_batch": holdout_metrics,
        "importance": importance.head(15).to_dict("records"),
        "data_source": "Synthétique (contrats/livraisons) + profils " + manifest.get("criteo", {}).get(
            "source", "paramétriques"),
        "is_real_data": False, "data_version": manifest.get("data_version", "n/a"),
        "tolerance": manifest.get("tolerance"), "prevalence": float(y.mean()),
        "features": {"num": num, "cat": cat}, "risk_bins": cfg["risk_bins"], "candidates": list(models),
    }
    bundle = {"model": best_est, "model_name": selected, "threshold": thr, "features": {"num": num, "cat": cat},
              "reference": reference, "risk_bins": cfg["risk_bins"]}
    version = save_model("under_delivery", bundle, metadata, paths)
    log_experiment("under_delivery", {
        "version": version, "selected": selected, "seed": seed, "data_version": metadata["data_version"],
        "threshold": thr, "split": {k: v for k, v in metadata["split"].items() if not k.endswith("_ids")},
        "test_pr_auc": {k: v["test"]["pr_auc"] for k, v in results.items()},
        "test_recall": {k: v["test"]["recall"] for k, v in results.items()},
        "holdout_pr_auc": holdout_metrics["pr_auc"]}, paths)
    say(f"Modèle sous-impression « {selected} » sauvegardé ({version})")
    return {"version": version, "metadata": metadata, "importance": importance, "by_snapshot": by_pct}


def run(paths: Paths | None = None, models: list[str] | None = None, seed: int | None = None,
        n_campaigns: int | None = None, progress=None) -> dict:
    paths = (paths or default_paths()).ensure()
    cfg = load_config("under_delivery", paths)
    manifest = prepare_data(paths, cfg, n_campaigns, seed)
    snaps = pd.read_parquet(paths.data_synthetic / "snapshots.parquet")
    return train_under_delivery(snaps, manifest, paths, cfg, models, seed, progress)
