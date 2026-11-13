"""Entraînement du module A : baseline + modèles candidats, mêmes splits, sélection sur validation."""
from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from ..common.config import Paths, default_paths, load_config
from ..common.logging import get_logger
from ..common.model_registry import log_experiment, save_model
from ..data.preprocessing import data_version, prepare_inventory_dataset
from .evaluation import (
    conversion_sensitivity,
    error_by_group,
    error_by_volume_quantile,
    interval_coverage,
    regression_metrics,
)
from .features import (
    TARGET,
    NaiveBaseline,
    add_engineered,
    available_features,
    make_preprocessor,
    make_split,
)

log = get_logger()
ALL_MODELS = ["naive", "ridge", "random_forest", "hist_gb", "lightgbm"]

GRIDS = {
    "hist_gb": [dict(learning_rate=0.1, max_iter=200, max_leaf_nodes=31),
                dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=31),
                dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=63, l2_regularization=1.0)],
    "lightgbm": [dict(n_estimators=300, learning_rate=0.05, num_leaves=31),
                 dict(n_estimators=500, learning_rate=0.03, num_leaves=63, min_child_samples=40),
                 dict(n_estimators=400, learning_rate=0.05, num_leaves=15, reg_lambda=1.0)],
}


def _log_target(reg):
    return TransformedTargetRegressor(regressor=reg, func=np.log1p, inverse_func=np.expm1,
                                      check_inverse=False)


def build_model(name: str, num, cat, seed: int, params: dict | None = None):
    params = params or {}
    if name == "naive":
        return NaiveBaseline(calibrate="day_week_sin" in num)
    if name == "ridge":
        return _log_target(Pipeline([("pre", make_preprocessor(num, cat, True)),
                                     ("model", Ridge(alpha=1.0, **params))]))
    if name == "random_forest":
        p = dict(n_estimators=150, max_depth=16, min_samples_leaf=5, n_jobs=-1, random_state=seed)
        return _log_target(Pipeline([("pre", make_preprocessor(num, cat, False)),
                                     ("model", RandomForestRegressor(**{**p, **params}))]))
    if name == "hist_gb":
        p = dict(learning_rate=0.1, max_iter=200, max_leaf_nodes=31, random_state=seed)
        return _log_target(Pipeline([("pre", make_preprocessor(num, cat, False)),
                                     ("model", HistGradientBoostingRegressor(**{**p, **params}))]))
    if name == "lightgbm":
        p = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=seed,
                 verbose=-1, n_jobs=2)
        return _log_target(Pipeline([("pre", make_preprocessor(num, cat, False)),
                                     ("model", LGBMRegressor(**{**p, **params}))]))
    raise ValueError(f"Modèle inconnu : {name}")


def _quantile_model(num, cat, alpha: float, seed: int):
    return _log_target(Pipeline([
        ("pre", make_preprocessor(num, cat, False)),
        ("model", LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=250, learning_rate=0.05,
                                num_leaves=31, random_state=seed, verbose=-1, n_jobs=2))]))


def train_inventory(df: pd.DataFrame | None = None, manifest: dict | None = None,
                    paths: Paths | None = None, cfg: dict | None = None,
                    models: list[str] | None = None, seed: int | None = None,
                    progress=None) -> dict:
    paths = (paths or default_paths()).ensure()
    cfg = cfg or load_config("inventory", paths)
    seed = int(seed if seed is not None else cfg.get("random_seed", 42))
    models = models or cfg.get("models", ALL_MODELS)
    say = progress or (lambda m: log.info(m))

    if df is None:
        f = paths.data_processed / "internos_clean.parquet"
        if not f.exists():
            raise FileNotFoundError("Dataset non préparé : lancez d'abord la préparation des données.")
        df = pd.read_parquet(f)
        mf = paths.data_processed / "inventory_manifest.json"
        if manifest is None and mf.exists():
            import json
            manifest = json.loads(mf.read_text(encoding="utf-8"))
    manifest = manifest or {}

    df = add_engineered(df).reset_index(drop=True)
    num, cat = available_features(df)
    split, split_type = make_split(df, cfg["split"]["val_frac"], cfg["split"]["test_frac"], seed)
    tr, va, te = (split == "train").values, (split == "val").values, (split == "test").values
    X, y = df[num + cat], df[TARGET].astype(float).values
    say(f"Split {split_type} : train={tr.sum()}, val={va.sum()}, test={te.sum()}")

    fitted, results = {}, {}
    for name in models:
        say(f"Entraînement : {name}")
        t0 = time.time()
        candidates = GRIDS.get(name, [None]) if cfg.get("hyperparameter_search", True) else [None]
        best = None
        for params in candidates:
            est = build_model(name, num, cat, seed, params).fit(X[tr], y[tr])
            m_val = regression_metrics(y[va], est.predict(X[va]))
            if best is None or m_val["wape"] < best[0]["wape"]:
                best = (m_val, est, params)
        m_val, est, params = best
        m_test = regression_metrics(y[te], est.predict(X[te]))
        fitted[name] = est
        results[name] = {"val": m_val, "test": m_test, "params": params or {},
                         "fit_seconds": round(time.time() - t0, 1)}
        say(f"  {name}: WAPE val={m_val['wape']:.3f} test={m_test['wape']:.3f}")

    # Sélection sur la VALIDATION uniquement (WAPE) ; le test n'intervient jamais dans le choix
    selected = min(results, key=lambda k: results[k]["val"]["wape"])
    best_est = fitted[selected]
    naive_wape = results.get("naive", {}).get("test", {}).get("wape")
    for name in results:
        results[name]["gain_vs_naive_wape"] = (
            None if naive_wape is None else 1 - results[name]["test"]["wape"] / naive_wape)

    # Intervalles d'incertitude (régression quantile LightGBM, log-cible)
    say("Modèles d'intervalle (quantiles)")
    iv = {"low": _quantile_model(num, cat, cfg["intervals"]["low"], seed).fit(X[tr], y[tr]),
          "high": _quantile_model(num, cat, cfg["intervals"]["high"], seed).fit(X[tr], y[tr])}
    lo, hi = np.minimum(iv["low"].predict(X[te]), best_est.predict(X[te])), np.maximum(
        iv["high"].predict(X[te]), best_est.predict(X[te]))
    coverage = interval_coverage(y[te], lo, hi)

    # Importance des variables (permutation sur validation, score = -MAE)
    say("Importance des variables")
    rng = np.random.default_rng(seed)
    vidx = np.where(va)[0]
    vidx = rng.choice(vidx, size=min(cfg.get("importance_sample", 4000), len(vidx)), replace=False)
    pi = permutation_importance(best_est, X.iloc[vidx], y[vidx], scoring="neg_mean_absolute_error",
                                n_repeats=3, random_state=seed, n_jobs=1)
    importance = (pd.DataFrame({"feature": num + cat, "importance": pi.importances_mean,
                                "std": pi.importances_std})
                  .sort_values("importance", ascending=False).reset_index(drop=True))

    # Prédictions test + analyse d'erreurs
    pred_te = pd.DataFrame({"y_true": y[te], "y_pred": np.maximum(best_est.predict(X[te]), 0),
                            "y_low": lo, "y_high": hi}, index=df.index[te])
    keep = [c for c in ("media_id", "date", "video_type", "content_category") if c in df.columns]
    pred_te = pd.concat([df.loc[te, keep], pred_te], axis=1)
    pred_te["abs_error"] = (pred_te["y_pred"] - pred_te["y_true"]).abs()
    pred_te.to_parquet(paths.data_processed / "inventory_test_predictions.parquet", index=False)
    err_type = error_by_group(pred_te, "video_type")
    err_vol = error_by_volume_quantile(pred_te)
    sens = conversion_sensitivity(float(pred_te["y_pred"].sum()))
    for nm, tbl in (("inventory_error_by_type", err_type), ("inventory_error_by_volume", err_vol),
                    ("inventory_conversion_sensitivity", sens), ("inventory_feature_importance", importance)):
        tbl.to_csv(paths.reports_metrics / f"{nm}.csv", index=False)

    dates = pd.to_datetime(df["date"])
    metadata = {
        "trained_at": datetime.now().isoformat(timespec="seconds"), "seed": seed,
        "selected_model": selected, "selection_metric": "WAPE (validation)",
        "selected_beats_naive": bool(selected != "naive"),
        "split": {"type": split_type, "n_train": int(tr.sum()), "n_val": int(va.sum()),
                  "n_test": int(te.sum()),
                  "train_end": None if split_type == "predefined" else str(dates[tr].max().date()),
                  "val_end": None if split_type == "predefined" else str(dates[va].max().date()),
                  "test_start": None if split_type == "predefined" else str(dates[te].min().date())},
        "has_media_id": bool(manifest.get("has_media_id", True)),
        "target_normalised": bool(manifest.get("target_normalised", False)),
        "metrics": results, "interval_coverage_test": coverage,
        "interval_nominal": cfg["intervals"]["high"] - cfg["intervals"]["low"],
        "importance": importance.head(15).to_dict("records"),
        "data_source": manifest.get("source", "inconnu"), "is_real_data": manifest.get("is_real", False),
        "data_version": manifest.get("data_version", data_version(df)),
        "conversion_defaults": cfg["conversion"], "n_rows": len(df),
        "features": {"num": num, "cat": cat}, "candidates": list(models),
    }
    bundle = {"model": best_est, "model_name": selected, "interval_models": iv,
              "features": {"num": num, "cat": cat}}
    version = save_model("inventory", bundle, metadata, paths)
    log_experiment("inventory", {
        "version": version, "selected": selected, "seed": seed, "data_source": metadata["data_source"],
        "data_version": metadata["data_version"], "split": metadata["split"],
        "test_wape": {k: v["test"]["wape"] for k, v in results.items()},
        "test_mae": {k: v["test"]["mae"] for k, v in results.items()}}, paths)
    say(f"Modèle inventaire « {selected} » sauvegardé ({version})")
    return {"version": version, "metadata": metadata, "importance": importance,
            "error_by_type": err_type, "error_by_volume": err_vol, "test_predictions": pred_te}


def run(paths: Paths | None = None, models: list[str] | None = None, seed: int | None = None,
        force_proxy: bool = False, progress=None) -> dict:
    paths = (paths or default_paths()).ensure()
    cfg = load_config("inventory", paths)
    df, manifest = prepare_inventory_dataset(paths, cfg, force_proxy=force_proxy)
    return train_inventory(df, manifest, paths, cfg, models, seed, progress)
