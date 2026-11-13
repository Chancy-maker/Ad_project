"""Métriques de classification et sélection de seuil (module B)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y, p, threshold: float) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    yhat = (p >= threshold).astype(int)
    one_class = len(np.unique(y)) < 2
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold), "n": int(len(y)), "prevalence": float(y.mean()),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "pr_auc": float("nan") if one_class else float(average_precision_score(y, p)),
        "roc_auc": float("nan") if one_class else float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "alert_rate": float(yhat.mean()),
    }


def select_threshold(y, p, strategy: str = "recall_target", recall_target: float = 0.8) -> float:
    """Seuil choisi sur la VALIDATION (jamais sur le test)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    if len(np.unique(y)) < 2:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    if strategy == "recall_target":
        ok = np.where(rec >= recall_target)[0]
        if len(ok):
            return float(thr[ok[np.argmax(prec[ok])]])   # meilleure précision à rappel suffisant
        return float(thr[0])
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return float(thr[int(np.argmax(f1))])


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    frac, mean_p = calibration_curve(np.asarray(y).astype(int), np.asarray(p, float), n_bins=n_bins,
                                     strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_p, "fraction_positive": frac})


def threshold_sweep(y, p, thresholds=None) -> pd.DataFrame:
    thresholds = thresholds if thresholds is not None else np.round(np.arange(0.1, 0.91, 0.1), 2)
    rows = []
    for t in thresholds:
        m = classification_metrics(y, p, t)
        rows.append({k: m[k] for k in ("threshold", "recall", "precision", "f1", "alert_rate")})
    return pd.DataFrame(rows)


def risk_class(p: np.ndarray, low_max: float, high_min: float) -> np.ndarray:
    p = np.asarray(p, float)
    return np.where(p < low_max, "faible", np.where(p >= high_min, "élevé", "modéré"))


def subgroup_metrics(df: pd.DataFrame, group: str, threshold: float, y="y_true", p="proba") -> pd.DataFrame:
    rows = []
    for g, sub in df.groupby(group, observed=True):
        m = classification_metrics(sub[y], sub[p], threshold)
        rows.append({group: g, "n": m["n"], "prevalence": m["prevalence"], "recall": m["recall"],
                     "precision": m["precision"], "pr_auc": m["pr_auc"]})
    return pd.DataFrame(rows)
