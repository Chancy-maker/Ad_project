"""Métriques et analyses d'erreurs du module A."""
from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    total = y.sum()
    mask = y > 0
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "wape": float(np.abs(err).sum() / total) if total > 0 else float("nan"),
        # MAPE calculé uniquement là où la cible est > 0 (précaution : cible souvent nulle)
        "mape_pos": float((np.abs(err[mask]) / y[mask]).mean()) if mask.any() else float("nan"),
        "bias": float(err.sum() / total) if total > 0 else float("nan"),
        "n": int(len(y)),
    }


def error_by_group(df: pd.DataFrame, group: str, y="y_true", p="y_pred") -> pd.DataFrame:
    rows = []
    for g, sub in df.groupby(group, observed=True):
        m = regression_metrics(sub[y], sub[p])
        rows.append({group: g, "n": m["n"], "volume": float(sub[y].sum()),
                     "mae": m["mae"], "wape": m["wape"], "bias": m["bias"]})
    return pd.DataFrame(rows).sort_values("volume", ascending=False).reset_index(drop=True)


def error_by_volume_quantile(df: pd.DataFrame, q: int = 5, y="y_true", p="y_pred") -> pd.DataFrame:
    """Analyse d'erreurs sur les volumes élevés : quantiles de la cible observée."""
    d = df.copy()
    d["volume_bin"] = pd.qcut(d[y].rank(method="first"), q, labels=[f"Q{i + 1}" for i in range(q)])
    return error_by_group(d, "volume_bin", y, p)


def interval_coverage(y, low, high) -> float:
    y, low, high = map(lambda a: np.asarray(a, float), (y, low, high))
    return float(((y >= low) & (y <= high)).mean())


def conversion_sensitivity(total_requests: float, opps_grid=(1.0, 1.5, 2.0, 2.5, 3.0),
                           elig_grid=(0.6, 0.7, 0.8, 0.9, 1.0)) -> pd.DataFrame:
    """Stock éligible total selon les hypothèses de conversion (impact de leur incertitude)."""
    rows = [{"ad_opportunities_per_view": o, "eligibility_rate": e,
             "eligible_impressions": total_requests * o * e}
            for o in opps_grid for e in elig_grid]
    return pd.DataFrame(rows)
