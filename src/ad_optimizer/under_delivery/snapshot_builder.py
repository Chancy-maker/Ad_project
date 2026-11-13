"""Construction des snapshots de campagne, SANS fuite de données.

Garantie structurelle : `compute_snapshot_features` ne reçoit que les séries tronquées à l'instant
de suivi (livraisons et inventaire des `elapsed` premiers jours). Il lui est donc impossible de lire
le futur. La cible est calculée séparément, à partir de la livraison finale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NUM_FEATURES = [
    "booked_impressions", "delivered_so_far", "remaining_impressions", "elapsed_days",
    "remaining_days", "campaign_duration_days", "snapshot_pct", "delivery_progress_rate",
    "progress_gap", "avg_daily_delivery_rate", "recent_3d_delivery_rate", "recent_7d_delivery_rate",
    "required_daily_rate", "recent_to_required_rate_ratio", "avg_to_required_rate_ratio",
    "available_inventory_estimate", "inventory_trend_ratio", "observed_capture_rate",
    "projected_capacity_ratio", "priority", "n_allowed_categories", "has_daily_cap",
    "daily_cap_to_required_ratio",
]
CAT_FEATURES = ["targeting_segment", "campaign_type"]
TARGET = "under_delivery_final"
# Jamais en entrée du modèle
FORBIDDEN_FEATURES = {"final_delivered_impressions", "under_delivery_final", "scenario",
                      "campaign_id", "split"}
RATIO_CAP = 10.0


def safe_ratio(a: float, b: float, cap: float = RATIO_CAP) -> float:
    return cap if b <= 1e-9 else float(min(a / b, cap))


def compute_snapshot_features(campaign: pd.Series | dict, delivered: np.ndarray, available: np.ndarray,
                              snapshot_pct: float) -> dict:
    """Variables à l'instant de suivi. `delivered`/`available` = jours ÉCOULÉS uniquement."""
    delivered = np.asarray(delivered, float)
    available = np.asarray(available, float)
    e = len(delivered)
    if e < 1 or len(available) != e:
        raise ValueError("Il faut au moins un jour écoulé, et des séries de même longueur.")
    booked = float(campaign["booked_impressions"])
    D = int(campaign["duration_days"])
    rem_days = D - e
    done = float(delivered.sum())
    remaining = max(booked - done, 0.0)
    required = remaining / rem_days if rem_days > 0 else remaining
    avg = done / e
    r3, r7 = float(delivered[-3:].mean()), float(delivered[-7:].mean())
    avail_est = float(available[-7:].mean())
    capture = done / max(float(available.sum()), 1e-9)
    cap = float(campaign.get("max_daily_impressions", 0) or 0)
    return {
        "booked_impressions": booked, "delivered_so_far": done, "remaining_impressions": remaining,
        "elapsed_days": e, "remaining_days": rem_days, "campaign_duration_days": D,
        "snapshot_pct": float(snapshot_pct), "delivery_progress_rate": done / booked,
        "progress_gap": done / booked - e / D, "avg_daily_delivery_rate": avg,
        "recent_3d_delivery_rate": r3, "recent_7d_delivery_rate": r7, "required_daily_rate": required,
        "recent_to_required_rate_ratio": safe_ratio(r7, required),
        "avg_to_required_rate_ratio": safe_ratio(avg, required),
        "available_inventory_estimate": avail_est,
        "inventory_trend_ratio": safe_ratio(float(available[-3:].mean()), float(available.mean())),
        "observed_capture_rate": capture,
        "projected_capacity_ratio": safe_ratio(avail_est * capture, required),
        "priority": int(campaign["priority"]), "n_allowed_categories": int(campaign["n_allowed_categories"]),
        "has_daily_cap": int(cap > 0),
        "daily_cap_to_required_ratio": safe_ratio(cap, required) if cap > 0 else RATIO_CAP,
        "targeting_segment": str(campaign["targeting_segment"]),
        "campaign_type": str(campaign["campaign_type"]),
    }


def snapshot_day_count(duration: int, pct: float) -> int:
    """Nombre de jours écoulés au snapshot ; au moins 1 jour écoulé et 1 jour restant."""
    return int(min(max(1, round(pct * duration)), duration - 1))


def build_snapshots(campaigns: pd.DataFrame, daily: pd.DataFrame, pcts=(0.25, 0.5, 0.75),
                    tolerance: float = 0.98) -> pd.DataFrame:
    """Plusieurs points de suivi par campagne ; cible = livraison finale < tolérance x objectif."""
    daily = daily.sort_values(["campaign_id", "day_index"])
    series = {cid: (g["delivered_impressions"].to_numpy(), g["available_inventory"].to_numpy(),
                    g["date"].to_numpy())
              for cid, g in daily.groupby("campaign_id", sort=False)}
    rows = []
    for _, c in campaigns.iterrows():
        delivered, available, dates = series[c["campaign_id"]]
        final = float(delivered.sum())
        label = int(final < tolerance * float(c["booked_impressions"]))
        for pct in pcts:
            e = snapshot_day_count(int(c["duration_days"]), pct)
            feats = compute_snapshot_features(c, delivered[:e], available[:e], pct)   # tronqué !
            rows.append({"campaign_id": c["campaign_id"], "snapshot_date": pd.Timestamp(dates[e - 1]),
                         **feats, "final_delivered_impressions": final, TARGET: label,
                         "scenario": c.get("scenario", "")})
    return pd.DataFrame(rows)
