"""Inférence du module B : probabilité, classe de risque, recommandations déterministes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import risk_class
from .snapshot_builder import CAT_FEATURES, NUM_FEATURES


def score_snapshots(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    cols = bundle["features"]["num"] + bundle["features"]["cat"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    out = df.copy()
    out["proba"] = bundle["model"].predict_proba(df[cols])[:, 1]
    b = bundle["risk_bins"]
    out["risk_class"] = risk_class(out["proba"], b["low_max"], b["high_min"])
    out["alert"] = (out["proba"] >= bundle["threshold"]).astype(int)
    out["recommendation"] = [recommend(r, p, c) for (_, r), p, c in
                             zip(out.iterrows(), out["proba"], out["risk_class"])]
    return out


def recommend(row: pd.Series, proba: float, risk: str) -> str:
    """Règles métier transparentes, calculées sur les variables réelles (aucune cause inventée)."""
    msgs = []
    if row["recent_to_required_rate_ratio"] < 0.8:
        msgs.append("Le rythme récent est inférieur au rythme quotidien nécessaire.")
    if row["avg_to_required_rate_ratio"] < 0.8:
        msgs.append("Le rythme moyen actuel est inférieur au rythme quotidien nécessaire.")
    if row["projected_capacity_ratio"] < 1.0:
        msgs.append("Le volume restant est élevé au regard de l'inventaire disponible et des jours restants.")
    if row["inventory_trend_ratio"] < 0.75:
        msgs.append("L'inventaire disponible a récemment baissé : envisager de réallouer du stock (simulé).")
    if row["has_daily_cap"] and row["daily_cap_to_required_ratio"] < 1.2:
        msgs.append("Le plafond journalier limite le rattrapage possible.")
    if row["n_allowed_categories"] <= 1 or row["observed_capture_rate"] < 0.1:
        msgs.append("Le ciblage semble restreint : envisager de l'élargir (simulé).")
    if risk == "élevé":
        msgs.append("La campagne nécessite une surveillance renforcée.")
    elif risk == "modéré":
        msgs.append("À surveiller.")
    return " ".join(msgs) if msgs else "Aucune action requise : la livraison est conforme à l'objectif."


def local_explanation(bundle: dict, row: pd.DataFrame, top: int = 6) -> pd.DataFrame:
    """Explication locale par occlusion : effet sur la probabilité du remplacement d'une variable
    par sa valeur de référence (médiane/mode du train). Méthode agnostique du modèle."""
    cols = bundle["features"]["num"] + bundle["features"]["cat"]
    base = row[cols].iloc[[0]].reset_index(drop=True)
    p0 = float(bundle["model"].predict_proba(base)[0, 1])
    ref = bundle["reference"]
    variants = []
    for c in cols:
        v = base.copy()
        v[c] = ref[c]
        variants.append(v)
    p_ref = bundle["model"].predict_proba(pd.concat(variants, ignore_index=True))[:, 1]
    res = pd.DataFrame({"variable": cols, "valeur": [base[c].iloc[0] for c in cols],
                        "reference": [ref[c] for c in cols], "contribution": p0 - p_ref})
    res["abs"] = res["contribution"].abs()
    return res.sort_values("abs", ascending=False).drop(columns="abs").head(top).reset_index(drop=True)


__all__ = ["score_snapshots", "recommend", "local_explanation", "NUM_FEATURES", "CAT_FEATURES"]
_ = np
