"""Inférence du module A : charge le modèle sauvegardé, ne réentraîne jamais."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..common.schemas import ConversionAssumptions
from .features import add_engineered


def predict_requests(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Prévision des requêtes à 24 h (+ intervalle si disponible). Index conservé."""
    X = add_engineered(df)
    cols = bundle["features"]["num"] + bundle["features"]["cat"]
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes pour la prédiction : {missing}")
    X = X[cols]
    y = np.maximum(bundle["model"].predict(X), 0)
    out = pd.DataFrame({"y_pred": y}, index=df.index)
    iv = bundle.get("interval_models")
    if iv:
        lo = np.maximum(iv["low"].predict(X), 0)
        hi = np.maximum(iv["high"].predict(X), 0)
        out["y_low"] = np.minimum(lo, y)
        out["y_high"] = np.maximum(hi, y)
    return out


def estimate_inventory(pred: pd.DataFrame, assumptions: ConversionAssumptions | dict) -> pd.DataFrame:
    """impressions_eligibles = requêtes_prévues x opportunités_par_lecture x taux_éligibilité."""
    a = assumptions if isinstance(assumptions, ConversionAssumptions) else ConversionAssumptions(**assumptions)
    out = pred.copy()
    out["theoretical_ad_opportunities"] = out["y_pred"] * a.ad_opportunities_per_view
    out["eligible_impressions"] = out["theoretical_ad_opportunities"] * a.eligibility_rate
    for s in ("low", "high"):
        if f"y_{s}" in out:
            out[f"eligible_impressions_{s}"] = out[f"y_{s}"] * a.ad_opportunities_per_view * a.eligibility_rate
    return out
