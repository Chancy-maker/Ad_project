"""Agrégation Criteo : profils de diffusion empiriques par campagne (volume, variabilité).

Le champ `timestamp` de Criteo est RELATIF (secondes depuis le début) : aucune date calendaire réelle
n'est attribuée. Criteo ne fournit ni objectifs contractuels ni stock : ils restent synthétiques.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.config import Paths, default_paths, load_config
from ..common.logging import get_logger

log = get_logger()
DAY_SECONDS = 86400


def aggregate_campaign_daily(path: Path, chunksize: int = 1_000_000, nrows: int | None = None) -> pd.DataFrame:
    """Impressions par (campagne, jour relatif). Lecture par blocs (fichier volumineux)."""
    parts = []
    reader = pd.read_csv(path, sep="\t", usecols=["timestamp", "campaign"], chunksize=chunksize,
                         nrows=nrows)
    for chunk in reader:
        chunk["rel_day"] = (chunk["timestamp"] // DAY_SECONDS).astype(int)
        parts.append(chunk.groupby(["campaign", "rel_day"]).size().rename("impressions").reset_index())
    daily = pd.concat(parts).groupby(["campaign", "rel_day"], as_index=False)["impressions"].sum()
    return daily


def extract_delivery_profiles(daily: pd.DataFrame, min_days: int = 5) -> pd.DataFrame:
    """Une ligne par campagne : volume quotidien moyen, coefficient de variation, jours actifs."""
    g = daily.groupby("campaign")["impressions"]
    prof = pd.DataFrame({"total_impressions": g.sum(), "active_days": g.size(),
                         "mean_daily": g.mean(), "std_daily": g.std(ddof=0)})
    prof["cv_daily"] = prof["std_daily"] / prof["mean_daily"].replace(0, np.nan)
    prof = prof[(prof["active_days"] >= min_days) & prof["cv_daily"].notna()]
    return prof.reset_index()


def prepare_delivery_profiles(paths: Paths | None = None) -> tuple[pd.DataFrame | None, dict]:
    """Profils Criteo si le fichier est présent ; sinon None (générateur paramétrique par défaut)."""
    paths = (paths or default_paths()).ensure()
    cfg = load_config("app", paths)
    raw = paths.data_raw / cfg["datasets"]["criteo_file"]
    if not raw.exists():
        info = {"source": "paramétrique (Criteo absent)", "is_real": False,
                "note": "Fichier Criteo absent : distributions de volume/variabilité paramétriques."}
        return None, info
    log.info("Agrégation Criteo (peut prendre quelques minutes)...")
    daily = aggregate_campaign_daily(raw)
    prof = extract_delivery_profiles(daily)
    prof.to_parquet(paths.data_processed / "criteo_profiles.parquet", index=False)
    info = {"source": "Criteo Attribution Modeling for Bidding (réel, CC BY-NC-SA 4.0)", "is_real": True,
            "n_campaigns_profiled": len(prof), "n_impressions": int(daily["impressions"].sum()),
            "profiled_at": datetime.now().isoformat(timespec="seconds")}
    (paths.data_processed / "criteo_manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return prof, info
