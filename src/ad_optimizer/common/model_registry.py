"""Registre local de modèles : versionnement, modèle actif, historique d'expériences."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import Paths, default_paths


class ModelNotFoundError(FileNotFoundError):
    pass


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if np.isnan(v) or np.isinf(v) else v
    if isinstance(o, float):
        return None if np.isnan(o) or np.isinf(o) else o
    if isinstance(o, (pd.Timestamp, datetime, Path)):
        return str(o)
    return o


def _dir(kind: str, paths: Paths) -> Path:
    d = paths.models / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_model(kind: str, bundle: dict, metadata: dict, paths: Paths | None = None) -> str:
    """Enregistre bundle + métadonnées sous models/<kind>/<version>/ et l'active."""
    paths = paths or default_paths()
    base = _dir(kind, paths)
    stamp = datetime.now().strftime("v%Y%m%d-%H%M%S")
    version, i = stamp, 1
    while (base / version).exists():
        i += 1
        version = f"{stamp}-{i}"
    vdir = base / version
    vdir.mkdir()
    metadata = {**metadata, "version": version, "kind": kind}
    joblib.dump(bundle, vdir / "model.joblib")
    (vdir / "metadata.json").write_text(
        json.dumps(to_jsonable(metadata), indent=2, ensure_ascii=False), encoding="utf-8")
    activate(kind, version, paths)
    return version


def activate(kind: str, version: str, paths: Paths | None = None) -> None:
    paths = paths or default_paths()
    base = _dir(kind, paths)
    if not (base / version / "model.joblib").exists():
        raise ModelNotFoundError(f"Version inconnue : {kind}/{version}")
    (base / "active.json").write_text(json.dumps({"version": version}), encoding="utf-8")


def active_version(kind: str, paths: Paths | None = None) -> str | None:
    paths = paths or default_paths()
    f = paths.models / kind / "active.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8")).get("version")


def active_metadata(kind: str, paths: Paths | None = None) -> dict | None:
    paths = paths or default_paths()
    v = active_version(kind, paths)
    if v is None:
        return None
    f = paths.models / kind / v / "metadata.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def load_active(kind: str, paths: Paths | None = None) -> tuple[dict, dict]:
    """Charge le modèle actif (sans réentraîner). Retourne (bundle, metadata)."""
    paths = paths or default_paths()
    v = active_version(kind, paths)
    if v is None:
        raise ModelNotFoundError(
            f"Aucun modèle actif pour « {kind} ». Lancez l'entraînement (page « Entraînement » "
            "ou scripts/train_*.py).")
    bundle = joblib.load(paths.models / kind / v / "model.joblib")
    meta = json.loads((paths.models / kind / v / "metadata.json").read_text(encoding="utf-8"))
    return bundle, meta


def list_versions(kind: str, paths: Paths | None = None) -> list[str]:
    paths = paths or default_paths()
    base = paths.models / kind
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if (p / "model.joblib").exists())


def log_experiment(kind: str, record: dict, paths: Paths | None = None) -> None:
    paths = paths or default_paths()
    paths.reports_metrics.mkdir(parents=True, exist_ok=True)
    rec = {"kind": kind, "logged_at": datetime.now().isoformat(timespec="seconds"), **record}
    with open(paths.reports_metrics / "experiments.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(to_jsonable(rec), ensure_ascii=False) + "\n")


def read_experiments(paths: Paths | None = None, kind: str | None = None) -> list[dict]:
    paths = paths or default_paths()
    f = paths.reports_metrics / "experiments.jsonl"
    if not f.exists():
        return []
    rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if kind is None or r.get("kind") == kind]
