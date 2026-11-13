"""État des datasets et lecture des manifestes (utilisé par l'interface)."""
from __future__ import annotations

import json
from pathlib import Path

from ..data.preprocessing import find_presplit_files
from .config import Paths, default_paths, load_config


def _read(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def inventory_manifest(paths: Paths | None = None) -> dict | None:
    paths = paths or default_paths()
    return _read(paths.data_processed / "inventory_manifest.json")


def under_delivery_manifest(paths: Paths | None = None) -> dict | None:
    paths = paths or default_paths()
    return _read(paths.data_synthetic / "under_delivery_manifest.json")


def dataset_status(paths: Paths | None = None) -> list[dict]:
    """Statut : présent / absent / synthétique, pour chaque source."""
    paths = paths or default_paths()
    app = load_config("app", paths)["datasets"]
    inv, ud = inventory_manifest(paths), under_delivery_manifest(paths)
    rows = []
    internos_raw = (paths.data_raw / app["internos_file"]).exists() or find_presplit_files(paths.data_raw) is not None
    rows.append({"Dataset": "Internos (Mendeley, module A)",
                 "Statut": "Présent" if internos_raw else "Absent",
                 "Utilisé": ("Réel" if inv and inv.get("is_real") else
                             "Substitut synthétique" if inv else "Non préparé"),
                 "Lignes": inv["n_rows"] if inv else None})
    criteo_raw = (paths.data_raw / app["criteo_file"]).exists()
    rows.append({"Dataset": "Criteo Attribution (profils, module B)",
                 "Statut": "Présent" if criteo_raw else "Absent",
                 "Utilisé": "Profils réels" if criteo_raw and ud else "Paramétrique" if ud else "Non préparé",
                 "Lignes": None})
    rows.append({"Dataset": "Campagnes synthétiques (contrats, livraisons)",
                 "Statut": "Synthétique" if ud else "Non généré", "Utilisé": "Synthétique",
                 "Lignes": ud["n_snapshots"] if ud else None})
    rows.append({"Dataset": "YTLive (facultatif)", "Statut": "Non utilisé", "Utilisé": "—", "Lignes": None})
    return rows
