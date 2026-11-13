"""Configuration centralisée et chemins relatifs du projet."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Paths:
    """Tous les chemins dérivent d'une racine unique (surchargeable via ADO_ROOT)."""

    root: Path

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_interim(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def data_synthetic(self) -> Path:
        return self.root / "data" / "synthetic"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def reports_metrics(self) -> Path:
        return self.root / "reports" / "metrics"

    @property
    def reports_figures(self) -> Path:
        return self.root / "reports" / "figures"

    def ensure(self) -> "Paths":
        for p in (
            self.data_raw, self.data_interim, self.data_processed, self.data_synthetic,
            self.models / "inventory", self.models / "under_delivery",
            self.reports_metrics, self.reports_figures,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self


def default_paths() -> Paths:
    return Paths(Path(os.environ.get("ADO_ROOT", str(PROJECT_ROOT))))


def deep_update(base: dict, overrides: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(name: str, paths: Paths | None = None, overrides: dict | None = None) -> dict:
    """Charge configs/<name>.yaml (repli sur la config du dépôt si absente de la racine)."""
    paths = paths or default_paths()
    f = paths.configs / f"{name}.yaml"
    if not f.exists():
        f = PROJECT_ROOT / "configs" / f"{name}.yaml"
    with open(f, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return deep_update(cfg, overrides)
