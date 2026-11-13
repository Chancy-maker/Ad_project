"""Orchestration de bout en bout (utilisée par les scripts CLI et par l'interface)."""
from __future__ import annotations

from ..inventory import train as inv_train
from ..under_delivery import train as ud_train
from .config import Paths, default_paths


def run_all(paths: Paths | None = None, seed: int | None = None, progress=None,
            n_campaigns: int | None = None, force_proxy: bool = False) -> dict:
    paths = (paths or default_paths()).ensure()
    inv = inv_train.run(paths, seed=seed, force_proxy=force_proxy, progress=progress)
    ud = ud_train.run(paths, seed=seed, n_campaigns=n_campaigns, progress=progress)
    return {"inventory": inv, "under_delivery": ud}
