"""Entraîne le module A (prévision des requêtes) : python scripts/train_inventory.py"""
import argparse

import _bootstrap  # noqa: F401

from ad_optimizer.inventory.train import ALL_MODELS, run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None, choices=ALL_MODELS)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--force-proxy", action="store_true")
    a = ap.parse_args()
    r = run(models=a.models, seed=a.seed, force_proxy=a.force_proxy)
    print("Version :", r["version"], "| modèle :", r["metadata"]["selected_model"])
