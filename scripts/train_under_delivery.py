"""Entraîne le module B (risque de sous-impression) : python scripts/train_under_delivery.py"""
import argparse

import _bootstrap  # noqa: F401

from ad_optimizer.under_delivery.train import ALL_MODELS, run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None, choices=ALL_MODELS)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-campaigns", type=int, default=None)
    a = ap.parse_args()
    r = run(models=a.models, seed=a.seed, n_campaigns=a.n_campaigns)
    print("Version :", r["version"], "| modèle :", r["metadata"]["selected_model"])
