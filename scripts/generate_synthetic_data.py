"""Prépare le dataset du module A et génère les campagnes synthétiques du module B."""
import argparse

import _bootstrap  # noqa: F401

from ad_optimizer.common.config import default_paths, load_config
from ad_optimizer.data.preprocessing import prepare_inventory_dataset
from ad_optimizer.under_delivery.train import prepare_data

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-campaigns", type=int, default=None)
    ap.add_argument("--force-proxy", action="store_true", help="ignorer data/raw/internos.csv")
    a = ap.parse_args()
    paths = default_paths().ensure()
    cfg = load_config("inventory", paths, {"random_seed": a.seed} if a.seed is not None else None)
    prepare_inventory_dataset(paths, cfg, force_proxy=a.force_proxy)
    prepare_data(paths, n_campaigns=a.n_campaigns, seed=a.seed)
