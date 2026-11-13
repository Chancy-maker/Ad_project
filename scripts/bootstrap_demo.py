"""Prépare tout pour la démo : données -> entraînement des 2 modules (≈ 1 minute)."""
import argparse

import _bootstrap  # noqa: F401

from ad_optimizer.common.pipeline import run_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--force-proxy", action="store_true")
    a = ap.parse_args()
    r = run_all(seed=a.seed, force_proxy=a.force_proxy)
    print("\nPrêt. Lancez : streamlit run src/ad_optimizer/app/streamlit_app.py")
