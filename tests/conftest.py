import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ad_optimizer.common.config import Paths, load_config  # noqa: E402


@pytest.fixture(scope="session")
def small_paths(tmp_path_factory) -> Paths:
    return Paths(tmp_path_factory.mktemp("ado_root")).ensure()


@pytest.fixture(scope="session")
def small_inv_cfg(small_paths):
    return load_config("inventory", small_paths, {
        "proxy": {"n_media": 80}, "hyperparameter_search": False, "importance_sample": 500,
        "models": ["naive", "ridge", "lightgbm"]})


@pytest.fixture(scope="session")
def small_ud_cfg(small_paths):
    return load_config("under_delivery", small_paths, {
        "generator": {"n_campaigns": 400}, "holdout": {"n_campaigns": 120},
        "hyperparameter_search": False, "models": ["dummy", "pacing_rule", "logistic_regression", "lightgbm"]})
