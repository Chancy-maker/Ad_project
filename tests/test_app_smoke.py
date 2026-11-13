"""Test de fumée de l'interface Streamlit sur un projet temporaire entraîné."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ad_optimizer.common.config import Paths
from ad_optimizer.data.preprocessing import prepare_inventory_dataset
from ad_optimizer.inventory import train as inv
from ad_optimizer.under_delivery import train as ud

APP = Path(__file__).resolve().parents[1] / "src" / "ad_optimizer" / "app" / "streamlit_app.py"
PAGES = ["Vue d'ensemble", "Prévision de l'inventaire", "Risque de sous-impression",
         "Entraînement et évaluation", "Données et méthodologie"]


@pytest.fixture(scope="module")
def trained_root(tmp_path_factory, ):
    from ad_optimizer.common.config import load_config
    root = tmp_path_factory.mktemp("app_root")
    paths = Paths(root).ensure()
    icfg = load_config("inventory", paths, {"proxy": {"n_media": 60}, "hyperparameter_search": False,
                                            "importance_sample": 300, "models": ["naive", "lightgbm"]})
    df, mf = prepare_inventory_dataset(paths, icfg)
    inv.train_inventory(df, mf, paths, icfg, progress=lambda m: None)
    ucfg = load_config("under_delivery", paths, {"generator": {"n_campaigns": 300}, "holdout": {"n_campaigns": 80},
                                                 "hyperparameter_search": False,
                                                 "models": ["dummy", "logistic_regression", "lightgbm"]})
    mf = ud.prepare_data(paths, ucfg)
    import pandas as pd
    ud.train_under_delivery(pd.read_parquet(paths.data_synthetic / "snapshots.parquet"), mf, paths, ucfg,
                            progress=lambda m: None)
    return root


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exception(trained_root, monkeypatch, page):
    monkeypatch.setenv("ADO_ROOT", str(trained_root))
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]


def test_untrained_app_shows_guidance_and_does_not_train(tmp_path, monkeypatch):
    monkeypatch.setenv("ADO_ROOT", str(Paths(tmp_path).ensure().root))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert any("Initialiser la démo" in b.label for b in at.button)
    assert not (tmp_path / "models" / "inventory" / "active.json").exists()   # pas d'entraînement auto
    at.sidebar.radio[0].set_value("Prévision de l'inventaire").run()
    assert not at.exception and at.warning
