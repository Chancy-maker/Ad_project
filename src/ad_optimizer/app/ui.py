"""Aides communes à l'interface Streamlit (chargement en cache, style des graphiques)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..common.config import Paths, default_paths
from ..common.model_registry import active_metadata, load_active

BLUE, ORANGE, AQUA, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#52514e"
RISK_COLORS = {"faible": AQUA, "modéré": "#fab219", "élevé": "#d03b3b"}


def get_paths() -> Paths:
    return default_paths()


def style(fig: go.Figure, height: int = 380, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, title=title, margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        font=dict(color=INK, size=13), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified" if not title or "dispersion" not in (title or "") else "closest")
    fig.update_xaxes(showgrid=False, linecolor="#d9d8d3")
    fig.update_yaxes(gridcolor="#eceae5", zeroline=False)
    return fig


def fmt(n, digits: int = 0) -> str:
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return "—"
    return f"{n:,.{digits}f}".replace(",", " ")


def pct(x, digits: int = 1) -> str:
    return "—" if x is None or pd.isna(x) else f"{100 * x:.{digits}f} %"


@st.cache_data(show_spinner=False)
def read_parquet_cached(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


def read_parquet(path: Path) -> pd.DataFrame | None:
    return read_parquet_cached(str(path), path.stat().st_mtime) if path.exists() else None


@st.cache_resource(show_spinner=False)
def _load_bundle(kind: str, version: str, root: str):
    return load_active(kind, Paths(Path(root)))


def get_model(kind: str):
    """Retourne (bundle, metadata) du modèle actif, ou (None, None)."""
    paths = get_paths()
    meta = active_metadata(kind, paths)
    if not meta:
        return None, None
    return _load_bundle(kind, meta["version"], str(paths.root))


def need_training(kind_label: str) -> None:
    st.warning(f"Aucun modèle « {kind_label} » n'est encore entraîné. Rendez-vous dans la page "
               "**Entraînement et évaluation** (ou lancez `python scripts/bootstrap_demo.py`).")


def synthetic_banner(is_real: bool, what: str) -> None:
    if is_real:
        st.info(f"Données : {what}.")
    else:
        st.warning(f"⚠️ **Données synthétiques / substitut** — {what}. Les résultats illustrent la "
                   "méthode et ne représentent aucune performance opérationnelle réelle (aucune donnée TF1+).")
