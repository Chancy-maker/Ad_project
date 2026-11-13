"""Variables, découpage et préprocesseurs du module A (prévision des requêtes à 24 h)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "requests"
NUM_FEATURES = [
    "requests_one_day", "requests_two_days", "requests_one_week", "requests_two_week",
    "day_of_week", "day_of_month", "day_of_year", "month",
    "comments", "comments_in_section", "publications", "publications_in_section",
    "votes", "votes_in_section", "days_published", "holidays_left", "next_holidays",
    "is_weekend", "ratio_day_week", "trend_week",
]
# Schéma des fichiers déjà normalisés (train/validation/test) : encodages cycliques + drapeaux de type
NUM_FEATURES_NORM = [
    "requests_one_day", "requests_two_days", "requests_one_week", "requests_two_week",
    "day_week_sin", "day_week_cos", "day_month_sin", "day_month_cos", "day_year_sin", "day_year_cos",
    "current_month_sin", "current_month_cos", "comments", "comments_in_section", "publications",
    "publications_in_section", "duration", "votes", "votes_in_section", "days_published",
    "holidays_left", "next_holidays", "vt_0", "vt_1", "vt_2", "vt_3",
]
CAT_FEATURES = ["video_type"]
# Les identifiants bruts ne sont JAMAIS des variables prédictives (voir cahier des charges §5.1)
FORBIDDEN_FEATURES = {"media_id", "date", "content_category", "split", TARGET}


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "day_of_week" in out:
        out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    if {"requests_one_day", "requests_one_week"} <= set(out.columns):
        out["ratio_day_week"] = out["requests_one_day"] / (out["requests_one_week"] / 7 + 1)
    if {"requests_one_week", "requests_two_week"} <= set(out.columns):
        out["trend_week"] = ((out["requests_one_week"] / 7 - out["requests_two_week"] / 14)
                             / (out["requests_two_week"] / 14 + 1))
    return out


def available_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    normalised = "day_week_sin" in df.columns
    num = [c for c in (NUM_FEATURES_NORM if normalised else NUM_FEATURES) if c in df.columns]
    # En schéma normalisé, le type est déjà porté par les drapeaux vt_* : pas de catégorielle redondante
    cat = [] if normalised else [c for c in CAT_FEATURES if c in df.columns]
    assert not (set(num + cat) & FORBIDDEN_FEATURES)
    return num, cat


def temporal_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15) -> pd.Series:
    """Split chronologique par quantiles de dates (train < val < test). Retourne 'train'/'val'/'test'."""
    d = pd.to_datetime(df["date"])
    q_val = d.quantile(1 - val_frac - test_frac)
    q_test = d.quantile(1 - test_frac)
    split = np.where(d <= q_val, "train", np.where(d <= q_test, "val", "test"))
    return pd.Series(split, index=df.index, name="split")


def group_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15,
                seed: int = 42, group: str = "media_id") -> pd.Series:
    """Repli : split groupé par ressource média (aucun contenu dans deux jeux)."""
    rng = np.random.default_rng(seed)
    groups = df[group].astype(str).unique()
    rng.shuffle(groups)
    n = len(groups)
    n_test, n_val = int(round(n * test_frac)), int(round(n * val_frac))
    m = {g: "test" for g in groups[:n_test]}
    m.update({g: "val" for g in groups[n_test:n_test + n_val]})
    m.update({g: "train" for g in groups[n_test + n_val:]})
    return df[group].astype(str).map(m).rename("split")


def make_split(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int) -> tuple[pd.Series, str]:
    if "split" in df.columns and set(df["split"].unique()) == {"train", "val", "test"}:
        return df["split"].rename("split"), "predefined"
    if "date" in df.columns and df["date"].nunique() > 10:
        return temporal_split(df, val_frac, test_frac), "temporal"
    return group_split(df, val_frac, test_frac, seed), "group"


def make_preprocessor(num: list[str], cat: list[str], scale: bool) -> ColumnTransformer:
    steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("sc", StandardScaler()))
    ct = ColumnTransformer(
        [("num", Pipeline(steps), num),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
        verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return ct


class NaiveBaseline(BaseEstimator, RegressorMixin):
    """Baseline : dernière valeur connue (requêtes des dernières 24 h)."""

    def __init__(self, column: str = "requests_one_day", calibrate: bool = False):
        self.column = column
        self.calibrate = calibrate

    def fit(self, X, y=None):
        # Si les variables sont normalisées à une échelle différente de la cible, on calibre
        # un simple facteur d'échelle sur le train (persistance recalibrée).
        self.scale_ = 1.0
        if self.calibrate and y is not None:
            denom = float(np.asarray(X[self.column], dtype=float).sum())
            self.scale_ = float(np.asarray(y, dtype=float).sum()) / denom if denom > 0 else 1.0
        return self

    def predict(self, X):
        return np.asarray(X[self.column], dtype=float) * getattr(self, "scale_", 1.0)
