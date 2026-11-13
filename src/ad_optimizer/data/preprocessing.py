"""Préparation des données du module A (Internos réel ou substitut synthétique)."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.config import Paths, default_paths, load_config
from ..common.logging import get_logger
from .synthetic_generator import (
    assign_content_category,
    build_inventory_daily,
    generate_internos_proxy,
    holiday_features,
)
from .validation import ValidationReport, raise_if_invalid, validate_dataframe

log = get_logger()

REQUIRED_INTERNOS = ["requests_one_day", "requests", "video_type"]
NON_NEGATIVE = ["requests", "requests_one_day", "requests_two_days", "requests_one_week",
                "requests_two_week", "comments", "votes", "publications", "days_published"]


def snake(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """'Requests one week' -> 'requests_one_week', etc."""
    out = df.copy()
    out.columns = [snake(c) for c in out.columns]
    return out


def load_internos(path: Path) -> pd.DataFrame:
    """Charge le fichier Internos réel (CSV/TSV/XLSX) et normalise les noms de colonnes."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep=None, engine="python")
    return normalize_columns(df)


PRESPLIT_RENAME = {
    "rep_one_week": "requests_one_week", "rep_two_week": "requests_two_week",
    "rep_one_day": "requests_one_day", "rep_two_day": "requests_two_days",
    "current_comments": "comments", "recent_publications": "publications",
    "current_votes": "votes", "next_holiday": "next_holidays", "reproductions": "requests",
}


def find_presplit_files(raw_dir: Path) -> dict[str, Path] | None:
    """Repère train/validation/test (fichiers déjà découpés et normalisés) dans data/raw/."""
    found: dict[str, Path] = {}
    for f in sorted(Path(raw_dir).glob("*.csv")):
        n = f.name.lower()
        for key, tags in (("train", ("train",)), ("val", ("valid", "val")), ("test", ("test",))):
            if key not in found and any(t in n for t in tags):
                found[key] = f
                break
    return found if len(found) == 3 else None


def infer_target_scale(y: pd.Series) -> tuple[float | None, str]:
    """Si la cible est normalisée min-max, retrouve l'échelle en nombre de requêtes (1 / plus petite
    valeur positive) et vérifie que les valeurs reconstruites sont bien entières."""
    pos = y[y > 0]
    if pos.empty:
        return None, "cible nulle partout"
    scale = 1.0 / float(pos.min())
    if scale <= 1.5:      # déjà en comptes
        return 1.0, "cible déjà en nombre de requêtes"
    k = pos * scale
    frac_int = float(((k - k.round()).abs() < 0.02).mean())
    if frac_int >= 0.95:
        return scale, f"échelle déduite : max ≈ {scale:,.0f} requêtes ({frac_int:.0%} de valeurs entières)"
    return None, f"échelle non déductible ({frac_int:.0%} de valeurs entières) : cible laissée normalisée"


def load_internos_presplit(files: dict[str, Path], target_max_requests: float | None = None
                           ) -> tuple[pd.DataFrame, list[str]]:
    """Charge les 3 fichiers déjà découpés/normalisés ; conserve le découpage fourni."""
    notes = []
    parts = []
    for key, name in (("train", "train"), ("val", "val"), ("test", "test")):
        d = pd.read_csv(files[key], sep=r"\s+", engine="python")
        d["split"] = name
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df = df.rename(columns=PRESPLIT_RENAME)
    # Type de vidéo : 4 colonnes « décalées » ; valeur > 0 = drapeau actif (plusieurs possibles)
    flags = [c for c in df.columns if re.fullmatch(r"video_type_\d+", c)]
    for c in flags:
        df[c.replace("video_type_", "vt_")] = (df[c] > 0).astype(int)
    vt = [c.replace("video_type_", "vt_") for c in flags]
    df = df.drop(columns=flags)
    n_act = df[vt].sum(axis=1)
    first = df[vt].values.argmax(axis=1)
    df["video_type"] = [f"type_{k}" if n == 1 else ("multi_types" if n > 1 else "aucun_drapeau")
                        for k, n in zip(first, n_act)]
    # Date approximative reconstruite depuis l'angle jour-de-l'année (année de référence fictive)
    ang = np.arctan2(df["day_year_sin"], df["day_year_cos"]) % (2 * np.pi)
    doy = np.clip(np.round(ang / (2 * np.pi) * 365).astype(int), 0, 364)
    df["date"] = pd.Timestamp("2017-01-01") + pd.to_timedelta(doy, unit="D")
    notes.append("Fichiers train/validation/test déjà normalisés (0–1) : découpage fourni conservé ; la date est "
                 "reconstruite à partir du jour de l'année (année fictive) et il n'y a pas d'identifiant de contenu.")
    if target_max_requests:
        scale, msg = float(target_max_requests), "échelle fournie dans configs/inventory.yaml"
    else:
        scale, msg = infer_target_scale(df["requests"])
    if scale:
        df["requests"] = (df["requests"] * scale).round()
    notes.append(f"Cible `requests` : {msg}. À confirmer sur la fiche Mendeley (surcharge possible via "
                 "`internos_target_max_requests`).")
    return df, notes


def ensure_ids_and_date(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Garantit media_id et date ; signale ce qui est dérivé (le schéma réel peut les omettre)."""
    notes = []
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "day_of_year" in df.columns:
        df["date"] = pd.Timestamp("2017-01-01") + pd.to_timedelta(df["day_of_year"].astype(int) - 1, unit="D")
        notes.append("Colonne `date` absente : dérivée de day_of_year (année de référence 2017). "
                     "Vérifier le schéma réel ; sinon, privilégier un split groupé.")
    else:
        df["date"] = pd.Timestamp("2017-01-01") + pd.to_timedelta(range(len(df)), unit="D")
        notes.append("Aucune information temporelle exploitable : date fictive par ordre de ligne.")
    if "media_id" not in df.columns:
        df["media_id"] = "ALL-" + df["video_type"].astype(str)
        notes.append("Identifiant de ressource absent : media_id = type de vidéo.")
    df["media_id"] = df["media_id"].astype(str)
    return df, notes


def clean_internos(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport, dict]:
    rep = validate_dataframe(df, REQUIRED_INTERNOS, NON_NEGATIVE, name="internos")
    raise_if_invalid(rep)
    n0 = len(df)
    df = df.drop_duplicates().dropna(subset=["requests"])
    df = df[df["requests"] >= 0]
    actions = {"rows_in": n0, "rows_out": len(df), "rows_dropped": n0 - len(df)}
    return df.reset_index(drop=True), rep, actions


def data_version(df: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(df, index=False).values
    return format(int(h.sum() % (1 << 48)), "012x")


def prepare_inventory_dataset(paths: Paths | None = None, cfg: dict | None = None,
                              force_proxy: bool = False) -> tuple[pd.DataFrame, dict]:
    """Prépare le dataset du module A et écrit data/processed/* + un manifeste."""
    paths = (paths or default_paths()).ensure()
    cfg = cfg or load_config("inventory", paths)
    app_cfg = load_config("app", paths)
    seed = int(cfg.get("random_seed", 42))
    raw = paths.data_raw / app_cfg["datasets"]["internos_file"]
    presplit = find_presplit_files(paths.data_raw)
    notes: list[str] = []
    predefined = False
    if raw.exists() and not force_proxy:
        df = load_internos(raw)
        source, is_real = "Internos (réel, Mendeley Data)", True
    elif presplit and not force_proxy:
        df, n0 = load_internos_presplit(presplit, cfg.get("internos_target_max_requests"))
        notes += n0
        source, is_real, predefined = "Internos (réel, Mendeley Data — fichiers train/validation/test)", True, True
    else:
        p = cfg["proxy"]
        df = generate_internos_proxy(p["n_media"], seed, p["start"], p["end"])
        df.to_parquet(paths.data_synthetic / "internos_proxy.parquet", index=False)
        source, is_real = "Substitut SYNTHÉTIQUE d'Internos (fichier réel absent)", False
        notes.append("Le fichier réel data/raw/internos.csv est absent : un substitut synthétique "
                     "au même schéma est utilisé. À remplacer par le vrai dataset pour le mémoire.")
    has_media_id = "media_id" in df.columns
    df, n2 = ensure_ids_and_date(df)
    notes += [n for n in n2 if not predefined]
    df, rep, actions = clean_internos(df)
    if "is_holiday" not in df.columns:
        df["is_holiday"] = holiday_features(df["date"])["is_holiday"].values
    df["content_category"] = (assign_content_category(df, seed) if has_media_id else df["video_type"])
    df = df.sort_values(["date", "media_id"]).reset_index(drop=True)
    df.to_parquet(paths.data_processed / "internos_clean.parquet", index=False)
    inv = build_inventory_daily(df, seed)
    inv.to_parquet(paths.data_synthetic / "synthetic_inventory_daily.parquet", index=False)
    manifest = {
        "source": source, "is_real": is_real, "n_rows": len(df), "n_media": int(df["media_id"].nunique()),
        "date_min": str(df["date"].min().date()), "date_max": str(df["date"].max().date()),
        "loaded_at": datetime.now().isoformat(timespec="seconds"), "data_version": data_version(df),
        "has_media_id": has_media_id, "predefined_split": predefined,
        "target_normalised": bool(predefined and df["requests"].max() <= 1.0),
        "columns": list(df.columns), "validation": rep.to_dict(), "cleaning": actions, "notes": notes,
        "seed": seed,
    }
    (paths.data_processed / "inventory_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Dataset inventaire prêt : %s lignes (%s)", len(df), source)
    return df, manifest
