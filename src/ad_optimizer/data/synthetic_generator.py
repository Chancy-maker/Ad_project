"""Générateurs synthétiques déterministes (graine configurable).

Contient :
- un SUBSTITUT d'Internos (même schéma, utilisé uniquement si le vrai fichier est absent) ;
- la table journalière d'inventaire (champs métier absents d'Internos) ;
- les campagnes, contrats et livraisons quotidiennes du module B.

Rien ici ne représente des données TF1+.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- calendrier
HOLIDAYS_MD = [(1, 1), (5, 1), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)]


def _holiday_array(y0: int, y1: int) -> np.ndarray:
    return np.array(sorted(np.datetime64(f"{y}-{m:02d}-{d:02d}")
                           for y in range(y0, y1 + 1) for m, d in HOLIDAYS_MD))


def holiday_features(dates: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(dates)
    hol = _holiday_array(dates.dt.year.min() - 1, dates.dt.year.max() + 1)
    d64 = dates.values.astype("datetime64[D]")
    idx = np.searchsorted(hol, d64, side="left")
    nxt = hol[np.minimum(idx, len(hol) - 1)]
    days_to_next = (nxt - d64).astype(int)
    is_hol = np.isin(d64, hol)
    year_end = np.array([np.datetime64(f"{y}-12-31") for y in dates.dt.year])
    left = np.array([((hol > d) & (hol <= ye)).sum() for d, ye in zip(d64, year_end)])
    return pd.DataFrame({"is_holiday": is_hol.astype(int), "holidays_left": left,
                         "next_holidays": days_to_next}, index=dates.index)


# --------------------------------------------------------------------------- Internos (substitut)
VIDEO_TYPES = ["lecture", "seminar", "tutorial", "event", "other"]
_TYPE_W = [0.40, 0.20, 0.20, 0.10, 0.10]
_TYPE_F = {"lecture": 1.0, "seminar": 0.7, "tutorial": 1.3, "event": 2.0, "other": 0.5}
CATEGORIES = ["Divertissement", "Information", "Sport", "Jeunesse", "Documentaire"]


def generate_internos_proxy(n_media: int = 900, seed: int = 42, start: str = "2016-01-01",
                            end: str = "2018-06-30") -> pd.DataFrame:
    """Substitut SYNTHÉTIQUE d'Internos : même schéma, mais dynamique inventée.

    Au jour t : requests_one_day = requêtes du jour t, requests_two_days = jour t-1,
    requests_one_week / two_week = sommes glissantes 7 / 14 jours, `requests` (cible) = jour t+1.
    """
    rng = np.random.default_rng(seed)
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    span = (t1 - t0).days
    parts = []
    for m in range(n_media):
        vt = rng.choice(VIDEO_TYPES, p=_TYPE_W)
        publish = t0 + pd.Timedelta(days=int(rng.integers(0, span)))
        life = int(rng.integers(40, 161))
        base = rng.lognormal(np.log(40), 1.0) * _TYPE_F[vt]
        tau = rng.uniform(15, 90)
        d = np.arange(life + 1)
        dates = pd.date_range(publish, periods=life + 1, freq="D")
        dow = dates.dayofweek.values
        doy = dates.dayofyear.values
        month = dates.month.values
        hol = np.isin(dates.values.astype("datetime64[D]"), _holiday_array(2015, 2019))
        level = (base * (np.exp(-d / tau) + 0.15)
                 * np.where(dow >= 5, 0.55, 1.0)
                 * (1 + 0.25 * np.cos(2 * np.pi * (doy - 30) / 365))
                 * np.where(np.isin(month, (7, 8)), 0.5, 1.0)
                 * np.where(hol, 0.6, 1.0)
                 * (1 + (rng.random(life + 1) < 0.03) * rng.uniform(1, 3, life + 1)))
        req = rng.poisson(level * rng.gamma(4.0, 1 / 4.0, life + 1)).astype(float)
        cs = np.concatenate([[0.0], np.cumsum(req)])
        t = np.arange(1, life)                      # lignes t = 1..life-1 (cible t+1 existe)
        one_week = cs[t + 1] - cs[np.maximum(t - 6, 0)]
        two_week = cs[t + 1] - cs[np.maximum(t - 13, 0)]
        df = pd.DataFrame({
            "media_id": f"M{m:04d}",
            "date": dates[t],
            "requests_one_week": one_week,
            "requests_two_week": two_week,
            "requests_one_day": req[t],
            "requests_two_days": req[t - 1],
            "day_of_week": dow[t],
            "day_of_month": dates.day.values[t],
            "day_of_year": doy[t],
            "month": month[t],
            "comments": rng.poisson(0.02 * one_week),
            "comments_in_section": rng.poisson(10 + 0.01 * one_week),
            "publications": rng.poisson(2, len(t)),
            "publications_in_section": rng.poisson(20, len(t)),
            "votes": rng.poisson(0.01 * one_week),
            "votes_in_section": rng.poisson(8 + 0.005 * one_week),
            "days_published": t,
            "video_type": vt,
            "requests": req[t + 1],
        })
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    hf = holiday_features(out["date"])
    out["holidays_left"] = hf["holidays_left"].values
    out["next_holidays"] = hf["next_holidays"].values
    out["is_holiday"] = hf["is_holiday"].values
    return out


def assign_content_category(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Catégorie éditoriale SYNTHÉTIQUE stable par contenu (absente d'Internos)."""
    ids = df["media_id"].astype(str).unique()
    rng = np.random.default_rng(seed + 7)
    mapping = dict(zip(ids, rng.choice(CATEGORIES, size=len(ids))))
    return df["media_id"].astype(str).map(mapping)


def build_inventory_daily(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Table `synthetic_inventory_daily` : champs métier simulés autour des requêtes."""
    rng = np.random.default_rng(seed + 11)
    n = len(df)
    cat = df["content_category"] if "content_category" in df else assign_content_category(df, seed)
    opps_by_cat = {"Divertissement": 2.4, "Information": 1.8, "Sport": 3.0, "Jeunesse": 1.5,
                   "Documentaire": 2.0}
    opps = cat.map(opps_by_cat).astype(float).values * rng.normal(1, 0.08, n)
    elig = np.clip(rng.normal(0.85, 0.05, n), 0.5, 1.0)
    views = df["requests"].astype(float).values
    eligible = views * opps * elig
    fill = np.clip(rng.beta(8, 2.5, n), 0.3, 1.0)
    d = pd.to_datetime(df["date"])
    return pd.DataFrame({
        "date": d.values, "content_id": df["media_id"].values, "content_category": cat.values,
        "video_type": df["video_type"].values, "day_of_week": d.dt.dayofweek.values,
        "month": d.dt.month.values, "is_weekend": (d.dt.dayofweek >= 5).astype(int).values,
        "is_holiday": holiday_features(d)["is_holiday"].values,
        "views_or_requests": views,
        "avg_watch_time_minutes": np.clip(rng.normal(22, 8, n), 2, 90),
        "ad_opportunities_per_view": opps, "eligibility_rate": elig,
        "eligible_ad_impressions": eligible,
        "delivered_ad_impressions": eligible * fill, "fill_rate": fill,
    })


# --------------------------------------------------------------------------- campagnes (module B)
SCENARIOS = {
    "normal": 0.28, "ambitious": 0.12, "inventory_drop": 0.15, "narrow_targeting": 0.12,
    "slow_start": 0.11, "irregular": 0.10, "easy": 0.12,
}
SEGMENTS = {"general": (0.6, 1.0), "sport": (0.2, 0.5), "news": (0.2, 0.5),
            "jeunesse": (0.15, 0.4), "premium": (0.1, 0.3)}
CAMPAIGN_TYPES = ["awareness", "performance", "branding"]
_WEEKLY = np.array([1.0, 0.98, 1.0, 1.05, 1.15, 1.25, 1.2])
_PRIO_F = {1: 1.0, 2: 0.8, 3: 0.6}


def _sample_volume_and_cv(rng, profiles: pd.DataFrame | None, median_inventory: float):
    """Volume quotidien relatif et variabilité : empiriques (Criteo) si dispo, sinon paramétriques."""
    if profiles is not None and len(profiles) > 0:
        row = profiles.iloc[int(rng.integers(0, len(profiles)))]
        med = max(float(profiles["mean_daily"].median()), 1e-9)
        return median_inventory * float(row["mean_daily"]) / med, float(np.clip(row["cv_daily"], 0.05, 1.2))
    return (median_inventory * float(rng.lognormal(0, 0.7)),
            float(np.clip(rng.lognormal(np.log(0.30), 0.35), 0.08, 1.0)))


def generate_campaigns(n: int = 3000, seed: int = 42, profiles: pd.DataFrame | None = None,
                       median_inventory: float = 60000, start_date: str = "2026-01-01",
                       tolerance: float = 0.98, id_prefix: str = "C") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Génère contrats (`synthetic_campaigns`) et livraisons (`synthetic_campaign_delivery_daily`)."""
    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp(start_date)
    names, probs = list(SCENARIOS), np.array(list(SCENARIOS.values()))
    probs = probs / probs.sum()
    camps, dailies = [], []
    for i in range(n):
        scen = str(rng.choice(names, p=probs))
        D = int(rng.integers(14, 61))
        start = base_date + pd.Timedelta(days=int(rng.integers(0, 300)))
        seg = str(rng.choice(list(SEGMENTS)))
        lo, hi = SEGMENTS[seg]
        share = float(rng.uniform(lo, hi))
        if scen == "narrow_targeting":
            seg = str(rng.choice(["premium", "jeunesse", "sport"]))
            share = float(rng.uniform(0.04, 0.12))
        n_cat = int(rng.integers(1, 5))
        cats = list(rng.choice(CATEGORIES, n_cat, replace=False))
        priority = int(rng.choice([1, 2, 3], p=[0.25, 0.45, 0.30]))
        ctype = str(rng.choice(CAMPAIGN_TYPES))
        V, cv = _sample_volume_and_cv(rng, profiles, median_inventory)
        if scen == "irregular":
            cv = float(np.clip(cv * 2.2, 0.5, 1.2))
        capture_base = float(rng.uniform(0.25, 0.45)) * _PRIO_F[priority]
        avail_share = share * (0.5 + 0.5 * n_cat / 4)

        t = np.arange(D)
        dows = ((start.dayofweek + t) % 7)
        drift = rng.normal(0, 0.15)
        exp_avail = V * avail_share * _WEEKLY[dows] * (1 + drift * t / D)
        capacity = float((exp_avail * capture_base).sum())
        ambition = {"normal": rng.uniform(0.55, 1.0), "ambitious": rng.uniform(1.0, 1.5),
                    "inventory_drop": rng.uniform(0.70, 1.05), "narrow_targeting": rng.uniform(0.9, 1.35),
                    "slow_start": rng.uniform(0.85, 1.2), "irregular": rng.uniform(0.9, 1.3),
                    "easy": rng.uniform(0.30, 0.60)}[scen]
        booked = int(max(1000, round(capacity * ambition)))
        max_daily = int(booked / D * rng.uniform(1.5, 3.0)) if rng.random() < 0.3 else 0

        shock = np.ones(D)
        if scen == "inventory_drop":
            td = int(rng.uniform(0.3, 0.8) * D)
            shock[td:] = rng.uniform(0.25, 0.6)
        pace = np.ones(D)
        if scen == "slow_start":
            pace[: max(1, int(rng.uniform(0.3, 0.4) * D))] = 0.25
        noise_av = np.clip(1 + rng.normal(0, cv, D), 0.2, 2.5)
        noise_cap = np.clip(1 + rng.normal(0, 0.45 if scen == "irregular" else 0.12, D), 0.0, 2.5)
        boost_max = float(rng.uniform(0.2, 0.6))
        avail = exp_avail * noise_av * shock

        delivered = np.zeros(D)
        remaining = float(booked)
        for d in range(D):
            required = remaining / (D - d)
            base_rate = avail[d] * capture_base * pace[d] * noise_cap[d]
            boost = np.clip(required / (avail[d] * capture_base + 1e-9), 1.0, 1.0 + boost_max)
            x = base_rate * boost
            if max_daily:
                x = min(x, max_daily)
            x = float(min(x, remaining, avail[d]))
            delivered[d] = round(x)
            remaining -= delivered[d]
        cum = np.cumsum(delivered)
        under = int(cum[-1] < tolerance * booked)
        cid = f"{id_prefix}{i:05d}"
        camps.append({
            "campaign_id": cid, "advertiser_id": f"A{int(rng.integers(1, 400)):04d}",
            "campaign_start_date": start, "campaign_end_date": start + pd.Timedelta(days=D - 1),
            "booked_impressions": booked, "priority": priority, "targeting_segment": seg,
            "allowed_content_categories": "|".join(cats), "n_allowed_categories": n_cat,
            "max_daily_impressions": max_daily, "campaign_type": ctype,
            "scenario": scen,   # métadonnée du générateur : JAMAIS une variable d'entrée
        })
        rem_after = booked - cum
        rem_days = D - 1 - t
        r3 = np.array([delivered[max(0, k - 2): k + 1].mean() for k in range(D)])
        r7 = np.array([delivered[max(0, k - 6): k + 1].mean() for k in range(D)])
        dailies.append(pd.DataFrame({
            "campaign_id": cid, "date": start + pd.to_timedelta(t, unit="D"), "day_index": t,
            "available_inventory": avail.round(), "delivered_impressions": delivered,
            "cumulative_delivered_impressions": cum, "remaining_impressions": np.maximum(rem_after, 0),
            "remaining_days": rem_days, "daily_delivery_rate": delivered,
            "recent_3d_delivery_rate": r3, "recent_7d_delivery_rate": r7,
            "required_daily_rate": np.where(rem_days > 0, np.maximum(rem_after, 0) / np.maximum(rem_days, 1),
                                            np.maximum(rem_after, 0)),
            "delivery_progress_rate": cum / booked, "under_delivery_final": under,
        }))
    campaigns = pd.DataFrame(camps)
    campaigns["duration_days"] = (campaigns["campaign_end_date"] - campaigns["campaign_start_date"]).dt.days + 1
    return campaigns, pd.concat(dailies, ignore_index=True)
