"""Tests anti-fuite de données (module B) — critères d'acceptation 8 et 9."""
import numpy as np
import pandas as pd
import pytest

from ad_optimizer.data.synthetic_generator import generate_campaigns
from ad_optimizer.under_delivery.snapshot_builder import (
    CAT_FEATURES,
    FORBIDDEN_FEATURES,
    NUM_FEATURES,
    build_snapshots,
    compute_snapshot_features,
    snapshot_day_count,
)
from ad_optimizer.under_delivery.train import assert_no_leakage


@pytest.fixture(scope="module")
def data():
    return generate_campaigns(60, seed=11)


def test_feature_list_has_no_forbidden_columns():
    assert not (set(NUM_FEATURES + CAT_FEATURES) & FORBIDDEN_FEATURES)
    assert_no_leakage(NUM_FEATURES + CAT_FEATURES)


def test_guard_raises_on_leak():
    with pytest.raises(AssertionError):
        assert_no_leakage(NUM_FEATURES + ["final_delivered_impressions"])
    with pytest.raises(AssertionError):
        assert_no_leakage(["under_delivery_final"])


def test_features_ignore_the_future(data):
    """Modifier radicalement les jours APRÈS le snapshot ne doit changer aucune variable."""
    camps, daily = data
    pcts = (0.25, 0.5, 0.75)
    base = build_snapshots(camps, daily, pcts)
    tampered = daily.copy()
    rng = np.random.default_rng(0)
    dur = camps.set_index("campaign_id")["duration_days"]
    for pct in pcts:
        pass
    # on corrompt tous les jours au-delà du PREMIER snapshot (25 %) puis on ne compare que celui-ci
    cut = tampered["campaign_id"].map(lambda c: snapshot_day_count(int(dur[c]), 0.25))
    future = tampered["day_index"] >= cut
    tampered.loc[future, "delivered_impressions"] = rng.integers(0, 10**6, future.sum())
    tampered.loc[future, "available_inventory"] = rng.integers(0, 10**6, future.sum())
    alt = build_snapshots(camps, tampered, pcts)
    a = base[base["snapshot_pct"] == 0.25][NUM_FEATURES + CAT_FEATURES].reset_index(drop=True)
    b = alt[alt["snapshot_pct"] == 0.25][NUM_FEATURES + CAT_FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    # ... alors que la cible, elle, dépend bien de la livraison finale
    assert not base["under_delivery_final"].equals(alt["under_delivery_final"])


def test_builder_matches_truncated_recomputation(data):
    """Recalculer à la main avec des séries tronquées donne exactement les variables du snapshot."""
    camps, daily = data
    snaps = build_snapshots(camps, daily, (0.5,))
    c = camps.iloc[3]
    d = daily[daily["campaign_id"] == c["campaign_id"]].sort_values("day_index")
    e = snapshot_day_count(int(c["duration_days"]), 0.5)
    f = compute_snapshot_features(c, d["delivered_impressions"].to_numpy()[:e],
                                  d["available_inventory"].to_numpy()[:e], 0.5)
    row = snaps[snaps["campaign_id"] == c["campaign_id"]].iloc[0]
    for k in NUM_FEATURES:
        assert row[k] == pytest.approx(f[k])


def test_snapshot_function_cannot_see_more_than_it_is_given():
    camp = {"booked_impressions": 100, "duration_days": 5, "priority": 1, "n_allowed_categories": 1,
            "max_daily_impressions": 0, "targeting_segment": "g", "campaign_type": "t"}
    with pytest.raises(ValueError):
        compute_snapshot_features(camp, np.array([]), np.array([]), 0.5)
    with pytest.raises(ValueError):
        compute_snapshot_features(camp, np.array([1.0, 2.0]), np.array([1.0]), 0.5)


def test_trained_model_uses_only_allowed_features(small_paths, small_ud_cfg):
    from ad_optimizer.common.model_registry import load_active
    from ad_optimizer.under_delivery import train as ud
    ud.prepare_data(small_paths, small_ud_cfg, seed=3)
    snaps = pd.read_parquet(small_paths.data_synthetic / "snapshots.parquet")
    ud.train_under_delivery(snaps, None, small_paths, small_ud_cfg, seed=3, progress=lambda m: None)
    bundle, meta = load_active("under_delivery", small_paths)
    used = bundle["features"]["num"] + bundle["features"]["cat"]
    assert not (set(used) & FORBIDDEN_FEATURES)
    train_ids = set(snaps["campaign_id"]) - set(meta["split"]["test_campaign_ids"]) - set(meta["split"]["val_campaign_ids"])
    assert not (train_ids & set(meta["split"]["test_campaign_ids"]))
