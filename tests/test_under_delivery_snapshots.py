import numpy as np
import pandas as pd
import pytest

from ad_optimizer.data.synthetic_generator import generate_campaigns
from ad_optimizer.under_delivery.evaluation import (
    classification_metrics,
    risk_class,
    select_threshold,
)
from ad_optimizer.under_delivery.predict import recommend
from ad_optimizer.under_delivery.snapshot_builder import (
    TARGET,
    build_snapshots,
    compute_snapshot_features,
    snapshot_day_count,
)
from ad_optimizer.under_delivery.train import group_split_ids


@pytest.fixture(scope="module")
def data():
    return generate_campaigns(120, seed=5)


def test_generator_is_deterministic():
    a = generate_campaigns(30, seed=9)
    b = generate_campaigns(30, seed=9)
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])


def test_delivery_never_exceeds_booking_and_is_consistent(data):
    camps, daily = data
    last = daily.sort_values("day_index").groupby("campaign_id").last()
    booked = camps.set_index("campaign_id")["booked_impressions"]
    assert (last["cumulative_delivered_impressions"] <= booked.loc[last.index] + 1e-6).all()
    assert (daily["delivered_impressions"] >= 0).all()


def test_snapshot_features_hand_computed():
    camp = {"booked_impressions": 1000, "duration_days": 10, "priority": 2, "n_allowed_categories": 2,
            "max_daily_impressions": 0, "targeting_segment": "general", "campaign_type": "awareness"}
    f = compute_snapshot_features(camp, np.array([50, 50, 50, 50, 50]), np.array([500.0] * 5), 0.5)
    assert f["delivered_so_far"] == 250 and f["remaining_impressions"] == 750
    assert f["remaining_days"] == 5 and f["required_daily_rate"] == 150
    assert f["delivery_progress_rate"] == pytest.approx(0.25)
    assert f["progress_gap"] == pytest.approx(0.25 - 0.5)
    assert f["recent_to_required_rate_ratio"] == pytest.approx(50 / 150)
    assert f["observed_capture_rate"] == pytest.approx(0.1)


def test_label_uses_final_delivery_and_tolerance():
    camps = pd.DataFrame([{"campaign_id": "X", "booked_impressions": 1000, "duration_days": 4, "priority": 1,
                           "n_allowed_categories": 1, "max_daily_impressions": 0, "targeting_segment": "general",
                           "campaign_type": "awareness", "scenario": "t"}])
    mk = lambda d: pd.DataFrame({"campaign_id": "X", "day_index": range(4),
                                 "date": pd.date_range("2026-01-01", periods=4),
                                 "delivered_impressions": d, "available_inventory": [1000.0] * 4})
    assert build_snapshots(camps, mk([250, 250, 250, 250]), (0.5,), 0.98)[TARGET].iloc[0] == 0
    assert build_snapshots(camps, mk([250, 250, 250, 240]), (0.5,), 0.98)[TARGET].iloc[0] == 0   # 99 %
    assert build_snapshots(camps, mk([250, 250, 250, 200]), (0.5,), 0.98)[TARGET].iloc[0] == 1   # 95 %
    assert build_snapshots(camps, mk([250, 250, 250, 240]), (0.5,), 1.0)[TARGET].iloc[0] == 1    # tolérance 100 %


def test_snapshot_keeps_at_least_one_day_remaining():
    assert snapshot_day_count(14, 0.75) == 10 and snapshot_day_count(2, 0.99) == 1


def test_split_by_campaign_never_mixes_snapshots(data):
    camps, daily = data
    snaps = build_snapshots(camps, daily, (0.25, 0.5, 0.75))
    ids = group_split_ids(snaps["campaign_id"].to_numpy(), 0.15, 0.15, seed=0)
    assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])
    assert ids["train"] | ids["val"] | ids["test"] == set(snaps["campaign_id"])
    for k in ids:
        m = snaps["campaign_id"].isin(ids[k])
        # tous les snapshots d'une campagne sont dans le même jeu
        assert snaps[m].groupby("campaign_id").size().eq(3).all()


def test_threshold_selection_reaches_recall_target():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.3 + rng.random(2000) * 0.7, 0, 1)
    thr = select_threshold(y, p, "recall_target", 0.8)
    assert classification_metrics(y, p, thr)["recall"] >= 0.8


def test_metrics_and_risk_classes():
    m = classification_metrics([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1], 0.5)
    assert m["confusion"] == [[1, 1], [1, 1]] and m["recall"] == 0.5
    assert risk_class(np.array([0.1, 0.5, 0.9]), 0.35, 0.65).tolist() == ["faible", "modéré", "élevé"]


def test_recommendation_is_rule_based():
    row = pd.Series({"recent_to_required_rate_ratio": 0.4, "avg_to_required_rate_ratio": 0.5,
                     "projected_capacity_ratio": 0.7, "inventory_trend_ratio": 0.5, "has_daily_cap": 0,
                     "daily_cap_to_required_ratio": 10, "n_allowed_categories": 3, "observed_capture_rate": 0.3})
    txt = recommend(row, 0.9, "élevé")
    assert "rythme récent est inférieur" in txt and "surveillance renforcée" in txt
