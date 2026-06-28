"""Tests for the SQLAlchemy database layer."""

from __future__ import annotations

from src.db.database import Database


def _make_db(tmp_path) -> Database:
    return Database(db_path=str(tmp_path / "test.db"))


def test_insert_and_get_snapshot(tmp_path):
    db = _make_db(tmp_path)
    row_id = db.insert_snapshot(
        freq_mean=50.01,
        delta_f=0.01,
        system_buy_price=60.0,
        system_sell_price=55.0,
        net_imbalance_volume=-10.0,
        wind_generation=8000.0,
        solar_generation=1000.0,
        national_demand=30000.0,
    )
    assert row_id == 1
    snapshots = db.get_recent_snapshots()
    assert len(snapshots) == 1
    assert snapshots[0]["freq_mean"] == 50.01
    assert "timestamp" in snapshots[0]


def test_insert_and_get_decision(tmp_path):
    db = _make_db(tmp_path)
    db.insert_decision(
        action=2,
        action_mw=-25.0,
        current_soc=0.47,
        reward=1.5,
        profit=0.75,
        freq_penalty=-2.0,
        degradation_cost=11.25,
        settlement_period=12,
    )
    decisions = db.get_recent_decisions()
    assert len(decisions) == 1
    assert decisions[0]["action"] == 2
    assert decisions[0]["settlement_period"] == 12


def test_recent_decisions_ordered_newest_first_and_limited(tmp_path):
    db = _make_db(tmp_path)
    for period in range(1, 6):
        db.insert_decision(
            action=0,
            action_mw=0.0,
            current_soc=0.5,
            reward=0.0,
            profit=0.0,
            freq_penalty=0.0,
            degradation_cost=0.0,
            settlement_period=period,
        )
    recent = db.get_recent_decisions(n=3)
    assert len(recent) == 3
    # Newest (settlement_period 5) first.
    assert recent[0]["settlement_period"] == 5
    assert recent[-1]["settlement_period"] == 3


def test_empty_database_returns_empty_lists(tmp_path):
    db = _make_db(tmp_path)
    assert db.get_recent_decisions() == []
    assert db.get_recent_snapshots() == []
