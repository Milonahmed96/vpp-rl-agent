"""Tests for the FastAPI control loop and endpoints.

Ingestion clients and the PPO model are faked so the tests run offline and do
not depend on the gitignored trained-model / scaler artifacts.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.api.main as api
from src.api.main import VPPController, app
from src.db.database import Database
from src.features.engineer import FeatureEngineer


class _FakeModel:
    """Stand-in PPO model that always returns the configured action."""

    def __init__(self, action: int = 1) -> None:
        self._action = action

    def predict(self, observation, deterministic: bool = True):
        assert observation.shape == (15,)
        return np.array(self._action), None


def _mock_neso() -> AsyncMock:
    neso = AsyncMock()
    neso.fetch_frequency.return_value = {"freq_mean": 49.95, "freq_std": 0.02}
    neso.fetch_generation.return_value = pd.DataFrame(
        [
            {
                "SETTLEMENT_DATE": "2024-03-01",
                "SETTLEMENT_PERIOD": 20,
                "WIND_GENERATION": 9000.0,
                "SOLAR_GENERATION": 1500.0,
                "ND": 31000.0,
            }
        ]
    )
    return neso


def _mock_elexon() -> AsyncMock:
    elexon = AsyncMock()
    elexon.fetch_imbalance_price.return_value = {
        "settlementPeriod": 20,
        "systemBuyPrice": 85.0,
        "systemSellPrice": 70.0,
        "netImbalanceVolume": -40.0,
    }
    return elexon


def _make_controller(tmp_path, action: int = 1) -> VPPController:
    engineer = FeatureEngineer(scaler_path=str(tmp_path / "scaler.pkl"))
    ctrl = VPPController(
        model_path="unused",
        db=Database(db_path=str(tmp_path / "api.db")),
        neso=_mock_neso(),
        elexon=_mock_elexon(),
        engineer=engineer,
        seed_path=str(tmp_path / "missing.csv"),  # forces synthetic seed
    )
    ctrl._model = _FakeModel(action=action)  # bypass real model load
    return ctrl


@pytest.mark.asyncio
async def test_tick_persists_decision_and_updates_state(tmp_path):
    ctrl = _make_controller(tmp_path, action=2)  # discharge
    decision = await ctrl.tick()

    assert decision["action"] == 2
    assert ctrl.last_action == 2
    assert ctrl.last_updated is not None
    decisions = ctrl.db.get_recent_decisions()
    assert len(decisions) == 1
    snapshots = ctrl.db.get_recent_snapshots()
    assert len(snapshots) == 1
    assert snapshots[0]["system_buy_price"] == 85.0


@pytest.mark.asyncio
async def test_metrics_and_reset(tmp_path):
    ctrl = _make_controller(tmp_path, action=2)
    await ctrl.tick()
    await ctrl.tick()
    metrics = ctrl.metrics()
    assert metrics["episodes_run"] == 2
    assert "net_profit_gbp" in metrics
    assert 0.0 <= metrics["stabilising_action_pct"] <= 100.0

    reset = ctrl.reset()
    assert reset["current_soc"] == 0.5
    assert ctrl.last_action is None
    assert ctrl.episodes_run == 0


class _DummyScheduler:
    """No-op scheduler so endpoint tests never fire a background tick."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def add_job(self, *args, **kwargs) -> None:
        pass

    def start(self) -> None:
        pass

    def shutdown(self, wait: bool = False) -> None:
        pass


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "AsyncIOScheduler", _DummyScheduler)
    api.controller = _make_controller(tmp_path, action=1)
    with TestClient(app) as test_client:
        yield test_client
    api.controller = None


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "timestamp" in body


def test_status_endpoint(client):
    response = client.get("/status")
    assert response.status_code == 200
    assert "current_soc" in response.json()


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "net_profit_gbp" in response.json()


def test_reset_endpoint(client):
    response = client.post("/reset")
    assert response.status_code == 200
    assert response.json()["current_soc"] == 0.5


def test_decisions_endpoint_limit_validation(client):
    assert client.get("/decisions?limit=10").status_code == 200
    # limit above the allowed maximum is rejected.
    assert client.get("/decisions?limit=99999").status_code == 422
