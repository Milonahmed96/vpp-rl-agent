"""Tests for the NESO and Elexon ingestion clients.

HTTP traffic is mocked with :class:`httpx.MockTransport`; no real network calls
are made. ``asyncio.sleep`` is neutralised so retry/backoff paths run instantly.
"""

from __future__ import annotations

import httpx
import pandas as pd
import pytest

from src.ingestion import elexon_client, neso_client
from src.ingestion.elexon_client import ElexonAPIError, ElexonClient
from src.ingestion.neso_client import NESOAPIError, NESOClient


def _install_mock_transport(monkeypatch, module, handler) -> None:
    """Patch ``module.httpx.AsyncClient`` to route through a mock transport."""
    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", factory)
    # Neutralise backoff / rate-limit sleeps for fast tests.
    monkeypatch.setattr(module.asyncio, "sleep", _async_noop)


async def _async_noop(*_args, **_kwargs) -> None:
    return None


# --------------------------------------------------------------------------- #
# NESO
# --------------------------------------------------------------------------- #
def _neso_frequency_handler(_request: httpx.Request) -> httpx.Response:
    records = [
        {"dtm": f"2024-01-01T00:00:{i:02d}", "f": 50.0 + (i % 3) * 0.01}
        for i in range(10)
    ]
    return httpx.Response(200, json={"success": True, "result": {"records": records}})


def _neso_generation_handler(_request: httpx.Request) -> httpx.Response:
    records = [
        {
            "SETTLEMENT_DATE": "2024-01-01",
            "SETTLEMENT_PERIOD": p,
            "WIND_GENERATION": 1000 + p,
            "SOLAR_GENERATION": 200 + p,
            "ND": 30000 - p,
        }
        for p in range(1, 49)
    ]
    return httpx.Response(200, json={"success": True, "result": {"records": records}})


@pytest.mark.asyncio
async def test_neso_fetch_frequency_returns_mean_std(monkeypatch):
    _install_mock_transport(monkeypatch, neso_client, _neso_frequency_handler)
    result = await NESOClient().fetch_frequency()
    assert set(result) == {"freq_mean", "freq_std"}
    assert 49.0 < result["freq_mean"] < 51.0
    assert result["freq_std"] >= 0.0


@pytest.mark.asyncio
async def test_neso_fetch_generation_shape(monkeypatch):
    _install_mock_transport(monkeypatch, neso_client, _neso_generation_handler)
    frame = await NESOClient().fetch_generation()
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (48, 5)
    assert list(frame.columns) == [
        "SETTLEMENT_DATE",
        "SETTLEMENT_PERIOD",
        "WIND_GENERATION",
        "SOLAR_GENERATION",
        "ND",
    ]


@pytest.mark.asyncio
async def test_neso_non_200_raises(monkeypatch):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    _install_mock_transport(monkeypatch, neso_client, handler)
    with pytest.raises(NESOAPIError):
        await NESOClient().fetch_frequency()


@pytest.mark.asyncio
async def test_neso_missing_field_handled(monkeypatch):
    def handler(_request: httpx.Request) -> httpx.Response:
        # 'f' field is absent -> must raise a clear domain error, not KeyError.
        records = [{"dtm": "2024-01-01T00:00:00"}]
        return httpx.Response(
            200, json={"success": True, "result": {"records": records}}
        )

    _install_mock_transport(monkeypatch, neso_client, handler)
    with pytest.raises(NESOAPIError):
        await NESOClient().fetch_frequency()


# --------------------------------------------------------------------------- #
# Elexon
# --------------------------------------------------------------------------- #
def _elexon_price_handler(_request: httpx.Request) -> httpx.Response:
    data = [
        {
            "settlementPeriod": p,
            "systemBuyPrice": 60.0 + p,
            "systemSellPrice": 55.0 + p,
            "netImbalanceVolume": -10.0 + p,
        }
        for p in range(1, 11)
    ]
    return httpx.Response(200, json={"data": data})


@pytest.mark.asyncio
async def test_elexon_fetch_imbalance_price(monkeypatch):
    _install_mock_transport(monkeypatch, elexon_client, _elexon_price_handler)
    result = await ElexonClient().fetch_imbalance_price()
    assert set(result) == {
        "settlementPeriod",
        "systemBuyPrice",
        "systemSellPrice",
        "netImbalanceVolume",
    }
    # Latest period (10) should be selected.
    assert result["settlementPeriod"] == 10
    assert result["systemBuyPrice"] == 70.0


@pytest.mark.asyncio
async def test_elexon_generation_mix_shape(monkeypatch):
    def handler(_request: httpx.Request) -> httpx.Response:
        data = [
            {"fuelType": "WIND", "generation": 1000, "settlementPeriod": 1},
            {"fuelType": "CCGT", "generation": 5000, "settlementPeriod": 1},
        ]
        return httpx.Response(200, json={"data": data})

    _install_mock_transport(monkeypatch, elexon_client, handler)
    frame = await ElexonClient().fetch_generation_mix()
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 2
    assert "fuelType" in frame.columns


@pytest.mark.asyncio
async def test_elexon_non_200_raises(monkeypatch):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    _install_mock_transport(monkeypatch, elexon_client, handler)
    with pytest.raises(ElexonAPIError):
        await ElexonClient().fetch_imbalance_price()


@pytest.mark.asyncio
async def test_elexon_empty_data_raises(monkeypatch):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    _install_mock_transport(monkeypatch, elexon_client, handler)
    with pytest.raises(ElexonAPIError):
        await ElexonClient().fetch_imbalance_price()
