"""Async client for the Elexon BMRS Insights API (v1).

The Elexon BMRS API is open and requires no API key. Two endpoints are used:

* **System prices** - half-hourly imbalance prices (buy/sell) and the net
  imbalance volume for a given settlement date.
* **FUELHH** - half-hourly generation broken down by fuel type.

IMPORTANT - data lag caveat:
    Settlement / imbalance data is published roughly **15-30 minutes after the
    settlement period ends**. The agent therefore always acts on the most
    recently *settled* half-hour, never on the live instant. This lag is a
    fundamental property of the data source and is surfaced to callers via the
    returned ``settlementPeriod``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/"

_REQUEST_TIMEOUT_SECONDS = 10.0
_MAX_RETRIES = 3


class ElexonAPIError(Exception):
    """Raised when the Elexon API returns an error or unexpected payload."""


class ElexonClient:
    """Asynchronous client for Elexon BMRS imbalance and generation data.

    Transient failures are retried with exponential backoff and every request
    carries an explicit 10-second timeout.
    """

    def __init__(self, base_url: str = BASE_URL) -> None:
        """Initialise the client.

        Args:
            base_url: Root URL of the BMRS v1 API.
        """
        self._base_url = base_url.rstrip("/") + "/"

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """Perform a GET request and return the decoded JSON payload.

        Args:
            path: Path relative to the API base URL (no leading slash).
            params: Query-string parameters.

        Returns:
            The decoded JSON object.

        Raises:
            ElexonAPIError: On non-200 responses or after retries are exhausted.
        """
        url = self._base_url + path.lstrip("/")

        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=_REQUEST_TIMEOUT_SECONDS
                ) as client:
                    response = await client.get(url, params=params)
                if response.status_code != 200:
                    raise ElexonAPIError(
                        f"Elexon API returned HTTP {response.status_code} for {path}"
                    )
                return response.json()
            except (httpx.HTTPError, ElexonAPIError, ValueError) as exc:
                last_exc = exc
                backoff = 2.0 ** (attempt - 1)
                logger.warning(
                    "Elexon request attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                    backoff,
                )
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(backoff)

        raise ElexonAPIError(
            f"Elexon request failed after {_MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc

    async def fetch_imbalance_price(self) -> dict[str, float | int]:
        """Fetch the most recent settled imbalance price for today.

        Calls ``/balancing/settlement/system-prices/{settlementDate}`` for the
        current UTC date and returns the latest available settlement period.

        NOTE: imbalance data lags ~15-30 minutes behind real time, so the
        "most recent" period returned is the most recently *settled* one.

        Returns:
            Mapping with keys ``settlementPeriod`` (int), ``systemBuyPrice``,
            ``systemSellPrice`` and ``netImbalanceVolume`` (floats).

        Raises:
            ElexonAPIError: If the request fails or no records are returned.
        """
        settlement_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        payload = await self._get(
            f"balancing/settlement/system-prices/{settlement_date}",
            params={"format": "json"},
        )
        records = payload.get("data", [])
        if not records:
            raise ElexonAPIError(
                f"Elexon returned no system-price records for {settlement_date}"
            )

        frame = pd.DataFrame(records)
        if "settlementPeriod" not in frame.columns:
            raise ElexonAPIError("Elexon system-price payload missing settlementPeriod")

        # Most recently settled period == highest settlementPeriod available.
        latest = frame.sort_values("settlementPeriod").iloc[-1]
        result = {
            "settlementPeriod": int(latest["settlementPeriod"]),
            "systemBuyPrice": float(latest.get("systemBuyPrice", float("nan"))),
            "systemSellPrice": float(latest.get("systemSellPrice", float("nan"))),
            "netImbalanceVolume": float(latest.get("netImbalanceVolume", float("nan"))),
        }
        logger.info(
            "Fetched Elexon imbalance price for %s period %d (SBP=%.2f, SSP=%.2f)",
            settlement_date,
            result["settlementPeriod"],
            result["systemBuyPrice"],
            result["systemSellPrice"],
        )
        return result

    async def fetch_generation_mix(self) -> pd.DataFrame:
        """Fetch the half-hourly generation mix for the last 24 hours.

        Calls the ``FUELHH`` dataset endpoint with a 24-hour ``from``/``to``
        window expressed in ISO 8601.

        Returns:
            A :class:`pandas.DataFrame` of the returned FUELHH records (one row
            per fuel type per settlement period).

        Raises:
            ElexonAPIError: If the request fails or returns no records.
        """
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=24)
        params = {
            "from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "format": "json",
        }
        payload = await self._get("datasets/FUELHH", params=params)
        records = payload.get("data", [])
        if not records:
            raise ElexonAPIError("Elexon returned no FUELHH generation records")

        frame = pd.DataFrame(records)
        logger.info("Fetched %d FUELHH generation-mix records", len(frame))
        return frame.reset_index(drop=True)
