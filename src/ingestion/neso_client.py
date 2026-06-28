"""Async client for the NESO (National Energy System Operator) CKAN API.

The NESO Data Portal exposes datasets through a CKAN ``datastore_search``
endpoint. Two resources are consumed here:

* **System Frequency** (``f93d1835-75bc-43e5-84ad-fba0a2ad8ba0``) - one-second
  frequency readings in Hz.
* **Historic Demand / Generation**
  (``177f6fa4-ae49-4182-81ea-0c6b35f26ca7``) - half-hourly demand and
  generation figures.

IMPORTANT - data freshness caveat:
    The NESO "System Frequency" resource is **historic monthly CSV data**, not
    a live real-time stream. The most recent record available is therefore not
    "now"; it is the latest published month. Callers must treat the returned
    statistics as representative historic samples, not live telemetry.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)

BASE_URL = "https://api.neso.energy/api/3/action/"
FREQUENCY_RESOURCE_ID = "f93d1835-75bc-43e5-84ad-fba0a2ad8ba0"
GENERATION_RESOURCE_ID = "177f6fa4-ae49-4182-81ea-0c6b35f26ca7"

# 1-second frequency records over a 5-minute window.
FREQUENCY_WINDOW_RECORDS = 300
# 48 half-hourly settlement periods == one day of generation data.
GENERATION_WINDOW_RECORDS = 48

_REQUEST_TIMEOUT_SECONDS = 10.0
_MAX_RETRIES = 3
_MIN_SECONDS_BETWEEN_REQUESTS = 30.0


class NESOAPIError(Exception):
    """Raised when the NESO API returns an error or unexpected payload."""


class NESOClient:
    """Asynchronous client for NESO CKAN datastore resources.

    The client enforces a minimum spacing of one request per 30 seconds and
    retries transient failures with exponential backoff. All network calls use
    an explicit 10-second timeout.
    """

    def __init__(self, base_url: str = BASE_URL) -> None:
        """Initialise the client.

        Args:
            base_url: Root URL of the CKAN ``action`` API.
        """
        self._base_url = base_url.rstrip("/") + "/"
        self._last_request_monotonic: float | None = None
        self._rate_limit_lock = asyncio.Lock()

    async def _respect_rate_limit(self) -> None:
        """Sleep if necessary to honour the 1-request-per-30s rate limit."""
        async with self._rate_limit_lock:
            if self._last_request_monotonic is not None:
                elapsed = time.monotonic() - self._last_request_monotonic
                wait = _MIN_SECONDS_BETWEEN_REQUESTS - elapsed
                if wait > 0:
                    logger.debug("Rate limiting NESO request: sleeping %.1fs", wait)
                    await asyncio.sleep(wait)
            self._last_request_monotonic = time.monotonic()

    async def _datastore_search(
        self, resource_id: str, limit: int, sort: str
    ) -> list[dict[str, Any]]:
        """Call ``datastore_search`` and return the list of records.

        Args:
            resource_id: CKAN resource identifier.
            limit: Maximum number of records to return.
            sort: CKAN sort expression (e.g. ``"dtm desc"``).

        Returns:
            The list of record dictionaries from the response.

        Raises:
            NESOAPIError: On non-200 responses, exhausted retries, or a payload
                that does not indicate success.
        """
        url = self._base_url + "datastore_search"
        params = {"resource_id": resource_id, "limit": limit, "sort": sort}

        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            await self._respect_rate_limit()
            try:
                async with httpx.AsyncClient(
                    timeout=_REQUEST_TIMEOUT_SECONDS
                ) as client:
                    response = await client.get(url, params=params)
                if response.status_code != 200:
                    raise NESOAPIError(
                        f"NESO API returned HTTP {response.status_code} "
                        f"for resource {resource_id}"
                    )
                payload = response.json()
                if not payload.get("success", False):
                    raise NESOAPIError(
                        f"NESO API reported failure for resource {resource_id}: "
                        f"{payload.get('error')}"
                    )
                return payload["result"]["records"]
            except (httpx.HTTPError, NESOAPIError, KeyError, ValueError) as exc:
                last_exc = exc
                backoff = 2.0 ** (attempt - 1)
                logger.warning(
                    "NESO request attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                    backoff,
                )
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(backoff)

        raise NESOAPIError(
            f"NESO request failed after {_MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc

    async def fetch_frequency(self) -> dict[str, float]:
        """Fetch recent system-frequency records and summarise them.

        Retrieves the most recent ``FREQUENCY_WINDOW_RECORDS`` one-second
        readings (a ~5-minute window) and returns their mean and standard
        deviation.

        NOTE: this resource is historic monthly CSV data, not a live stream.

        Returns:
            Mapping with keys ``freq_mean`` and ``freq_std`` (both Hz).

        Raises:
            NESOAPIError: If the request fails or no usable records are found.
        """
        records = await self._datastore_search(
            FREQUENCY_RESOURCE_ID,
            limit=FREQUENCY_WINDOW_RECORDS,
            sort="dtm desc",
        )
        frame = pd.DataFrame(records)
        if frame.empty or "f" not in frame.columns:
            raise NESOAPIError("NESO frequency response contained no 'f' field")

        freq = pd.to_numeric(frame["f"], errors="coerce").dropna()
        if freq.empty:
            raise NESOAPIError("NESO frequency response had no numeric values")

        result = {
            "freq_mean": float(freq.mean()),
            "freq_std": float(freq.std(ddof=0)),
        }
        logger.info(
            "Fetched %d frequency records (mean=%.4f Hz, std=%.4f Hz)",
            len(freq),
            result["freq_mean"],
            result["freq_std"],
        )
        return result

    async def fetch_generation(self) -> pd.DataFrame:
        """Fetch the most recent half-hourly demand/generation records.

        Returns:
            A :class:`pandas.DataFrame` with columns ``SETTLEMENT_DATE``,
            ``SETTLEMENT_PERIOD``, ``WIND_GENERATION``, ``SOLAR_GENERATION`` and
            ``ND`` (national demand), one row per settlement period.

        Raises:
            NESOAPIError: If the request fails or returns no records.
        """
        records = await self._datastore_search(
            GENERATION_RESOURCE_ID,
            limit=GENERATION_WINDOW_RECORDS,
            sort="SETTLEMENT_DATE desc, SETTLEMENT_PERIOD desc",
        )
        frame = pd.DataFrame(records)
        if frame.empty:
            raise NESOAPIError("NESO generation response contained no records")

        expected = [
            "SETTLEMENT_DATE",
            "SETTLEMENT_PERIOD",
            "WIND_GENERATION",
            "SOLAR_GENERATION",
            "ND",
        ]
        # Keep only expected columns that are present; tolerate missing extras.
        available = [col for col in expected if col in frame.columns]
        frame = frame[available].copy()
        for col in available:
            if col not in ("SETTLEMENT_DATE",):
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

        logger.info("Fetched %d generation records from NESO", len(frame))
        return frame.reset_index(drop=True)
