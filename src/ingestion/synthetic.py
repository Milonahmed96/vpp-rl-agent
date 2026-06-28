"""Synthetic merged-data generator (development fallback).

The real data sources cannot be bulk-downloaded in a reproducible offline build:
the NESO system-frequency resource is published as historic *monthly CSVs* and
the Elexon imbalance feed lags real time by 15-30 minutes. To let the full
pipeline (feature engineering -> environment -> PPO training -> backtest -> API)
run end-to-end deterministically, this module synthesises a merged half-hourly
dataset with the exact schema the :class:`~src.features.engineer.FeatureEngineer`
expects.

This is **development/test data only** - it is not a substitute for the real
feeds and must not be presented as real grid history.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)


def generate_synthetic_merged(
    n_days: int = 90, seed: int = 42, start: date = date(2024, 1, 1)
) -> pd.DataFrame:
    """Generate a synthetic merged NESO + Elexon half-hourly dataset.

    Args:
        n_days: Number of days to generate (48 settlement periods each).
        seed: RNG seed for reproducibility.
        start: First settlement date.

    Returns:
        A :class:`pandas.DataFrame` with the canonical FeatureEngineer input
        schema, ordered chronologically.
    """
    rng = np.random.default_rng(seed)
    n = n_days * 48
    periods = np.tile(np.arange(1, 49), n_days)
    day_offsets = np.repeat(np.arange(n_days), 48)
    dates = [(start + timedelta(days=int(d))).isoformat() for d in day_offsets]
    hour = (periods - 1) / 2.0

    # Diurnal demand curve (MW) with a morning and evening peak.
    demand = (
        28000
        + 6000 * np.sin((hour - 7) / 24.0 * 2 * np.pi)
        + 3000 * np.sin((hour - 18) / 12.0 * 2 * np.pi)
        + rng.normal(0, 400, n)
    )
    # Solar follows daylight; wind is broad and noisier.
    solar = np.clip(6000 * np.sin((hour - 6) / 12.0 * np.pi), 0, None)
    solar *= hour < 20
    wind = np.clip(
        7000 + 4000 * np.sin(day_offsets / 3.0) + rng.normal(0, 800, n), 0, None
    )

    # Imbalance prices loosely track residual demand (demand minus renewables).
    residual = demand - wind - solar
    base_price = 40 + 0.004 * (residual - residual.mean())
    system_buy_price = base_price + rng.normal(0, 6, n) + 8
    system_sell_price = base_price + rng.normal(0, 6, n) - 8
    net_imbalance_volume = rng.normal(0, 120, n) + 0.002 * (residual - residual.mean())

    # Frequency drifts around 50 Hz, dipping when residual demand is high.
    frequency = 50.0 - 0.0000015 * (residual - residual.mean()) + rng.normal(0, 0.03, n)

    frame = pd.DataFrame(
        {
            "settlement_date": dates,
            "settlement_period": periods.astype(int),
            "frequency": frequency,
            "system_buy_price": system_buy_price,
            "system_sell_price": system_sell_price,
            "net_imbalance_volume": net_imbalance_volume,
            "wind_generation": wind,
            "solar_generation": solar,
            "nd": demand,
        }
    )
    logger.info("Generated %d synthetic merged rows (%d days)", len(frame), n_days)
    return frame
