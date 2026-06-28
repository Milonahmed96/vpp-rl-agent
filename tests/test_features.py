"""Tests for the feature-engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import FEATURE_NAMES, FeatureEngineer


def _synthetic_merged(n_periods: int = 200, seed: int = 0) -> pd.DataFrame:
    """Build a deterministic synthetic merged NESO+Elexon frame."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_periods):
        day = i // 48
        period = (i % 48) + 1
        rows.append(
            {
                "settlement_date": f"2024-01-{day + 1:02d}",
                "settlement_period": period,
                "frequency": 50.0 + rng.normal(0.0, 0.05),
                "system_buy_price": 60.0 + 20.0 * np.sin(i / 7.0) + rng.normal(0, 3),
                "system_sell_price": 55.0 + 18.0 * np.sin(i / 7.0) + rng.normal(0, 3),
                "net_imbalance_volume": rng.normal(0.0, 50.0),
                "wind_generation": 8000 + 2000 * np.sin(i / 11.0),
                "solar_generation": max(0.0, 3000 * np.sin(i / 24.0)),
                "nd": 30000 + 5000 * np.sin(i / 24.0),
            }
        )
    return pd.DataFrame(rows)


def test_latest_vector_shape_and_no_nan(tmp_path):
    engineer = FeatureEngineer(scaler_path=str(tmp_path / "scaler.pkl"))
    merged = _synthetic_merged()
    vector = engineer.fit_transform(merged)  # fit + transform
    latest = engineer.latest_vector(merged)

    assert latest.shape == (14,)
    assert not np.isnan(latest).any()
    assert latest.dtype == np.float32
    # All 14 named feature columns present in transform output.
    assert set(FEATURE_NAMES).issubset(vector.columns)


def test_transform_values_normalised_in_unit_range(tmp_path):
    engineer = FeatureEngineer(scaler_path=str(tmp_path / "scaler.pkl"))
    merged = _synthetic_merged()
    transformed = engineer.fit_transform(merged)
    feats = transformed[FEATURE_NAMES].to_numpy()
    assert not np.isnan(feats).any()
    # MinMax scaler maps the training window into [0, 1].
    assert feats.min() >= -1e-9
    assert feats.max() <= 1.0 + 1e-9


def test_scaler_persisted_to_disk(tmp_path):
    scaler_path = tmp_path / "scaler.pkl"
    engineer = FeatureEngineer(scaler_path=str(scaler_path))
    engineer.fit(_synthetic_merged())
    assert scaler_path.exists()


def test_missing_columns_raise(tmp_path):
    engineer = FeatureEngineer(scaler_path=str(tmp_path / "scaler.pkl"))
    bad = pd.DataFrame({"settlement_period": [1, 2, 3]})
    with pytest.raises(ValueError):
        engineer.build_raw_features(bad)
