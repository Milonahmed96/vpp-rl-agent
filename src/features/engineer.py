"""Feature engineering for the VPP RL agent.

The :class:`FeatureEngineer` turns a merged NESO + Elexon half-hourly time
series into a normalised feature matrix suitable for the Gymnasium environment
and PPO policy.

Reconciling the 14-feature requirement
--------------------------------------
The project brief lists twelve named features but requires an observation
feature vector of shape ``(14,)`` (the environment then appends fleet SoC for a
15-dim observation). The two remaining slots are filled with the **normalised
price levels** ``system_buy_price`` and ``system_sell_price``. The listed
features capture price *dynamics* (momentum, z-score); the raw levels give the
policy the absolute price context it also needs. This keeps the contract at
exactly 14 features while remaining well motivated.

Input schema (canonical, lower-case columns)
---------------------------------------------
``settlement_date`` (str/date), ``settlement_period`` (int, 1-48),
``frequency`` (Hz), ``system_buy_price``, ``system_sell_price``,
``net_imbalance_volume``, ``wind_generation``, ``solar_generation`` and ``nd``
(national demand). ``hour`` is optional; when absent it is derived from the
settlement period.

Time-series integrity
---------------------
Rows are sorted chronologically by ``(settlement_date, settlement_period)``
before any rolling computation. Data is **never shuffled**.
"""

from __future__ import annotations

import os
from typing import Final

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_SCALER_PATH: Final[str] = "data/processed/scaler.pkl"

#: The 14 ordered feature columns that form the observation feature vector.
FEATURE_NAMES: Final[list[str]] = [
    "freq_mean",
    "freq_std",
    "delta_f",
    "price_momentum_5",
    "price_momentum_15",
    "price_zscore",
    "wind_solar_ratio",
    "renewable_momentum",
    "hour_sin",
    "hour_cos",
    "period_sin",
    "period_cos",
    "system_buy_price",
    "system_sell_price",
]

#: Raw (un-normalised) columns carried alongside features for the environment's
#: reward calculation.
AUX_COLUMNS: Final[list[str]] = [
    "raw_delta_f",
    "raw_system_buy_price",
    "raw_system_sell_price",
    "settlement_period",
]

_ZSCORE_WINDOW: Final[int] = 48


class FeatureEngineer:
    """Build, fit and apply the normalised feature pipeline.

    A :class:`~sklearn.preprocessing.MinMaxScaler` is fitted on a training
    window and persisted to disk so that inference (the live API loop) applies
    exactly the same transform.
    """

    def __init__(self, scaler_path: str = DEFAULT_SCALER_PATH) -> None:
        """Initialise the engineer.

        Args:
            scaler_path: Filesystem path where the fitted scaler is stored.
        """
        self._scaler_path = scaler_path
        self._scaler: MinMaxScaler | None = None

    # ------------------------------------------------------------------ #
    # Raw feature construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_raw_features(merged: pd.DataFrame) -> pd.DataFrame:
        """Compute raw (un-normalised) features from merged source data.

        Args:
            merged: Merged NESO + Elexon half-hourly data (see module docstring
                for the expected schema).

        Returns:
            A DataFrame containing the 14 feature columns plus the auxiliary raw
            columns, with all rows that contain NaN (from rolling warm-up)
            dropped.

        Raises:
            ValueError: If required input columns are missing.
        """
        required = {
            "settlement_date",
            "settlement_period",
            "frequency",
            "system_buy_price",
            "system_sell_price",
            "wind_generation",
            "solar_generation",
            "nd",
        }
        missing = required - set(merged.columns)
        if missing:
            raise ValueError(
                f"Merged frame missing required columns: {sorted(missing)}"
            )

        frame = merged.sort_values(
            ["settlement_date", "settlement_period"]
        ).reset_index(drop=True)

        if "hour" in frame.columns:
            hour = frame["hour"].astype(float)
        else:
            # Each settlement period is 30 min; period 1 starts at 00:00.
            hour = (frame["settlement_period"].astype(float) - 1.0) // 2.0

        out = pd.DataFrame(index=frame.index)

        freq = frame["frequency"].astype(float)
        out["freq_mean"] = freq.rolling(5).mean()
        out["freq_std"] = freq.rolling(5).std(ddof=0)
        out["delta_f"] = out["freq_mean"] - 50.0

        sbp = frame["system_buy_price"].astype(float)
        out["price_momentum_5"] = sbp.rolling(5).mean()
        out["price_momentum_15"] = sbp.rolling(15).mean()
        roll = sbp.rolling(_ZSCORE_WINDOW)
        roll_std = roll.std(ddof=0)
        out["price_zscore"] = (sbp - roll.mean()) / roll_std.replace(0.0, np.nan)

        wind_solar_ratio = (
            frame["wind_generation"].astype(float)
            + frame["solar_generation"].astype(float)
        ) / frame["nd"].astype(float).replace(0.0, np.nan)
        out["wind_solar_ratio"] = wind_solar_ratio
        out["renewable_momentum"] = wind_solar_ratio.diff(3)

        out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        period = frame["settlement_period"].astype(float)
        out["period_sin"] = np.sin(2.0 * np.pi * period / 48.0)
        out["period_cos"] = np.cos(2.0 * np.pi * period / 48.0)

        out["system_buy_price"] = sbp
        out["system_sell_price"] = frame["system_sell_price"].astype(float)

        # Auxiliary raw columns for the environment's reward computation.
        out["raw_delta_f"] = out["delta_f"]
        out["raw_system_buy_price"] = sbp
        out["raw_system_sell_price"] = frame["system_sell_price"].astype(float)
        out["settlement_period"] = frame["settlement_period"].astype(int)

        before = len(out)
        out = out.dropna().reset_index(drop=True)
        logger.info(
            "Built raw features: %d rows (dropped %d warm-up/NaN rows)",
            len(out),
            before - len(out),
        )
        return out

    # ------------------------------------------------------------------ #
    # Scaler fit / transform
    # ------------------------------------------------------------------ #
    def fit(self, merged: pd.DataFrame) -> "FeatureEngineer":
        """Fit the MinMax scaler on the feature columns and persist it.

        Args:
            merged: Training-window merged source data.

        Returns:
            ``self`` for chaining.
        """
        raw = self.build_raw_features(merged)
        scaler = MinMaxScaler()
        scaler.fit(raw[FEATURE_NAMES].to_numpy(dtype=np.float64))
        self._scaler = scaler

        os.makedirs(os.path.dirname(self._scaler_path) or ".", exist_ok=True)
        joblib.dump(scaler, self._scaler_path)
        logger.info("Fitted and saved MinMaxScaler to %s", self._scaler_path)
        return self

    def _ensure_scaler(self) -> MinMaxScaler:
        """Return the in-memory scaler, loading it from disk if needed."""
        if self._scaler is None:
            if not os.path.exists(self._scaler_path):
                raise FileNotFoundError(
                    f"Scaler not found at {self._scaler_path}; call fit() first."
                )
            self._scaler = joblib.load(self._scaler_path)
            logger.info("Loaded MinMaxScaler from %s", self._scaler_path)
        return self._scaler

    def transform(self, merged: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted scaler, returning normalised features plus aux cols.

        Args:
            merged: Merged source data to transform.

        Returns:
            DataFrame with the 14 normalised :data:`FEATURE_NAMES` columns and
            the :data:`AUX_COLUMNS` raw columns.
        """
        scaler = self._ensure_scaler()
        raw = self.build_raw_features(merged)
        scaled = scaler.transform(raw[FEATURE_NAMES].to_numpy(dtype=np.float64))
        result = pd.DataFrame(scaled, columns=FEATURE_NAMES, index=raw.index)
        for col in AUX_COLUMNS:
            result[col] = raw[col].to_numpy()
        return result

    def fit_transform(self, merged: pd.DataFrame) -> pd.DataFrame:
        """Convenience: :meth:`fit` then :meth:`transform` on the same data."""
        self.fit(merged)
        return self.transform(merged)

    def latest_vector(self, merged: pd.DataFrame) -> np.ndarray:
        """Return the most recent normalised feature vector of shape ``(14,)``.

        Args:
            merged: Merged source data (the last row is the most recent period).

        Returns:
            A ``float32`` numpy array of shape ``(14,)`` with no NaN values.
        """
        transformed = self.transform(merged)
        vector = transformed[FEATURE_NAMES].iloc[-1].to_numpy(dtype=np.float32)
        return vector
