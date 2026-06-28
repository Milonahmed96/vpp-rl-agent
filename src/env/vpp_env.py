"""Gymnasium environment simulating a Virtual Power Plant of home EV batteries.

The environment models an aggregated fleet of 5,000 EV batteries that the agent
can collectively **hold**, **charge** (draw from the grid) or **discharge**
(inject to the grid) once per half-hourly settlement period. The reward trades
off arbitrage profit, a frequency-stability service term and battery
degradation.

Sign conventions (and two deliberate corrections to the brief)
--------------------------------------------------------------
``action_mw`` is positive when **drawing from the grid** (charging) and negative
when **injecting to the grid** (discharging), per the brief.

Two formulas in the brief are internally inconsistent with that convention and
with the brief's own stated behaviour; they are corrected here and documented:

* **State of charge.** The brief gives ``delta_soc = -action_mw * ...`` which
  would make charging (``action_mw = +37``) *lower* SoC. Charging must *raise*
  SoC, so the implementation uses ``+action_mw``.
* **Frequency penalty.** The brief gives ``freq_penalty = delta_f * action_mw``
  but labels it "positive = agent worsened frequency, negative = agent
  stabilised" and requires "reward positive when discharging during a
  low-frequency period". Drawing power lowers grid frequency and injecting
  raises it, so *worsening* (e.g. injecting while frequency is already high)
  corresponds to ``-delta_f * action_mw > 0``. The implementation therefore uses
  ``freq_penalty = -delta_f * action_mw``, which satisfies both the stated label
  and the required behaviour.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.features.engineer import FEATURE_NAMES
from src.logging_config import get_logger

logger = get_logger(__name__)

# --- Fleet / battery constants -------------------------------------------- #
N_BATTERIES: int = 5000
BATTERY_CAPACITY_KWH: float = 75.0
MAX_CHARGE_RATE_KW: float = 7.4
MAX_DISCHARGE_RATE_KW: float = 5.0
FLEET_MAX_CHARGE_MW: float = 37.0
FLEET_MAX_DISCHARGE_MW: float = 25.0
MIN_SOC: float = 0.1
MAX_SOC: float = 0.9
CYCLE_DEGRADATION_COST: float = 0.003  # GBP per kWh of throughput

# Aggregate fleet energy capacity in MWh (5000 * 75 kWh / 1000).
FLEET_CAPACITY_MWH: float = N_BATTERIES * BATTERY_CAPACITY_KWH / 1000.0

# Action index -> signed fleet power in MW (positive draws, negative injects).
ACTION_TO_MW: dict[int, float] = {
    0: 0.0,
    1: FLEET_MAX_CHARGE_MW,
    2: -FLEET_MAX_DISCHARGE_MW,
}

# Default reward weights: profit, frequency penalty, degradation cost.
DEFAULT_WEIGHTS: tuple[float, float, float] = (1.0, 0.5, 0.3)

_HALF_HOUR_FRACTION: float = 0.5
_TERMINATION_BOUND_STEPS: int = 3
_REQUIRED_COLUMNS: tuple[str, ...] = (
    "raw_delta_f",
    "raw_system_buy_price",
    "raw_system_sell_price",
)


class VPPEnv(gym.Env):
    """A single-step-per-period VPP dispatch environment.

    Observation space is ``Box(shape=(15,))`` - the 14 normalised features plus
    the current fleet mean state of charge. Action space is ``Discrete(3)``:
    ``0`` hold, ``1`` charge, ``2`` discharge.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        mode: str = "train",
        train_frac: float = 0.8,
        max_episode_steps: int = 48,
        reward_weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    ) -> None:
        """Initialise the environment over a feature-engineered dataset.

        Args:
            data: Output of :class:`~src.features.engineer.FeatureEngineer`,
                containing the 14 :data:`FEATURE_NAMES` columns plus the raw
                auxiliary columns used for the reward.
            mode: ``"train"`` (random start within the first ``train_frac`` of
                the data) or ``"eval"`` (sequential start at the ``train_frac``
                boundary - the held-out tail).
            train_frac: Chronological train/eval split fraction. Data is never
                shuffled.
            max_episode_steps: Maximum number of steps before truncation.
            reward_weights: ``(w1, w2, w3)`` weights for profit, frequency
                penalty and degradation cost.

        Raises:
            ValueError: If required columns are missing or ``mode`` is invalid.
        """
        super().__init__()
        missing = [
            c for c in (*FEATURE_NAMES, *_REQUIRED_COLUMNS) if c not in data.columns
        ]
        if missing:
            raise ValueError(f"Environment data missing columns: {missing}")
        if mode not in ("train", "eval"):
            raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")

        self._data = data.reset_index(drop=True)
        self._features = self._data[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
        self._delta_f = self._data["raw_delta_f"].to_numpy(dtype=np.float64)
        self._sbp = self._data["raw_system_buy_price"].to_numpy(dtype=np.float64)
        self._ssp = self._data["raw_system_sell_price"].to_numpy(dtype=np.float64)

        self._mode = mode
        self._n = len(self._data)
        self._split_index = int(self._n * train_frac)
        self._max_episode_steps = max_episode_steps
        self._w1, self._w2, self._w3 = reward_weights

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self.current_soc: float = 0.5
        self._index: int = 0
        self._steps_taken: int = 0
        self._bound_streak: int = 0

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the episode.

        In ``train`` mode the start index is sampled uniformly from the first
        ``train_frac`` of the data; in ``eval`` mode it begins at the split
        boundary and proceeds sequentially. ``current_soc`` is initialised to
        ``0.5``.

        Returns:
            A tuple ``(observation, info)``.
        """
        super().reset(seed=seed)
        if self._mode == "train":
            high = max(1, self._split_index - self._max_episode_steps)
            self._index = int(self.np_random.integers(0, high))
        else:
            self._index = self._split_index

        self.current_soc = 0.5
        self._steps_taken = 0
        self._bound_streak = 0
        return self._observation(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the environment by one settlement period.

        Args:
            action: ``0`` hold, ``1`` charge, ``2`` discharge.

        Returns:
            ``(observation, reward, terminated, truncated, info)``.
        """
        action = int(action)
        action_mw = ACTION_TO_MW[action]
        idx = self._index

        delta_f = float(self._delta_f[idx])
        sbp = float(self._sbp[idx])
        ssp = float(self._ssp[idx])

        # --- Profit (GBP, scaled per brief by 0.5h and /1000) ------------- #
        # Charging (action_mw > 0) buys energy at the system sell price (SSP);
        # discharging (action_mw < 0) sells energy at the system buy price (SBP).
        price = ssp if action_mw > 0 else sbp
        profit = -action_mw * price * _HALF_HOUR_FRACTION / 1000.0

        # --- Frequency stability term ------------------------------------- #
        # See module docstring: positive => agent worsened frequency.
        freq_penalty = -delta_f * action_mw

        # --- Degradation cost (GBP) --------------------------------------- #
        energy_throughput_kwh = abs(action_mw) * 1000.0 * _HALF_HOUR_FRACTION
        degradation_cost = energy_throughput_kwh * CYCLE_DEGRADATION_COST

        reward = (
            self._w1 * profit - self._w2 * freq_penalty - self._w3 * degradation_cost
        )

        # --- State-of-charge update --------------------------------------- #
        # Charging (action_mw > 0) raises SoC; discharging lowers it.
        delta_soc = action_mw * _HALF_HOUR_FRACTION / FLEET_CAPACITY_MWH
        self.current_soc = float(
            np.clip(self.current_soc + delta_soc, MIN_SOC, MAX_SOC)
        )

        # --- Termination / truncation ------------------------------------- #
        at_bound = self.current_soc in (MIN_SOC, MAX_SOC)
        self._bound_streak = self._bound_streak + 1 if at_bound else 0
        terminated = self._bound_streak >= _TERMINATION_BOUND_STEPS

        self._steps_taken += 1
        self._index += 1
        truncated = (
            self._steps_taken >= self._max_episode_steps or self._index >= self._n
        )

        info = {
            "action_mw": action_mw,
            "profit": profit,
            "freq_penalty": freq_penalty,
            "degradation_cost": degradation_cost,
            "current_soc": self.current_soc,
            "settlement_period": (
                int(self._data["settlement_period"].iloc[min(idx, self._n - 1)])
                if "settlement_period" in self._data.columns
                else -1
            ),
            "stabilising": freq_penalty < 0.0,
        }

        observation = (
            self._observation() if not truncated else self._observation(clamp=True)
        )
        return observation, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _observation(self, clamp: bool = False) -> np.ndarray:
        """Build the 15-dim observation for the current index.

        Args:
            clamp: When True, clamp the index to the last valid row (used after
                the final step to avoid an out-of-range read).

        Returns:
            ``float32`` array of shape ``(15,)``.
        """
        idx = min(self._index, self._n - 1) if clamp else self._index
        idx = min(idx, self._n - 1)
        feature_row = self._features[idx]
        return np.concatenate(
            [feature_row, np.array([self.current_soc], dtype=np.float32)]
        ).astype(np.float32)
