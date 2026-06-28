"""Tests for the VPP Gymnasium environment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.env.vpp_env import MAX_SOC, MIN_SOC, VPPEnv
from src.features.engineer import FEATURE_NAMES


def _make_env_data(
    n: int = 300,
    delta_f: float | np.ndarray = 0.0,
    sbp: float = 60.0,
    ssp: float = 55.0,
) -> pd.DataFrame:
    """Build a synthetic feature-engineered dataset for the environment."""
    rng = np.random.default_rng(0)
    data = {name: rng.random(n).astype(np.float32) for name in FEATURE_NAMES}
    delta_f_arr = np.full(n, delta_f) if np.isscalar(delta_f) else np.asarray(delta_f)
    data["raw_delta_f"] = delta_f_arr
    data["raw_system_buy_price"] = np.full(n, sbp)
    data["raw_system_sell_price"] = np.full(n, ssp)
    data["settlement_period"] = ((np.arange(n) % 48) + 1).astype(int)
    return pd.DataFrame(data)


def test_step_returns_correct_tuple_shape():
    env = VPPEnv(_make_env_data())
    obs, info = env.reset(seed=0)
    assert obs.shape == (15,)
    assert isinstance(info, dict)

    obs, reward, terminated, truncated, info = env.step(1)
    assert obs.shape == (15,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "profit" in info and "current_soc" in info


def test_soc_stays_in_bounds_over_100_random_steps():
    env = VPPEnv(_make_env_data(n=400))
    env.reset(seed=1)
    for _ in range(100):
        action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(action)
        assert MIN_SOC - 1e-9 <= info["current_soc"] <= MAX_SOC + 1e-9
        if terminated or truncated:
            env.reset()


def test_discharge_reward_positive_in_low_frequency():
    # Low frequency: delta_f = -1.0 (49 Hz). Discharging (action 2) injects
    # power and stabilises the grid -> reward should be positive.
    env = VPPEnv(_make_env_data(delta_f=-1.0), mode="eval")
    env.reset()
    _, reward, _, _, info = env.step(2)
    assert info["stabilising"] is True
    assert reward > 0.0


def test_discharge_reward_negative_in_high_frequency():
    # High frequency: delta_f = +1.0 (51 Hz). Discharging injects more power
    # into an already over-supplied grid -> destabilising -> reward negative.
    env = VPPEnv(_make_env_data(delta_f=1.0), mode="eval")
    env.reset()
    _, reward, _, _, info = env.step(2)
    assert info["stabilising"] is False
    assert reward < 0.0


def test_charging_increases_soc():
    env = VPPEnv(_make_env_data(), mode="eval")
    env.reset()
    start = env.current_soc
    env.step(1)  # charge
    assert env.current_soc > start


def test_eval_mode_starts_at_split_boundary():
    data = _make_env_data(n=100)
    env = VPPEnv(data, mode="eval", train_frac=0.8)
    env.reset()
    assert env._index == 80
