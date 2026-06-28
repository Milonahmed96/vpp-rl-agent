"""PPO training and backtesting for the VPP RL agent.

Run with::

    python -m src.agent.train                 # full 500k-step run (per brief)
    python -m src.agent.train --timesteps 5000  # quick smoke run

Time-series discipline
----------------------
The dataset is split **chronologically**: the first 80% trains the agent and
fits the feature scaler; the final 20% is held out for evaluation and the
backtest. The data is **never shuffled**.

Device
------
Training runs on ``device="cpu"``. MPS (Apple-Silicon GPU) is intentionally
avoided: it is unstable with Stable-Baselines3.
"""

from __future__ import annotations

import argparse
import os
from typing import Final

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.env.vpp_env import VPPEnv
from src.features.engineer import FeatureEngineer
from src.ingestion.synthetic import generate_synthetic_merged
from src.logging_config import get_logger

logger = get_logger(__name__)

TRAIN_FRAC: Final[float] = 0.8
CHECKPOINT_DIR: Final[str] = "models/checkpoints"
# EvalCallback saves "<best_model_save_path>/best_model.zip"; pointing it at the
# checkpoint dir yields models/checkpoints/best_model.zip, which is what the API
# loads via MODEL_PATH ("models/checkpoints/best_model").
BEST_MODEL_PATH: Final[str] = "models/checkpoints/best_model"
TENSORBOARD_DIR: Final[str] = "./logs/tensorboard/"
BACKTEST_PATH: Final[str] = "data/processed/backtest_results.csv"
MERGED_DATA_PATH: Final[str] = "data/processed/merged.csv"
DEFAULT_TIMESTEPS: Final[int] = 500_000


def load_merged_data(path: str = MERGED_DATA_PATH) -> pd.DataFrame:
    """Load merged source data, falling back to synthetic dev data.

    Args:
        path: CSV path for a previously-saved merged dataset.

    Returns:
        The merged half-hourly DataFrame.
    """
    if os.path.exists(path):
        logger.info("Loading merged data from %s", path)
        return pd.read_csv(path)
    logger.warning(
        "No merged data at %s; generating synthetic development data instead.", path
    )
    frame = generate_synthetic_merged()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def build_feature_frame(merged: pd.DataFrame) -> pd.DataFrame:
    """Fit the scaler on the training window and transform the full series.

    The scaler is fitted on the first ``TRAIN_FRAC`` of the data only, so no
    information from the held-out tail leaks into normalisation.

    Args:
        merged: Merged source data.

    Returns:
        Feature-engineered DataFrame (features + raw aux columns).
    """
    split = int(len(merged) * TRAIN_FRAC)
    engineer = FeatureEngineer()
    engineer.fit(merged.iloc[:split])
    return engineer.transform(merged)


def make_model(train_env: Monitor) -> PPO:
    """Construct the PPO model with the brief's hyperparameters."""
    return PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device="cpu",  # MPS is unstable with SB3 on Apple Silicon.
        tensorboard_log=TENSORBOARD_DIR,
    )


def backtest(model: PPO, features: pd.DataFrame) -> dict[str, float]:
    """Backtest a trained model on the held-out final 20% of the data.

    The held-out tail is swept in consecutive episodes so every settlement
    period is evaluated exactly once.

    Args:
        model: Trained PPO model.
        features: Full feature-engineered dataset.

    Returns:
        A dictionary of backtest metrics.
    """
    env = VPPEnv(features, mode="eval", train_frac=TRAIN_FRAC)
    split = env._split_index  # noqa: SLF001 - intentional internal use
    n = env._n  # noqa: SLF001
    horizon = env._max_episode_steps  # noqa: SLF001

    episode_rewards: list[float] = []
    total_profit = 0.0
    total_degradation = 0.0
    stabilising = 0
    destabilising = 0

    for start in range(split, n - 1, horizon):
        obs, _ = env.reset(options={"start_index": start})
        done = False
        episode_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += reward
            total_profit += info["profit"]
            total_degradation += info["degradation_cost"]
            if info["freq_penalty"] < 0:
                stabilising += 1
            elif info["freq_penalty"] > 0:
                destabilising += 1
            done = terminated or truncated
        episode_rewards.append(episode_reward)

    rewards = np.asarray(episode_rewards, dtype=np.float64)
    mean_reward = float(rewards.mean()) if rewards.size else 0.0
    std_reward = float(rewards.std()) if rewards.size else 0.0
    sharpe = mean_reward / std_reward if std_reward > 1e-9 else 0.0

    metrics = {
        "mean_episode_reward": mean_reward,
        "std_episode_reward": std_reward,
        "total_profit_gbp": total_profit,
        "total_degradation_cost_gbp": total_degradation,
        "stabilising_actions": float(stabilising),
        "destabilising_actions": float(destabilising),
        "sharpe_equivalent": sharpe,
        "n_episodes": float(len(episode_rewards)),
    }
    logger.info("Backtest metrics: %s", metrics)
    return metrics


def train(
    timesteps: int = DEFAULT_TIMESTEPS, data_path: str = MERGED_DATA_PATH
) -> dict[str, float]:
    """Train PPO, evaluate, and write backtest results.

    Args:
        timesteps: Total training timesteps.
        data_path: Path to the merged dataset CSV.

    Returns:
        The backtest metrics dictionary.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(BACKTEST_PATH) or ".", exist_ok=True)

    merged = load_merged_data(data_path)
    features = build_feature_frame(merged)
    logger.info("Feature frame: %d rows", len(features))

    train_env = Monitor(VPPEnv(features, mode="train", train_frac=TRAIN_FRAC))
    eval_env = Monitor(VPPEnv(features, mode="eval", train_frac=TRAIN_FRAC))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=CHECKPOINT_DIR,
        log_path="./logs/eval/",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000, save_path=CHECKPOINT_DIR, name_prefix="ppo_vpp"
    )

    model = make_model(train_env)
    logger.info("Starting PPO training for %d timesteps (device=cpu)", timesteps)
    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=False,
    )

    # Ensure a loadable model exists even if EvalCallback never improved (e.g.
    # a short run where eval_freq is never reached).
    final_path = os.path.join(CHECKPOINT_DIR, "final_model")
    model.save(final_path)
    if not os.path.exists(BEST_MODEL_PATH + ".zip"):
        model.save(BEST_MODEL_PATH)
        logger.info("EvalCallback produced no best model; saved final as best.")
    logger.info("Saved final model to %s", final_path)

    metrics = backtest(model, features)
    pd.DataFrame([metrics]).to_csv(BACKTEST_PATH, index=False)
    logger.info("Wrote backtest results to %s", BACKTEST_PATH)
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the VPP PPO agent.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=int(os.getenv("VPP_TIMESTEPS", DEFAULT_TIMESTEPS)),
        help="Total training timesteps (default: 500000).",
    )
    parser.add_argument(
        "--data-path", type=str, default=MERGED_DATA_PATH, help="Merged data CSV path."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(timesteps=args.timesteps, data_path=args.data_path)
