"""FastAPI service exposing the autonomous VPP control loop.

An APScheduler job runs the control loop every ``LOOP_INTERVAL_MINUTES``
(default 30). Each tick ingests the most recently settled grid conditions,
engineers features, asks the PPO policy for an action, applies it to the
in-memory fleet, and persists a snapshot and decision.

DATA-LAG NOTICE (known limitation)
----------------------------------
Both upstream feeds are delayed: NESO system-frequency data is published as
historic monthly CSVs, and Elexon imbalance prices lag 15-30 minutes behind the
settlement period. The agent therefore always acts on the **most recently
settled half-hour**, not on the live instant. This is inherent to the data
sources and is not a real-time control system.

Run with::

    uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from stable_baselines3 import PPO

from src.db.database import Database
from src.env.vpp_env import simulate_step
from src.features.engineer import FEATURE_NAMES, FeatureEngineer
from src.ingestion.elexon_client import ElexonClient
from src.ingestion.neso_client import NESOClient
from src.ingestion.synthetic import generate_synthetic_merged
from src.logging_config import get_logger

logger = get_logger(__name__)

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/checkpoints/best_model")
MERGED_SEED_PATH = "data/processed/merged.csv"
LOOP_INTERVAL_MINUTES = int(os.getenv("LOOP_INTERVAL_MINUTES", "30"))
_BUFFER_ROWS = 60  # enough history for the 48-period rolling features
ACTION_NAMES = {0: "hold", 1: "charge", 2: "discharge"}


class VPPController:
    """Owns the model, feature pipeline, fleet state and persistence.

    The control-loop logic lives in :meth:`tick`, which accepts injected
    ingestion clients so it can be exercised without real network access.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        db: Database | None = None,
        neso: NESOClient | None = None,
        elexon: ElexonClient | None = None,
        engineer: FeatureEngineer | None = None,
        seed_path: str = MERGED_SEED_PATH,
    ) -> None:
        """Initialise the controller and seed the rolling feature buffer.

        Args:
            model_path: Path to the trained PPO model (without ``.zip``).
            db: Database repository (a default one is created if omitted).
            neso: NESO client (defaults to a real client).
            elexon: Elexon client (defaults to a real client).
            engineer: Feature engineer (defaults to a new one).
            seed_path: CSV used to seed the rolling buffer with recent history.
        """
        self.db = db or Database()
        self.neso = neso or NESOClient()
        self.elexon = elexon or ElexonClient()
        self.engineer = engineer or FeatureEngineer()

        self._model: PPO | None = None
        self._model_path = model_path

        # Seed the rolling buffer so rolling features are computable immediately.
        self._buffer = self._load_seed(seed_path)
        # Ensure a fitted scaler exists (fit on seed data if training never ran).
        self._ensure_scaler()

        # In-memory fleet / episode state.
        self.current_soc: float = 0.5
        self.last_action: int | None = None
        self.last_reward: float | None = None
        self.last_price: float | None = None
        self.last_freq: float | None = None
        self.last_updated: str | None = None
        self.episodes_run: int = 0

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_seed(seed_path: str) -> pd.DataFrame:
        """Load recent merged history, falling back to synthetic data."""
        if os.path.exists(seed_path):
            frame = pd.read_csv(seed_path)
        else:
            logger.warning("Seed data %s missing; using synthetic seed.", seed_path)
            frame = generate_synthetic_merged(n_days=3)
        return frame.tail(_BUFFER_ROWS).reset_index(drop=True)

    def _ensure_scaler(self) -> None:
        """Fit and persist a scaler from the seed buffer if none exists."""
        if not os.path.exists(self.engineer._scaler_path):  # noqa: SLF001
            logger.warning("No scaler found; fitting one on seed data.")
            self.engineer.fit(self._buffer)

    @property
    def model(self) -> PPO:
        """Lazily load and cache the PPO model."""
        if self._model is None:
            logger.info("Loading PPO model from %s", self._model_path)
            self._model = PPO.load(self._model_path, device="cpu")
        return self._model

    # ------------------------------------------------------------------ #
    # Control loop
    # ------------------------------------------------------------------ #
    async def _fetch_latest_row(self) -> dict[str, Any]:
        """Fetch the most recently settled merged grid row from both APIs.

        Returns:
            A dict in the canonical FeatureEngineer input schema.
        """
        frequency = await self.neso.fetch_frequency()
        generation = await self.neso.fetch_generation()
        price = await self.elexon.fetch_imbalance_price()

        gen_row = generation.iloc[0]  # newest record (sorted desc)
        return {
            "settlement_date": str(gen_row.get("SETTLEMENT_DATE", "")),
            "settlement_period": int(price["settlementPeriod"]),
            "frequency": float(frequency["freq_mean"]),
            "system_buy_price": float(price["systemBuyPrice"]),
            "system_sell_price": float(price["systemSellPrice"]),
            "net_imbalance_volume": float(price["netImbalanceVolume"]),
            "wind_generation": float(gen_row.get("WIND_GENERATION", 0.0)),
            "solar_generation": float(gen_row.get("SOLAR_GENERATION", 0.0)),
            "nd": float(gen_row.get("ND", 0.0)),
        }

    async def tick(self) -> dict[str, Any]:
        """Run one control-loop iteration.

        Steps: fetch -> feature-engineer -> predict -> apply -> persist.

        Returns:
            The decision dict that was persisted.
        """
        row = await self._fetch_latest_row()

        # Append to the rolling buffer, keeping a bounded history window.
        self._buffer = (
            pd.concat([self._buffer, pd.DataFrame([row])], ignore_index=True)
            .tail(_BUFFER_ROWS)
            .reset_index(drop=True)
        )

        features = self.engineer.transform(self._buffer)
        latest = features.iloc[-1]
        vector = latest[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
        observation = np.concatenate(
            [vector, np.array([self.current_soc], dtype=np.float32)]
        )

        action, _ = self.model.predict(observation, deterministic=True)
        action = int(action)

        outcome = simulate_step(
            action,
            delta_f=float(latest["raw_delta_f"]),
            sbp=float(latest["raw_system_buy_price"]),
            ssp=float(latest["raw_system_sell_price"]),
            current_soc=self.current_soc,
        )

        now = datetime.now(timezone.utc).isoformat()
        self.db.insert_snapshot(
            timestamp=now,
            freq_mean=row["frequency"],
            delta_f=float(latest["raw_delta_f"]),
            system_buy_price=row["system_buy_price"],
            system_sell_price=row["system_sell_price"],
            net_imbalance_volume=row["net_imbalance_volume"],
            wind_generation=row["wind_generation"],
            solar_generation=row["solar_generation"],
            national_demand=row["nd"],
        )
        decision = {
            "timestamp": now,
            "action": action,
            "action_mw": outcome["action_mw"],
            "current_soc": outcome["new_soc"],
            "reward": outcome["reward"],
            "profit": outcome["profit"],
            "freq_penalty": outcome["freq_penalty"],
            "degradation_cost": outcome["degradation_cost"],
            "settlement_period": int(row["settlement_period"]),
        }
        self.db.insert_decision(**decision)

        # Update in-memory fleet/episode state.
        self.current_soc = outcome["new_soc"]
        self.last_action = action
        self.last_reward = outcome["reward"]
        self.last_price = row["system_buy_price"]
        self.last_freq = row["frequency"]
        self.last_updated = now
        self.episodes_run += 1

        logger.info(
            "Tick: action=%s soc=%.3f reward=%.4f",
            ACTION_NAMES[action],
            self.current_soc,
            outcome["reward"],
        )
        return decision

    async def safe_tick(self) -> None:
        """Run :meth:`tick`, logging and swallowing errors so the loop survives."""
        try:
            await self.tick()
        except Exception:  # noqa: BLE001 - the scheduler must keep running.
            logger.exception("Control-loop tick failed; will retry next interval.")

    # ------------------------------------------------------------------ #
    # Read models for the API
    # ------------------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        """Return the current fleet/episode status."""
        return {
            "current_soc": self.current_soc,
            "last_action": (
                ACTION_NAMES[self.last_action] if self.last_action is not None else None
            ),
            "last_reward": self.last_reward,
            "last_price": self.last_price,
            "last_freq": self.last_freq,
            "last_updated": self.last_updated,
        }

    def metrics(self) -> dict[str, Any]:
        """Aggregate cumulative metrics across all persisted decisions."""
        decisions = self.db.get_recent_decisions(n=100_000)
        total_profit = sum(d["profit"] for d in decisions)
        total_degradation = sum(d["degradation_cost"] for d in decisions)
        stabilising = sum(1 for d in decisions if d["freq_penalty"] < 0)
        destabilising = sum(1 for d in decisions if d["freq_penalty"] > 0)
        acting = stabilising + destabilising
        pct = 100.0 * stabilising / acting if acting else 0.0
        return {
            "total_profit_gbp": total_profit,
            "total_degradation_cost_gbp": total_degradation,
            "net_profit_gbp": total_profit - total_degradation,
            "stabilising_action_pct": pct,
            "episodes_run": self.episodes_run,
        }

    def reset(self) -> dict[str, Any]:
        """Reset SoC to 0.5 and clear in-memory episode state."""
        self.current_soc = 0.5
        self.last_action = None
        self.last_reward = None
        self.episodes_run = 0
        logger.info("Episode state reset (SoC=0.5).")
        return {"status": "reset", "current_soc": self.current_soc}


# --------------------------------------------------------------------------- #
# FastAPI application
# --------------------------------------------------------------------------- #
controller: VPPController | None = None
scheduler: AsyncIOScheduler | None = None


def get_controller() -> VPPController:
    """Return the process-wide controller, creating it on first use."""
    global controller
    if controller is None:
        controller = VPPController()
    return controller


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start and stop the APScheduler control loop with the app."""
    global scheduler
    ctrl = get_controller()
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        ctrl.safe_tick,
        "interval",
        minutes=LOOP_INTERVAL_MINUTES,
        id="vpp_control_loop",
        next_run_time=datetime.now(timezone.utc),
    )
    scheduler.start()
    logger.info("Control loop scheduled every %d minutes.", LOOP_INTERVAL_MINUTES)
    try:
        yield
    finally:
        if scheduler is not None:
            scheduler.shutdown(wait=False)
            logger.info("Scheduler shut down.")


app = FastAPI(title="VPP RL Agent", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/status")
def status() -> dict[str, Any]:
    """Return the current fleet/episode status."""
    return get_controller().status()


@app.get("/decisions")
def decisions(limit: int = Query(default=50, ge=1, le=1000)) -> list[dict[str, Any]]:
    """Return the most recent agent decisions."""
    return get_controller().db.get_recent_decisions(n=limit)


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    """Return cumulative profit / degradation / stability metrics."""
    return get_controller().metrics()


@app.post("/reset")
def reset() -> dict[str, Any]:
    """Reset the fleet SoC and episode state."""
    return get_controller().reset()
