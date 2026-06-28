"""SQLAlchemy persistence layer for the VPP RL agent.

Two tables are maintained:

* ``grid_snapshots`` - the most recently settled grid conditions ingested from
  NESO and Elexon.
* ``agent_decisions`` - the agent's action and the reward decomposition for
  each control-loop tick.

The database is SQLite; its path comes from the ``DB_PATH`` environment
variable (default ``data/vpp.db``).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import (
    Float,
    Integer,
    String,
    create_engine,
    desc,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from src.logging_config import get_logger

logger = get_logger(__name__)

load_dotenv()

DEFAULT_DB_PATH = "data/vpp.db"


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class GridSnapshot(Base):
    """A snapshot of settled grid conditions at a point in time."""

    __tablename__ = "grid_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[str] = mapped_column(String, nullable=False)
    freq_mean: Mapped[float] = mapped_column(Float)
    delta_f: Mapped[float] = mapped_column(Float)
    system_buy_price: Mapped[float] = mapped_column(Float)
    system_sell_price: Mapped[float] = mapped_column(Float)
    net_imbalance_volume: Mapped[float] = mapped_column(Float)
    wind_generation: Mapped[float] = mapped_column(Float)
    solar_generation: Mapped[float] = mapped_column(Float)
    national_demand: Mapped[float] = mapped_column(Float)


class AgentDecision(Base):
    """A single agent action and its reward decomposition."""

    __tablename__ = "agent_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[str] = mapped_column(String, nullable=False)
    action: Mapped[int] = mapped_column(Integer)
    action_mw: Mapped[float] = mapped_column(Float)
    current_soc: Mapped[float] = mapped_column(Float)
    reward: Mapped[float] = mapped_column(Float)
    profit: Mapped[float] = mapped_column(Float)
    freq_penalty: Mapped[float] = mapped_column(Float)
    degradation_cost: Mapped[float] = mapped_column(Float)
    settlement_period: Mapped[int] = mapped_column(Integer)


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


class Database:
    """Thin repository wrapper around a SQLite SQLAlchemy engine."""

    def __init__(self, db_path: str | None = None) -> None:
        """Initialise the engine and create tables if needed.

        Args:
            db_path: SQLite file path. When ``None`` the ``DB_PATH`` environment
                variable is used, defaulting to ``data/vpp.db``.
        """
        self._db_path = db_path or os.getenv("DB_PATH", DEFAULT_DB_PATH)
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._engine = create_engine(f"sqlite:///{self._db_path}", future=True)
        Base.metadata.create_all(self._engine)
        logger.info("Initialised database at %s", self._db_path)

    def insert_snapshot(self, **fields: Any) -> int:
        """Insert a grid snapshot.

        Args:
            **fields: Column values for :class:`GridSnapshot`. ``timestamp``
                defaults to the current UTC time if omitted.

        Returns:
            The primary key of the inserted row.
        """
        fields.setdefault("timestamp", _utc_now_iso())
        snapshot = GridSnapshot(**fields)
        with Session(self._engine) as session:
            session.add(snapshot)
            session.commit()
            session.refresh(snapshot)
            return snapshot.id

    def insert_decision(self, **fields: Any) -> int:
        """Insert an agent decision.

        Args:
            **fields: Column values for :class:`AgentDecision`. ``timestamp``
                defaults to the current UTC time if omitted.

        Returns:
            The primary key of the inserted row.
        """
        fields.setdefault("timestamp", _utc_now_iso())
        decision = AgentDecision(**fields)
        with Session(self._engine) as session:
            session.add(decision)
            session.commit()
            session.refresh(decision)
            return decision.id

    def get_recent_decisions(self, n: int = 100) -> list[dict[str, Any]]:
        """Return the most recent ``n`` agent decisions, newest first.

        Args:
            n: Maximum number of rows to return.

        Returns:
            A list of column-value dictionaries.
        """
        stmt = select(AgentDecision).order_by(desc(AgentDecision.id)).limit(n)
        with Session(self._engine) as session:
            rows = session.scalars(stmt).all()
            return [self._row_to_dict(row) for row in rows]

    def get_recent_snapshots(self, n: int = 100) -> list[dict[str, Any]]:
        """Return the most recent ``n`` grid snapshots, newest first.

        Args:
            n: Maximum number of rows to return.

        Returns:
            A list of column-value dictionaries.
        """
        stmt = select(GridSnapshot).order_by(desc(GridSnapshot.id)).limit(n)
        with Session(self._engine) as session:
            rows = session.scalars(stmt).all()
            return [self._row_to_dict(row) for row in rows]

    @staticmethod
    def _row_to_dict(row: Base) -> dict[str, Any]:
        """Convert an ORM row to a plain dictionary."""
        return {col.name: getattr(row, col.name) for col in row.__table__.columns}
