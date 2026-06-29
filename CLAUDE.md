# CLAUDE.md

Guidance for Claude Code (and other AI assistants) working in this repository.

## What this is

A reinforcement-learning agent that dispatches a Virtual Power Plant of 5,000
home EV batteries to arbitrage UK imbalance prices while providing grid
frequency-stability services. A PPO policy decides each half-hour whether to
**hold**, **charge** (draw from grid) or **discharge** (inject to grid).

See `README.md` for the full architecture, data sources, reward derivation,
fleet parameters, and known limitations. `project_context.md` covers the
domain background and design rationale.

## Layout

```
src/
├── ingestion/   NESO + Elexon async clients (httpx), synthetic dev data
├── features/    feature engineering → 14-feature vector + MinMaxScaler
├── env/         VPPEnv (gymnasium) + shared reward function
├── agent/       PPO training (stable-baselines3) + backtest
├── db/          SQLAlchemy persistence (SQLite)
└── api/         FastAPI app + APScheduler control loop
tests/           pytest suite — fully offline, HTTP mocked
```

## Common commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# Tests (offline, no network)
pytest --cov=src

# Lint / format
flake8 .
black . && isort .

# Train (synthetic data if no real merged dataset present)
python -m src.agent.train                    # full 500k-step run
python -m src.agent.train --timesteps 5000   # quick smoke run

# Serve the control loop
uvicorn src.api.main:app --reload --port 8000
```

## Conventions

- **Python ≥ 3.11.** Line length 88 (black + flake8). isort uses the black
  profile; `src` is first-party.
- **flake8 ignores** `E203, W503`; `__init__.py` may keep unused imports (`F401`).
- **Tests are offline.** Mock all HTTP (NESO/Elexon). `asyncio_mode = "auto"`,
  so async tests need no explicit marker. Never make real network calls in tests.
- **Config via env.** Copy `.env.example` to `.env`; override DB path, model
  path, loop interval, log level. Use `python-dotenv` / pydantic settings.
- **Torch runs on CPU** (`device = cpu`) by design.

## Reward sign convention (important)

Action power `a` is positive for charge/draw, negative for discharge/inject.
Drawing power lowers grid frequency; injecting raises it. *Stabilising* opposes
the deviation and increases reward. Two formulas in the original brief were
inverted relative to this physics; both are corrected in `src/env/vpp_env.py`
and validated by behavioural tests — do not "restore" the brief's versions.
Default reward weights `(w1, w2, w3) = (1.0, 0.5, 0.01)`; the degradation
weight was rebalanced from the brief's `0.3` so the agent actively trades.
Weights stay tunable via `VPPEnv(reward_weights=...)`.

## Data caveat

Both grid feeds are lagged (NESO frequency is historic monthly CSV; Elexon
prices publish 15-30 min late), so the agent always acts on the most recently
**settled** half-hour, never the live instant. Without a real merged dataset,
`src/ingestion/synthetic.py` produces schema-correct development data — it is
not real grid history.

## Working agreements

- Don't commit data/model artifacts: `data/`, `models/checkpoints/*.zip`, and
  `logs/` are gitignored (only `.gitkeep` files are tracked).
- Keep tests green and offline before committing.
- Match existing module structure and naming when adding code.
