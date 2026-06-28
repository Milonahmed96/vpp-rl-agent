# VPP RL Agent

A reinforcement-learning agent that autonomously dispatches a Virtual Power
Plant of 5,000 home EV batteries to arbitrage UK imbalance prices while
providing grid frequency-stability services.

The agent observes settled grid conditions from NESO and Elexon, engineers a
14-feature state vector, and a PPO policy decides each half-hour whether to
**hold**, **charge** (draw from the grid) or **discharge** (inject to the grid).

---

## Architecture

```
            +---------------+      +----------------+
            |   NESO API    |      |   Elexon API   |
            | frequency,    |      | imbalance px,  |
            | demand/gen    |      | generation mix |
            +-------+-------+      +--------+-------+
                    |                       |
                    +-----------+-----------+
                                |
                        +-------v--------+
                        |   Feature      |   14 features + SoC
                        |  Engineering   |   (MinMaxScaler)
                        +-------+--------+
                                |
                        +-------v--------+
                        |  VPP Gym Env   |   reward = profit
                        | (5,000 EVs)    |   - freq penalty
                        +-------+--------+   - degradation
                                |
                        +-------v--------+
                        |   PPO Agent    |   stable-baselines3
                        | (MlpPolicy)    |   device = cpu
                        +-------+--------+
                                |
                        +-------v--------+
                        | FastAPI Loop   |   APScheduler, 30 min
                        | (control loop) |
                        +-------+--------+
                                |
                        +-------v--------+
                        |    SQLite      |   snapshots + decisions
                        +----------------+
```

---

## Quickstart

```bash
# 1. Create the environment and install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# 2. Run the test suite
pytest --cov=src

# 3. Train the agent (writes models/checkpoints/best_model.zip and
#    data/processed/backtest_results.csv). Uses synthetic dev data if no
#    real merged dataset is present.
python -m src.agent.train                 # full 500k-step run
python -m src.agent.train --timesteps 5000  # quick smoke run

# 4. Serve the autonomous control loop
uvicorn src.api.main:app --reload --port 8000
curl localhost:8000/health
```

Copy `.env.example` to `.env` to override defaults (DB path, model path, loop
interval, log level).

---

## Project structure

```
vpp-rl-agent/
├── .github/workflows/ci.yml     # lint + test + coverage
├── src/
│   ├── ingestion/               # NESO + Elexon async clients, synthetic data
│   ├── features/                # feature engineering + MinMaxScaler
│   ├── env/                     # VPPEnv (gymnasium) + shared reward function
│   ├── agent/                   # PPO training + backtest
│   ├── db/                      # SQLAlchemy persistence
│   └── api/                     # FastAPI app + APScheduler control loop
├── tests/                       # pytest suite (mocked HTTP, offline)
├── data/                        # raw / processed artifacts (gitignored)
├── models/checkpoints/          # trained models (gitignored)
└── requirements*.txt, pyproject.toml, .flake8
```

---

## Data sources

| Source | Endpoint | Fields used | Freshness caveat |
|--------|----------|-------------|------------------|
| NESO System Frequency | CKAN `datastore_search` (resource `f93d1835-…`) | `dtm`, `f` (Hz) | **Historic monthly CSVs, not a live stream.** |
| NESO Demand/Generation | CKAN `datastore_search` (resource `177f6fa4-…`) | `SETTLEMENT_DATE`, `SETTLEMENT_PERIOD`, `WIND_GENERATION`, `SOLAR_GENERATION`, `ND` | Half-hourly. |
| Elexon System Prices | `/balancing/settlement/system-prices/{date}` | `settlementPeriod`, `systemBuyPrice`, `systemSellPrice`, `netImbalanceVolume` | **Published ~15-30 min after the period ends.** |
| Elexon Generation Mix | `/datasets/FUELHH` | per-fuel half-hourly generation | Lagged. |

> Because both feeds are lagged, the agent always acts on the **most recently
> settled half-hour**, never the live instant. When no real merged dataset is
> available, `src/ingestion/synthetic.py` generates schema-correct **development
> data** so the full pipeline runs offline — it is not real grid history.

---

## Reward function

For action power `a` (MW, positive = charge/draw, negative = discharge/inject),
frequency deviation `Δf = f̄ − 50`, system buy price `SBP` and sell price `SSP`:

```
profit          = -a · price · 0.5 / 1000          (price = SSP if charging else SBP)
freq_penalty    = -Δf · a                            (positive ⇒ worsened frequency)
degradation     = |a| · 1000 · 0.5 · 0.003           (£, throughput × cycle cost)

reward          = w1·profit − w2·freq_penalty − w3·degradation
                  with (w1, w2, w3) = (1.0, 0.5, 0.01)
```

**Sign convention.** Drawing power lowers grid frequency; injecting raises it.
*Stabilising* means opposing the deviation (inject when `Δf < 0`, draw when
`Δf > 0`), which yields `freq_penalty < 0` and therefore **increases** reward.
*Worsening* yields `freq_penalty > 0` and is penalised. The state of charge
rises when charging and falls when discharging.

> **Note on the brief.** Two formulas in the original brief were inverted
> relative to this stated physics (`delta_soc = -a·…` would make charging
> *lower* SoC, and `freq_penalty = Δf·a` inverts the stabilise/worsen label).
> Both are corrected here and documented in `src/env/vpp_env.py`; the corrections
> are validated by the environment's behavioural tests.

---

## Battery fleet parameters

| Parameter | Value |
|-----------|-------|
| Number of batteries | 5,000 |
| Battery capacity | 75.0 kWh |
| Max charge rate (per EV) | 7.4 kW |
| Max discharge rate (per EV) | 5.0 kW |
| Fleet max charge | 37.0 MW |
| Fleet max discharge | 25.0 MW |
| Fleet energy capacity | 375 MWh |
| SoC bounds | 0.1 – 0.9 |
| Cycle degradation cost | £0.003 / kWh throughput |

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe with timestamp. |
| `GET` | `/status` | Current SoC, last action/reward/price/freq, last update. |
| `GET` | `/decisions?limit=50` | Most recent agent decisions from the DB. |
| `GET` | `/metrics` | Total profit, degradation, net profit, stabilising-action %, episodes run. |
| `POST` | `/reset` | Reset SoC to 0.5 and clear episode state. |

---

## Known limitations

- **Data lag.** NESO frequency is historic monthly CSV data and Elexon prices
  lag 15-30 minutes, so the agent acts on settled — not live — data.
- **No physical grid feedback loop.** The agent's dispatch does not actually
  influence measured grid frequency; the frequency term is a modelled incentive,
  not a closed-loop response.
- **Linear degradation model.** Battery wear is modelled as a flat £/kWh
  throughput cost, ignoring depth-of-discharge, temperature and calendar ageing.
- **Reward-term commensurability.** The brief divides profit by 1000 but leaves
  degradation in raw £, so the terms are on different scales. With the brief's
  `w3 = 0.3` the degradation term dominated (~50×) and the agent learned to
  always Hold. The default degradation weight is therefore rebalanced to
  `w3 = 0.01` so the terms are comparable and the agent actively trades; the
  weights remain tunable via `VPPEnv(reward_weights=...)`. A fuller fix would
  express all three terms in a single, consistent unit.
- **Synthetic fallback data.** Without a real merged dataset, training and the
  API loop use synthetic development data, which only approximates real
  market/grid dynamics.

---

## Backtest results

Populated by `python -m src.agent.train` into
`data/processed/backtest_results.csv` (placeholder values shown):

| Metric | Value |
|--------|-------|
| Mean episode reward | _TBD_ |
| Std episode reward | _TBD_ |
| Total profit (£) | _TBD_ |
| Total degradation cost (£) | _TBD_ |
| Stabilising actions | _TBD_ |
| Destabilising actions | _TBD_ |
| Sharpe-equivalent | _TBD_ |

---

## Future work

- Integrate the **Octopus Agile** API for retail price signals.
- **Multi-agent coordination** across regional fleets with shared constraints.
- An **LSTM price forecaster** feeding predicted prices into the state.
- Rebalanced / learned reward weights and a non-linear degradation model.
