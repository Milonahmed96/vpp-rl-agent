# Project Context

Background, motivation, and design rationale for the VPP RL Agent. For
operational details (commands, structure, API) see `README.md`; for AI-assistant
working conventions see `CLAUDE.md`.

## Problem

A **Virtual Power Plant (VPP)** aggregates many small, distributed energy
resources — here, 5,000 home EV batteries — and operates them as a single
controllable asset. The goal is twofold:

1. **Arbitrage.** Buy energy (charge) when imbalance prices are low and sell
   (discharge) when they are high, profiting from UK Balancing Mechanism price
   swings.
2. **Grid stability.** The UK grid targets 50 Hz. Demand/supply mismatch pushes
   frequency off target. A responsive fleet can help by drawing power when
   frequency is high and injecting when it is low.

These objectives partly conflict (the most profitable action is not always the
most stabilising), and battery cycling has a real wear cost. The agent must
balance all three — profit, frequency support, and degradation — each half-hour.

## Why reinforcement learning

The dispatch decision is sequential and state-dependent: state of charge,
current and recent prices, frequency, and generation mix all matter, and today's
action constrains tomorrow's options (you can't discharge an empty battery).
This maps naturally onto an RL Markov decision process. We use **PPO**
(stable-baselines3, `MlpPolicy`) over a custom `gymnasium` environment.

## State, action, reward

- **State** — a 14-feature vector engineered from NESO + Elexon settled data
  (prices, frequency, demand, wind/solar generation, time encodings) plus the
  battery state of charge, scaled with a `MinMaxScaler`.
- **Action** — discrete: hold / charge / discharge.
- **Reward** — `w1·profit − w2·freq_penalty − w3·degradation`. See `README.md`
  and `src/env/vpp_env.py` for the exact formulas and the corrected sign
  conventions.

## Data sources and the lag reality

The agent consumes **settled** grid data, not a live feed:

- **NESO** system frequency and demand/generation (CKAN datastore). Frequency
  is published as historic monthly CSVs, not a real-time stream.
- **Elexon** system buy/sell prices and generation mix, published ~15-30 minutes
  after each settlement period ends.

Consequently the agent always acts on the **most recently settled half-hour**.
This is an honest constraint of freely available UK grid data, not an
implementation shortcut. When no real merged dataset is present,
`src/ingestion/synthetic.py` generates schema-correct **development** data so the
full pipeline (training, API loop, tests) runs offline.

## Key design decisions

- **Degradation weight rebalanced.** The original brief's `w3 = 0.3`, combined
  with profit being divided by 1000 while degradation stays in raw £, made the
  degradation term dominate (~50×). The agent learned to always Hold. We set the
  default to `w3 = 0.01` so the three terms are comparable and the agent actively
  trades. Weights remain tunable via `VPPEnv(reward_weights=...)`.
- **Corrected brief formulas.** Two formulas in the brief were physically
  inverted (a charging action that *lowers* SoC; a frequency penalty that rewards
  *worsening* the deviation). Both are corrected and locked in by behavioural
  tests.
- **CPU-only training.** Torch runs on CPU; the model is small enough that GPU
  provides no meaningful benefit and CPU keeps the dev loop portable.
- **Offline-first.** Tests mock all HTTP and synthetic data backs every stage, so
  the whole system is reproducible without API access.

## Known limitations

- No physical closed loop — the fleet's dispatch does not actually move measured
  grid frequency; the frequency term is a modelled incentive.
- Linear degradation model ignores depth-of-discharge, temperature, and calendar
  ageing.
- Reward terms are not on a single consistent unit; weights paper over the scale
  mismatch rather than fixing it fundamentally.
- Synthetic fallback data only approximates real market/grid dynamics.

## Future direction

Octopus Agile retail price signals, multi-agent coordination across regional
fleets, an LSTM price forecaster feeding predicted prices into the state, and a
non-linear degradation model with learned reward weights.
