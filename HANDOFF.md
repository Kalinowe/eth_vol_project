# ETH Vol Project

Sequential Bayesian pipeline for detecting multi-well potential structure in the ETH/USDT drift field. Answers: is ETH in a single-well (trending) or multi-well (mean-reverting) regime?

---

## Files

| File | Role |
|---|---|
| `RunGP.py` | Stationary Kalman-GP pipeline. Reprojection per window. HP-opt optional. Outputs → `gp_results/{si}s/{no_hp\|hp}/` |
| `RunGP_update.py` | Incremental Kalman-GP update for new daily data. Outputs → `GP_updates/{date}/` |
| `phase_GP.py` | `KalmanGPDriftModel`, `topology_from_gp`, `optimise_hp`, `reproject_to_range`, `forecast_topology` |
| `regime_estimation.py` | Phase A: per-window KM drift → GP potential → well count → regime label |
| `markov_chain.py` | Phase B: empirical transition matrix + dwell times |
| `data_collection.py` | Download + aggregate Binance trades |
| `force_field_estimation.py` | KM bin estimator (μ, σ²) used by Phase A and the update-loop recalibration |
| `plots.py` | `plot_topology_snapshots`, `plot_logprice_topology`, `plot_drift_with_km`, Phase B MC plots |
| `find_well_jumps.py` | Causal daily Kalman replay through 2025; emits `well_jumps_2025.csv` + `well_jumps_2025_daily.csv` |
| `backtest_jumps.py` | Pre-jump leading-indicator backtest against the daily topology |

---

## Pipeline

```
r_hat(x,t) = μ(x,t) + ε
μ ~ GP(0, k_rbf(x,x') × k_matern32(t,t'))
```

Per window: `reproject_to_range()` moves all `N_INDUCING` points into the
window's observed x-range, then a Kalman update sweeps the window. Topology
is read from posterior samples of μ at the window end.

Key knobs in `RunGP.py`:

- `SPATIAL_VAR_SOURCE` (`km`/`fixed`)
- `HP_OPT_MODE` (`none`/`ls_only`/`all`)
- `USE_REPROJECT` (default `True`)
- `N_INDUCING = 10`
- `SPATIAL_LENGTHSCALE_INIT = None` → `(x_hi − x_lo) / (3 × N_INDUCING)`
- `TEMPORAL_LENGTHSCALE_DAYS_INIT = 5.0`

---

## Kalman-GP state (phase_GP.py)

```
state = [f_1, f'_1, ..., f_M, f'_M]   shape (2M,)
```

Transition: block-diagonal Matern 3/2 SDE discretised via matrix exponential.
Observation: `H = K_{xZ} K_{ZZ}⁻¹` (consistent with `predict()`).
Units: everything annualised [/yr]. `sigma2 = var(r_hat·dt)/dt` [/yr²·sec].

Snapshot tuple: `(dt_query, state_mean, state_cov, topo_range, inducing_x, window_start)`.

---

## Topology output keys (Phase C)

| Key | Meaning |
|---|---|
| `p_multiwell` | fraction of posterior drift samples with ≥2 stable wells |
| `barrier_mean / barrier_std` | mean & std of the potential barrier `U(x)` height |
| `mu_std_to_mean` | posterior `σ(μ) / |μ|` — noise calibration |
| `kramers_mean` | Kramers escape rate `≈ exp(−barrier_mean)` |

---

## Run

```powershell
$env:PYTHONIOENCODING='utf-8'; & ".venv/Scripts/python.exe" RunGP.py
```

---

## Incremental update — `RunGP_update.py`

`RunGP.py` persists a `*_state.pkl` next to its CSVs. `RunGP_update.py` picks
up that state (or `GP_updates/chain_state.pkl` if a previous update run
exists), ingests new daily data from `last_dt + 1d` through `NEW_END_DATE`
(default = next single day), runs the Kalman filter one observation at a
time, and writes to `GP_updates/{date}/`:

| Output | Contents |
|---|---|
| `drift_snapshot.png` | GP drift ±2σ + KM bins from new data, zoomed to today's price range |
| `potential_snapshot.png` | GP `U(x)` ±2σ + `U_KM` (baseline-aligned), zoomed to today's price range |
| `topology_trajectory.png` | `p_multiwell`, `barrier_mean`, `kramers_mean` history → new point → forecast |
| `innovations.png` | per-obs `z = innov/√S` + rolling `mean|z|` vs H0 line `√(2/π) ≈ 0.80` |
| `fragility.png` | top: `barrier_mean ± barrier_std` daily; bottom: SNR = `barrier_mean/barrier_std` |
| `forecast.csv` | topology forecasts at `{1, 3, 7, 14}` day horizons |
| `diagnostics.json` | topology, slope/z of `p_multiwell` & `barrier_mean`, `mean|z|`, sigma2 factor |
| `update_state.pkl` | date-stamped archive copy of chained state |
| `chain_state.pkl` | *(in `GP_updates/`)* fixed-path state consumed by the next run |

**Chaining.** Each run writes both an archive copy and
`GP_updates/chain_state.pkl`. The next run auto-picks `chain_state.pkl` so
no config change is needed. Delete it to restart from the `RunGP.py` base
state.

**sigma2 auto-calibration** (`AUTOADJUST_SIGMA2 = True`). After each run the
model computes `sigma2_factor = (mean|z| / √(2/π))²` from today's
innovations and updates an EWMA scale: `sigma2 = sigma2_base × ewma_factor`
(α = 0.2, clipped to `[0.25×, 4×]`). `sigma2_base` is frozen at the `RunGP.py`
value so the EWMA is fully reversible when volatility recovers.

**Reprojection cadence.** Monthly (same as `RunGP.py`). The daily topology
cache also uses monthly reproject with a fixed monthly x-range so
`barrier_mean` is continuous within each month.

**Diagnostics to watch.**

- Rolling `mean|z| ≈ 0.80` → well-calibrated. Persistently below → obs noise
  too large (will auto-correct). Persistently above → regime shift / fat
  tails.
- `SNR = barrier_mean / barrier_std > 1` → barrier reliably detected;
  `< 1` → topology uncertain.
- Slope z-stats on `p_multiwell` and `barrier_mean` → directional regime
  signal.

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'; & ".venv/Scripts/python.exe" RunGP_update.py
```

---

## Backtesting — `find_well_jumps.py` + `backtest_jumps.py`

`find_well_jumps.py` warm-starts from a 2024-11-30 snapshot of the Phase C
state and replays the Kalman-GP day by day through 2025. At each day it
records the topology and runs a state machine on log-price:

```
STABLE   : trailing STABLE_DAYS log-price range < STABLE_THR
JUMPING  : current log-price has moved > JUMP_THR from the anchor
SETTLED  : rolling range < STABLE_THR again in the new price zone
```

Outputs:

- `well_jumps_2025_daily.csv` — daily causal topology (state at day `d`
  uses observations up to and including day `d`).
- `well_jumps_2025.csv` — one row per detected jump event.

`backtest_jumps.py` then tests whether the daily topology gives any signal
*before* `jump_start`. For each event it samples signals at offsets
`{−21, −14, −7, −3, −1}` days and compares against a null built from
"calm" dates (≥ `NULL_BUFFER_DAYS = 21` from any `jump_start`).
Signals tested:

```
p_multiwell, barrier_mean, barrier_snr, kramers,
slope_z_p_multiwell, slope_z_barrier_mean   (OLS slope z over 7-day window)
```

One-sided Mann-Whitney U with hypothesis direction declared per signal in
`ALT_HYPOTHESIS`. Outputs in the run dir: `per_event.csv`, `null_samples.csv`,
`summary.csv`, `per_signal_boxes.png`, `events_overview.png`.

**Current finding.** Of the signals tested, only the *rising slope of*
`p_multiwell` is a statistically significant leading indicator at the
tested offsets. `barrier_mean` trends in the opposite of the hypothesised
direction before jumps. The level signals (`p_multiwell`,
`barrier_mean`, `kramers`) do not separate from null at conventional
significance.

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'; & ".venv/Scripts/python.exe" find_well_jumps.py
$env:PYTHONIOENCODING='utf-8'; & ".venv/Scripts/python.exe" backtest_jumps.py
```

---

## See also

- `STREAMLINING.md` — proposals for further refactors (extract Kalman-GP
  step primitive, split `RunGP_update.py`, paths helper, etc.).
