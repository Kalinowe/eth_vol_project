# ETH Volatility Project

Sequential Bayesian pipeline for detecting multi-well structure in the ETH/USDT
drift field. The central question: is ETH currently in a **single-well** (trending)
or **multi-well** (mean-reverting between price levels) regime?

---

## Architecture

```
data_collection.py       Download + aggregate Binance aggTrades
        │
        ▼
regime_estimation.py     Phase A — KM drift estimation
        │                (estimate_km lives here)
        ▼
phase_GP.py              KalmanGPDriftModel — sequential Kalman-GP over μ(x,t)
        │
   ┌────┴────┐
   ▼         ▼
RunGP.py    RunGP_update.py     Full-history run  /  incremental daily update
                │
           update/              State I/O, daily topology cache, update plots
```

---

## Files

| File | Role |
|---|---|
| `data_collection.py` | Download Binance aggTrades zips, aggregate into log-return bars, EMA-demean drift |
| `regime_estimation.py` | KM bin estimator (`estimate_km`); Phase A per-window pipeline: writes per-window KM drift/diffusion CSVs to `regime_results/km/` |
| `phase_GP.py` | `KalmanGPDriftModel`, `topology_from_gp`, `forecast_topology`, `daily_replay`; Matern 3/2 state-space kernel in time × RBF kernel in space |
| `RunGP.py` | Full-history Kalman-GP run over a configurable date range; writes state pickle to `gp_results/{si}s/` |
| `RunGP_update.py` | Incremental update — loads saved state, ingests new data, writes diagnostics + forecasts to `GP_updates/{date}/` |
| `find_well_jumps.py` | Day-by-day Kalman replay through 2025; detects well-jump events using a price-geometric state machine; writes `well_jumps_2025.csv` |
| `backtest_jumps.py` | Self-contained backtest of GP topology signals as leading indicators of well-jump events; runs its own GP replay and price-geometric jump detection inline; writes per-event stats and summary to `backtest_results/` |
| `paths.py` | Path helpers: `gp_output_dir`, `gp_state_stem` |
| `plots.py` | GP visualisations: potential snapshots, drift + KM overlay, log-price topology |
| `update/state_io.py` | Load/restore/recalibrate Kalman-GP model state from pickles |
| `update/daily_cache.py` | Build and maintain daily topology cache for the update dashboard |
| `update/plots_update.py` | Plots for `RunGP_update.py` output |

---

## Model

The drift field follows a spatio-temporal GP prior:

$$\mu(x, t) \sim \mathcal{GP}\!\left(0,\; k_\text{RBF}(x, x') \times k_{\text{Matern}_{3/2}}(t, t')\right)$$

Observations are EMA-demeaned annualised drift rates:

$$\hat{r}_t = \frac{dx_t}{dt} \cdot \text{sec\_per\_year} - \text{EMA}(\ldots)$$

The Matern 3/2 kernel is represented as an exact state-space SDE, giving $O(N)$
Kalman-filter inference. All units are annualised $[/\text{yr}]$.

**Topology signal** — at each time step the posterior over $\mu(x)$ is integrated
into a potential $U(x) = -\!\int \mu \,dx$ and sampled. Each sample is scanned for
local minima (stable wells); $p_\text{multiwell}$ is the fraction of samples with
$\ge 2$ wells.

---

## Running the pipeline

### Full history run

```powershell
$env:PYTHONIOENCODING='utf-8'; & ".venv/Scripts/python.exe" RunGP.py
```

Configure the date range and GP knobs in the `CONFIGURATION` block at the top of
`RunGP.py`. Key parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `SPATIAL_VAR_SOURCE` | `"km"` | `"km"` = estimate from Phase A KM bins; `"fixed"` = constant |
| `USE_REPROJECT` | `True` | Move inducing grid to each window's x-range |
| `N_INDUCING` | `10` | Number of inducing points |
| `TEMPORAL_LENGTHSCALE_DAYS_INIT` | `5.0` | Initial Matern temporal lengthscale |
| `SPATIAL_LENGTHSCALE_INIT` | `None` | `None` → `(x_hi − x_lo) / N_INDUCING` |
| `WINDOW_TYPE` | `"monthly"` | Window cadence: `"weekly"`, `"biweekly"`, `"monthly"` |

Outputs land in `gp_results/{si}s/`.

### Incremental daily update

```powershell
$env:PYTHONIOENCODING='utf-8'; & ".venv/Scripts/python.exe" RunGP_update.py
```

Loads the most recent state pickle, ingests new data since `last_dt`, and writes
to `GP_updates/{date}/`:

| Output | Contents |
|---|---|
| `drift_snapshot.png` | GP drift ±2σ + KM bins from new data |
| `potential_snapshot.png` | GP $U(x)$ ±2σ + $U_\text{KM}$ baseline-aligned |
| `innovations.png` | Per-obs z-scores + rolling `mean|z|` vs $\sqrt{2/\pi}$ |
| `fragility.png` | `barrier_mean ± barrier_std` + SNR = `barrier_mean / barrier_std` |
| `forecast.csv` | Topology forecasts at 1, 3, 7, 14 day horizons |
| `diagnostics.json` | Topology, slopes, `mean|z|`, sigma2 scale factor |
| `update_state.pkl` | Archive copy of chained state |

Each run also writes `GP_updates/chain_state.pkl` so the next call picks up
exactly where this one ended.

### Well-jump replay

```powershell
$env:PYTHONIOENCODING='utf-8'; & ".venv/Scripts/python.exe" find_well_jumps.py
```

Replays the saved state day-by-day through 2025 and emits `well_jumps_2025.csv`.

### Backtest

```powershell
$env:PYTHONIOENCODING='utf-8'; & ".venv/Scripts/python.exe" backtest_jumps.py
```

Runs a fresh GP from scratch over the configured backtest window, detects
well-jumps from price geometry alone, then measures whether `p_multiwell` and
`barrier_snr` are elevated before each jump vs. calm periods (Mann-Whitney test).
Results in `backtest_results/` (or the configured `BACKTEST_DIR`).

---

## Topology output keys

| Key | Meaning |
|---|---|
| `p_multiwell` | Fraction of posterior drift samples with ≥ 2 stable wells |
| `barrier_mean / barrier_std` | Mean & std of the potential barrier height $U(x)$ |
| `barrier_snr` | `barrier_mean / barrier_std` — uncertainty-adjusted barrier strength |
| `mu_std_to_mean` | Posterior $\sigma(\mu) / |\mu|$ — noise-to-signal calibration |

---

## Kalman-GP state

```
state = [f_1, f'_1, ..., f_M, f'_M]   shape (2M,)
```

Transition: block-diagonal Matern 3/2 SDE discretised via matrix exponential.  
Observation: $H = K_{xZ} K_{ZZ}^{-1}$ (consistent with `predict()`).  
Snapshot tuple: `(dt_query, state_mean, state_cov, topo_range, inducing_x, window_start)`.

---

## Data

Raw data is downloaded on demand from [Binance Vision](https://data.binance.vision/)
(`ETHUSDT-aggTrades-{date}.zip`) and cached under `data/`. Aggregated log-return
CSVs are also cached there; re-aggregation is skipped if the file already exists.

---

## Dependencies

```
pip install -r requirements.txt
```
