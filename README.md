# ETH Volatility Project

A sequential Bayesian pipeline for detecting and forecasting **multi-well structure** in the ETH/USDT drift field. The system models the log-price of Ethereum as a particle moving through a time-varying potential landscape, and asks in real-time: is ETH currently trapped in a **single well** (trending / momentum regime) or oscillating between **multiple wells** (mean-reverting between discrete price levels)?

---

## Theoretical Background

This project extends the framework introduced by **Igor Halperin** in *"Non-Linear and Meta-Stable Dynamics in Financial Markets: Evidence from High Frequency Crypto Currency Market Makers"*, which treats a financial asset's price as a **particle moving through a potential field**. In that formulation:

- The log-price $x_t$ evolves under an SDE whose drift $\mu(x)$ is the negative gradient of a potential: $\mu(x) = -\nabla U(x)$.
- **Wells** (local minima of $U$) correspond to stable price attractors — the price tends to revert towards them.
- **Barriers** (local maxima of $U$ between wells) determine how "sticky" a regime is — low barriers mean the price can easily escape to another well.
- The **Kramers-Moyal (KM) expansion** provides a non-parametric way to estimate the drift and diffusion coefficients directly from high-frequency tick data, without imposing parametric assumptions on the dynamics.

Halperin's insight is powerful but static: KM estimation produces a time-averaged snapshot of the potential landscape. **This project makes the drift field explicit and dynamic** — rather than estimating a single average $\mu(x)$ over a fixed window, we model $\mu(x, t)$ as a function of *both* price level and time via a sequential Gaussian Process, allowing the potential landscape to evolve continuously as new data arrives.

The key advance is turning a retrospective diagnostic ("what did the landscape look like?") into a **real-time, forward-looking signal** ("is the landscape currently multi-well, and is that structure strengthening or weakening?").

---

## Pipeline Overview

The system is organised into three sequential phases:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                          │
│  data_collection.py: Binance aggTrades → log-return bars (30s, 900s)        │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     KRAMERS-MOYAL ESTIMATION                                 │
│  kramers_moyal.py: per-window KM estimation of drift μ(x) and               │
│  diffusion D(x) from 30-second log-return bars                              │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │  spatial_var, prior calibration
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SEQUENTIAL KALMAN-GP                                     │
│  gaussian_process.py: KalmanGPDriftModel — spatio-temporal GP over μ(x,t)  │
│  with O(N) Matern 3/2 state-space inference in time × RBF in space          │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐
     │ Initialise_GP.py │  │GP_Daily_Update.py│  │ Backtest.py  │
     │  full-hist init  │  │  daily incr      │  │  signal bt   │
     └──────────────────┘  └──────────────────┘  └──────────────┘
```

### Production vs. Research Scripts

| Script | Role |
|--------|------|
| **`Initialise_GP.py`** | **Production** — full-history Kalman-GP run. Initialises the model over a date range, processes all windows sequentially, persists state. |
| **`GP_Daily_Update.py`** | **Production** — incremental daily update. Loads the chained state, ingests one day of new data, produces topology diagnostics + forecasts, saves the updated state for the next day. This is what runs daily in production. |
| `Backtest.py` | **Research** — mirrors the production pipeline but runs autonomously over a configurable historical window. Detects well-jump events from price geometry alone, then evaluates topology signals as leading indicators. |

The backtest uses `init_gp_pipeline()` from `Initialise_GP.py` directly, guaranteeing identical model initialisation between production and evaluation.

---

## Methods in Detail

### 1. Data Collection & Aggregation

Raw tick-level data (`aggTrades`) is downloaded from [Binance Vision](https://data.binance.vision/) and aggregated into fixed-interval log-return bars using a backward moving average (BMA) kernel:

$$r_k = \log\!\left(\text{BMA}(t_{k+1})\right) - \log\!\left(\text{BMA}(t_k)\right)$$

Two timescales are used:
- **30-second bars** (KM estimation): high-frequency resolution for KM drift/diffusion estimation.
- **900-second (15-min) bars** (GP): coarser observations for the sequential GP, balancing information density with computational tractability.

An optional **EMA demeaning** step removes slow-moving trend components so the GP focuses on the *shape* of the drift field rather than its level.

### 2. Kramers-Moyal Estimation

For each calendar-month window, the **Kramers-Moyal expansion** is applied to the 30-second log-price series to estimate the first two coefficients:

$$D^{(1)}(x) = \lim_{\Delta t \to 0} \frac{1}{\Delta t} \langle x_{t+\Delta t} - x_t \mid x_t = x \rangle \quad \text{(drift)}$$

$$D^{(2)}(x) = \lim_{\Delta t \to 0} \frac{1}{2\Delta t} \langle (x_{t+\Delta t} - x_t)^2 \mid x_t = x \rangle \quad \text{(diffusion)}$$

The implementation uses the `kramersmoyal` library with histogram binning (200 bins by default) and a minimum weight threshold to suppress noisy tail estimates.

KM outputs serve two purposes:
1. **Calibration** — the variance of KM drift bins sets `spatial_var`, the prior amplitude of the GP.
2. **Validation overlay** — KM "red dots" are plotted alongside the GP posterior as a non-parametric sanity check.

### 3. Sequential Kalman-GP

The core model places a **separable spatio-temporal GP prior** over the drift field:

$$\mu(x, t) \sim \mathcal{GP}\!\left(0,\; k_\text{RBF}(x, x') \times k_{\text{Matérn}_{3/2}}(t, t')\right)$$

**Why this factorisation?**

- The **RBF kernel in log-price space** $x$ captures smooth spatial structure — wells, barriers, slopes — in the drift field.
- The **Matérn 3/2 kernel in time** allows the landscape to evolve (wells appear, deepen, disappear) while maintaining temporal continuity. Its state-space SDE representation enables exact $O(N)$ Kalman filtering:

$$\frac{d}{dt}\begin{pmatrix} f \\ f' \end{pmatrix} = \begin{pmatrix} 0 & 1 \\ -\lambda^2 & -2\lambda \end{pmatrix}\begin{pmatrix} f \\ f' \end{pmatrix} + \begin{pmatrix} 0 \\ 1 \end{pmatrix} w(t), \quad \lambda = \frac{\sqrt{3}}{\ell_t}$$

The full state vector has dimension $2M$ (position + velocity per inducing point):

$$\mathbf{s} = [f_1, f'_1, \ldots, f_M, f'_M]^T$$

**Inducing-point reprojection**: as the price drifts over weeks and months, the inducing grid is periodically relocated to cover the current price range (with a margin), preserving the accumulated posterior via GP interpolation.

### 4. Topology Extraction

At each time step, the posterior $\mu(x \mid \text{data})$ is integrated into a potential:

$$U(x) = -\int \mu(x)\, dx$$

The posterior uncertainty is propagated by drawing $N_\text{samples}$ from the GP, integrating each sample, and scanning for local minima. The **topology signal** summarises the landscape:

| Signal | Definition |
|--------|-----------|
| $p_\text{multiwell}$ | Fraction of posterior samples with $\ge 2$ stable wells |
| `barrier_mean` | Average barrier height between adjacent wells (across samples) |
| `barrier_snr` | `barrier_mean / barrier_std` — a high SNR means the multi-well structure is statistically robust |

### 5. Well-Jump Detection (Backtest)

The backtest detects **regime transitions** (jumps between wells) using a purely price-geometric criterion — no topology information leaks into the labels:

1. A rolling window of `STABLE_DAYS` is classified as "stable" if the log-price range is below a threshold.
2. A "jump" begins when displacement from the stable anchor exceeds `JUMP_THR`.
3. The jump ends when price stabilises in a new region for `SETTLE_DAYS`.

Topology signals (level and trend) at configurable look-back offsets before each jump are compared to null (calm period) samples via **Mann-Whitney U tests**, yielding rank-biserial effect sizes and significance levels.

---

## Production Workflow

### Full History Initialisation

```powershell
$env:PYTHONIOENCODING='utf-8'; & ".venv/Scripts/python.exe" Initialise_GP.py
```

Steps performed:
1. Download & aggregate raw data for the configured date range.
2. Run KM on monthly windows to calibrate spatial variance.
3. Initialise the Kalman-GP model with the estimated hyperparameters.
4. Process all windows sequentially: reproject → Kalman update → topology extraction.
5. Save the full model state to `gp_results/900s/`.

### Daily Incremental Update

```powershell
$env:PYTHONIOENCODING='utf-8'; & ".venv/Scripts/python.exe" GP_Daily_Update.py
```

Steps performed:
1. Load the chained state pickle (`GP_updates/chain_state.pkl` or latest from `gp_results/`).
2. Download & aggregate new data since `last_dt`.
3. Reproject the GP to the new day's price range.
4. Run one-at-a-time Kalman updates, recording per-observation innovation z-scores.
5. Extract topology (p_multiwell, barriers, fragility).
6. Compute trend slopes and forecasts at multiple horizons.
7. Produce diagnostic plots and save the updated chain state.

### Backtest

```powershell
$env:PYTHONIOENCODING='utf-8'; & ".venv/Scripts/python.exe" Backtest.py
```

Mirrors the production pipeline with a burn-in phase followed by a backtest window. No data leakage: labels are purely price-geometric, and topology at day $d$ uses only data $\le d$.

Outputs are written to `backtests/{start}_{end}/`:
- `daily_topology.csv` — day-by-day topology across the full period
- `events.csv` — detected well-jump events
- `per_event.csv` — signal values at each (event, offset) pair
- `null_samples.csv` — same schema for null (calm) dates
- `summary.csv` — per-signal Mann-Whitney stats + rank-biserial IC vs null
- `plots/all_months_drift.png` — GP drift ±2σ with KM overlay, one panel per backtest month
- `plots/all_months_potential.png` — integrated potential U(x) ±2σ, one panel per backtest month
- `plots/YYYY-MM_overview.png` — per-month: ETH/USD price + p_multiwell + barrier_snr, with red jump lines and green calm-day shading
- `plots/per_signal_boxes.png` — pre-jump vs null signal distributions
- `plots/events_overview.png` — full-period price + signal panels with jump markers

---

## Key Configuration Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `SPATIAL_VAR_SOURCE` | `"km"` | `"km"` = calibrate from KM bins; `"fixed"` = constant |
| `USE_REPROJECT` | `True` | Relocate inducing grid to each window's price range |
| `N_INDUCING` | `10` | Number of inducing points (spatial resolution) |
| `TEMPORAL_LENGTHSCALE_DAYS_INIT` | `5.0` | Matérn temporal memory (days) |
| `SPATIAL_LENGTHSCALE_INIT` | `None` | `None` → `(x_hi − x_lo) / N_INDUCING` |
| `EMA_HALFLIFE_DAYS` | `30` | Half-life for drift demeaning |
| `GP_SI` | `900` | GP observation interval (seconds) |
| `KM_SI` | `30` | KM estimation interval (seconds) |

---

## Output Gallery

### Backtest — All Months Drift

Grid of GP posterior drift ±2σ panels with KM estimates (red dots) overlaid — one panel per backtest calendar month. Consistent y-axis limits across all panels.

### Backtest — All Months Potential

Grid of integrated potential U(x) ±2σ panels — one per backtest month. Wells appear as valleys; barriers as peaks.

### Backtest — Monthly Overview

Per-month three-panel figure: ETH/USD price (top), p_multiwell (middle), barrier_snr (bottom). Green shading marks calm (low-volatility) days; red vertical lines mark detected jump_start dates. Data plotted as daily steps to show discrete resolution.

### Backtest — Events Overview

Full backtest period: price with topology signal panels and markers at detected well-jump events.

![Backtest events overview](backtests/2024-07-01_2025-12-31/plots/events_overview.png)

### Backtest — Signal Distributions

Box plots comparing signal values at look-back offsets before well-jump events (red) vs. null/calm periods (blue). Significant separation confirms predictive value.

![Per-signal box plots](backtests/2024-07-01_2025-12-31/plots/per_signal_boxes.png)

### Daily Update — Drift Snapshot

A single day's incremental update showing the GP drift posterior updated with new observations, overlaid with KM estimates.

![Daily drift snapshot](GP_updates/2026-01-01/drift_snapshot.png)

### Daily Update — Potential Snapshot

The potential $U(x)$ on the update day. Well positions and barrier heights are directly readable.

![Daily potential snapshot](GP_updates/2026-01-01/potential_snapshot.png)

### Daily Update — Innovation Diagnostics

Per-observation z-scores for the new data. A running mean $|z|$ near $\sqrt{2/\pi} \approx 0.80$ indicates well-calibrated observation noise.

![Innovation diagnostics](GP_updates/2026-01-01/innovations.png)

### Daily Update — Fragility Monitor

Barrier mean ± std over time, with the SNR (barrier_mean / barrier_std). Declining SNR signals weakening regime structure — a potential precursor to a regime change.

![Fragility monitor](GP_updates/2026-01-01/fragility.png)

---

## Topology Output Keys

| Key | Meaning |
|-----|---------|
| `p_multiwell` | Fraction of posterior drift samples with ≥ 2 stable wells |
| `barrier_mean` | Mean potential barrier height between adjacent wells |
| `barrier_std` | Standard deviation of barrier height across posterior samples |
| `barrier_snr` | `barrier_mean / barrier_std` — uncertainty-adjusted barrier strength |
| `slope_p_multiwell` | OLS slope of $p_\text{multiwell}$ over trailing window (trend) |
| `slope_z_p_multiwell` | t-statistic of the slope (trend significance) |

---

## File Reference

| File | Role |
|------|------|
| `data_collection.py` | Download Binance aggTrades zips, aggregate into log-return bars, EMA-demean drift |
| `kramers_moyal.py` | KM bin estimator (`estimate_km`); per-window KM pipeline |
| `gaussian_process.py` | `KalmanGPDriftModel`, `topology_from_gp`, `daily_replay`; Matérn 3/2 state-space kernel × RBF spatial kernel |
| `Initialise_GP.py` | **Production** — full-history Kalman-GP run; writes state pickle to `gp_results/` |
| `GP_Daily_Update.py` | **Production** — incremental daily update; writes to `GP_updates/{date}/` |
| `Backtest.py` | **Research** — self-contained backtest; produces drift/potential grids, monthly overviews, and signal-analysis plots under `backtests/{tag}/plots/` |
| `paths.py` | Path helpers: `gp_output_dir` |
| `plots.py` | All visualisations: `plot_all_months_drift`, `plot_all_months_potential`, `plot_monthly_overview`, backtest box plots, backtest overview, `detect_price_jumps` |
| `update/state_io.py` | Load/restore/recalibrate Kalman-GP model state from pickles |
| `update/daily_cache.py` | Build and maintain daily topology cache |
| `update/plots_update.py` | Diagnostic plots for `GP_Daily_Update.py` output |

---

## Data

Raw tick data is downloaded on demand from [Binance Vision](https://data.binance.vision/) (`ETHUSDT-aggTrades-{date}.zip`) and cached under `data/`. Aggregated log-return CSVs are also cached there; re-aggregation is skipped if the file already exists. First execution of the pipeline is slow, because files need to be downloaded and aggregated. Subsequent executions with the same parameters are faster.

---

## Dependencies

```
pip install -r requirements.txt
```

Key libraries: `numpy`, `scipy`, `pandas`, `matplotlib`, `kramersmoyal`, `rich`.

---

## References

- Halperin, I. (2020). *"Non-equilibrium skewness, market crises, and option pricing: Non-linear Langevin model of markets with variable liquidity and non-linear feedback."* — Introduces the potential-field analogy for asset prices and the use of Kramers-Moyal coefficients for empirical drift/diffusion estimation.
- Hartikainen, J. & Särkkä, S. (2010). *"Kalman filtering and smoothing solutions to temporal Gaussian process regression models."* — State-space GP inference via Matérn SDE representations.
