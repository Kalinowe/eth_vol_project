# ETH Volatility Project

A sequential Bayesian pipeline for detecting and forecasting **multi-well structure** in the ETH/USDT drift field. The system models the log-price of Ethereum as a particle moving through a time-varying potential landscape, and asks in real-time: is ETH currently trapped in a **single well** (trending / momentum regime) or oscillating between **multiple wells** (mean-reverting between discrete price levels)?

---

## Theoretical Background

This project extends the framework introduced by **Igor Halperin** in *"Non-linear meta-stable dynamics of a particle with a position-dependent mass"* (and related work on *"QLBS: Q-Learner in the Black-Scholes(-Merton) Worlds"*), which treats a financial asset's price as a **particle moving through a potential field**. In that formulation:

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
│                     PHASE A — KRAMERS-MOYAL                                  │
│  regime_estimation.py: per-window KM estimation of drift μ(x) and           │
│  diffusion D(x) from 30-second log-return bars                              │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │  spatial_var, prior calibration
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PHASE GP — SEQUENTIAL KALMAN-GP                           │
│  phase_GP.py: KalmanGPDriftModel — spatio-temporal GP over μ(x,t)           │
│  with O(N) Matern 3/2 state-space inference in time × RBF in space          │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     ┌────────────┐  ┌──────────────┐  ┌──────────────────┐
     │  RunGP.py  │  │RunGP_update  │  │ backtest_jumps.py│
     │ full-hist  │  │  daily incr  │  │  signal backtest │
     └────────────┘  └──────────────┘  └──────────────────┘
```

### Production vs. Research Scripts

| Script | Role |
|--------|------|
| **`RunGP.py`** | **Production** — full-history Kalman-GP run. Initialises the model over a date range, processes all windows sequentially, persists state. |
| **`RunGP_update.py`** | **Production** — incremental daily update. Loads the chained state, ingests one day of new data, produces topology diagnostics + forecasts, saves the updated state for the next day. This is what runs daily in production. |
| `backtest_jumps.py` | **Research** — mirrors the production pipeline but runs autonomously over a configurable historical window. Detects well-jump events from price geometry alone, then evaluates topology signals as leading indicators. |

The backtest uses `init_gp_pipeline()` from `RunGP.py` directly, guaranteeing identical model initialisation between production and evaluation.

---

## Methods in Detail

### 1. Data Collection & Aggregation

Raw tick-level data (`aggTrades`) is downloaded from [Binance Vision](https://data.binance.vision/) and aggregated into fixed-interval log-return bars using a backward moving average (BMA) kernel:

$$r_k = \log\!\left(\text{BMA}(t_{k+1})\right) - \log\!\left(\text{BMA}(t_k)\right)$$

Two timescales are used:
- **30-second bars** (Phase A): high-frequency resolution for KM drift/diffusion estimation.
- **900-second (15-min) bars** (Phase GP): coarser observations for the sequential GP, balancing information density with computational tractability.

An optional **EMA demeaning** step removes slow-moving trend components so the GP focuses on the *shape* of the drift field rather than its level.

### 2. Phase A — Kramers-Moyal Estimation

For each calendar-month window, the **Kramers-Moyal expansion** is applied to the 30-second log-price series to estimate the first two coefficients:

$$D^{(1)}(x) = \lim_{\Delta t \to 0} \frac{1}{\Delta t} \langle x_{t+\Delta t} - x_t \mid x_t = x \rangle \quad \text{(drift)}$$

$$D^{(2)}(x) = \lim_{\Delta t \to 0} \frac{1}{2\Delta t} \langle (x_{t+\Delta t} - x_t)^2 \mid x_t = x \rangle \quad \text{(diffusion)}$$

The implementation uses the `kramersmoyal` library with histogram binning (200 bins by default) and a minimum weight threshold to suppress noisy tail estimates.

Phase A outputs serve two purposes:
1. **Calibration** — the variance of KM drift bins sets `spatial_var`, the prior amplitude of the GP.
2. **Validation overlay** — KM "red dots" are plotted alongside the GP posterior as a non-parametric sanity check.

### 3. Phase GP — Sequential Kalman-GP

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
$env:PYTHONIOENCODING='utf-8'; & ".venv/Scripts/python.exe" RunGP.py
```

Steps performed:
1. Download & aggregate raw data for the configured date range.
2. Run Phase A (KM) on monthly windows to calibrate spatial variance.
3. Initialise the Kalman-GP model with the estimated hyperparameters.
4. Process all windows sequentially: reproject → Kalman update → topology extraction.
5. Save the full model state to `gp_results/900s/`.

### Daily Incremental Update

```powershell
$env:PYTHONIOENCODING='utf-8'; & ".venv/Scripts/python.exe" RunGP_update.py
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
$env:PYTHONIOENCODING='utf-8'; & ".venv/Scripts/python.exe" backtest_jumps.py
```

Mirrors the production pipeline with a burn-in phase followed by a backtest window. No data leakage: labels are purely price-geometric, and topology at day $d$ uses only data $\le d$.

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
| `PHASE_GP_SI` | `900` | GP observation interval (seconds) |
| `PHASE_A_SI` | `30` | KM estimation interval (seconds) |

---

## Output Gallery

### Log-Price with Topology Overlay

The top panel shows ETH log-price over the full history; the bottom panel shows the evolving $p_\text{multiwell}$ signal. Shaded regions indicate periods where the model confidently identifies multi-well structure.

![Log-price with topology overlay](gp_results/900s/gp_2024-01-01_to_2025-12-31_900s_kmvar_reproject_logprice_topology.png)

### Potential Landscape Snapshots

Each panel shows the posterior potential $U(x)$ at a different point in time. Wells (local minima) appear as valleys; barriers as peaks. The ±2σ envelope reflects model uncertainty.

![Topology snapshots](gp_results/900s/gp_2024-01-01_to_2025-12-31_900s_kmvar_reproject_topology_snapshots.png)

### GP Drift with KM Overlay

The GP posterior drift $\mu(x)$ (blue ±2σ band) vs. the non-parametric KM estimate (red dots). Agreement validates the GP; disagreement signals model adaptation lag.

![Drift with KM overlay](gp_results/900s/gp_2024-01-01_to_2025-12-31_900s_kmvar_reproject_drift_km.png)

### Daily Update — Drift Snapshot

A single day's incremental update showing the GP drift posterior updated with new observations, overlaid with KM estimates from the day's data.

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

### Backtest — Events Overview

Price with topology signal panels and markers at detected well-jump events. The topology signals visibly elevate before jumps.

![Backtest events overview](backtest_results/events_overview.png)

### Backtest — Signal Distributions

Box plots comparing signal values at look-back offsets before well-jump events (orange) vs. null/calm periods (blue). Significant separation confirms predictive value.

![Per-signal box plots](backtest_results/per_signal_boxes.png)

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
| `regime_estimation.py` | KM bin estimator (`estimate_km`); per-window Phase A pipeline |
| `phase_GP.py` | `KalmanGPDriftModel`, `topology_from_gp`, `forecast_topology`, `daily_replay`; Matérn 3/2 state-space kernel × RBF spatial kernel |
| `RunGP.py` | **Production** — full-history Kalman-GP run; writes state to `gp_results/` |
| `RunGP_update.py` | **Production** — incremental daily update; writes to `GP_updates/{date}/` |
| `backtest_jumps.py` | **Research** — self-contained backtest mirroring the production pipeline |
| `paths.py` | Path helpers: `gp_output_dir`, `gp_state_stem` |
| `plots.py` | GP visualisations: potential snapshots, drift + KM overlay, log-price topology |
| `update/state_io.py` | Load/restore/recalibrate Kalman-GP model state from pickles |
| `update/daily_cache.py` | Build and maintain daily topology cache |
| `update/plots_update.py` | Diagnostic plots for `RunGP_update.py` output |

---

## Data

Raw tick data is downloaded on demand from [Binance Vision](https://data.binance.vision/) (`ETHUSDT-aggTrades-{date}.zip`) and cached under `data/`. Aggregated log-return CSVs are also cached there; re-aggregation is skipped if the file already exists.

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
