# ETH Vol Project — Handoff

## What this project is

A sequential Bayesian pipeline for detecting multi-well potential structure
(regime topology) in the ETH/USDT drift field, using Binance aggregate-trade
data. The question it answers: is ETH currently in a single-well (trending)
or multi-well (range-bound, bimodal) regime, and how likely is a regime
transition in the next 1–7 days?

---

## Pipeline overview

```
RunGP_simple.py  (main execution script — edit CONFIGURATION block and run)
    │
    ├── Step 1  data_collection.ensure_data()
    │           Download missing Binance daily CSV dumps
    │
    ├── Step 2  data_collection.aggregate_log_returns_range()
    │           Resample raw trades → fixed-interval log-returns (phase_a + phase_gp intervals)
    │
    ├── Step 3  regime_estimation.run_phase_a()     [Phase A]
    │           Per-window KM drift estimate → GP potential → well count → regime label
    │           Output: regime_results/km/<stem>_<window_type>_<si>s.csv (one per window)
    │                   regime_results/regime_labels_<stem>_<window_type>.csv
    │
    ├── Step 4  markov_chain.run_markov_chain()     [Phase B]
    │           Empirical transition matrix + mean dwell times from Phase A labels.
    │           Output: regime_results/mc_<stem>_results.csv + PNGs
    │
    ├── Steps 5–6  KM-anchored spatial_var + model init
    │           spatial_var = median KM drift-variance across high-weight bins (annualised)
    │           sl_init = (x_hi − x_lo) / n_inducing_eff
    │           temporal_ls = TEMPORAL_LENGTHSCALE_DAYS_INIT (overridden for weekly)
    │
    ├── Step 7  HP optimisation (if HP_OPT_MODE != 'none')
    │           Multi-restart L-BFGS-B MLL on subsampled data (≤ HP_OPT_MAX_SAMPLES).
    │           ls bounds computed from median per-window x-range (not global range).
    │           HP_OPT_MODE='ls_only' fixes spatial_var; 'all' optimises all three HPs.
    │
    ├── Step 8  Sequential Kalman updates + topology per window
    │           For each window: optional reproject_to_range → Kalman update → topology_from_gp
    │           Logs: p_multi_gp, p_multi_a, barrier, sigma/mu per window
    │           Output: gp_results/<si>s/<stem>_topology.csv + _params.csv
    │
    ├── Step 9  GP drift topology plot  (_km_vs_gp.png)
    │           One panel per snapshot: GP posterior mean + ±2σ bands, inducing points.
    │           Y-axis from GP posterior percentiles across snapshots.
    │
    ├── Step 10 Log-price vs topology plot  (_logprice_topology.png)
    │           Log-price time series per window with GP zero-crossing lines overlaid.
    │           Green dashed = stable well, red dotted = saddle.
    │
    └── Step 11 GP drift + KM overlay plot  (_drift_km.png)
                GP posterior + KM bin scatter (size ∝ bin weight) per snapshot.
                Y-axis from weighted KM percentile across all windows.
```

---

## Key files

| File | Role |
|---|---|
| `RunGP_simple.py` | Main execution script. All knobs in the `CONFIGURATION` block at the top. Run with `python RunGP_simple.py`. |
| `phase_GP.py` | All Kalman-GP logic: `KalmanGPDriftModel`, `topology_from_gp`, `optimise_hp`, `reproject_to_range`. |
| `regime_estimation.py` | Phase A: KM drift, GP potential, well detection, `run_phase_a`. |
| `markov_chain.py` | Phase B: empirical Markov chain on regime labels. |
| `data_collection.py` | Download + aggregate Binance data. |
| `plots.py` | All visualisation: `plot_km_vs_gp_simple`, `plot_logprice_topology`, `plot_drift_with_km`, `plot_gp_topology_series`. |
| `RunGP.py` | Legacy full pipeline — not yet updated with current GP fixes. |

---

## Current configuration defaults (RunGP_simple.py)

```python
start_date = datetime(2024, 1, 1)
end_date   = datetime(2024, 12, 31)
phase_a_seconds_interval  = 30
phase_gp_seconds_interval = 900
window_type               = 'monthly'   # 'weekly' | 'biweekly' | 'monthly'

SPATIAL_VAR_SOURCE = 'km'              # 'km' | 'fixed'
HP_OPT_MODE        = 'none'            # 'none' | 'ls_only' | 'all'
HP_OPT_N_RESTARTS  = 3
HP_OPT_MAX_SAMPLES = 5000
USE_REPROJECT      = True
REPROJECT_MARGIN   = 0.1

SPATIAL_LENGTHSCALE_INIT       = None  # None → (x_hi − x_lo) / n_inducing_eff
TEMPORAL_LENGTHSCALE_DAYS_INIT = 7.0
N_INDUCING = 10
N_GRID = 200;  N_SAMPLES = 200;  MIN_CROSSING_SEP = 10
SEED = 42
```

**Weekly window automatic overrides** (applied when `window_type='weekly'`):
- `N_INDUCING → min(N_INDUCING, 6)`
- `temporal_ls → min(TEMPORAL_LENGTHSCALE_DAYS_INIT, 4.0)`
- `HP_OPT_MODE → 'none'`

---

## `KalmanGPDriftModel` design (phase_GP.py)

### State layout
```
state = [f_1, f'_1, f_2, f'_2, ..., f_M, f'_M]   shape (2M,)
```
`f_j` = drift value at inducing point `z_j`.  `f'_j` = temporal derivative.

### Temporal kernel
Matern 3/2 in exact SDE form. Block-diagonal over M inducing points.
Discretised per step by `matern32_discrete()` via matrix exponential.

### Spatial kernel
RBF: `k(x, x') = spatial_var * exp(−0.5*(x−x')²/spatial_ls²)`.

### Observation model
```
r_hat(x_i) = H_i @ f + ε,   ε ~ N(0, obs_noise)
H_i = K_{x_i,Z} @ K_{ZZ}^{-1}    ← cached as self._K_zz_inv
```
`_K_zz_inv` is computed in `initialise()` and recomputed by `_recompute_hp_dependent()`.
Both `predict()` and `_obs_matrix()` use it — always call `_recompute_hp_dependent()`
after manually changing `inducing_x` or any HP.

### Units (critical)
Everything in **[/year]**. KM CSV `drift` column is [/sec] → multiply by
`_SEC_PER_YEAR = 365.25×24×3600`. `sigma2 = var(r_hat × dt_sec) / dt_sec`
[/yr²·sec]; `obs_noise = sigma2 / dt_sec` [/yr²].

### reproject_to_range()
Moves inducing points to `[x_lo, x_hi]`, rebuilds `_K_zz_inv` etc., and
GP-interpolates the state mean/cov to the new inducing positions. Use this
before each window when `USE_REPROJECT=True`.

### HP-opt bounds
`optimise_hp()` accepts `bounds_range` (separate from `x_range` used for
inducing placement). Pass the median per-window x-range here so that ls bounds
are calibrated to window scale, not the full multi-year global range.
`ls_lo = x_width_window / n_ind`, `ls_hi = min(x_width_window / 2, 0.1)`.

### Snapshot tuple (6 elements)
```python
(dt_query, state_mean, state_cov, topo_range, inducing_x, window_start)
```
`window_start` is used to slice the log-price time series in `plot_logprice_topology`.

---

## topology_from_gp() output keys

| Key | Meaning |
|---|---|
| `p_multiwell` | Fraction of posterior samples with ≥ 2 stable wells |
| `mean_n_wells` | Mean well count across samples |
| `barrier_mean/std` | Mean potential barrier height (annualised units) |
| `kramers_mean` | Mean Kramers escape rate exp(−barrier/D) |
| `u_range` | Potential range of mean-field U(x) |
| `mu_std_to_mean` | Posterior std / mean drift magnitude (identifiability diagnostic; ≳1 means noise-dominated) |

---

## Virtual environment
```
.venv/Scripts/python.exe
$env:PYTHONIOENCODING='utf-8'; & ".venv/Scripts/python.exe" RunGP_simple.py
```

