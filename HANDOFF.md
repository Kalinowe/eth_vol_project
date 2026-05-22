# ETH Vol Project

Sequential Bayesian pipeline for detecting multi-well potential structure in the ETH/USDT drift field. Answers: is ETH in a single-well (trending) or multi-well (mean-reverting) regime?

---

## Files

| File | Role |
|---|---|
| `RunGP_simple.py` | **Stationary** pipeline. Reprojection per window. HP-opt optional. Outputs → `gp_results/{si}s_simple/{no_hp\|hp}/` |
| `RunGP.py` | **Dynamic/trend** pipeline. Explicit Kalman-tracked scalar trend β(t). No reprojection. Outputs → `gp_results_dynamic/{si}s_trend/no_hp/` |
| `phase_GP.py` | `KalmanGPDriftModel`, `topology_from_gp`, `optimise_hp`, `reproject_to_range` |
| `phase_GP_trend.py` | `KalmanGPDriftWithTrendModel` — augments state with β(t) Matern block |
| `regime_estimation.py` | Phase A: per-window KM drift → GP potential → well count → regime label |
| `markov_chain.py` | Phase B: empirical transition matrix + dwell times |
| `data_collection.py` | Download + aggregate Binance trades |
| `plots.py` | `plot_topology_snapshots`, `plot_logprice_topology`, `plot_drift_with_km` |

---

## Two pipelines

### RunGP_simple — stationary GP with reprojection

```
r_hat(x,t) = μ(x,t) + ε
μ ~ GP(0, k_rbf(x,x') × k_matern32(t,t'))
```

Per window: `reproject_to_range()` moves all N_INDUCING points into the window's observed x-range, then Kalman update. Topology evaluated from μ only.

Key knobs: `SPATIAL_VAR_SOURCE` (`km`/`fixed`), `HP_OPT_MODE` (`none`/`ls_only`/`all`), `USE_REPROJECT`, `N_INDUCING=10`, `spatial_ls = (x_hi−x_lo)/(3×N_INDUCING)`.

Weekly overrides: `N→6`, `temporal_ls→4d`, `HP_OPT_MODE→none`.

### RunGP — dynamic GP with trend state

```
r_hat(x,t) = β(t) + μ(x,t) + ε
β(t) ~ Matern32(trend_ls, trend_var)   ← slow x-independent trend
μ(x,t) ~ GP(0, k_rbf × k_matern32)    ← residual x-shape
```

Inducing points fixed over global (p1, p99) range; no reprojection. Topology from μ only — β absorbs trending drift so μ captures wells cleanly. Variances split from KM total: `spatial_var = 0.3×KM_var`, `trend_var = 0.7×KM_var`.

Key knobs: `TREND_LENGTHSCALE_DAYS=30` (must be >> `TEMPORAL_LENGTHSCALE_DAYS_INIT=7`), `N_INDUCING=30`.

---

## Kalman-GP state (phase_GP.py)

```
state = [f_1, f'_1, ..., f_M, f'_M]   shape (2M,)
```

Transition: block-diagonal Matern 3/2 SDE discretised via matrix exponential.  
Observation: `H = K_{xZ} K_{ZZ}⁻¹` (consistent with `predict()`).  
Units: everything annualised [/yr]. `sigma2 = var(r_hat·dt)/dt` [/yr²·sec].

Trend state (phase_GP_trend.py) prepends `[β, β']` → shape `(2+2M,)`. `H[0,0]=1`.

Snapshot tuple: `(dt_query, state_mean, state_cov, topo_range, inducing_x, window_start)`.

---

## Topology output keys

`p_multiwell` — fraction of posterior drift samples with ≥2 stable wells  
`barrier_mean/std` — mean potential barrier U(x) height  
`mu_std_to_mean` — posterior σ(μ)/|μ| (>1 = noise-dominated, calibration check)  
`kramers_mean` — Kramers escape rate exp(−barrier/D)

---

## Run

```powershell
$env:PYTHONIOENCODING='utf-8'; & ".venv/Scripts/python.exe" RunGP_simple.py
$env:PYTHONIOENCODING='utf-8'; & ".venv/Scripts/python.exe" RunGP.py
```

---

## Incremental update — `RunGP_update.py`

After `RunGP_simple.py` runs, it persists a `*_state.pkl` next to its CSVs containing `{model_params, inducing_x, state_mean, state_cov, x_range_global, snapshots, topology_history, last_dt, config}`.

`RunGP_update.py` consumes that pickle, ingests new daily data from `last_dt + 1d` through `NEW_END_DATE` (default = yesterday UTC), reprojects to the new window's x-range, runs the Kalman filter one observation at a time (so per-obs innovations are captured), and writes to `GP_updates/{NEW_END_DATE}/`:

| Output | Contents |
|---|---|
| `drift_snapshot.png` | GP drift ±2σ + crimson dots = KM bins from new data |
| `potential_snapshot.png` | GP U(x) ±2σ + crimson dots/line = U_KM from new data (baseline-aligned) |
| `topology_trajectory.png` | `p_multiwell`, `barrier_mean`, `kramers_mean` history → new point (red) → forecast (orange triangles) |
| `innovations.png` | per-obs z = innov/√S + rolling mean &#124;z&#124; |
| `fragility.png` | σ(μ)/&#124;μ&#124; and `barrier_std / barrier_mean` history |
| `forecast.csv` | topology at horizons {1, 3, 7, 14, 30} days |
| `topology_history.csv` | appended history (chained) |
| `diagnostics.json` | new-window topology, slope/z of `p_multiwell` & `barrier_mean` over last `TREND_LOOKBACK` windows, mean &#124;z&#124; surprise, fragility |
| `update_state.pkl` | chained state so successive updates compose |

New diagnostics:
- **innovation z-score**: `innov / √S` per obs. Mean &#124;z&#124; under correct model ≈ √(2/π) ≈ 0.80. Persistent values >1 mean the GP is being surprised → upcoming regime shift.
- **fragility** = `barrier_std / barrier_mean`. High (>0.5) ⇒ topology is uncertain; wells about to merge or split.
- **trend slope + z** on `p_multiwell` and `barrier_mean` over the last `TREND_LOOKBACK` windows → directional answer to "is p_multiwell growing?" / "is barrier lowering?".

Run:
```powershell
$env:PYTHONIOENCODING='utf-8'; & ".venv/Scripts/python.exe" RunGP_update.py
```


