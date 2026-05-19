# ETH Vol Project — Handoff

## What this project is

A sequential Bayesian pipeline for detecting multi-well potential structure
(regime topology) in the ETH/USDT drift field, using Binance aggregate-trade
data. The question it answers: is ETH currently in a single-well (trending)
or multi-well (range-bound, bimodal) regime, and how likely is a regime
transition in the next 1–14 days?

---

## Pipeline overview

```
RunGP.py  (execution button — edit CONFIGURATION block and run)
    │
    ├── Step 1  data_collection.ensure_data()
    │           Download missing Binance daily CSV dumps
    │
    ├── Step 2  data_collection.aggregate_log_returns_range()
    │           Resample raw trades → fixed-interval log-returns
    │
    ├── Step 3  regime_estimation.run_phase_a()     [Phase A]
    │           Per-window Kramers–Moyal drift estimate → GP potential →
    │           well count → regime label (single-well / multi-well / uncertain)
    │           Output: regime_results/regime_labels_<stem>_<window_type>.csv
    │
    ├── Step 4  markov_chain.run_markov_chain()     [Phase B]  (inside run_phase_gp)
    │           Empirical transition matrix + mean dwell times from Phase A labels.
    │           Mean dwell time → temporal_lengthscale_days + ema_halflife_days
    │           Output: regime_results/mc_<stem>_results.csv + PNGs
    │
    └── Step 5  phase_GP.run_phase_gp()             [Phase GP / Phase C]
                Sequential Kalman-filter GP over EMA-demeaned drift mu(x,t).
                Kernel: RBF(x) × Matern-3/2(t) in exact SDE state-space form.
                Outputs:
                  phase_gp_<stem>_topology.csv   — p_multiwell, barrier, Kramers per week
                  phase_gp_<stem>_forecast.csv   — topology at 1/3/7/14-day horizons
                  phase_gp_<stem>_params.csv     — learned kernel hyperparameters
                  phase_gp_<stem>_topology.png
                  phase_gp_<stem>_potential.png
```

---

## Key files

| File | Role |
|---|---|
| `RunGP.py` | Single execution entry point. All user-facing knobs are in the `CONFIGURATION` block at the top. Run with `python RunGP.py`. |
| `phase_GP.py` | All Kalman-GP logic: `KalmanGPDriftModel` class, `topology_from_gp`, `forecast_topology`, `check_topology_correlation`, `run_phase_gp`. |
| `regime_estimation.py` | Phase A: KM drift, GP potential, well detection, `run_phase_a`. |
| `markov_chain.py` | Phase B: empirical Markov chain on regime labels. |
| `data_collection.py` | Download + aggregate Binance data. |
| `plots.py` | All visualisation. |

---

## `KalmanGPDriftModel` design (phase_GP.py)

### State layout
```
state = [f_1, f'_1, f_2, f'_2, ..., f_M, f'_M]   shape (2M,)
```
`f_j` = drift function value at inducing point `z_j`.
`f'_j` = temporal derivative (from Matern SDE).
M = `n_inducing` (default 20).

### Temporal kernel
Matern 3/2 in exact SDE form. `F`, `L`, `Qc`, `P_inf` from `matern32_sde()`.
Discretised per step by `matern32_discrete()` via matrix exponential.
Block-diagonal over M inducing points (independent temporal process per point).

### Spatial kernel
RBF: `k(x, x') = spatial_var * exp(-0.5*(x-x')²/spatial_ls²)`.

### Observation model — IMPORTANT
```
r_hat(x_i) = H_i @ f + epsilon,   epsilon ~ N(0, obs_noise)
H_i = K_{x_i,Z} @ K_{ZZ}^{-1}    ← K_ZZ^{-1} cached as self._K_zz_inv
```
`_K_zz_inv` is computed once in `initialise()` with jitter `1e-6 * I`.
**Do not copy a model manually without also copying `_K_zz_inv`** — `predict()`
and `_obs_matrix()` both use it. If you build a forward-propagated model by
hand (as `forecast_topology` does), copy `_K_zz_inv` explicitly.

### Why H = K_{xZ} @ K_{ZZ}^{-1}, not K_{xZ}
`predict()` computes `mu(x) = K_{xZ} @ K_{ZZ}^{-1} @ f_mean`.
Using raw `K_{xZ}` as H inflates the effective observation magnitude ~2.9×
relative to what `predict()` reads back, causing over-aggressive Kalman gain,
P collapsing toward zero, and eventual FP instability. Fixed in this session.

### Initial state covariance P_inf
```
P_inf[f-f block]  = K_ZZ + 1e-6*I   (GP spatial prior)
P_inf[f'-f' block] = lam² * (K_ZZ + 1e-6*I)
```
`predict()` recovers K(x,x) prior variance only when P_ff starts at K_ZZ.
Previously was `spatial_var * I` (wrong — inconsistent with predict formula).

### Numerical safety nets in `update()` (all emit RuntimeWarning when triggered)
1. **Overflow early-exit**: if `m_pred` or `P_pred` is non-finite → set
   `log_lik = -inf` and break. Happens with extreme HPs during optimisation.
2. **S floor**: `s_val = max(S[0,0], 1e-15)`. Warns if triggered.
3. **P symmetrisation**: `P = (P + P.T) / 2` after every Joseph-form update.
   Warns if asymmetry > 1e-10 × max(|P|).
4. **HP optimiser penalty**: non-finite log_lik → return 1e10 (finite penalty)
   so L-BFGS-B can compute gradients. Warns with the offending parameter triple.

Warnings are **not suppressed** — they indicate genuine numerical events.
Routine P-asymmetry warnings at ~1e-10 relative scale are borderline noise;
consider raising the threshold from 1e-10 to 1e-8 if they are too frequent.

---

## Seconds interval — design note

**BM invariance**: for a fixed window length T, the total SNR for drift
estimation is `T * drift² / sigma²_per_second` — independent of `dt`.
Shorter intervals give more, noisier observations; longer give fewer, cleaner
ones; information content is identical.

**Practical implications**:
- Phase A (Kramers–Moyal): prefers **smaller dt** (30–300 s). KM is a
  finite-dt approximation with O(dt) bias; large dt causes inter-basin jumps
  within one step, blurring well structure.
- Phase GP (Kalman filter): prefers **larger dt** (1800–3600 s). Same SNR,
  smaller `obs_noise`, fewer Kalman iterations, more numerically stable.

Currently `RunGP.py` uses a single `seconds_interval` for both phases.
Splitting into `phase_a_seconds_interval` and `phase_gp_seconds_interval`
is the next logical improvement. The coupling between phases goes through
calendar-time quantities only (regime labels in weeks, temporal_ls in days,
ema_halflife in days) so the split is clean.

---

## Spatial structure — known limitation

With `spatial_ls = 0.32` and 20 inducing points over a log-price range of
~0.8 (typical weekly ETH range), K_ZZ has effective rank ~8 and condition
number ~10^30. The typical weekly x-range is ~0.10 log-price units, far less
than one spatial correlation length (0.32). This means **the spatial drift
structure is essentially unidentifiable from a single week of data** — the GP
cannot distinguish a multi-well potential from a flat one based on x-variation
alone. p_multiwell estimates are driven primarily by posterior uncertainty.

To improve spatial identifiability:
- Use longer windows (monthly or rolling multi-week)
- Reduce `spatial_ls` to match typical weekly x-excursion (~0.05–0.10)
- Or reduce `n_inducing` to match the effective spatial rank (~5–8)

---

## Topology correlation (Phase A vs Phase GP)

`check_topology_correlation(df_topology)` is called at end of `run_phase_gp`.
Computes Pearson and Spearman correlation of `p_multiwell_gp` vs `p_multiwell_a`.

On the existing 300s run (2024-12-30 to 2025-06-01, 22 weekly windows):
- Pearson r = +0.052 (p = 0.82) — statistically uncorrelated
- Spearman r = +0.180 (p = 0.42)

Root causes of poor correlation (partially addressed by H-fix and P_inf-fix
in this session; re-running will give updated numbers):
1. Prior-dominated first window before HP optimisation
2. Temporal lag: long temporal_ls (17.65 d) keeps GP elevated weeks after
   Phase A events
3. Spatial unidentifiability (see above)

---

## Existing results (pre-session, 300 s, 2024-12-30 → 2025-06-01)

Learned HPs: spatial_ls=0.322, temporal_ls=17.65 d, spatial_var=0.419,
obs_noise=6.3e-11. These were learned with the old (incorrect) observation
matrix and P_inf, so should be re-optimised.

Phase A found 4 multi-well weeks out of 22 (weeks ending 2025-01-05,
2025-01-19, 2025-03-16, 2025-05-25). Phase GP showed p_multiwell ~ 0.5–0.8
for most of Feb, which is inconsistent with Phase A (single-well throughout
Feb) — consistent with the spatial unidentifiability problem.

---

## What changed in this session

| Change | File | Why |
|---|---|---|
| Removed misleading "singularity" comment from `matern32_sde` | phase_GP.py | det(F) = λ² ≠ 0 always; comment referred to a stale wrong implementation |
| Created `RunGP.py` | RunGP.py | Pipeline execution button with all params visible |
| `topology_every_n_obs` derived, not configurable | RunGP.py | Always one snapshot per week = 7×86400/dt |
| `check_topology_correlation()` added | phase_GP.py | Pearson + Spearman, Phase A vs Phase GP p_multiwell |
| **H = K_{xZ} @ K_{ZZ}^{-1}** (was K_{xZ}) | phase_GP.py:196 | Consistency with predict(); old H inflated observations ~2.9× |
| **P_inf = K_ZZ block** (was spatial_var×I) | phase_GP.py:171–173 | predict() formula correct only when P_ff starts at K_ZZ |
| K_ZZ_inv cached in `initialise()`, reused in predict() | phase_GP.py | Single consistent inversion; predict() no longer re-solves |
| `_K_zz_inv` propagated in `forecast_topology` model copy | phase_GP.py:648 | Bug: AttributeError on predict() call; model copy was missing the attribute |
| Overflow early-exit in `update()` with RuntimeWarning | phase_GP.py:226 | HP explorer hits extreme params → inf P → cascade |
| S floor `max(S[0,0], 1e-15)` with RuntimeWarning | phase_GP.py:240 | P FP errors can make S slightly negative → log(negative) = nan |
| P symmetrisation after Joseph form with RuntimeWarning | phase_GP.py:254 | Antisymmetric FP accumulation over thousands of steps |
| HP optimiser: 1e10 penalty + RuntimeWarning on non-finite ll | phase_GP.py:446 | L-BFGS-B poisoned by nan objective; now gets finite gradient signal |
| Removed RuntimeWarning suppression from HP optimiser | phase_GP.py | User wants warnings when safety nets fire |
