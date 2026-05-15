# Handoff — eth_vol_project

Context for a fresh Claude Code conversation. The user (Jan) switches between
machines and pulls this file via git, so it must stay tracked — there is no
`*.md` entry in `.gitignore`. **EM is now in a working, validated state. The
next focus is MCMC.** Section 12 below lists concrete entry points.

## 1. Purpose of the study

End goal: a **market-instability prediction machine** for ETH/USDT. The
hypothesis is that the price process can be described as Brownian motion on a
slowly-changing **potential field** `U(x)` (with `x = log(price)`) whose
topology — number of stable wells — flips between *single-well* (trending /
one attractor) and *multi-well* (range-bound between two attractors) on a
*weekly-to-monthly* timescale. A flip from single → multi typically precedes
wider price dispersion, which is the signal we want to detect *in advance*.

Two complementary inference routes for the regime-switching SDE:

- **Phase C EM** — fast point estimates via Expectation–Maximisation
  (Hamilton filter + Kim smoother + closed-form/profiled WLS M-step).
- **Phase C MCMC** — full posterior via Forward Filter Backward Sample
  (FFBS) Gibbs, warm-started from EM. Used for credible intervals and forecast
  uncertainty. **This is where the next conversation should pick up.**

## 2. Mathematical model

Two-state hidden chain `S_t ∈ {single-well, multi-well}` selects between two
parametric drifts on log-price `x_t`:

- State 0 — OU: `μ₁(x) = −κ (x − m)`
- State 1 — cubic (Halperin-style): `μ₂(x) = −α (x − c)³ + β (x − c)`

Emission, Euler–Maruyama with step `dt = seconds_interval`:

```
Δx_t | x_{t-1}, S_t = s   ~   N( μ_s(x_{t-1}) · dt ,  σ²_s · dt )
```

`σ²` can be either shared across states or per-state — toggled by
`regime_specific_sigma2` in [RunEM.py](RunEM.py). MCMC currently still treats
σ² as a scalar (see §11 known issue).

**Unit convention.** All rate parameters (κ, α, β, σ²) are stored and computed
internally in **per-second** units. User-facing reports — console summary
tables, the κ(t) plot, the fitted-drift plot — are **annualised** via
`× SEC_PER_YEAR` (or `× sqrt(SEC_PER_YEAR)` for σ). Helpers in
[ExpectationMaximisation.py](ExpectationMaximisation.py) named
`annualize_theta()` and `sigma_annualized()`. The theta CSV stores **both**
raw and `*_ann` so MCMC can round-trip the per-second values.

## 3. File map

| File | Role |
|---|---|
| [RunEM.py](RunEM.py) | **Single-button entry point.** Runs the whole pipeline end-to-end. All knobs that affect Phase C are in its CONFIGURATION block. |
| [data_collection.py](data_collection.py) | Binance daily ETHUSDT dumps: downloads, unzips, aggregates trades into log-return bars at requested `seconds_interval`. Caches aggregated CSVs under `data/` keyed by `(start, end, interval, kernel_half_width, trim_quantile, detrend)`. Also defines `window_seconds(window_type)` used by RunEM for the dwell target. |
| [force_field_estimation.py](force_field_estimation.py) | Kramers–Moyal drift/diffusion estimator. Returns per-bin `drift` (per-second), `diffusion` (= σ²/2), `weight`. |
| [regime_estimation.py](regime_estimation.py) | **Phase A.** Per-window topology classifier using KM drift + a GP posterior over potentials. Outputs `p_multiwell ∈ [0,1]` per window. Single-interval driver — see §7. |
| [markov_chain.py](markov_chain.py) | **Phase B.** Empirical 2-state Markov chain over the per-window regime labels. Computes both MLE and Dirichlet-smoothed posterior. Console shows MLE; smoothed `pi` is returned for Phase C `pi_init`. Soft transition counts weighted by `p_multiwell` (default). |
| [ExpectationMaximisation.py](ExpectationMaximisation.py) | **Phase C — EM.** Hamilton filter + Kim smoother E-step; WLS for OU, profiled NLS for cubic, Dirichlet-prior P update in the M-step. Mean-dwell prior + GP-label tilt + regime-specific σ² + KM-pooled warm start all live here. |
| [phase_c_mcmc.py](phase_c_mcmc.py) | **Phase C — MCMC.** FFBS Gibbs with conjugate OU + MH cubic + Dirichlet P. Warm-started from EM cache. Outputs into an **interval-specific subfolder** under `phase_c_results/` — see §11. |
| [plots.py](plots.py) | All matplotlib plotting (KM potential, MC heatmap, EM γ-κ overlay, fitted drifts, MCMC posterior). |
| [handoff.md](handoff.md) | This file. |

## 4. Pipeline orchestration ([RunEM.py](RunEM.py))

`main()` runs four steps in order. The CONFIGURATION block at the top of the
file is the **single source of truth** — standalone `python regime_estimation.py`
and `python markov_chain.py` are intentionally disabled and print a redirect.

1. **Download & aggregate** — `data_collection.ensure_data(start, end)`
   downloads any missing daily zip; `data_collection.aggregate_log_returns_range`
   produces the bar series for `seconds_interval` (and a second pass for
   `theta_init_seconds_interval` if it differs). Aggregated CSVs are cached
   on disk and only rebuilt if their params change.

2. **Phase A — `regime_estimation.run_phase_a`** at one `seconds_interval`.
   For every window, run KM → fit a GP over the drift → sample from the GP
   posterior → integrate each sample into a potential → count wells with the
   greedy barrier-height filter → `p_multiwell = P(n_wells ≥ 2 | data)`.
   Writes `regime_results/regime_labels_<start>_to_<end>_<int>s_<window>.csv`
   with columns: `window_start, window_end, seconds_interval, regime,
   n_wells, p_multiwell, well_locations, barriers, u_range, n_observations`.
   Per-window KM CSVs go to `regime_results/km/km_<ws>_to_<we>_<int>s.csv`.

   When the KM warm start uses a different interval, a second Phase A pass
   runs at that interval too (cache-hit if already on disk).

3. **Phase B — `markov_chain.run_markov_chain`** at one `seconds_interval`.
   Loads the single-interval labels CSV. With `use_soft_counts=True`
   (default), every consecutive adjacent pair contributes
   `outer([1−p_mw_t, p_mw_t], [1−p_mw_{t+1}, p_mw_{t+1}])` to `N`. Returns
   *both* MLE and Dirichlet-smoothed posterior; console + plots show MLE,
   while `mc['pi']` (Dirichlet posterior stationary) is fed forward to EM.

4. **Phase C EM — `ExpectationMaximisation.run_phase_c`**. Pipeline inside:
   - `load_series` reads the cached aggregated returns CSV (trim + detrend
     already baked in) and emits `(x_prev, dx, dt, dt_t)` with cross-gap
     increments dropped (`dt > 1.5 · nominal`).
   - σ² initialised from `var(dx)/dt` (post trim/detrend by construction).
     If `regime_specific_sigma2=True`, it's a length-2 vector refined per
     state in the M-step.
   - Initial θ: KM-pooled or moments-based — see §5.
   - `P0` near-identity, `pi_init` = Phase B smoothed stationary.
   - `alpha_prior_P = build_dwell_alpha_prior(...)` adds a Dirichlet prior
     pulling `P[i,i]` toward `1 − dt / target_dwell_seconds` (default one
     window) with mass `dwell_prior_strength · len(dx)` (default 10×). This
     is what enforces the *weeks-scale* persistence at 30s ticks.
   - `build_window_assignments` maps each observation to its Phase A weekly
     window; `p_label_pairs[w] = [1−p_mw, p_mw]`. If `eta_label > 0` and
     the labels CSV lacks `p_multiwell` for the requested interval,
     `run_phase_c` raises (no silent fallbacks).
   - `run_em`: iterate `_emit → Hamilton filter → Kim smoother → m_step`.
     `_emit` calls `emission_log_b` then `augmented_log_b` to add
     `η · log p_multiwell_window(t)` to each row of `log_b`.
   - Outputs to `phase_c_results/` (flat for EM at the moment — see §11):
     `phase_c_<from>_to_<to>_<int>s_probs.csv` (filtered + smoothed),
     `_theta.csv` (raw + annualised + per-state σ²), `_loglik.csv`,
     `_kappa.csv`, `_drifts.png`, `_gamma_kappa.png`.

## 5. θ initialisation modes

**Default is now `theta_init='moments'`** (changed from `'km'` after empirical
testing). With the moments init, EM converges to a clear double-well drift in
windows that Phase A flagged as multi-well. With the KM-pooled init, EM
typically settles on a near-flat cubic in the same windows: the pooled
multi-well KM drift averages many week-scale topologies together, which
smears the would-be barrier out and leaves the cubic basin too shallow for
EM to escape the OU attractor.

[ExpectationMaximisation.fit_initial_theta_moments](ExpectationMaximisation.py#L323)
is therefore the active initialiser. It does a global OLS for `(κ, m)` and a
2-means k-means on `(x_prev, dx/dt)` to seed `(α, β, c)` from the
high-residual cluster — no dependence on Phase A.

[ExpectationMaximisation.fit_initial_theta_from_km](ExpectationMaximisation.py#L268)
is retained behind `theta_init='km'`. It also falls back to moments
automatically when the KM pools are too sparse. The cross-interval pooling
infrastructure below stays in place so future tweaks (different weighting,
within-regime sub-clustering) can be tried without further plumbing.

### KM warm start at a different sampling interval

`run_phase_c` accepts `theta_init_seconds_interval` (default 30). The KM
estimator divides by `Δt`, so the resulting drift is **per-second** regardless
of source interval — using 30s KM curves to seed a 300s EM is valid (and
typically *less* Euler-biased than 300s curves).

RunEM auto-runs a second Phase A pass at `theta_init_seconds_interval` when
it differs from `seconds_interval`, so the corresponding labels CSV +
`km/km_*_<init>s.csv` files exist for `fit_initial_theta_from_km`. The path
is passed via `theta_init_labels_csv`; without it, `run_phase_c` derives the
sibling CSV by swapping the interval token in the filename. Hard-fails with
a clear `FileNotFoundError` if the file isn't on disk.

### Weighted KM pooling

`pool_km_drift_curves(..., regime_label, ...)` pools per-window km_df rows
filtered by regime. Each row's WLS weight is `KM bin count × regime
confidence` where confidence = `p_multiwell` for the multi-well pool and
`1 − p_multiwell` for the single-well pool. That way uncertain windows
contribute proportionally and binwise KM noise is also respected.

## 6. Non-agnostic injections into EM (where the priors come from)

A vanilla Markov-switching SDE EM has none of these. We have all of them:

1. **`pi_init` from Phase B** — Dirichlet-smoothed stationary of the
   empirical weekly Markov chain.
2. **Dirichlet prior on P (mean-dwell enforcement)** — domain prior
   ("regimes flip on a weeks timescale, not every tick"). Prior mass
   ≈ 10 × T, so EM's `xi_sum + α` is dominated by the prior unless the data
   shows extremely persistent transitions.
3. **GP label tilt** — Phase A's `p_multiwell` per weekly window adds
   `η · log p_multiwell` to each row of `log_b`. Couples Phase A's
   week-scale drift-shape inference to Phase C's tick-scale dynamics.
4. **KM-pooled θ warm start** *(optional, off by default)* — Phase A's
   per-window KM drift curves, pooled by regime label, give an initial
   `(κ, m, α, β, c)`. Can use a finer interval than EM (see §5).
   Currently *not* the default after the finding that it converges to a
   flat cubic in multi-well windows; use `theta_init='km'` to re-enable.
5. **σ² from data** — initialised at `var(dx)/dt`; held fixed in shared
   mode, re-estimated per-state in regime-specific mode.

## 7. Single-interval Phase A & cross-interval naming

Phase A used to run several intervals in one CSV and then majority-vote in
Phase B. **That's gone.** Now:

- Each `run_phase_a` call handles **one** `seconds_interval`. The interval
  is part of the filename:
  `regime_labels_<start>_to_<end>_<int>s_<window_type>.csv`
- `_load_cached_windows` globs `regime_labels_*_<int>s_<window_type>.csv`,
  so different intervals are isolated.
- `markov_chain` no longer aggregates votes — one row per window, the
  `confidence` column comes from `p_multiwell` directly.
- MC artefacts are named off the labels stem (which already contains the
  interval) → `mc_regime_labels_..._<int>s_<window>_results.csv` etc.

The fast path in `run_phase_a` checks that all rows have non-NaN
`p_multiwell` before short-circuiting; the partial-cache helper drops stale
rows so `drop_duplicates(keep='last')` always favours fresh values.

## 8. Caching layers (this matters — compute is expensive)

| Layer | Where | Cache key |
|---|---|---|
| Raw Binance dumps | `data/*.zip` and `*.csv` | (symbol, date) |
| Aggregated bars | `data/ETHUSDT-aggReturns-<start>_to_<end>-<int>sec_k<kw>_trim<tq>[_detrended].csv` | (range, interval, kernel_half_width, trim_quantile, detrend) |
| Per-window KM | `regime_results/km/km_<ws>_to_<we>_<int>s.csv` | (window, interval) |
| Phase A labels | `regime_results/regime_labels_<start>_to_<end>_<int>s_<window_type>.csv` | (range, interval, window_type) |
| Phase B MC | `regime_results/mc_<labels_stem>_results.csv` | derived from labels CSV name |
| Phase C EM | `phase_c_results/phase_c_<from>_to_<to>_<int>s_*.csv/.png` | (range, interval) **flat directory** |
| Phase C MCMC | `phase_c_results/<int>/phase_c_mcmc_<from>_to_<to>_<int>s_chain.npz` | (range, interval) **interval-specific subdir** |

`phase_c_mcmc.run_phase_c_mcmc` tries hard to *not* re-run upstream:
- Phase B counts → recovered from the prior block in the MC CSV
  (`alpha = 1 + λ·N` is invertible).
- EM warm start → loaded from the cached theta + probs CSVs.
- Final chain → if the npz exists, short-circuit and reload summaries.

`phase_b_dir` defaults to the directory of the labels CSV, so Phase B
artefacts live next to Phase A and aren't duplicated in `phase_c_results/`.

## 9. Plots produced by EM

Only two PNGs come out of EM per run:

- `phase_c_<...>_drifts.png` — fitted parametric drifts μ₁(x) (OU) and μ₂(x)
  (cubic), **annualised**, with a log-price histogram below for context.
  See [plots.plot_phase_c_drifts](plots.py#L273).
- `phase_c_<...>_gamma_kappa.png` — overlay diagnostic:
  filled-area γ_t(multi-well) on the right axis [0,1], **smoothed**
  annualised κ(t) line on top (left axis). Designed to inspect whether a
  κ-decline precedes a multi-well regime. See
  [plots.plot_gamma_kappa_overlay](plots.py#L309). The standalone γ-only
  plot was removed; the standalone κ-only plot exists in plots.py but is
  no longer wired in.

`estimate_kappa_series` post-smooths the rolling-regression κ(t) with a
centred rolling mean of width `smooth_window=30` samples and exposes both
`kappa_ann` and `kappa_ann_smoothed` columns. The plot consumes the
smoothed column.

## 10. The annualisation audit

Earlier the codebase mixed per-second and annualised drifts. The current
convention (locked in across all reports):

| Place | What's shown |
|---|---|
| `force_field_estimation.estimate_km` | Per-second drift; docstring notes the conversion. Not user-facing. |
| `plots.plot_potentials_from_km_results` | Drift and vol **annualised**. |
| `regime_estimation.classify_potential_topology` | Annualises drift internally before the GP fit; `p_multiwell` is scale-invariant. |
| EM final summary table | **Annualised** κ, α, β; m, c (positions) unscaled. |
| EM volatility table | Per-state σ_ann (per-second σ² shown alongside). |
| Theta CSV | Both raw (per-second) and `*_ann` rows. Raw is what MCMC loads. |
| `estimate_kappa_series` + plot | **Annualised** column `kappa_ann` + smoothed. |
| `plot_phase_c_drifts` | **Annualised** drift, y-axis `yr⁻¹`. |

## 11. Known caveats / WIP

1. **EM vs MCMC output directories are inconsistent** ([phase_c_mcmc.py](phase_c_mcmc.py#L595)).
   The most recent commit (`59ecf70` — "improved initialisation logic and
   other fixes") moved MCMC outputs into `phase_c_results/<interval>/`.
   `_try_load_em_result` was updated to look for the EM cache in that
   subdirectory too, but [run_phase_c](ExpectationMaximisation.py#L1108)
   still writes the EM cache to `phase_c_results/` (flat). Net effect: when
   MCMC tries to warm-start from a previous EM run, the lookup misses and
   EM re-runs. Either (a) make EM write to the same per-interval subdir,
   or (b) revert the MCMC subfolder change. Worth resolving before serious
   MCMC work — see §12 item 1.
2. **MCMC σ² is still scalar** even when EM produces per-state σ². EM
   writes `sigma2 = mean(sigma2_0, sigma2_1)` to the theta CSV so MCMC's
   `_try_load_em_result` keeps working. A conjugate Inverse-Gamma update
   per state is the natural next extension — TODO marker at
   [phase_c_mcmc.py:186](phase_c_mcmc.py#L186).
3. **`window_seconds` defined in two places.** Identical implementation
   lives in both [regime_estimation.py:37](regime_estimation.py#L37) and
   [data_collection.py:280](data_collection.py#L280). RunEM uses the
   `data_collection` copy. Harmless but worth deduplicating.
4. **Legacy phase_c_* filenames.** The module was renamed to
   `ExpectationMaximisation` but output filenames, the standalone
   `__main__` stem variable, and the `phase_c_results/` directory still
   say `phase_c_`. **Don't change these** — cached artefacts depend on
   the stable naming.
5. **Regime-collapse hypothesis.** Earlier in the dev cycle the EM was
   collapsing to one regime. We traced it to two bugs: (a) a dtype
   mismatch silently zeroed the GP-tilt lookup, (b) the Phase A cache
   shadowing fresh `p_multiwell` values. Both are fixed. If you see a
   degenerate share-of-time in the EM summary after this, check the
   `"GP label tilt: eta=…, X% of observations covered…"` line in the
   console — `X` should be ≈100%. If it's not, the labels CSV's intervals
   don't match the EM call.
6. **Phase A `regime` values.** Now only `'single-well'`, `'multi-well'`,
   `'uncertain'` are accepted as "good"; the earlier `'no-equilibrium'`
   tag was removed from the cache's good-regime set in `59ecf70`.

## 12. Where to start (MCMC focus)

Pick the user's first message. Common starting points:

1. **Fix the EM/MCMC output-directory mismatch** (§11 item 1) so MCMC can
   actually short-circuit to a cached EM run instead of recomputing.
   Cleanest: have EM also write to `phase_c_results/<int>/`. Update
   handoff.md when done.
2. **Validate MCMC posterior vs EM point estimate** on a window with a
   known regime flip. The posterior means in `_summary.csv` should bracket
   the EM `_theta.csv` values. The new γ-κ overlay plot used by EM is
   *not* yet produced by MCMC — adding it (with posterior-mean γ and
   maybe a κ posterior band) is a natural addition for visual checking.
3. **Per-state σ² in MCMC** (§11 item 2). Add a fourth Gibbs block per
   sweep: conjugate Inverse-Gamma σ²_s | S, x, μ_s. Wire it through
   `run_gibbs` and into the summary table. Also update
   `_try_load_em_result` to actually use the per-state values when
   `regime_specific_sigma2=True` was used by the EM that produced the
   cache.
4. **Calibrate the dwell prior from Phase B** — currently
   `target_dwell_seconds` is set hand-tuned (default `window_seconds(window_type)`).
   We could replace it with a shrunk Phase B estimate
   `D = w · D_phaseB + (1 − w) · D_default` with a floor. Idea was
   sketched in an earlier conversation but not implemented.
5. **Dirichlet prior on MCMC P** — `draw_P` uses `alpha_prior = 1 + λ·N_PhaseB`.
   The mean-dwell Dirichlet that EM uses is a different (and stronger)
   prior. Decide whether MCMC should adopt the same dwell-based prior or
   keep the Phase-B-informed one. They're not directly comparable.

## 13. Conventions / pitfalls

- **Date snapping.** Weekly mode snaps `start_date` to the preceding Monday
  and `end_date` to the following Sunday via
  `regime_estimation.normalize_window_boundaries` — RunEM passes the
  snapped dates to every downstream stage so all caches use the same
  range string.
- **`seconds_interval` in CSVs.** Always coerce to int after `pd.read_csv`;
  don't trust the inferred dtype.
- **Soft counts ↔ irreducibility check.** The Phase B chain irreducibility
  check uses a 0.5-transition threshold to allow non-integer counts. See
  [markov_chain.check_chain](markov_chain.py#L254).
- **Standalone runs disabled.** `python regime_estimation.py` and
  `python markov_chain.py` print a redirect to RunEM and exit. They used
  to have hardcoded params that diverged from RunEM — a real footgun.
- **Old `regime_labels_*_<window_type>.csv` (no interval token).** Such
  legacy CSVs are silently ignored by the new globs. Re-run RunEM to
  populate the interval-tagged naming. Old `mc_*_<int>s_results.csv` files
  are also stale — the new MC artefacts don't have the trailing
  `_<int>s` suffix.

## 14. Status snapshot (as of commit `59ecf70`)

- EM is functional end-to-end with sensible warm starts, GP-tilted E-step,
  and a strong week-scale Dirichlet prior on P.
- Default warm start is moments-based (`theta_init='moments'`) — produces
  clear double-well drifts in identified multi-well windows. KM-based init
  is retained but does not push EM out of the OU basin on real data; it
  remains validated on synthetic drifts that match the parametric forms.
- The γ-κ overlay is the headline EM diagnostic plot.
- MCMC scaffold runs, but the EM cache lookup is broken (§11 item 1) and
  it doesn't yet use per-state σ² (§11 item 2).
- All non-Phase-C scripts (`regime_estimation`, `markov_chain`) are
  pipeline-only — RunEM is the single source of truth.
