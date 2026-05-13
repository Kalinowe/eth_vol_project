# Handoff — eth_vol_project

This document hands off project context for a fresh Claude Code conversation.
The user (Jan) often switches between machines and pulls this file via git, so
it must stay in version control — there is no .md entry in `.gitignore`.

## 1. Purpose of the study

The end goal is a **market-instability prediction machine** for ETH/USDT. The
hypothesis is that the price process can be described as Brownian motion on a
slowly-changing **potential field** `U(x)` (with `x = log(price)`), and that the
field's topology — number of stable wells — flips between *single-well*
(trending / one attractor) and *multi-well* (range-bound between two
attractors) on a *weekly-to-monthly* timescale. A flip from single → multi
typically precedes wider price dispersion, which is what we want to detect
*in advance* via early-warning signals (declining mean-reversion rate `κ(t)`,
rising emission variance, GP posterior over topology widening).

Two complementary inference routes for the regime-switching SDE:

- **Phase C EM** — fast point estimates via Expectation–Maximisation
  (Hamilton filter + Kim smoother + closed-form/profiled WLS M-step).
- **Phase C MCMC** — full posterior via Forward Filter Backward Sample
  (FFBS) Gibbs, warm-started from EM. Used for credible intervals and for
  forecast uncertainty.

> **We are currently finalising EM.** MCMC has a working scaffold but is not
> the focus of edits right now.

## 2. Mathematical model

Two-state hidden chain `S_t ∈ {single-well, multi-well}` selects between two
parametric drifts on log-price `x_t`:

- State 0 — OU: `μ₁(x) = −κ(x − m)`
- State 1 — cubic (Halperin-style): `μ₂(x) = −α(x − c)³ + β(x − c)`

Emission, Euler–Maruyama with step `dt = seconds_interval`:

```
Δx_t | x_{t-1}, S_t = s   ~   N( μ_s(x_{t-1}) · dt ,  σ²_s · dt )
```

`σ²` can be either shared across states or per-state — toggled by
`regime_specific_sigma2` in [RunEM.py](RunEM.py).

## 3. File map

| File | Role |
|---|---|
| [RunEM.py](RunEM.py) | **Single-button entry point.** Runs the whole pipeline end-to-end. All Phase C-relevant knobs are in its CONFIGURATION block. |
| [data_collection.py](data_collection.py) | Binance daily ETHUSDT dumps: downloads, unzips, aggregates trades into log-return bars at requested `seconds_interval`. Caches aggregated CSVs under `data/` keyed by `(start, end, interval, kernel_half_width, trim_quantile, detrend)`. |
| [force_field_estimation.py](force_field_estimation.py) | Kramers–Moyal drift/diffusion estimator on a 1-D log-price series. Returns per-bin `drift` (per-second), `diffusion` (= σ²/2), `weight`. |
| [regime_estimation.py](regime_estimation.py) | **Phase A.** Per-window (weekly/biweekly/monthly) topology classifier using KM drift + a Gaussian Process posterior over potentials. Outputs `p_multiwell ∈ [0,1]` per window. |
| [markov_chain.py](markov_chain.py) | **Phase B.** Empirical 2-state Markov chain over the weekly regime labels. Computes both MLE and Dirichlet-smoothed posterior. Console shows MLE; smoothed `pi` is returned for downstream Phase C use. Supports soft transition counts weighted by `p_multiwell`. |
| [ExpectationMaximisation.py](ExpectationMaximisation.py) | **Phase C — EM.** Hamilton forward filter + Kim smoother E-step; WLS for OU, profiled NLS for cubic, Dirichlet-prior P update in the M-step. Mean-dwell prior + GP-label tilt + regime-specific σ² all live here. |
| [phase_c_mcmc.py](phase_c_mcmc.py) | **Phase C — MCMC.** FFBS Gibbs with conjugate OU + MH cubic + Dirichlet P. Warm-started from EM cache. |
| [plots.py](plots.py) | All matplotlib plotting (KM potential, MC heatmap, EM gamma timeline, kappa series, forecast, MCMC posterior). |
| [handoff.md](handoff.md) | This file. |

## 4. Pipeline orchestration ([RunEM.py](RunEM.py))

`main()` runs four steps in order:

1. **Download & aggregate**
   `data_collection.ensure_data(start, end)` downloads any missing daily zip
   from Binance; `data_collection.aggregate_log_returns_range(..., kernel_half_width,
   trim_quantile, detrend)` produces the bar series for *each* interval in
   `seconds_intervals_phase_a` plus the EM target `seconds_interval`. Aggregated
   CSVs are cached on disk and only rebuilt if their params change.

2. **Phase A — `regime_estimation.run_phase_a`**
   For every (window, interval) pair, run KM → fit a GP over the drift → sample
   from the GP posterior → integrate each sample into a potential → count
   wells using `_greedy_well_filter` with a barrier-height filter →
   `p_multiwell = P(n_wells ≥ 2 | data)`. Writes
   `regime_results/regime_labels_<start>_to_<end>_<window_type>.csv` with
   columns: `window_start, window_end, seconds_interval, regime, n_wells,
   p_multiwell, well_locations, barriers, u_range, n_observations`.
   Per-window KM CSVs go to `regime_results/km/`.

   Cache strategy is layered:
   - **Fast path**: existing `regime_labels_*.csv` with non-NaN `p_multiwell`
     for every "good" regime row → load and return.
   - **Partial cache**: `_load_cached_windows` collects rows from *other*
     existing CSVs in `regime_results/` that fall within the requested date
     range and have `p_multiwell`. Only windows missing all intervals
     get recomputed.
   - The final merge uses `drop_duplicates(keep='last')` so freshly computed
     rows always shadow stale ones — fixes a subtle bug where NaN
     `p_multiwell` rows from older runs would persist.

3. **Phase B — `markov_chain.run_markov_chain`**
   Loads the Phase A labels CSV, collapses multi-interval votes to one
   regime per window (`aggregate_label_votes`), then builds the transition
   count matrix `N`. With `use_soft_counts=True` (default), every consecutive
   adjacent pair contributes `outer([1−p_mw_t, p_mw_t], [1−p_mw_{t+1}, p_mw_{t+1}])`
   to `N`, so uncertain windows split mass across all four cells. Two
   matrices are computed:
   - **MLE** `A_mle = N / row_sums` — what the console + plots show
     (matches the *"Empirical Markov Chain"* table title).
   - **Posterior** `A_map = (N + α) / row_sums` with
     `α = 1 + prior_lambda · prior_counts` (uniform add-one by default).
   The returned dict carries *both*; `mc['pi']` is the Dirichlet-smoothed
   stationary, used as `pi_init` for EM.

4. **Phase C EM — `ExpectationMaximisation.run_phase_c`**
   Pipeline inside the function:
   1. `load_series` reads the cached aggregated returns CSV (after trim +
      detrend) and emits `(x_prev, dx, dt, dt_t)` with cross-gap increments
      dropped (`dt > 1.5 · nominal`).
   2. σ² initialised from `var(dx)/dt` — **note: by construction this is
      already post trim/detrend** because `load_series` reads the
      trim/detrend-tagged cached file. If `regime_specific_sigma2=True`,
      it's a length-2 vector `[var(dx)/dt, var(dx)/dt]` (refined per state
      in the M-step).
   3. Initial θ from data: `fit_initial_theta_moments` does a global OLS for
      `(κ, m)` then 2-means k-means on `(x_prev, dx/dt)` to seed `(α, β, c)`
      from the high-residual cluster. *Does not use Phase A/B.*
   4. `P0` near-identity, `pi_init` = Phase B smoothed stationary.
   5. `alpha_prior_P = build_dwell_alpha_prior(...)` — Dirichlet
      pseudo-counts pulling `P[i,i]` toward `1 − dt / target_dwell_seconds`
      (default one week) with mass `dwell_prior_strength · len(dx)`
      (default 10×). This is what enforces the *weeks-scale* persistence.
   6. `build_window_assignments` maps every observation to its Phase A
      weekly window; `p_label_pairs[w] = [1 − p_mw, p_mw]`. Errors out
      with a clear message if `eta_label > 0` but the CSV lacks
      `p_multiwell` for the requested `seconds_interval`.
   7. `run_em`: iterate `_emit → Hamilton filter → Kim smoother → m_step`
      until `|Δlog-lik| / max(|prev_ll|, 1) < tol` or `max_iter` hit.
      `_emit` calls `emission_log_b` then `augmented_log_b` to add
      `η · log p_multiwell_window(t)` to each row of `log_b`.
   8. Outputs to `phase_c_results/`:
      `phase_c_<from>_to_<to>_<int>s_probs.csv` (filtered + smoothed),
      `_theta.csv` (raw + annualised + per-state σ²),
      `_loglik.csv`, `_forecast.csv`, `_kappa.csv`,
      `_gamma.png`, `_kappa.png`, `_forecast_k5.png`.
      Console prints the final-parameters table (annualised) and a
      separate volatility table tagged `[regime-specific]` or `[shared]`,
      plus a share-of-time table from `mean(γ)` and `mean(p_filt)`.

## 5. RunEM.py knobs that affect Phase C outcome

Grouped in [RunEM.py](RunEM.py)'s CONFIGURATION block:

| Group | Knob | Effect |
|---|---|---|
| Date | `start_date`, `end_date` | window of analysis (snapped to weekly Monday/Sunday boundaries) |
| Data agg / KM | `seconds_interval` | bar size used inside EM |
| | `seconds_intervals_phase_a` | bar sizes scanned by Phase A's GP topology classifier |
| | `kernel_half_width` | smoothing of log-prices in the aggregator |
| | `trim_quantile` | symmetric tail trim on log-prices |
| | `n_bins`, `weight_threshold` | KM histogram bins / minimum bin count |
| | `detrend` | 1 = linear-detrend log-prices before KM |
| Phase A topology | `window_type` | weekly / biweekly / monthly |
| | `min_barrier_fraction`, `min_well_separation` | well-counting filters |
| Phase B MC | `prior_lambda` | strength of Dirichlet prior on transitions |
| | `use_soft_counts` | weight transitions by `p_multiwell` |
| Phase C EM convergence | `max_iter`, `tol`, `epsilon_P` | standard EM controls |
| | `sigma2` | None → estimate from data |
| | `regime_specific_sigma2` | False → shared σ²; True → per-state σ² re-estimated in M-step |
| Phase C priors | `target_dwell_seconds` | mean dwell the prior pulls toward (default 7 days) |
| | `dwell_prior_strength` | prior mass = strength · len(dx); larger = stricter |
| | `eta_label` | strength of the Phase A GP `p_multiwell` tilt on log_b |

## 6. Non-agnostic injections into EM (where the priors come from)

A "vanilla" Markov-switching SDE EM has none of these. We have all of them:

1. **`pi_init` from Phase B** — Dirichlet-smoothed stationary of the empirical
   weekly Markov chain. Without it the Hamilton filter would default to
   uniform `[0.5, 0.5]`.
2. **Dirichlet prior on P (mean-dwell enforcement)** — domain prior
   ("regimes flip on a weekly timescale, not every 30s"). Massive: prior
   pseudo-count `≈ 10 · T`, so the EM M-step's `xi_sum + α` is dominated by
   the prior unless the data shows extremely persistent transitions.
3. **GP label tilt** — Phase A's `p_multiwell` per weekly window is added
   (in log-space, with weight `η`) to each row of `log_b`. Couples Phase A
   (which sees the *shape* of the drift across a week) to Phase C (which
   sees ticks).
4. **Initial θ from data clustering** — not really a "prior" but not
   agnostic either: k-means picks an initial multi-well cluster from the
   residuals against the global OU fit.
5. **σ² fixed (or per-state) at `var(dx)/dt`** — not re-estimated each M-step
   in shared mode.

A dormant alternative (`fit_initial_theta_from_km`) would seed θ from
pooled Phase A KM drift curves — currently unused; the `annualize=False`
flag was added to it during the annualisation audit so a future re-enable
won't surprise anyone.

## 7. Caching strategy (this matters — compute is expensive)

The codebase deliberately avoids redoing anything it can avoid. Key cache
layers:

| Layer | Where | Cache key |
|---|---|---|
| Raw Binance dumps | `data/*.zip` and `*.csv` | (symbol, date) |
| Aggregated bars | `data/ETHUSDT-aggReturns-<start>_to_<end>-<interval>sec_k<kw>_trim<tq>[_detrended].csv` | (range, interval, kernel_half_width, trim_quantile, detrend) |
| Per-window KM | `regime_results/km/km_<ws>_to_<we>_<interval>s.csv` | (window, interval) |
| Phase A labels | `regime_results/regime_labels_<start>_to_<end>_<window_type>.csv` | (range, window_type) |
| Phase B MC results | `regime_results/mc_<labels_stem>_<int>s_results.csv` | derived from labels CSV name |
| Phase C EM artefacts | `phase_c_results/phase_c_<from>_to_<to>_<int>s_*.csv/.png` | (range, interval) |
| Phase C MCMC chain | `phase_c_results/phase_c_mcmc_<from>_to_<to>_<int>s_chain.npz` | (range, interval) |

For partial caches in Phase A, `_load_cached_windows` collects rows from
*all* existing labels CSVs in `regime_results/` that fall within the
requested range, and only the windows missing entirely (or missing
`p_multiwell`) get recomputed.

`phase_c_mcmc.run_phase_c_mcmc` tries hard to *not* re-run upstream:
- Phase B counts → recovered from the prior block in the MC CSV
  (`alpha = 1 + λ·N` is invertible).
- EM warm start → loaded from the theta + probs CSVs.
- Final chain → if the npz exists and the user hasn't disabled it,
  short-circuit and just reload summaries.

`phase_b_dir` defaults to the directory of the labels CSV, so Phase B
artefacts always live next to Phase A and aren't duplicated when Phase C
writes to a different `output_dir` (`phase_c_results/`).

## 8. Annualisation convention

EM works internally in **per-second** units (κ, α, β, σ² all per second
because `dt` is in seconds). User-facing reports are **annualised**
(`× SEC_PER_YEAR` for rates, `× sqrt(SEC_PER_YEAR)` for σ):

- Console summary tables — annualised columns.
- `estimate_kappa_series` output CSV column → `kappa_ann`.
- `plot_kappa_series` plots `kappa_ann` with y-label `κ_ann (yr⁻¹)`.
- Theta CSV carries **both** raw (for round-trip with MCMC) and
  `*_ann` diagnostic rows.

Helpers in [ExpectationMaximisation.py](ExpectationMaximisation.py):
`annualize_theta(theta)` and `sigma_annualized(sigma2)`.

## 9. Most recent problem & what was fixed

The EM was producing chains that collapsed almost entirely into one
regime (state 1, "multi-well"). Suspected cause: the Phase A GP label
tilt was *silently* not being applied because `run_phase_c` was hitting
a yellow-warning path that swallowed the failure and ran EM with `eta=0`
effectively. Two failure modes were collapsed into one warning:

1. The labels CSV `seconds_interval` column had a dtype mismatch
   (string/float in the cached CSV vs the `int` `seconds_interval`
   argument) → the row filter wiped the dataframe.
2. Older CSVs lacked `p_multiwell` entirely (column never written).

The current code in [run_phase_c](ExpectationMaximisation.py#L840):
- coerces `seconds_interval` to `Int64` before filtering
- distinguishes "column missing" / "no rows for this interval" /
  "all-NaN" as separate `ValueError`s
- raises (not warns) when `eta_label > 0` and labels aren't usable
- the user opted **against** a "fallback to nearest interval" — phase_c
  errors out hard if the requested interval isn't in the CSV. The
  message tells them which intervals *are* available.

**The single-regime collapse hypothesis still needs validation** by
running EM on the same window with the new code and checking that the
GP tilt covers (≫0%) of observations (the new console line:
`"  GP label tilt: eta=…, X% of observations covered by a Phase A
window."`). If γ_t is still pinned to one state, the next thing to
investigate is whether the **dwell prior** is too strong — at default
`prior_strength=10` it dominates the data by 10:1; lower it to 1.0 and
see if the chain starts mixing.

## 10. Recently resolved bugs worth remembering

- **`p_multi` empty in regime table** — *two* bugs: (a)
  `_load_cached_windows` returned `cached_df` (with stale NaN rows)
  instead of `valid_rows`; (b) the merge then used
  `drop_duplicates(keep='first')`, favouring stale over fresh. Both
  fixed; the merge is now `keep='last'`.
- **sklearn ConvergenceWarning** in the GP fit — widened `WhiteKernel`
  bounds to `(1e-10, 1e3)` and wrapped the fit in
  `warnings.catch_warnings()`. The per-bin α already absorbs noise so
  the warning was benign.
- **Markov chain "empirical" tables actually showed posterior** — split:
  console + plots + saved CSV show MLE; `pi`, `A`, `dwells` in the
  returned dict remain Dirichlet-smoothed for Phase C consumers; new
  keys `A_mle`, `pi_mle`, `dwells_mle` are exposed for explicit access.

## 11. Conventions / pitfalls

- **Date snapping.** Weekly mode snaps `start_date` to the preceding Monday
  and `end_date` to the following Sunday via
  `regime_estimation.normalize_window_boundaries` — RunEM passes the
  snapped dates to every downstream stage, so all caches use the same
  range string.
- **`seconds_interval` in CSVs.** Always coerce to int after `pd.read_csv`;
  don't trust the inferred dtype.
- **Soft counts ↔ irreducibility check.** The Phase B chain irreducibility
  check now uses a 0.5-transition threshold (because soft counts are
  non-integer) — see [markov_chain.check_chain](markov_chain.py#L270).
- **MCMC σ²** is still scalar even when EM is regime-specific. EM writes
  `sigma2 = mean(sigma2_0, sigma2_1)` to the theta CSV so MCMC's
  `_try_load_em_result` keeps working. A proper per-state σ² in MCMC
  (conjugate Inverse-Gamma) is the natural next extension — there is a
  TODO at the top of [phase_c_mcmc.py](phase_c_mcmc.py#L186).
- **"phase_c" string references.** The module was renamed to
  `ExpectationMaximisation` (file: [ExpectationMaximisation.py](ExpectationMaximisation.py))
  but legacy strings still exist:
  - Output filenames are still `phase_c_*.csv/.png` (kept stable so
    cached artefacts aren't invalidated)
  - Output directory `phase_c_results/`
  - Internal stem `phase_c_<from>_to_<to>_<int>s`
  These are not bugs — leave them alone.

## 12. Where to start (for the next conversation)

Pick the user's first message. Common starting points:

1. **Validate the GP-tilt fix** — run `python RunEM.py` and confirm the
   console line `"  GP label tilt: eta=0.3, X% of observations covered…"`
   appears with `X` close to 100%. Look at the share-of-time table at the
   end — if it's not 50/50-ish for a healthy mixed-regime window, the
   collapse may be a separate issue (dwell prior too strong, etc.).
2. **MCMC bring-up** — the EM is in good shape now; the natural next step
   is to validate MCMC posteriors match EM point estimates on the same
   window, then add per-state σ² in MCMC (open TODO).
3. **Calibrate dwell prior from Phase B** — the user asked about replacing
   the hand-set `target_dwell_seconds` with a shrunk Phase B estimate
   (`D = w·D_phaseB + (1−w)·D_default`, with a floor). Idea was sketched
   but not implemented; see the discussion turn before the
   ExpectationMaximisation rename.
