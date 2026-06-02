"""
backtest_jumps.py — self-contained backtest of Kalman-GP topology signals as
predictors of ETH price well-jumps.

STRUCTURE
---------
There is a single continuous GP run from BURN_IN_START to BACKTEST_END:

  [BURN_IN_START, BACKTEST_START)  — burn-in: model warms up, no topology
                                     recorded, no events classified.
  [BACKTEST_START, BACKTEST_END]   — backtest: topology recorded every day,
                                     well-jumps detected, signals evaluated.

The same daily_replay loop runs over the entire period; BACKTEST_START is
passed as record_from so burn-in days advance model state but produce no
output.  There is no separate train/eval split and no model state reset at
BACKTEST_START.

GP INITIALISATION
-----------------
  • KM (Phase A) runs on KM_INIT_MONTHS of data ending the day before
    BURN_IN_START.  This period is used ONLY to compute spatial_var for the
    GP prior; the GP itself does not see this data.
  • SPATIAL_VAR_MODE = "km"    → spatial_var = Var[KM annualised drift bins];
                                 data download + aggregation guaranteed by the
                                 pipeline so this never silently falls back.
  • SPATIAL_VAR_MODE = "fixed" → spatial_var = SPATIAL_VAR_FIXED (constant);
                                 KM init steps are skipped entirely.
  • sigma2 = Var[r_hat·dt]/dt − spatial_var, estimated over the burn-in;\n    subtracting spatial_var removes the drift-variation component so that\n    sigma2 reflects pure observation noise rather than regime-to-regime drift.
  • sl_init = (x_hi - x_lo) / (3 * N_INDUCING) over the KM init x-range
  • KM CSVs are written to a backtest-local subdirectory (REGIME_OUTPUT_DIR)
    and are not shared with RunGP.py.

DATA-LEAKAGE GUARANTEE
-----------------------
1. The GP runs strictly forward in time; topology at day d uses data ≤ d only.
2. Jump labels use a price-only geometric criterion (rolling range + displacement
   thresholds) with no reference to any topology signal.
3. The backtest analysis reads only from the two CSV files written by the GP run.

SIGNALS TESTED
--------------
  p_multiwell           — level: fraction of posterior samples showing ≥2 wells
  barrier_snr           — level: barrier_mean / barrier_std (uncertainty-adj.)
  slope_p_multiwell     — trend: OLS slope of p_multiwell over TREND_WINDOW days
  slope_z_p_multiwell   — trend: OLS z-stat of p_multiwell over TREND_WINDOW days

OUTPUTS  (all under BACKTEST_DIR / {backtest_tag}/)
-------
  daily_topology.csv    — day-by-day topology across [BACKTEST_START, BACKTEST_END]
  events.csv            — detected well-jump events
  per_event.csv         — signal values at each (event, offset) pair
  null_samples.csv      — same schema for null (calm) dates
  summary.csv           — per-signal Mann-Whitney stats + rank-biserial IC vs null
  per_signal_boxes.png  — boxplots: pre-jump vs null at each look-back offset
  events_overview.png   — price + signal panels with jump markers
"""

from __future__ import annotations

import gc
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy.stats import mannwhitneyu

import data_collection as dc
from data_collection import load_series, ema_demean_drift
from phase_GP import KalmanGPDriftModel, daily_replay, _SEC_PER_YEAR
from regime_estimation import (
    iter_windows,
    run_phase_a,
)
from RunGP import compute_km_spatial_var


# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Backtest period ---------------------------------------------------------
# Burn-in: GP runs but topology is NOT recorded and events are NOT classified.
BURN_IN_START = datetime(2025, 1, 1)  # first day the GP runs
BACKTEST_START = datetime(2025, 7, 1)  # first day topology is recorded
BACKTEST_END = datetime(2026, 3, 31)  # last day

# --- KM initialisation data --------------------------------------------------
# Phase A (KM) runs on this period to estimate spatial_var for the GP prior.
# It ends the day before BURN_IN_START; the GP itself never sees this data.
KM_INIT_MONTHS = 12  # months of history to use for KM
# KM_INIT_END and KM_INIT_START are derived in code from BURN_IN_START.

# --- Phase A (KM) ------------------------------------------------------------
PHASE_A_SI = 30  # aggregation interval for KM estimation (seconds)
KERNEL_HW_PHASE_A = 3  # kernel half-width for Phase A
N_BINS = 200  # KM histogram bins
WEIGHT_THRESHOLD = 5  # min weight for KM bin to be included
MIN_BARRIER_FRACTION = 0.1  # min barrier / well-depth fraction (used in daily_replay)
# KM output dir — resolved to {BACKTEST_DIR}/{tag}/regime_results/ at runtime
# so it is always backtest-local and never shared with RunGP.py.
REGIME_OUTPUT_DIR = "regime_results"  # relative sub-path appended to out_dir

# --- GP config ---------------------------------------------------------------
PHASE_GP_SI = 900  # aggregation interval for the GP (seconds)
KERNEL_HW = 100  # kernel half-width for 900 s aggregation
TRIM_QUANTILE = 0.01  # trim extreme micro-returns
N_INDUCING = 10  # inducing points
TEMPORAL_LS_DAYS = 6.0  # Matern temporal lengthscale (days)
SPATIAL_LENGTHSCALE_INIT = None  # None → (x_hi-x_lo)/(3*N_INDUCING)
# spatial_var mode: "km" = estimate from Phase A KM bins (preferred);
#                   "fixed" = use SPATIAL_VAR_FIXED constant (skip KM)
SPATIAL_VAR_MODE = "km"
SPATIAL_VAR_FIXED = 300.0  # used only when SPATIAL_VAR_MODE == "fixed"
SIGMA2 = None  # None → var(r_hat*dt)/dt from calm periods within burn-in data
USE_REPROJECT = True
REPROJECT_WINDOW_DAYS = 30  # trailing days for rolling x-range reproject

# --- EMA demeaning -----------------------------------------------------------
EMA_HALFLIFE_DAYS = 30  # None → no demeaning; set e.g. 14.0 to enable

# --- Topology ----------------------------------------------------------------
N_GRID = 200
N_SAMPLES = 120
MIN_CROSSING_SEP = 10
SEED = 42

# --- Well-jump detection (price-geometric; no topology used here) -------------
STABLE_DAYS = 4  # rolling-window size to assess price stability
STABLE_THR = 0.05  # log-price range < this → "stable"
JUMP_THR = 0.1  # displacement from anchor > this → "jumping"
SETTLE_DAYS = 4  # consecutive stable days needed to declare new well

# --- Backtest analysis -------------------------------------------------------
OFFSETS = [-2, -1]  # days before jump_start to sample signals
TREND_WINDOW = 10  # look-back for slope-z statistics (must be ≥ 4 for _slope_z)
NULL_BUFFER_DAYS = 10  # calm date must be ≥ this far from any jump
NULL_SAMPLE_SIZE = 200  # null draws per offset

# --- Output ------------------------------------------------------------------
BACKTEST_DIR = "backtests"


# =============================================================================
# SIGNALS
# =============================================================================

SIGNALS = [
    "p_multiwell",
    "barrier_snr",
    "slope_p_multiwell",
    "slope_z_p_multiwell",
]

ALT_HYPOTHESIS = {
    "p_multiwell": "higher",  # more credible second well → instability
    "barrier_snr": "lower",  # shrinking signal/noise → barrier fading
    "slope_p_multiwell": "higher",  # p_multiwell raw slope positive → trending up
    "slope_z_p_multiwell": "higher",  # p_multiwell trending up
}


# =============================================================================
# GP RUN + BACKTEST REPLAY
# =============================================================================


def _run_gp_and_record(console: Console, rng, out_dir: str) -> tuple[str, str]:
    """Run the Kalman-GP over [BURN_IN_START, BACKTEST_END] as a single loop.

    Burn-in [BURN_IN_START, BACKTEST_START) advances the model state but does
    not record topology and is not used for jump classification.

    KM (Phase A) is run on [KM_INIT_START, KM_INIT_END] — one year of data
    ending the day before BURN_IN_START — solely to compute spatial_var for
    the GP prior.  The GP itself never receives those observations.

    Writes:
      {out_dir}/daily_topology.csv   (BACKTEST_START − TREND_WINDOW days →
                                      BACKTEST_END; the extra leading rows
                                      serve as look-back for slope signals)
      {out_dir}/events.csv           (events detected in [BACKTEST_START,
                                      BACKTEST_END] only)

    Returns (daily_path, events_path).
    """
    from dateutil.relativedelta import relativedelta

    km_init_end = BURN_IN_START - pd.Timedelta(days=1)
    km_init_start = BURN_IN_START - relativedelta(months=KM_INIT_MONTHS)
    # Regime output dir is placed under the backtest output dir so it is
    # never shared with RunGP.py and the run_phase_a cache fast-path never
    # masks missing km CSVs from a different run.
    regime_output_dir = os.path.join(out_dir, REGIME_OUTPUT_DIR)
    console.print(
        f"[bold]Backtest run[/bold]  "
        f"KM init=[{km_init_start.date()}, {km_init_end.date()}]  "
        f"burn-in=[{BURN_IN_START.date()}, {BACKTEST_START.date()})  "
        f"backtest=[{BACKTEST_START.date()}, {BACKTEST_END.date()}]  "
        f"→ {out_dir}/"
    )

    # ---- Step 1: download ---------------------------------------------------
    console.rule("[bold cyan]Step 1 — Download raw data")
    dc.ensure_data(km_init_start, BACKTEST_END)

    # ---- Step 2: aggregate 900s GP returns (per monthly window) -------------
    # The filename encodes KERNEL_HW so changing it invalidates the cache and
    # forces re-aggregation.  load_series in Step 5 reads these files directly.
    console.rule("[bold cyan]Step 2 — Aggregate 900 s log-returns")
    for _w_start, _w_end in iter_windows(BURN_IN_START, BACKTEST_END, "monthly"):
        dc.aggregate_log_returns_range(
            _w_start,
            _w_end,
            PHASE_GP_SI,
            kernel_half_width=KERNEL_HW,
            trim_quantile=TRIM_QUANTILE,
        )

    # ---- Steps 3-4: KM init (only when SPATIAL_VAR_MODE == "km") -----------
    if SPATIAL_VAR_MODE not in ("km", "fixed"):
        raise ValueError(
            f"SPATIAL_VAR_MODE must be 'km' or 'fixed', got {SPATIAL_VAR_MODE!r}"
        )

    if SPATIAL_VAR_MODE == "km":
        # run_phase_a writes per-window KM CSVs to {regime_output_dir}/km/ —
        # that is the only output consumed below.  The return value is discarded.
        # regime_output_dir is backtest-local so the KM CSV cache never masks
        # missing files from a prior run elsewhere.
        # analyze_window (called internally) handles data download and 30s
        # aggregation per window, so no explicit pre-aggregation is needed.
        console.rule(
            f"[bold cyan]Step 3 — Phase A (KM) on init period "
            f"[{km_init_start.date()} → {km_init_end.date()}]"
        )
        run_phase_a(
            km_init_start,
            km_init_end,
            PHASE_A_SI,
            kernel_half_width=KERNEL_HW_PHASE_A,
            trim_quantile=TRIM_QUANTILE,
            n_bins=N_BINS,
            weight_threshold=WEIGHT_THRESHOLD,
            window_type="monthly",
            output_dir=regime_output_dir,
            console=console,
        )

        # ---- Step 4: prepare KM window list --------------------------------
        console.rule("[bold cyan]Step 4 — Prepare KM window list and x_range")
        km_windows = list(iter_windows(km_init_start, km_init_end, "monthly"))
        km_dir = os.path.join(regime_output_dir, "km")
        console.print(f"  {len(km_windows)} monthly windows over KM init period")
        # Load 30s log-prices from the KM init period purely to derive the
        # x_range used to filter KM drift bins when computing spatial_var.
        x_km, _, _, _ = load_series(
            km_init_start,
            km_init_end,
            PHASE_A_SI,
            kernel_half_width=KERNEL_HW_PHASE_A,
            trim_quantile=TRIM_QUANTILE,
            window_type="monthly",
        )
        x_lo_km = float(np.percentile(x_km, 1))
        x_hi_km = float(np.percentile(x_km, 99))
        x_range_km = (x_lo_km, x_hi_km)
        del x_km
        console.print(f"  x_range(KM init)={x_range_km}")

        sp_var = compute_km_spatial_var(
            km_dir,
            km_windows,
            PHASE_A_SI,
            x_range=x_range_km,
            kernel_half_width=KERNEL_HW_PHASE_A,
            trim_quantile=TRIM_QUANTILE,
            console=console,
        )
        if sp_var is None:
            raise RuntimeError(
                f"compute_km_spatial_var found no usable KM CSVs in {km_dir} "
                "after run_phase_a completed. Check that analyze_window succeeded "
                "for at least one window in the KM init period."
            )
        if sp_var < 1.0:
            raise RuntimeError(
                f"KM spatial variance is degenerate ({sp_var:.4g} /yr\u00b2). "
                "Check Phase A parameters or the quality of the KM init period data."
            )
    else:  # "fixed"
        sp_var = SPATIAL_VAR_FIXED
        console.print(
            f"  [yellow]SPATIAL_VAR_MODE='fixed': using constant spatial_var={sp_var}[/yellow]"
        )
    console.print(f"  spatial_var={sp_var:.4g}")

    # ---- Step 5: load 900 s series for the full GP run ----------------------
    console.rule(
        f"[bold cyan]Step 5 — Load 900 s series "
        f"[{BURN_IN_START.date()} → {BACKTEST_END.date()}]"
    )
    x_all, dx_all, dt_step, dt_t_all = load_series(
        BURN_IN_START,
        BACKTEST_END,
        PHASE_GP_SI,
        kernel_half_width=KERNEL_HW,
        trim_quantile=TRIM_QUANTILE,
        window_type="monthly",
    )
    dt_t_pd = pd.to_datetime(dt_t_all)
    r_hat_all = (dx_all / dt_step) * _SEC_PER_YEAR
    if EMA_HALFLIFE_DAYS is not None:
        r_hat_all, _ = ema_demean_drift(
            r_hat_all, dt_t_pd, halflife_days=EMA_HALFLIFE_DAYS
        )
        console.print(f"  EMA demeaning applied: halflife={EMA_HALFLIFE_DAYS}d")
    # x_range for initial inducing grid — derived from burn-in only.
    # reproject_to_range keeps the grid current throughout the replay.
    burnin_mask = dt_t_pd < pd.Timestamp(BACKTEST_START)
    x_burnin = x_all[burnin_mask]
    x_lo_g = float(np.percentile(x_burnin, 1))
    x_hi_g = float(np.percentile(x_burnin, 99))
    x_range_global = (x_lo_g, x_hi_g)
    console.print(
        f"  N={len(x_all)}  dt={dt_step:.0f}s  x_range(inducing init)={x_range_global}  "
        f"r_hat mean={r_hat_all.mean():+.3e}/yr  std={r_hat_all.std():.3e}/yr"
    )

    # ---- Step 6: initialise model -------------------------------------------
    console.rule("[bold cyan]Step 6 — Initialise Kalman-GP model")
    sl_init = (
        SPATIAL_LENGTHSCALE_INIT
        if SPATIAL_LENGTHSCALE_INIT is not None
        else (x_hi_g - x_lo_g) / N_INDUCING
    )
    r_hat_burnin = r_hat_all[burnin_mask]  # burnin_mask computed in Step 5
    if SIGMA2 is not None:
        sigma2 = SIGMA2
    else:
        # Euler-Maruyama: r_hat * dt ~ N(mu*dt, sigma2*dt), so
        #   Var[r_hat * dt] / dt  =  sigma2  +  Var[mu(x,t)]
        # During calm periods the price range is narrow so the visited x-values
        # span a small slice of the drift field — Var[mu(x,t)] is inherently
        # small there, giving a cleaner obs-noise estimate even before subtracting
        # sp_var.  We use the same price-range criterion as jump detection.
        # Fall back to the full burn-in if fewer than 300 calm obs are found
        # (too short a burn-in or an unusually volatile warm-up period).
        _burnin_dates = dt_t_pd[burnin_mask]
        _daily_lp = (
            pd.Series(x_all[burnin_mask], index=_burnin_dates)
            .groupby(pd.Grouper(freq="D"))
            .median()
            .dropna()
        )
        _roll_range = _daily_lp.rolling(STABLE_DAYS, min_periods=STABLE_DAYS).apply(
            lambda w: float(w.max() - w.min()), raw=True
        )
        _calm_day_set = set(_roll_range[_roll_range < STABLE_THR].index.normalize())
        _calm_obs_mask = _burnin_dates.normalize().isin(_calm_day_set)
        r_hat_calm = r_hat_burnin[_calm_obs_mask]
        _MIN_CALM_OBS = 300
        if len(r_hat_calm) >= _MIN_CALM_OBS:
            raw_sigma2 = float(np.var(r_hat_calm * dt_step) / dt_step)
            _sigma2_method = (
                f"calm burn-in ({len(r_hat_calm)} obs, {len(_calm_day_set)} days)"
            )
        else:
            raw_sigma2 = float(np.var(r_hat_burnin * dt_step) / dt_step)
            _sigma2_method = (
                f"full burn-in ({len(r_hat_burnin)} obs; "
                f"only {len(r_hat_calm)} calm obs < {_MIN_CALM_OBS})"
            )
        sigma2 = max(raw_sigma2 - sp_var, 0.01 * raw_sigma2)
        console.print(
            f"  sigma2 estimation [{_sigma2_method}]: "
            f"raw={raw_sigma2:.4g}  sp_var={sp_var:.4g}  corrected={sigma2:.4g}"
        )
    obs_noise = sigma2 / dt_step
    gp_snr = sp_var / obs_noise
    # GP_SNR = spatial_var / obs_noise = (prior drift amplitude²) / (per-obs noise).
    # Financial drift has inherently low SNR at intraday frequencies, so values
    # well below 1 are expected.  However, very small SNR (≲ 0.05) means the
    # Kalman update barely moves the posterior from the prior, topology signals
    # will be structurally suppressed, and barrier_snr will have little dynamic
    # range.  If GP_SNR is unexpectedly low, consider whether spatial_var is
    # under-estimated (check KM drift bins) or sigma2 is inflated by a volatile
    # burn-in period.
    snr_msg = (
        f"  spatial_var={sp_var:.4g}  sigma2={sigma2:.4g}  "
        f"obs_noise={obs_noise:.4g}  GP_SNR={gp_snr:.4g}"
    )
    if gp_snr < 0.05:
        console.print(
            f"[bold red]{snr_msg}[/bold red]\n"
            "  [red]WARNING: GP_SNR < 0.05 — posterior nearly identical to prior; "
            "topology signals will have very low dynamic range.[/red]"
        )
    elif gp_snr < 0.2:
        console.print(
            f"[yellow]{snr_msg}[/yellow]\n"
            "  [yellow]NOTE: GP_SNR < 0.2 — observation noise dominates; "
            "topology signals may have limited dynamic range.[/yellow]"
        )
    else:
        console.print(snr_msg)
    model = KalmanGPDriftModel(
        spatial_lengthscale=sl_init,
        temporal_lengthscale_days=TEMPORAL_LS_DAYS,
        spatial_variance=sp_var,
        sigma2=sigma2,
        dt=float(dt_step),
    )
    model.initialise(x_range=x_range_global, n_inducing=N_INDUCING, data_x=x_burnin)
    console.print(
        f"  M={model.M}  sl={model.spatial_ls:.4f}  "
        f"tls={model.temporal_ls:.2f}d  sp_var={model.spatial_var:.4g}"
    )

    # ---- Step 7: single daily_replay loop over burn-in + backtest -----------
    console.rule(
        f"[bold cyan]Step 7 — GP replay  "
        f"burn-in [{BURN_IN_START.date()}, {BACKTEST_START.date()})  "
        f"+ backtest [{BACKTEST_START.date()}, {BACKTEST_END.date()}]"
    )
    # Record topology for the entire burn-in period so slope-z signals have
    # the maximum available look-back on every backtest day, and so burn-in
    # topology can be inspected post hoc.  Jump detection is still restricted
    # to [BACKTEST_START, BACKTEST_END] (enforced below).
    topology_start = pd.Timestamp(BURN_IN_START)
    console.print(
        f"  [dim]Topology recording from burn-in start ({topology_start.date()}); "
        "jump detection restricted to BACKTEST_START.[/dim]"
    )
    rows = []
    for d, topo_d, x_d in daily_replay(
        model,
        x_all,
        dx_all,
        dt_t_pd,
        dt_step,
        record_from=topology_start,
        r_hat_all=r_hat_all,
        reproject_window_days=REPROJECT_WINDOW_DAYS,
        n_inducing=N_INDUCING,
        n_grid=N_GRID,
        n_samples=N_SAMPLES,
        min_crossing_sep=MIN_CROSSING_SEP,
        min_barrier_fraction=MIN_BARRIER_FRACTION,
        rng=rng,
    ):
        rows.append(
            {
                "date": d,
                "log_price": float(np.median(x_d)),
                "price_usd": float(np.exp(np.median(x_d))),
                "p_multiwell": topo_d["p_multiwell"],
                "barrier_mean": topo_d["barrier_mean"],
                "barrier_std": topo_d["barrier_std"],
            }
        )
        console.print(
            f"  {d.date()}  $={rows[-1]['price_usd']:.0f}"
            f"  p_multi={topo_d['p_multiwell']:.2f}"
            f"  barrier={topo_d['barrier_mean']:.3f}±{topo_d['barrier_std']:.3f}"
        )

    # ---- Detect well-jumps on backtest period only --------------------------
    console.rule(
        "[bold cyan]Phase — Well-jump detection (price-geometric, backtest period)"
    )
    daily_df = pd.DataFrame(rows).set_index("date")
    daily_df.index = pd.to_datetime(daily_df.index)
    # Pass only the true backtest window to jump detection; the warmup rows
    # recorded before BACKTEST_START are retained in daily_df for slope look-back.
    events_df = _detect_jumps(
        daily_df[daily_df.index >= pd.Timestamp(BACKTEST_START)], console
    )

    # ---- Write CSV files ----------------------------------------------------
    daily_path = os.path.join(out_dir, "daily_topology.csv")
    events_path = os.path.join(out_dir, "events.csv")
    daily_df.reset_index().to_csv(daily_path, index=False)
    console.print(f"  Wrote {daily_path}")
    events_df.to_csv(events_path, index=False)
    if not events_df.empty:
        console.print(f"  Wrote {events_path} ({len(events_df)} events)")
    else:
        console.print("[yellow]  No events detected; wrote empty events.csv[/yellow]")

    del model, x_all, dx_all, r_hat_all, dt_t_all, dt_t_pd, rows, daily_df, events_df
    gc.collect()

    return daily_path, events_path


# =============================================================================
# JUMP DETECTION  (pure price; topology not used)
# =============================================================================


def _detect_jumps(daily_df: pd.DataFrame, console: Console) -> pd.DataFrame:
    """Price-geometric well-jump detection.

    Only log_price is used.  No topology signal influences the labels.
    """
    lp = daily_df["log_price"]
    roll_range = lp.rolling(STABLE_DAYS, min_periods=STABLE_DAYS).apply(
        lambda w: w.max() - w.min(), raw=True
    )
    dates = daily_df.index.tolist()
    n = len(dates)
    events = []
    i = 0
    while i < n:
        if pd.isna(roll_range.iloc[i]) or roll_range.iloc[i] >= STABLE_THR:
            i += 1
            continue
        stable_end_idx = i
        anchor_lp = lp.iloc[i]

        j = i + 1
        while j < n and abs(lp.iloc[j] - anchor_lp) < JUMP_THR:
            j += 1
        if j >= n:
            break
        jump_start_date = dates[j]

        k = j + 1
        settled = False
        while k + SETTLE_DAYS <= n:
            window = lp.iloc[k : k + SETTLE_DAYS]
            if window.max() - window.min() < STABLE_THR:
                settled = True
                break
            k += 1
        if not settled:
            i = j + 1
            continue

        post_stable_start = dates[k]
        post_lp_median = float(lp.iloc[k : k + SETTLE_DAYS].median())

        pre_start_idx = stable_end_idx
        while pre_start_idx > 0:
            c = pre_start_idx - 1
            if not pd.isna(roll_range.iloc[c]) and roll_range.iloc[c] < STABLE_THR:
                pre_start_idx = c
            else:
                break
        pre_stable_start = dates[pre_start_idx]
        pre_lp_median = float(lp.iloc[pre_start_idx : stable_end_idx + 1].median())

        log_change = post_lp_median - pre_lp_median
        events.append(
            {
                "event_id": len(events) + 1,
                "pre_stable_start": pre_stable_start.date(),
                "pre_stable_end": dates[stable_end_idx].date(),
                "jump_start": jump_start_date.date(),
                "post_stable_start": post_stable_start.date(),
                "pre_log_price": round(pre_lp_median, 5),
                "post_log_price": round(post_lp_median, 5),
                "pre_price_usd": round(float(np.exp(pre_lp_median)), 1),
                "post_price_usd": round(float(np.exp(post_lp_median)), 1),
                "log_price_change": round(log_change, 5),
                "direction": "up" if log_change > 0 else "down",
            }
        )
        console.print(
            f"  [bold]Event {len(events)}[/bold]  "
            f"{'UP' if log_change > 0 else 'DOWN'}  "
            f"{pre_stable_start.date()} → {jump_start_date.date()} → {post_stable_start.date()}  "
            f"${np.exp(pre_lp_median):.0f} → ${np.exp(post_lp_median):.0f}  "
            f"Δlog={log_change:+.3f}"
        )
        i = k + SETTLE_DAYS

    if not events:
        console.print("[yellow]  No well-jumps detected.[/yellow]")
    return pd.DataFrame(events)


# =============================================================================
# PHASE 2 — backtest analysis (reads CSV only; no model state in scope)
# =============================================================================


def _slope_z(y: np.ndarray) -> tuple[float, float]:
    """OLS slope vs index and two-sided z-stat. (nan, nan) if degenerate."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 3:
        return float("nan"), float("nan")
    x = np.arange(n, dtype=float)
    x_c = x - x.mean()
    sxx = float(np.sum(x_c * x_c))
    if sxx <= 0:
        return float("nan"), float("nan")
    y_c = y - y.mean()
    slope = float(np.sum(x_c * y_c) / sxx)
    resid = y_c - slope * x_c
    s2 = float(np.sum(resid * resid) / (n - 2))
    se = float(np.sqrt(s2 / sxx)) if s2 > 0 else 0.0
    z = slope / se if se > 0 else float("nan")
    return slope, z


def _sample_signals(daily: pd.DataFrame, date: pd.Timestamp) -> dict | None:
    """Compute all signals at *date* using only data ≤ date."""
    if date not in daily.index:
        return None
    row = daily.loc[date]
    bm = float(row["barrier_mean"])
    bst = float(row["barrier_std"])
    snr = bm / bst if bst > 0 else np.nan

    lo = date - pd.Timedelta(days=TREND_WINDOW - 1)
    tail = daily.loc[(daily.index >= lo) & (daily.index <= date)]
    sl_pm, z_pm = _slope_z(tail["p_multiwell"].values)
    return {
        "date": date,
        "p_multiwell": float(row["p_multiwell"]),
        "barrier_snr": snr,
        "slope_p_multiwell": sl_pm,
        "slope_z_p_multiwell": z_pm,
    }


def _bh_correct(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction over a 1-D array of p-values.

    NaN entries are passed through unchanged.  The correction uses
    m = number of finite p-values actually present (not a fixed constant),
    so it adapts automatically when some tests cannot be run.
    """
    out = np.full_like(pvals, np.nan, dtype=float)
    finite = np.isfinite(pvals)
    idx = np.where(finite)[0]
    if len(idx) == 0:
        return out
    m = len(idx)
    # Sort finite p-values ascending; track original positions.
    order = idx[np.argsort(pvals[idx])]
    ranks = np.arange(1, m + 1)
    p_adj = np.minimum(1.0, pvals[order] * m / ranks)
    # Enforce monotone decrease from largest rank downward.
    for i in range(m - 2, -1, -1):
        p_adj[i] = min(p_adj[i], p_adj[i + 1])
    out[order] = p_adj
    return out


def _run_backtest(
    console: Console,
    rng,
    out_dir: str,
    daily_path: str,
    events_path: str,
) -> None:
    """Statistical backtest: are signals elevated before detected jumps?"""
    console.rule("[bold cyan]Phase 6 — Load topology + events from CSV")
    daily = pd.read_csv(daily_path, parse_dates=["date"]).set_index("date").sort_index()
    events = pd.read_csv(events_path, parse_dates=["jump_start"])
    console.print(
        f"  {len(daily)} daily rows  "
        f"({daily.index.min().date()} → {daily.index.max().date()})"
    )
    console.print(f"  {len(events)} jump events")

    if events.empty:
        console.print("[yellow]No events — backtest analysis skipped.[/yellow]")
        return

    # ---- Pre-jump samples ---------------------------------------------------
    console.rule("[bold cyan]Phase 7 — Pre-jump signal samples")
    pre_rows = []
    for _, ev in events.iterrows():
        jstart = pd.Timestamp(ev["jump_start"]).normalize()
        for off in OFFSETS:
            d = jstart + pd.Timedelta(days=off)
            s = _sample_signals(daily, d)
            if s is None:
                continue
            s.update(
                {
                    "event_id": int(ev["event_id"]),
                    "offset_days": off,
                    "jump_start": jstart,
                    "direction": ev["direction"],
                }
            )
            pre_rows.append(s)
    per_event = pd.DataFrame(pre_rows)
    per_event.to_csv(os.path.join(out_dir, "per_event.csv"), index=False)
    console.print(f"  {len(per_event)} pre-jump sample rows")

    # ---- Null samples --------------------------------------------------------
    console.rule("[bold cyan]Phase 8 — Null (calm) samples")
    jump_dates = pd.to_datetime(events["jump_start"]).dt.normalize().values
    min_abs = np.min(
        np.abs(np.array([(daily.index - jd).days.astype(int) for jd in jump_dates])),
        axis=0,
    )
    calm_mask = min_abs >= NULL_BUFFER_DAYS
    calm_dates = daily.index[calm_mask]
    console.print(
        f"  {len(calm_dates)} calm dates (≥ {NULL_BUFFER_DAYS}d from any jump)"
    )
    if len(calm_dates) < 10:
        console.print("[red]  Too few calm dates — null distribution unreliable.[/red]")

    null_rows = []
    for off in OFFSETS:
        if len(calm_dates) == 0:
            break
        # Sample without replacement when the pool is large enough; fall back
        # to replacement only when NULL_SAMPLE_SIZE exceeds available dates.
        picks = rng.choice(
            calm_dates,
            size=NULL_SAMPLE_SIZE,
            replace=len(calm_dates) < NULL_SAMPLE_SIZE,
        )
        for d in picks:
            s = _sample_signals(daily, pd.Timestamp(d))
            if s is None:
                continue
            s["offset_days"] = off
            null_rows.append(s)
    null_df = pd.DataFrame(null_rows)
    null_df.to_csv(os.path.join(out_dir, "null_samples.csv"), index=False)

    # ---- Summary statistics -------------------------------------------------
    console.rule("[bold cyan]Phase 9 — Mann-Whitney tests")
    summary_rows = []
    for sig in SIGNALS:
        alt = ALT_HYPOTHESIS[sig]
        mw_alt = "greater" if alt == "higher" else "less"
        for off in OFFSETS:
            pre_vals = (
                per_event.loc[per_event["offset_days"] == off, sig].dropna().values
            )
            null_vals = null_df.loc[null_df["offset_days"] == off, sig].dropna().values
            if len(pre_vals) == 0 or len(null_vals) < 5:
                p_val, u_stat = np.nan, np.nan
            else:
                try:
                    u_stat, p_val = mannwhitneyu(
                        pre_vals, null_vals, alternative=mw_alt
                    )
                except ValueError:
                    p_val, u_stat = np.nan, np.nan
            # Rank-biserial IC: r = 2U/(n1*n2) - 1 ∈ [-1, +1].
            # +1 = all pre-jump values above all null; -1 = all below.
            # For "lower" hypothesis signals a negative IC is the expected
            # direction.  Derived directly from U so no extra compute needed.
            _n1, _n2 = len(pre_vals), len(null_vals)
            ic = (
                float(2 * u_stat / (_n1 * _n2) - 1)
                if np.isfinite(u_stat) and _n1 > 0 and _n2 > 0
                else np.nan
            )
            summary_rows.append(
                {
                    "signal": sig,
                    "offset_days": off,
                    "alt": alt,
                    "n_pre": len(pre_vals),
                    "pre_mean": float(np.mean(pre_vals)) if len(pre_vals) else np.nan,
                    "pre_median": float(np.median(pre_vals))
                    if len(pre_vals)
                    else np.nan,
                    "null_mean": float(np.mean(null_vals))
                    if len(null_vals)
                    else np.nan,
                    "null_median": float(np.median(null_vals))
                    if len(null_vals)
                    else np.nan,
                    "mw_u": float(u_stat) if np.isfinite(u_stat) else np.nan,
                    "mw_p_raw": float(p_val) if np.isfinite(p_val) else np.nan,
                    "ic": ic,
                }
            )
    summary = pd.DataFrame(summary_rows)

    # BH correction over the actual number of finite tests.
    p_raw = summary["mw_p_raw"].to_numpy(dtype=float)
    summary["mw_p_bh"] = _bh_correct(p_raw)
    console.print(
        f"  BH correction over {int(np.isfinite(p_raw).sum())} finite tests "
        f"(out of {len(p_raw)} total = {len(SIGNALS)} signals × {len(OFFSETS)} offsets)"
    )
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    # ---- Console table -------------------------------------------------------
    table = Table(title="Pre-jump vs null  (one-sided Mann-Whitney)")
    table.add_column("signal", style="bold")
    table.add_column("alt")
    for off in OFFSETS:
        table.add_column(f"d={off}  p_raw / p_bh / IC", justify="right")
    for sig in SIGNALS:
        row = [sig, ALT_HYPOTHESIS[sig]]
        for off in OFFSETS:
            s = summary[(summary["signal"] == sig) & (summary["offset_days"] == off)]
            if s.empty:
                row.append("-")
                continue
            p_r = float(s["mw_p_raw"].values[0])
            p_b = float(s["mw_p_bh"].values[0])
            ic_v = float(s["ic"].values[0])
            if not np.isfinite(p_r):
                row.append("-")
                continue
            raw_mark = "**" if p_r < 0.01 else ("*" if p_r < 0.05 else "")
            bh_mark = (
                "‡"
                if np.isfinite(p_b) and p_b < 0.05
                else ("†" if np.isfinite(p_b) and p_b < 0.10 else "")
            )
            p_b_str = f"{p_b:.3f}{bh_mark}" if np.isfinite(p_b) else "nan"
            ic_str = f"{ic_v:+.3f}" if np.isfinite(ic_v) else "nan"
            row.append(f"{p_r:.3f}{raw_mark} / {p_b_str} / {ic_str}")
        table.add_row(*row)
    console.print(table)
    console.print("  * p_raw<0.05  ** p_raw<0.01  † p_bh<0.10  ‡ p_bh<0.05")

    # ---- Plots ---------------------------------------------------------------
    console.rule("[bold cyan]Phase 10 — Plots")
    _plot_boxes(per_event, null_df, os.path.join(out_dir, "per_signal_boxes.png"))
    _plot_overview(daily, events, os.path.join(out_dir, "events_overview.png"))
    console.print(f"[green]All outputs written to {out_dir}/[/green]")


# =============================================================================
# PLOTS
# =============================================================================


def _plot_boxes(pre: pd.DataFrame, null: pd.DataFrame, out_path: str) -> None:
    n_events = pre["event_id"].nunique() if "event_id" in pre.columns else len(pre)
    offsets_str = ", ".join(f"{o:+d}d" for o in OFFSETS)
    fig, axes = plt.subplots(len(SIGNALS), 1, figsize=(10, 2.4 * len(SIGNALS)))
    if len(SIGNALS) == 1:
        axes = [axes]
    for ax, sig in zip(axes, SIGNALS):
        data, labels, colors = [], [], []
        for off in OFFSETS:
            pv = pre.loc[pre["offset_days"] == off, sig].dropna().values
            nv = null.loc[null["offset_days"] == off, sig].dropna().values
            data.extend([pv, nv])
            labels.extend(
                [f"pre {off:+d}d  (n={len(pv)})", f"null {off:+d}d  (n={len(nv)})"]
            )
            colors.extend(["#d6604d", "#4393c3"])
        positions = np.arange(len(data))
        bp = ax.boxplot(
            data, positions=positions, widths=0.7, patch_artist=True, showfliers=False
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(sig, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(
        f"Pre-jump (red) vs null calm (blue)\n"
        f"offsets: {offsets_str}   events: {n_events}   null buffer: {NULL_BUFFER_DAYS}d   "
        f"null draws/offset: {NULL_SAMPLE_SIZE}   trend window: {TREND_WINDOW}d\n"
        f"burn-in: {BURN_IN_START.date()} → {BACKTEST_START.date()}   "
        f"backtest: {BACKTEST_START.date()} → {BACKTEST_END.date()}   "
        f"GP si: {PHASE_GP_SI}s k={KERNEL_HW}   KM si: {PHASE_A_SI}s k={KERNEL_HW_PHASE_A}",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_overview(daily: pd.DataFrame, events: pd.DataFrame, out_path: str) -> None:
    """Price + signal panels with jump markers."""
    daily = daily.copy()
    # Compute rolling OLS z-stat of p_multiwell for the overview panel.
    slope_z_vals = np.full(len(daily), np.nan)
    for i in range(len(daily)):
        lo = daily.index[i] - pd.Timedelta(days=TREND_WINDOW - 1)
        tail_pm = np.asarray(
            daily.loc[
                (daily.index >= lo) & (daily.index <= daily.index[i]), "p_multiwell"
            ],
            dtype=float,
        )
        _, z = _slope_z(tail_pm)
        slope_z_vals[i] = z
    daily["slope_z_p_multiwell"] = slope_z_vals
    panels = [
        ("price_usd", "ETH/USDT [$]", "log"),
        ("p_multiwell", "p_multiwell", None),
        ("slope_z_p_multiwell", "slope_z_p_multiwell", None),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 8), sharex=True)
    for ax, (col, ylab, yscale) in zip(axes, panels):
        if col in daily.columns:
            ax.plot(daily.index, daily[col], color="#1f4e79", lw=0.9)
        if yscale:
            ax.set_yscale(yscale)
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(alpha=0.3)
        for _, ev in events.iterrows():
            ax.axvline(
                pd.Timestamp(ev["jump_start"]), color="crimson", lw=0.6, alpha=0.7
            )
    axes[-1].set_xlabel("date")
    n_events = len(events)
    offsets_str = ", ".join(f"{o:+d}d" for o in OFFSETS)
    fig.suptitle(
        f"Kalman-GP topology   backtest: {BACKTEST_START.date()} → {BACKTEST_END.date()}   "
        f"({n_events} jumps, red lines = jump_start)\n"
        f"burn-in: {BURN_IN_START.date()} → {BACKTEST_START.date()}   "
        f"GP si: {PHASE_GP_SI}s k={KERNEL_HW}   KM si: {PHASE_A_SI}s k={KERNEL_HW_PHASE_A}   "
        f"offsets: {offsets_str}   trend window: {TREND_WINDOW}d",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# =============================================================================
# ENTRY POINT
# =============================================================================


def main() -> None:
    console = Console()
    # Two independent RNG streams: one for GP posterior sampling, one for null
    # sampling.  Splitting via SeedSequence means changing null-draw size or
    # adding more GP samples never shifts the other stream.
    ss = np.random.SeedSequence(SEED)
    gp_rng, analysis_rng = [np.random.default_rng(s) for s in ss.spawn(2)]

    backtest_tag = (
        f"{BACKTEST_START.strftime('%Y-%m-%d')}_{BACKTEST_END.strftime('%Y-%m-%d')}"
    )
    out_dir = os.path.join(BACKTEST_DIR, backtest_tag)
    os.makedirs(out_dir, exist_ok=True)

    # Phase 1: run the GP (burn-in + backtest) and detect jumps.
    daily_path, events_path = _run_gp_and_record(console, gp_rng, out_dir)

    # Phase 2: pure CSV-based signal analysis.
    _run_backtest(console, analysis_rng, out_dir, daily_path, events_path)


if __name__ == "__main__":
    main()
