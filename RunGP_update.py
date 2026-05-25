"""
RunGP_update — incremental Kalman-GP update with new data.

Loads the persisted state pickle from `RunGP.py`, ingests new daily
data appended after the saved `last_dt`, runs a Kalman update on the new
window, and produces diagnostics + forecasts in:

    GP_updates/{NEW_END_DATE}/

WHAT THIS SCRIPT DOES
---------------------
1.  Loads the most recent `*_state.pkl` from `gp_results/{si}s/{no_hp|hp}/`.
2.  Ensures + aggregates new data from the day after `state['last_dt']` through
    NEW_END_DATE (today by default).
3.  Reprojects the GP to the new window's x-range (mirrors RunGP.py).
4.  Runs Kalman updates ONE observation at a time so per-observation
    innovations (and their z-scores) can be recorded for diagnostics.
5.  Computes:
      - new-window topology  (p_multiwell, barrier_mean, ...)
      - trend slope of `p_multiwell` and `barrier_mean` over recent history
      - "fragility" = barrier_std / barrier_mean   (low barrier + high std
        ⇒ topology near collapse / regime-change candidate)
      - mean innovation z-score (a positive surprise rate flags model mis-fit)
      - forecast topology at horizons {1, 3, 7, 14, 30} days using
        `forecast_topology()`.
6.  Plots:
      - drift snapshot @ end of new window + KM red dots from new data
      - potential U(x) snapshot + observed U_KM(x) red dots
      - p_multiwell history + new point + forecast band
      - barrier_mean history + ±std band + new point + forecasts
      - innovation z-score time series for the new window
      - σ/μ + fragility history
7.  Writes a chained `update_state.pkl` so successive updates compose.

Run:
    $env:PYTHONIOENCODING='utf-8'; & ".venv/Scripts/python.exe" RunGP_update.py
"""

from __future__ import annotations

import os
import glob
import json
import pickle
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from rich.console import Console

import data_collection as dc
from data_collection import load_series
from phase_GP import (
    KalmanGPDriftModel,
    topology_from_gp,
    forecast_topology,
    _SEC_PER_YEAR,
)
from force_field_estimation import estimate_km
from paths import gp_output_dir


# =============================================================================
# CONFIGURATION
# =============================================================================

# Source of the previous Kalman-GP state.
# - None: auto-pick the newest `*_state.pkl` under STATE_SEARCH_DIR.
# - explicit path: load exactly this pickle.
STATE_PATH: str | None = None
STATE_SEARCH_DIR: str = gp_output_dir(900, "none")

# End of the update window. None ⇒ today (UTC date).
# NB the start of the update window is auto-derived from the loaded state.
NEW_END_DATE: datetime | None = None

# Override the auto-derived start date.
# - None  : start from last_dt + 1 day (normal chained behaviour).
# - explicit datetime : start from this date regardless of chain state.
#   Use this to reprocess days whose output was deleted without resetting
#   chain_state.pkl, e.g. NEW_START_DATE = datetime(2026, 1, 1).
NEW_START_DATE: datetime | None = None

# Reprojection margin (same convention as RunGP.py).
REPROJECT_MARGIN = 0.1

# KM bins for the "observed drift red dots" overlay.
KM_N_BINS = 80
KM_WEIGHT_THRESHOLD = 5

# Forecast horizons in days.
FORECAST_HORIZONS = [1, 3, 7, 14]

# Trend window: number of trailing topology rows used for slope tests.
TREND_LOOKBACK = 6

# If True, scale model.sigma2 (and obs_noise) after each run based on the
# empirical mean |z| of innovations.  Keeps the Kalman gain well-calibrated
# as market volatility drifts away from the original sigma2 estimate.
# The scaling factor is (mean_abs_z / sqrt(2/pi))^2, clipped to [0.25, 4.0].
AUTOADJUST_SIGMA2: bool = True
# EWMA smoothing coefficient for the sigma2 scale factor (0 = no adaptation,
# 1 = instantly tracks today's innovations).  0.2 ≈ half-life of 3 days.
SIGMA2_EWMA_ALPHA: float = 0.2

# Reprojection cadence and rolling-window size.
# The inducing grid is moved every REPROJECT_CADENCE_DAYS days to the
# x-range spanned by the trailing REPROJECT_WINDOW_DAYS of log-prices.
# A 30-day window keeps most of the trained state in view when the price
# drifts, while a 7-day cadence prevents any single large move from causing
# a hard state-reset on a fixed calendar boundary.
REPROJECT_WINDOW_DAYS: int = 30
REPROJECT_CADENCE_DAYS: int = 7

# Topology + sampling.
N_GRID = 200
N_SAMPLES = 200
MIN_CROSSING_SEP = 10
MIN_BARRIER_FRACTION = 0.1
SEED = 42

# Output root.
OUTPUT_ROOT = "GP_updates"

DAILY_TOPO_LOOKBACK_DAYS = 90  # calendar days shown in the fragility plot
DAILY_TOPO_N_SAMPLES = 100  # fewer samples than main run — speed
DAILY_TOPO_CACHE_FNAME = "daily_topology_cache.csv"


# =============================================================================
# HELPERS AND PLOTS (imported from update/ sub-package)
# =============================================================================

from update.state_io import (
    _autopick_state,
    _restore_model,
    _restore_model_from_snap,
    _predict_at,
    _km_bins_for_overlay,
    _is_last_day_of_month,
    _recalibrate_from_km,
)
from update.daily_cache import (
    _build_daily_topo_cache,
    _load_or_update_daily_cache,
    _trend_slope,
)
from update.plots_update import (
    _plot_drift_snapshot,
    _plot_potential_snapshot,
    _plot_history_with_forecast,
    _plot_innovations,
    _plot_fragility,
)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    console = Console(
        file=open(
            sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False
        )
    )
    rng = np.random.default_rng(SEED)

    # ---------------- Load previous state -------------------------------------
    console.rule("[bold cyan]Step 1 — Load previous Kalman-GP state")
    state_path = STATE_PATH or _autopick_state(
        STATE_SEARCH_DIR, output_root=OUTPUT_ROOT
    )
    console.print(f"  state: {state_path}")
    with open(state_path, "rb") as fh:
        blob = pickle.load(fh)
    cfg = blob["config"]
    si = int(cfg["phase_gp_seconds_interval"])
    si_a = int(cfg["phase_a_seconds_interval"])
    khw = int(cfg["kernel_half_width"])
    trim = float(cfg["trim_quantile"])
    n_ind = int(cfg["n_inducing"])
    last_dt = pd.Timestamp(blob["last_dt"])
    console.print(
        f"  pipeline={blob['pipeline']}  si={si}s  last_dt={last_dt}  "
        f"M={len(blob['inducing_x'])}  hp={cfg['hp_opt_mode']}"
    )

    model = _restore_model(blob)

    # Initialise sigma2 EWMA tracking on first update run.
    # sigma2_base is frozen at the RunGP-calibrated value so the
    # EWMA scale factor can move both up and down relative to that baseline.
    if "sigma2_base" not in blob:
        blob["sigma2_base"] = float(blob["sigma2"])
        blob["sigma2_ewma_factor"] = 1.0

    # ---------------- Determine new date range --------------------------------
    new_start_date = (last_dt + pd.Timedelta(days=1)).floor("D").to_pydatetime()
    if NEW_START_DATE is not None:
        new_start_date = NEW_START_DATE
        console.print(
            f"  [yellow]NEW_START_DATE override: starting from "
            f"{new_start_date.date()} (chain last_dt={last_dt.date()})[/yellow]"
        )
    # Default: process exactly ONE day (new_start_date).
    # Set NEW_END_DATE explicitly to process a larger range.
    # Binance data lags ~1 day → hard cap at yesterday regardless.
    yesterday = (datetime.utcnow() - timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end_arg = NEW_END_DATE if NEW_END_DATE is not None else new_start_date
    new_end_date = min(end_arg, yesterday)
    if new_end_date < new_start_date:
        console.print(
            f"[yellow]Nothing to update: new_start={new_start_date.date()} > "
            f"new_end={new_end_date.date()}.[/yellow]"
        )
        return
    console.rule(
        f"[bold cyan]Step 2 — Update window "
        f"{new_start_date.date()} → {new_end_date.date()}"
    )

    out_dir = os.path.join(OUTPUT_ROOT, new_end_date.strftime("%Y-%m-%d"))
    os.makedirs(out_dir, exist_ok=True)
    console.print(f"  out: {out_dir}")

    # ---------------- Fetch + aggregate new data ------------------------------
    console.rule("[bold cyan]Step 3 — Ensure raw data + aggregate")
    dc.ensure_data(new_start_date, new_end_date)
    # Aggregate at both Phase-A and Phase-GP intervals so we can build KM bins.
    for s in sorted({si_a, si}):
        dc.aggregate_log_returns_range(
            new_start_date,
            new_end_date,
            s,
            kernel_half_width=khw,
            trim_quantile=trim,
        )

    # ---------------- Load Phase GP series for new window ---------------------
    console.rule("[bold cyan]Step 4 — Load new-window Phase GP series")
    x_prev, dx, dt_step, dt_t = load_series(
        new_start_date,
        new_end_date,
        si,
        kernel_half_width=khw,
        trim_quantile=trim,
        window_type=None,
    )
    N = len(dx)
    if N == 0:
        console.print("[yellow]No usable new-window increments. Abort.[/yellow]")
        return
    r_hat = (dx / dt_step) * _SEC_PER_YEAR
    t_seconds = (pd.to_datetime(dt_t).astype(np.int64) / 1e9).values.astype(float)
    snr = float(blob["model_params"]["spatial_variance"]) / (
        float(np.var(r_hat * dt_step) / dt_step) / dt_step
    )
    console.print(
        f"  N={N}  dt={dt_step:.0f}s  "
        f"x_window=[{x_prev.min():.4f},{x_prev.max():.4f}]  "
        f"r_hat std={r_hat.std():.2e}/yr  "
        f"SNR(spatial_var/obs_noise)={snr:.2f}"
    )
    if snr < 2.0:
        console.print(
            f"  [yellow]SNR={snr:.2f} < 2: GP posterior will barely move from the "
            f"reprojected prior on this window — drift curve will look flat. "
            f"This is expected at sub-weekly scales.[/yellow]"
        )

    # ---------------- Weekly rolling-window reproject + KM recalibration ------
    console.rule("[bold cyan]Step 5 — Reproject + Kalman update")
    # Reproject every REPROJECT_CADENCE_DAYS days to the x-range of the last
    # REPROJECT_WINDOW_DAYS of log-prices.  This keeps the inducing grid
    # centred on live prices without the hard state-reset that a fixed monthly
    # boundary caused when the price had drifted far between months.
    last_reproject_date = pd.Timestamp(blob.get("last_reproject_date", blob["last_dt"]))
    days_since_reproject = (pd.Timestamp(new_end_date) - last_reproject_date).days
    needs_reproject = days_since_reproject >= REPROJECT_CADENCE_DAYS

    if needs_reproject:
        roll_start = (
            pd.Timestamp(new_end_date) - pd.Timedelta(days=REPROJECT_WINDOW_DAYS - 1)
        ).to_pydatetime()
        try:
            x_roll, _, _, _ = load_series(
                roll_start,
                new_end_date,
                si,
                kernel_half_width=khw,
                trim_quantile=trim,
                window_type=None,
            )
            if len(x_roll) < 2:
                raise ValueError("insufficient observations in rolling window")
            x_w_width = max(float(x_roll.max() - x_roll.min()), 1e-4)
            x_lo_r, x_hi_r = float(x_roll.min()), float(x_roll.max())
        except Exception as exc:
            console.print(
                f"  [yellow]Rolling range load failed ({exc}); "
                "using current inducing range.[/yellow]"
            )
            x_lo_r = float(model.inducing_x.min())
            x_hi_r = float(model.inducing_x.max())
            x_w_width = max(x_hi_r - x_lo_r, 1e-4)
        topo_range = model.reproject_to_range(x_lo_r, x_hi_r, n_inducing=n_ind)
        blob["last_reproject_date"] = new_end_date
        blob["last_topo_range"] = list(topo_range)
        console.print(
            f"  Reprojected ({days_since_reproject}d since last, "
            f"window={REPROJECT_WINDOW_DAYS}d) → [{topo_range[0]:.4f}, {topo_range[1]:.4f}]"
        )
    else:
        topo_range = tuple(
            blob.get(
                "last_topo_range",
                [float(model.inducing_x.min()), float(model.inducing_x.max())],
            )
        )
        console.print(
            f"  No reproject ({days_since_reproject}d < {REPROJECT_CADENCE_DAYS}d cadence).  "
            f"topo_range=[{topo_range[0]:.4f}, {topo_range[1]:.4f}]"
        )

    # --- Step 5b: end-of-month KM recalibration --------------------------------
    # When the new window closes on the last day of a calendar month we have a
    # complete month of observations — the same unit Phase A operates on.  We
    # re-run the full KM estimation (n_bins from config, identical weight
    # threshold) so that:
    #   1. The canonical km_*.csv lands in regime_results/km/ and is found by
    #      future plot_drift_with_km calls.
    #   2. spatial_var is updated to the new month's drift variability.  This
    #      recalibrates Q_block (process noise) and K_zz_inv so the GP amplitude
    #      stays matched to the live regime.  A month whose volatility is 2×
    #      higher than the training period will have 2× larger spatial_var,
    #      widening the posterior and raising p_multiwell appropriately.
    recalib_meta: dict = {"recalibrated": False, "reason": "not_month_end"}
    if _is_last_day_of_month(new_end_date):
        console.rule(
            f"[bold yellow]Step 5b — Month-end KM recalibration "
            f"({new_end_date.strftime('%Y-%m')})"
        )
        recalib_meta = _recalibrate_from_km(
            model,
            new_start_date,
            new_end_date,
            si_a=si_a,
            khw=khw,
            trim=trim,
            cfg=cfg,
            console=console,
        )
    else:
        console.print(
            f"  [dim]{new_end_date.date()} is not month-end — skipping KM recalibration.[/dim]"
        )

    # Process one obs at a time so we can record innovations.
    innov = np.full(N, np.nan)
    innov_S = np.full(N, np.nan)
    for i in range(N):
        # Pre-update prediction at this x.
        mu_pred, mu_var_pred = _predict_at(model, x_prev[i])
        S = mu_var_pred + model.obs_noise
        innov[i] = float(r_hat[i] - mu_pred)
        innov_S[i] = float(S)
        # Apply single-obs Kalman update.
        model.update(
            np.array([x_prev[i]]),
            np.array([r_hat[i]]),
            np.array([t_seconds[i]]),
        )
    z_innov = innov / np.sqrt(np.maximum(innov_S, 1e-30))
    mean_abs_z = float(np.mean(np.abs(z_innov)))
    frac_z_gt2 = float(np.mean(np.abs(z_innov) > 2.0))
    _E0 = float(np.sqrt(2.0 / np.pi))  # ≈ 0.798 under H0
    sigma2_factor = float(np.clip((mean_abs_z / _E0) ** 2, 0.25, 4.0))
    _calib_msg = (
        f"  [yellow]sigma2 over-estimated ({sigma2_factor:.3f}x); "
        f"reducing obs_noise[/yellow]"
        if sigma2_factor < 0.90
        else (
            f"  [yellow]sigma2 under-estimated ({sigma2_factor:.3f}x); "
            f"increasing obs_noise[/yellow]"
            if sigma2_factor > 1.10
            else f"  sigma2 well-calibrated (factor={sigma2_factor:.3f})"
        )
    )
    console.print(
        f"  mean|z|={mean_abs_z:.2f}  (E[|z|]≈{_E0:.2f} under H0)  "
        f"frac|z|>2={frac_z_gt2:.2f}  sigma2_factor={sigma2_factor:.3f}"
    )
    if AUTOADJUST_SIGMA2 and abs(sigma2_factor - 1.0) > 0.05:
        console.print(_calib_msg)
        # EWMA update of the scale factor relative to the frozen base.
        # This prevents ratchet drift: if vol recovers, the factor moves back up.
        old_ewma = float(blob.get("sigma2_ewma_factor", 1.0))
        new_ewma = (
            SIGMA2_EWMA_ALPHA * sigma2_factor + (1.0 - SIGMA2_EWMA_ALPHA) * old_ewma
        )
        blob["sigma2_ewma_factor"] = new_ewma
        model.sigma2 = float(blob["sigma2_base"]) * new_ewma
        model.obs_noise = model.sigma2 / model.dt
        console.print(
            f"  sigma2_ewma_factor: {old_ewma:.3f} → {new_ewma:.3f}  "
            f"sigma2: {blob['sigma2_base']:.4e} × {new_ewma:.3f} = {model.sigma2:.4e}"
        )

    # ---------------- Topology for the new window -----------------------------
    console.rule("[bold cyan]Step 6 — Topology on updated state")
    topo = topology_from_gp(
        model,
        topo_range,
        n_grid=N_GRID,
        n_samples=N_SAMPLES,
        min_crossing_sep=MIN_CROSSING_SEP,
        min_barrier_fraction=MIN_BARRIER_FRACTION,
        rng=rng,
    )

    # Recompute σ/μ restricted to the x-range actually observed this window.
    # The global topo_range includes margin padding and regions the price never
    # visited; the posterior mean → 0 there (prior), dragging the denominator
    # toward zero and inflating σ/μ to 10–30 even on quiet days.  Restricting
    # to [x_obs_lo, x_obs_hi] makes the metric comparable to the monthly-window
    # values stored in topology_history.
    _obs_x_grid = np.linspace(float(x_prev.min()), float(x_prev.max()), N_GRID)
    _mu_obs, _var_obs = model.predict(_obs_x_grid, full_cov=False)
    _mu_std_obs = float(np.mean(np.sqrt(np.maximum(_var_obs, 0.0))))
    _mu_mag_obs = float(np.mean(np.abs(_mu_obs)))
    mu_std_to_mean_obs = (
        float(_mu_std_obs / _mu_mag_obs) if _mu_mag_obs > 1e-12 else float("inf")
    )

    dt_query = pd.Timestamp(dt_t[-1])
    new_row = {
        "datetime": dt_query,
        "window_start": str(pd.Timestamp(new_start_date).date()),
        "p_multiwell_gp": topo["p_multiwell"],
        "p_multiwell": topo["p_multiwell"],
        "mean_n_wells": topo["mean_n_wells"],
        "barrier_mean": topo["barrier_mean"],
        "barrier_std": topo["barrier_std"],
        "barrier_snr": (
            topo["barrier_mean"] / topo["barrier_std"]
            if topo["barrier_std"] > 0
            else 0.0
        ),
        "fragility": (
            topo["barrier_std"] / topo["barrier_mean"]
            if topo["barrier_mean"] > 0
            else float("nan")
        ),
        "kramers_mean": topo["kramers_mean"],
        "u_range": topo["u_range"],
        "mu_std_to_mean": topo["mu_std_to_mean"],
    }
    console.print(
        f"  p_multi={topo['p_multiwell']:.2f}  "
        f"barrier={topo['barrier_mean']:.3f}±{topo['barrier_std']:.3f}  "
        f"σ/μ(obs-range)={mu_std_to_mean_obs:.2f}  "
        f"σ/μ(full-range)={topo['mu_std_to_mean']:.2f}  "
        f"kramers={topo['kramers_mean']:.2e}"
    )

    # ---------------- Daily barrier cache (for fragility plot) ----------------
    console.rule("[bold cyan]Step 6b — Daily barrier cache")
    cache_path = os.path.join(
        cfg.get("gp_dir", cfg.get("gp_output_dir", ".")), DAILY_TOPO_CACHE_FNAME
    )
    daily_cache_plot = _load_or_update_daily_cache(
        blob,
        cache_path,
        new_date=dt_query.strftime("%Y-%m-%d"),
        new_topo=new_row,
        si=si,
        khw=khw,
        trim=trim,
        n_ind=n_ind,
        rng=np.random.default_rng(SEED),
        console=console,
        daily_topo_lookback_days=DAILY_TOPO_LOOKBACK_DAYS,
        reproject_cadence_days=REPROJECT_CADENCE_DAYS,
        reproject_window_days=REPROJECT_WINDOW_DAYS,
        n_grid=N_GRID,
        daily_topo_n_samples=DAILY_TOPO_N_SAMPLES,
        min_crossing_sep=MIN_CROSSING_SEP,
        min_barrier_fraction=MIN_BARRIER_FRACTION,
    )

    # ---------------- KM bins for red-dot overlay -----------------------------
    console.rule("[bold cyan]Step 7 — Empirical KM bins (red-dot overlay)")
    # Use phase-A resolution data for the overlay (denser; matches existing plots).
    x_prev_km, _dx_km, _dt_km, _ = load_series(
        new_start_date,
        new_end_date,
        si_a,
        kernel_half_width=khw,
        trim_quantile=trim,
        window_type=None,
    )
    log_prices_km = (
        np.concatenate([x_prev_km, x_prev_km[-1:] + _dx_km[-1:]])
        if len(_dx_km)
        else x_prev_km
    )
    km_df = _km_bins_for_overlay(
        log_prices_km,
        seconds_interval=si_a,
        n_bins=KM_N_BINS,
        weight_threshold=KM_WEIGHT_THRESHOLD,
        x_range=topo_range,
    )
    console.print(f"  KM bins kept: {len(km_df)}")

    # Forecasts disabled.
    fc = pd.DataFrame()

    # ---------------- Trend tests --------------------------------------------
    console.rule("[bold cyan]Step 9 — Trend tests over recent history")
    history = blob["topology_history"].copy()
    history["datetime"] = pd.to_datetime(history["datetime"])
    history = history.sort_values("datetime").reset_index(drop=True)

    tail = history.tail(TREND_LOOKBACK)
    slope_pm, z_pm = _trend_slope(tail["p_multiwell_gp"].values)
    slope_bm, z_bm = _trend_slope(tail["barrier_mean"].values)
    slope_kr, _ = _trend_slope(
        np.log(np.maximum(tail["kramers"].values, 1e-30))
        if "kramers" in tail.columns
        else np.array([])
    )
    diagnostics = {
        "new_window_start": str(new_start_date.date()),
        "new_window_end": str(new_end_date.date()),
        "n_obs_processed": int(N),
        "innov_mean_abs_z": mean_abs_z,
        "innov_frac_abs_z_gt2": frac_z_gt2,
        "new_p_multiwell": topo["p_multiwell"],
        "new_barrier_mean": topo["barrier_mean"],
        "new_barrier_std": topo["barrier_std"],
        "new_fragility": (
            topo["barrier_std"] / topo["barrier_mean"]
            if topo["barrier_mean"] > 0
            else None
        ),
        "new_mu_std_to_mean_full_range": topo["mu_std_to_mean"],
        "new_mu_std_to_mean_obs_range": mu_std_to_mean_obs,
        "trend_lookback": TREND_LOOKBACK,
        "slope_p_multiwell_per_window": slope_pm,
        "z_p_multiwell_slope": z_pm,
        "slope_barrier_mean_per_window": slope_bm,
        "z_barrier_mean_slope": z_bm,
        "slope_log_kramers_per_window": slope_kr,
        "month_end_recalibration": recalib_meta,
        "forecasts": fc.to_dict(orient="records"),
    }
    with open(os.path.join(out_dir, "diagnostics.json"), "w") as fh:
        json.dump(diagnostics, fh, indent=2, default=str)

    def _verdict(slope, z, label, good_direction=None):
        sign = "↑" if slope > 0 else ("↓" if slope < 0 else "·")
        sig = (
            ""
            if not np.isfinite(z)
            else (
                "  [bold]SIG[/bold]" if abs(z) > 2 else ("  weak" if abs(z) > 1 else "")
            )
        )
        return f"  {label}: slope/window={slope:+.4g} {sign}  z={z:+.2f}{sig}"

    console.print(_verdict(slope_pm, z_pm, "p_multiwell"))
    console.print(_verdict(slope_bm, z_bm, "barrier_mean"))
    console.print(
        f"  SNR (barrier_mean/barrier_std): {diagnostics['new_fragility']}"
        if diagnostics["new_fragility"] is not None
        else "  SNR: undefined (barrier_mean=0)"
    )

    # ---------------- Plots ---------------------------------------------------
    console.rule("[bold cyan]Step 10 — Plots")
    # Observation range with a 10 % margin on each side — used to zoom plots
    # so the x-axis shows only where data actually appeared today.
    x_obs_w = max(float(x_prev.max() - x_prev.min()), 1e-4)
    x_obs_margin = 0.10 * x_obs_w
    x_obs_range = (
        float(x_prev.min()) - x_obs_margin,
        float(x_prev.max()) + x_obs_margin,
    )
    _plot_drift_snapshot(
        model,
        x_obs_range,
        km_df,
        topo,
        os.path.join(out_dir, "drift_snapshot.png"),
        dt_query,
        n_grid=N_GRID,
    )
    _plot_potential_snapshot(
        model,
        x_obs_range,
        km_df,
        topo,
        os.path.join(out_dir, "potential_snapshot.png"),
        dt_query,
        rng,
    )
    # Use daily-resolution cache for the trajectory plot so each day has its own
    # point.  The monthly topology_history (from RunGP) is still used for
    # trend tests above; the plot just substitutes the finer-grained daily cache.
    # Exclude today — it is plotted separately as the crimson "new update" dot.
    _daily_hist = daily_cache_plot.rename(columns={"date": "datetime"}).copy()
    _daily_hist["datetime"] = pd.to_datetime(_daily_hist["datetime"])
    _daily_hist = _daily_hist[
        _daily_hist["datetime"].dt.strftime("%Y-%m-%d") != dt_query.strftime("%Y-%m-%d")
    ]
    _plot_history_with_forecast(
        _daily_hist,
        new_row,
        fc,
        fields=["p_multiwell", "barrier_mean", "barrier_snr"],
        out_path=os.path.join(out_dir, "topology_trajectory.png"),
        title_suffix=f"updated through {new_end_date.date()}",
    )
    _plot_innovations(
        pd.to_datetime(dt_t),
        z_innov,
        os.path.join(out_dir, "innovations.png"),
        dt_query,
    )
    _plot_fragility(
        daily_cache_plot,
        os.path.join(out_dir, "fragility.png"),
        daily_topo_lookback_days=DAILY_TOPO_LOOKBACK_DAYS,
    )

    # ---------------- Chained state pickle ------------------------------------
    console.rule("[bold cyan]Step 11 — Save chained state")
    updated_history = pd.concat(
        [
            history,
            pd.DataFrame(
                [
                    {
                        "datetime": new_row["datetime"],
                        "window_start": new_row["window_start"],
                        "p_multiwell_gp": new_row["p_multiwell_gp"],
                        "mean_n_wells": new_row["mean_n_wells"],
                        "barrier_mean": new_row["barrier_mean"],
                        "barrier_std": new_row["barrier_std"],
                        "kramers": new_row["kramers_mean"],
                        "u_range": new_row["u_range"],
                        "mu_std_to_mean": new_row["mu_std_to_mean"],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    updated_history.to_csv(
        os.path.join(out_dir, "topology_history.csv"),
        index=False,
    )

    new_snapshots = list(blob.get("snapshots", []))
    new_snapshots.append(
        (
            dt_query,
            model.state_mean.copy(),
            model.state_cov.copy(),
            topo_range,
            model.inducing_x.copy(),
            pd.Timestamp(new_start_date),
        )
    )
    new_blob = dict(blob)
    new_blob.update(
        {
            "model_params": model.get_params(),
            "inducing_x": model.inducing_x.copy(),
            "state_mean": model.state_mean.copy(),
            "state_cov": model.state_cov.copy(),
            "sigma2": model.sigma2,  # current effective sigma2
            "sigma2_base": blob.get("sigma2_base", model.sigma2),
            "sigma2_ewma_factor": blob.get("sigma2_ewma_factor", 1.0),
            "last_dt": dt_query,
            "last_reproject_date": blob.get("last_reproject_date", new_end_date),
            "last_topo_range": blob.get("last_topo_range", list(topo_range)),
            "snapshots": new_snapshots,
            "topology_history": updated_history,
        }
    )
    chained_path = os.path.join(out_dir, "update_state.pkl")
    with open(chained_path, "wb") as fh:
        pickle.dump(new_blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
    console.print(f"  wrote {chained_path}")

    console.rule("[bold green]Done")


if __name__ == "__main__":
    main()
