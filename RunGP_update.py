"""
RunGP_update — incremental Kalman-GP update with new data.

Loads the persisted state pickle from `RunGP.py`, ingests new daily
data appended after the saved `last_dt`, runs a Kalman update on the new
window, and produces diagnostics + forecasts in:

    GP_updates/{NEW_END_DATE}/

WHAT THIS SCRIPT DOES
---------------------
1.  Loads the most recent `*_state.pkl` from `gp_results/{si}s/`.
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
from rich.table import Table

import data_collection as dc
from data_collection import load_series
from phase_GP import (
    KalmanGPDriftModel,
    topology_from_gp,
    forecast_topology,
    _SEC_PER_YEAR,
)
from regime_estimation import estimate_km
from paths import gp_output_dir


# =============================================================================
# CONFIGURATION
# =============================================================================

# Source of the previous Kalman-GP state.
# - None: auto-pick the newest `*_state.pkl` under STATE_SEARCH_DIR.
# - explicit path: load exactly this pickle.
STATE_PATH: str | None = None
STATE_SEARCH_DIR: str = gp_output_dir(900)

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

# Trend window: number of trailing *daily* topology rows used for slope signals.
# Must match TREND_WINDOW in backtest_jumps.py so the signals are comparable.
SIGNAL_TREND_DAYS = 10

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
    _plot_innovations,
    _plot_fragility,
)


# =============================================================================
# GP UPDATE — core computation function
# =============================================================================


def gp_update(
    blob: dict,
    new_start_date: datetime,
    new_end_date: datetime,
    out_dir: str,
    *,
    reproject_cadence_days: int = REPROJECT_CADENCE_DAYS,
    reproject_window_days: int = REPROJECT_WINDOW_DAYS,
    autoadjust_sigma2: bool = AUTOADJUST_SIGMA2,
    sigma2_ewma_alpha: float = SIGMA2_EWMA_ALPHA,
    n_grid: int = N_GRID,
    n_samples: int = N_SAMPLES,
    min_crossing_sep: int = MIN_CROSSING_SEP,
    min_barrier_fraction: float = MIN_BARRIER_FRACTION,
    km_n_bins: int = KM_N_BINS,
    km_weight_threshold: int = KM_WEIGHT_THRESHOLD,
    daily_topo_lookback_days: int = DAILY_TOPO_LOOKBACK_DAYS,
    daily_topo_n_samples: int = DAILY_TOPO_N_SAMPLES,
    daily_topo_cache_fname: str = DAILY_TOPO_CACHE_FNAME,
    signal_trend_days: int = SIGNAL_TREND_DAYS,
    seed: int = SEED,
    console: Console | None = None,
) -> dict:
    """Run one Kalman-GP incremental update step.

    Performs all mathematical operations (data loading, Kalman update,
    topology, signal computation) and writes persistent outputs
    (diagnostics.json, topology_history.csv, update_state.pkl) to *out_dir*.

    Designed to be called from ``main()`` for the production workflow and
    importable by ``backtest_jumps.py`` when replaying history day-by-day to
    ensure the backtest exercises the exact same production code path.

    Parameters
    ----------
    blob : dict
        State pickle from a previous RunGP or RunGP_update run.  The dict is
        shallow-copied internally so the caller's reference is never mutated.
    new_start_date, new_end_date : datetime
        Inclusive date range to process.
    out_dir : str
        Directory for output files (created by the caller).
    console : Console | None
        Rich Console for step-level logging.  Pass ``None`` (default) to run
        silently — appropriate when replaying many days in a backtest loop.

    Returns
    -------
    dict or ``{}``
        Returns an empty dict if there are no usable observations for the
        requested window (caller should treat this as a no-op).  Otherwise:

        ``"diagnostics"``
            The 4 backtest signals + supporting fields, also written to
            ``diagnostics.json``.
        ``"new_blob"``
            Updated state dict — pass as *blob* on the next call or pickle
            externally.
        ``"chained_path"``
            Path of the written ``update_state.pkl``.
        ``"topo"``
            Raw ``topology_from_gp`` result dict.
        ``"daily_cache_plot"``
            DataFrame (≤ *daily_topo_lookback_days* rows) for the fragility
            plot.
        ``"km_df"``
            KM drift-bin DataFrame for drift/potential overlay plots.
        ``"model"``
            Updated ``KalmanGPDriftModel`` instance.
        ``"x_prev"``
            Log-price array for the new window.
        ``"dt_t"``
            Timestamp array for the new window.
        ``"z_innov"``
            Per-observation innovation z-scores.
        ``"dt_query"``
            ``pd.Timestamp`` of the last processed observation.
        ``"topo_range"``
            ``(x_lo, x_hi)`` tuple used for topology sampling.
    """
    # Shallow-copy so internal mutations (sigma2_base, last_reproject_date,
    # etc.) never propagate back to the caller's blob reference.
    blob = dict(blob)
    rng = np.random.default_rng(seed)

    cfg = blob["config"]
    si = int(cfg["phase_gp_seconds_interval"])
    si_a = int(cfg["phase_a_seconds_interval"])
    khw = int(cfg["kernel_half_width"])
    trim = float(cfg["trim_quantile"])
    n_ind = int(cfg["n_inducing"])

    model = _restore_model(blob)

    # Initialise sigma2 EWMA tracking on first update run.
    # sigma2_base is frozen at the RunGP-calibrated value so the
    # EWMA scale factor can move both up and down relative to that baseline.
    if "sigma2_base" not in blob:
        blob["sigma2_base"] = float(blob["sigma2"])
        blob["sigma2_ewma_factor"] = 1.0

    # ---- Step 3: ensure data + aggregate ------------------------------------
    if console:
        console.rule("[bold cyan]Step 3 — Ensure raw data + aggregate")
    dc.ensure_data(new_start_date, new_end_date)
    for s in sorted({si_a, si}):
        dc.aggregate_log_returns_range(
            new_start_date,
            new_end_date,
            s,
            kernel_half_width=khw,
            trim_quantile=trim,
        )

    # ---- Step 4: load Phase GP series ---------------------------------------
    if console:
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
        if console:
            console.print("[yellow]No usable new-window increments. Abort.[/yellow]")
        return {}
    r_hat = (dx / dt_step) * _SEC_PER_YEAR
    t_seconds = (pd.to_datetime(dt_t).astype(np.int64) / 1e9).values.astype(float)
    snr = float(blob["model_params"]["spatial_variance"]) / (
        float(np.var(r_hat * dt_step) / dt_step) / dt_step
    )
    if console:
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

    # ---- Step 5: rolling-window reproject -----------------------------------
    if console:
        console.rule("[bold cyan]Step 5 — Reproject + Kalman update")
    last_reproject_date = pd.Timestamp(blob.get("last_reproject_date", blob["last_dt"]))
    days_since_reproject = (pd.Timestamp(new_end_date) - last_reproject_date).days
    needs_reproject = days_since_reproject >= reproject_cadence_days

    if needs_reproject:
        roll_start = (
            pd.Timestamp(new_end_date) - pd.Timedelta(days=reproject_window_days - 1)
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
            x_lo_r, x_hi_r = float(x_roll.min()), float(x_roll.max())
        except Exception as exc:
            if console:
                console.print(
                    f"  [yellow]Rolling range load failed ({exc}); "
                    "using current inducing range.[/yellow]"
                )
            x_lo_r = float(model.inducing_x.min())
            x_hi_r = float(model.inducing_x.max())
        topo_range = model.reproject_to_range(x_lo_r, x_hi_r, n_inducing=n_ind)
        blob["last_reproject_date"] = new_end_date
        blob["last_topo_range"] = list(topo_range)
        if console:
            console.print(
                f"  Reprojected ({days_since_reproject}d since last, "
                f"window={reproject_window_days}d) → "
                f"[{topo_range[0]:.4f}, {topo_range[1]:.4f}]"
            )
    else:
        topo_range = tuple(
            blob.get(
                "last_topo_range",
                [float(model.inducing_x.min()), float(model.inducing_x.max())],
            )
        )
        if console:
            console.print(
                f"  No reproject ({days_since_reproject}d < "
                f"{reproject_cadence_days}d cadence).  "
                f"topo_range=[{topo_range[0]:.4f}, {topo_range[1]:.4f}]"
            )

    # ---- Step 5b: end-of-month KM recalibration -----------------------------
    recalib_meta: dict = {"recalibrated": False, "reason": "not_month_end"}
    if _is_last_day_of_month(new_end_date):
        if console:
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
    elif console:
        console.print(
            f"  [dim]{new_end_date.date()} is not month-end — "
            "skipping KM recalibration.[/dim]"
        )

    # ---- Innovations + Kalman update (one obs at a time) --------------------
    innov = np.full(N, np.nan)
    innov_S = np.full(N, np.nan)
    for i in range(N):
        mu_pred, mu_var_pred = _predict_at(model, x_prev[i])
        S = mu_var_pred + model.obs_noise
        innov[i] = float(r_hat[i] - mu_pred)
        innov_S[i] = float(S)
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
    if console:
        console.print(
            f"  mean|z|={mean_abs_z:.2f}  (E[|z|]≈{_E0:.2f} under H0)  "
            f"frac|z|>2={frac_z_gt2:.2f}  sigma2_factor={sigma2_factor:.3f}"
        )
    if autoadjust_sigma2 and abs(sigma2_factor - 1.0) > 0.05:
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
        if console:
            console.print(_calib_msg)
        old_ewma = float(blob.get("sigma2_ewma_factor", 1.0))
        new_ewma = (
            sigma2_ewma_alpha * sigma2_factor + (1.0 - sigma2_ewma_alpha) * old_ewma
        )
        blob["sigma2_ewma_factor"] = new_ewma
        model.sigma2 = float(blob["sigma2_base"]) * new_ewma
        model.obs_noise = model.sigma2 / model.dt
        if console:
            console.print(
                f"  sigma2_ewma_factor: {old_ewma:.3f} → {new_ewma:.3f}  "
                f"sigma2: {blob['sigma2_base']:.4e} × {new_ewma:.3f} = "
                f"{model.sigma2:.4e}"
            )

    # ---- Step 6: topology ---------------------------------------------------
    if console:
        console.rule("[bold cyan]Step 6 — Topology on updated state")
    topo = topology_from_gp(
        model,
        topo_range,
        n_grid=n_grid,
        n_samples=n_samples,
        min_crossing_sep=min_crossing_sep,
        min_barrier_fraction=min_barrier_fraction,
        rng=rng,
    )
    # σ/μ restricted to the observed x-range (avoids prior-dominated regions).
    _obs_x_grid = np.linspace(float(x_prev.min()), float(x_prev.max()), n_grid)
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
        "u_range": topo["u_range"],
        "mu_std_to_mean": topo["mu_std_to_mean"],
    }
    if console:
        console.print(
            f"  p_multi={topo['p_multiwell']:.2f}  "
            f"barrier={topo['barrier_mean']:.3f}±{topo['barrier_std']:.3f}  "
            f"σ/μ(obs-range)={mu_std_to_mean_obs:.2f}  "
            f"σ/μ(full-range)={topo['mu_std_to_mean']:.2f}"
        )

    # ---- Step 6b: daily cache -----------------------------------------------
    if console:
        console.rule("[bold cyan]Step 6b — Daily barrier cache")
    cache_path = os.path.join(
        cfg.get("gp_dir", cfg.get("gp_output_dir", ".")), daily_topo_cache_fname
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
        rng=np.random.default_rng(seed),
        console=console,
        daily_topo_lookback_days=daily_topo_lookback_days,
        reproject_cadence_days=reproject_cadence_days,
        reproject_window_days=reproject_window_days,
        n_grid=n_grid,
        daily_topo_n_samples=daily_topo_n_samples,
        min_crossing_sep=min_crossing_sep,
        min_barrier_fraction=min_barrier_fraction,
    )

    # ---- Step 7: KM bins for red-dot overlay --------------------------------
    if console:
        console.rule("[bold cyan]Step 7 — Empirical KM bins (red-dot overlay)")
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
        n_bins=km_n_bins,
        weight_threshold=km_weight_threshold,
        x_range=topo_range,
    )
    if console:
        console.print(f"  KM bins kept: {len(km_df)}")

    # ---- Signal computation -------------------------------------------------
    # Use the daily cache so signal windows are in calendar days — identical
    # to backtest_jumps.py: lo = today - (signal_trend_days - 1) days.
    history = blob["topology_history"].copy()
    if not daily_cache_plot.empty and "p_multiwell" in daily_cache_plot.columns:
        daily_sig = daily_cache_plot.copy()
        if "date" in daily_sig.columns:
            daily_sig["date"] = pd.to_datetime(daily_sig["date"])
            daily_sig = daily_sig.sort_values("date").set_index("date")
        # Use dt_query's date as "today" so that a cache seeded with future
        # rows (e.g. by RunGP Step 7b) doesn't make every run read the same
        # last row regardless of which day is being processed.
        target_date = pd.Timestamp(dt_query.date())
        if target_date in daily_sig.index:
            today_date = target_date
        else:
            today_date = (
                daily_sig.index[daily_sig.index <= target_date].max()
                if any(daily_sig.index <= target_date)
                else daily_sig.index[-1]
            )
        today_row = daily_sig.loc[today_date]
        p_multi_now = float(today_row["p_multiwell"])
        bm_now = float(today_row["barrier_mean"])
        bst_now = float(today_row["barrier_std"])
        snr_now = bm_now / bst_now if bst_now > 0 else float("nan")
        lo = today_date - pd.Timedelta(days=signal_trend_days - 1)
        sig_window = daily_sig.loc[
            (daily_sig.index >= lo) & (daily_sig.index <= today_date)
        ]
        slope_pm, z_pm = _trend_slope(sig_window["p_multiwell"].values)
    else:
        p_multi_now = topo["p_multiwell"]
        bm_now = topo["barrier_mean"]
        bst_now = topo["barrier_std"]
        snr_now = bm_now / bst_now if bst_now > 0 else float("nan")
        slope_pm = z_pm = float("nan")

    diagnostics = {
        "new_window_start": str(new_start_date.date()),
        "new_window_end": str(new_end_date.date()),
        "n_obs_processed": int(N),
        "innov_mean_abs_z": mean_abs_z,
        "innov_frac_abs_z_gt2": frac_z_gt2,
        # 4 backtest-equivalent signals (same names as backtest_jumps.py)
        "p_multiwell": p_multi_now,
        "barrier_snr": snr_now,
        "slope_p_multiwell": slope_pm,
        "slope_z_p_multiwell": z_pm,
        # supporting fields
        "barrier_mean": bm_now,
        "barrier_std": bst_now,
        "fragility": 1.0 / snr_now if np.isfinite(snr_now) and snr_now > 0 else None,
        "mu_std_to_mean_full_range": topo["mu_std_to_mean"],
        "mu_std_to_mean_obs_range": mu_std_to_mean_obs,
        "signal_trend_days": signal_trend_days,
        "month_end_recalibration": recalib_meta,
        "forecasts": [],
    }
    with open(os.path.join(out_dir, "diagnostics.json"), "w") as fh:
        json.dump(diagnostics, fh, indent=2, default=str)

    # ---- Topology history CSV -----------------------------------------------
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
                        "u_range": new_row["u_range"],
                        "mu_std_to_mean": new_row["mu_std_to_mean"],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    updated_history.to_csv(os.path.join(out_dir, "topology_history.csv"), index=False)

    # ---- Step 11: chained state pickle --------------------------------------
    if console:
        console.rule("[bold cyan]Step 11 — Save chained state")
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
            "sigma2": model.sigma2,
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
    if console:
        console.print(f"  wrote {chained_path}")

    return {
        "diagnostics": diagnostics,
        "new_blob": new_blob,
        "chained_path": chained_path,
        "topo": topo,
        "daily_cache_plot": daily_cache_plot,
        "km_df": km_df,
        "model": model,
        "x_prev": x_prev,
        "dt_t": dt_t,
        "z_innov": z_innov,
        "dt_query": dt_query,
        "topo_range": topo_range,
    }


# =============================================================================
# MAIN — orchestration + reporting
# =============================================================================


def main() -> None:
    console = Console(
        file=open(
            sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False
        )
    )

    # ---- Step 1: Load previous state ----------------------------------------
    console.rule("[bold cyan]Step 1 — Load previous Kalman-GP state")
    state_path = STATE_PATH or _autopick_state(
        STATE_SEARCH_DIR, output_root=OUTPUT_ROOT
    )
    console.print(f"  state: {state_path}")
    with open(state_path, "rb") as fh:
        blob = pickle.load(fh)
    cfg = blob["config"]
    si = int(cfg["phase_gp_seconds_interval"])
    last_dt = pd.Timestamp(blob["last_dt"])
    console.print(
        f"  pipeline={blob['pipeline']}  si={si}s  last_dt={last_dt}  "
        f"M={len(blob['inducing_x'])}"
    )

    # ---- Step 2: Determine update window ------------------------------------
    new_start_date = (last_dt + pd.Timedelta(days=1)).floor("D").to_pydatetime()
    if NEW_START_DATE is not None:
        new_start_date = NEW_START_DATE
        console.print(
            f"  [yellow]NEW_START_DATE override: starting from "
            f"{new_start_date.date()} (chain last_dt={last_dt.date()})[/yellow]"
        )
    # Binance data lags ~1 day → hard cap at yesterday.
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

    # ---- Steps 3–11: computation (delegate entirely to gp_update) -----------
    result = gp_update(
        blob,
        new_start_date,
        new_end_date,
        out_dir,
        console=console,
    )
    if not result:
        return  # no observations — gp_update already printed the reason

    # ---- Step 9: Signal report (reporting only) -----------------------------
    console.rule("[bold cyan]Step 9 — Signal report (backtest-equivalent)")
    diag = result["diagnostics"]
    p_multi_now = diag["p_multiwell"]
    snr_now = diag["barrier_snr"]
    slope_pm = diag["slope_p_multiwell"]
    z_pm = diag["slope_z_p_multiwell"]

    # --- Backtest reference thresholds (2025-07-01 → 2026-03-31, offset −1d) -
    # Each entry: (null_mean, pre_jump_mean, direction)
    # direction "higher" → signal elevated before jumps
    #           "lower"  → signal depressed before jumps
    _BT_REF = {
        "p_multiwell": (0.0789, 0.1033, "higher"),
        "barrier_snr": (0.4553, 0.2880, "lower"),
        "slope_p_multiwell": (-0.00143, 0.00426, "higher"),
        "slope_z_p_multiwell": (-0.2082, 1.0016, "higher"),
    }

    def _criticality(value: float, signal: str) -> str:
        if not np.isfinite(value) or signal not in _BT_REF:
            return ""
        null_m, pre_m, direction = _BT_REF[signal]
        halfway = (null_m + pre_m) / 2.0
        if direction == "higher":
            if value >= pre_m:
                return "[bold red]jump very likely, passed pre-jump mean[/bold red]"
            if value >= halfway:
                return "[yellow]passed halfway to pre-jump mean[/yellow]"
            return "typical of calm periods"
        else:  # lower
            if value <= pre_m:
                return "[bold red]jump very likely, passed pre-jump mean[/bold red]"
            if value <= halfway:
                return "[yellow]passed halfway to pre-jump mean[/yellow]"
            return "typical of calm periods"

    sig_table = Table(
        title="Jump signals  (backtest-equivalent)",
        show_header=True,
        header_style="bold cyan",
        box=None,
        pad_edge=False,
        min_width=60,
    )
    sig_table.add_column("signal", style="bold", min_width=24)
    sig_table.add_column("value", justify="right", min_width=10)
    sig_table.add_column("criticality", min_width=38)
    sig_table.add_row(
        "p_multiwell",
        f"{p_multi_now:.3f}" if np.isfinite(p_multi_now) else "n/a",
        _criticality(p_multi_now, "p_multiwell"),
    )
    sig_table.add_row(
        "barrier_snr",
        f"{snr_now:.3f}" if np.isfinite(snr_now) else "n/a",
        _criticality(snr_now, "barrier_snr"),
    )
    sig_table.add_row(
        "slope_p_multiwell",
        f"{slope_pm:+.6f}" if np.isfinite(slope_pm) else "n/a",
        _criticality(slope_pm, "slope_p_multiwell"),
    )
    sig_table.add_row(
        "slope_z_p_multiwell",
        f"{z_pm:+.3f}" if np.isfinite(z_pm) else "n/a",
        _criticality(z_pm, "slope_z_p_multiwell"),
    )
    console.print(sig_table)

    # ---- Step 10: Plots (reporting only) ------------------------------------
    console.rule("[bold cyan]Step 10 — Plots")
    model = result["model"]
    topo = result["topo"]
    km_df = result["km_df"]
    x_prev = result["x_prev"]
    dt_t = result["dt_t"]
    z_innov = result["z_innov"]
    dt_query = result["dt_query"]
    daily_cache_plot = result["daily_cache_plot"]
    topo_range = result["topo_range"]

    x_obs_w = max(float(x_prev.max() - x_prev.min()), 1e-4)
    x_obs_margin = 0.10 * x_obs_w
    x_obs_range = (
        float(x_prev.min()) - x_obs_margin,
        float(x_prev.max()) + x_obs_margin,
    )
    plot_rng = np.random.default_rng(SEED)
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
        plot_rng,
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

    console.rule("[bold green]Done")


if __name__ == "__main__":
    main()
