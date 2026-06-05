"""
RunGP — stationary Kalman-GP pipeline.

Main execution script for the sequential GP pipeline.
All configuration is in the CONFIGURATION block below.  Run with:

    $env:PYTHONIOENCODING='utf-8'; & ".venv/Scripts/python.exe" RunGP.py

Pipeline steps
--------------
    1  Download raw data
    2  Aggregate log-returns
    3  Phase A (KM estimation)
    4  Load Phase GP series
    5  Determine spatial_var from KM or fixed value
    6  Initialise Kalman-GP model
    7  Sequential Kalman updates + topology per window
         For each window: optional reproject_to_range Kalman update topology_from_gp
    8  GP potential U(x) topology snapshots
    9  Log-price vs topology plot
   10  GP drift + KM overlay plot
   11  Persist model state
"""

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from rich.console import Console

import data_collection as dc
from data_collection import load_series
from regime_estimation import (
    iter_windows,
    normalize_window_boundaries,
    run_phase_a,
)
from phase_GP import KalmanGPDriftModel, topology_from_gp, _SEC_PER_YEAR
from paths import gp_output_dir
from plots import (
    detect_price_jumps,
    plot_topology_snapshots,
    plot_logprice_topology,
    plot_drift_with_km,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Date range ---------------------------------------------------------------
start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 12, 31)

# --- Data aggregation ---------------------------------------------------------
phase_a_seconds_interval = 30
phase_gp_seconds_interval = 900
kernel_half_width = 50
kernel_half_width_phase_a = 3
trim_quantile = 0.01

# --- Phase A ------------------------------------------------------------------
window_type = "monthly"
n_bins = 200
weight_threshold = 5
min_barrier_fraction = 0.1


# =============================================================================
# MODEL CONFIG
# =============================================================================

# --- Spatial variance source --------------------------------------------------
# 'km'    — compute var(Kramers-Moyal annualised drift bins) from Phase A output
# 'fixed' — use SPATIAL_VAR_FIXED directly
SPATIAL_VAR_SOURCE = "fixed"
SPATIAL_VAR_FIXED = 2000.0  # used when SPATIAL_VAR_SOURCE='fixed' or as fallback

# --- Reprojection (move inducing points into each window's observed x-range) --
USE_REPROJECT = True
REPROJECT_MARGIN = 0.1  # fraction of window x-width added on each side
# Rolling window used to compute the x-range at each weekly reproject.
# Wider than a single window to keep the state warm when price drifts.
REPROJECT_WINDOW_DAYS = 30

# --- Lengthscales (initial values) -------------------------------------------
# spatial_ls: None -> (x_hi - x_lo) / (3 * N_INDUCING)  — smaller than the
#   inducing spacing so the posterior can capture sharper well/barrier
#   features in drift.
SPATIAL_LENGTHSCALE_INIT = None  # None -> (x_hi - x_lo) / (N_INDUCING)
TEMPORAL_LENGTHSCALE_DAYS_INIT = 10.0  # days

# --- EMA demeaning -----------------------------------------------------------
# Remove a slow-moving mean from the observed drift before GP inference.
# None  -> no demeaning (raw r_hat passed to the model)
# float -> half-life in calendar days of the exponential moving average
EMA_HALFLIFE_DAYS = 30  # e.g. 30.0 to enable

# --- Observation noise -------------------------------------------------------
SIGMA2 = None  # None -> estimated from calm periods (rolling-range filter)
# Calm-period filter for sigma2 calibration (same geometric criterion as jump detection)
SIGMA2_STABLE_DAYS = 4  # rolling window size (days) to assess price stability
SIGMA2_STABLE_THR = 0.05  # log-price range < this -> day counted as calm

# --- Model size --------------------------------------------------------------
N_INDUCING = 10

# --- Topology ----------------------------------------------------------------
N_GRID = 200
N_SAMPLES = 200
MIN_CROSSING_SEP = 10

# --- Reproducibility ---------------------------------------------------------
SEED = 42

# --- Daily topology look-back for RunGP_update bootstrap ---------------------
# Number of trailing calendar days for which daily topology is pre-computed
# and stored in the pickle so RunGP_update can immediately produce slope
# signals on its first run without a full 90-day state replay.
RECENT_DAILY_DAYS = 14

# --- Output ------------------------------------------------------------------
OUTPUT_DIR = "regime_results"
GP_OUTPUT_DIR_ROOT = "gp_results"


# =============================================================================
# HELPERS
# =============================================================================


def _estimate_calm_sigma2(
    x_all: np.ndarray,
    dt_t_pd,
    r_hat_all: np.ndarray,
    dt_step: float,
    stable_days: int = 4,
    stable_thr: float = 0.05,
    min_calm_obs: int = 300,
    console=None,
) -> float:
    """Raw sigma2 estimated from price-calm periods only.

    A day is 'calm' if its ``stable_days``-day rolling log-price range is
    below ``stable_thr`` — the same geometric criterion used for jump
    detection.  Falls back to all observations if fewer than ``min_calm_obs``
    calm observations are found.
    """
    _daily_lp = (
        pd.Series(x_all, index=dt_t_pd).groupby(pd.Grouper(freq="D")).median().dropna()
    )
    _roll_range = _daily_lp.rolling(stable_days, min_periods=stable_days).apply(
        lambda w: float(w.max() - w.min()), raw=True
    )
    _calm_day_set = set(_roll_range[_roll_range < stable_thr].index.normalize())
    _calm_mask = dt_t_pd.normalize().isin(_calm_day_set)
    r_hat_calm = r_hat_all[_calm_mask]

    if len(r_hat_calm) >= min_calm_obs:
        raw_sigma2 = float(np.var(r_hat_calm * dt_step) / dt_step)
        method = f"calm ({len(r_hat_calm)} obs, {len(_calm_day_set)} days)"
    else:
        raw_sigma2 = float(np.var(r_hat_all * dt_step) / dt_step)
        method = (
            f"all obs ({len(r_hat_all)}; "
            f"only {len(r_hat_calm)} calm obs < {min_calm_obs})"
        )
    if console is not None:
        console.print(f"  sigma2 raw [{method}]: {raw_sigma2:.4g}")
    return raw_sigma2


def compute_km_spatial_var(
    km_dir,
    window_list,
    phase_a_si,
    x_range=None,
    drift_trim_pct=0.02,
    kernel_half_width=0,
    trim_quantile=0.0,
    ema_halflife_windows=None,
    console=None,
):
    """Spatial variance Var[mu(x)] in /year^2, estimated from KM drift bins.

    Reads all per-window KM CSVs, concatenates the annualised drift values,
    applies a light trim to remove sparse-bin outliers, and returns the
    variance.  Used as spatial_var in the GP prior so that the GP amplitude
    is calibrated to the observed drift variability in x.

    Parameters
    ----------
    ema_halflife_windows : float or None
        When set, bins from each window are weighted by an exponential decay
        w_i = 2^(-(n-1-i) / halflife) where i=0 is the oldest loaded window
        and i=n-1 is the newest.  The weighted variance is returned instead of
        the plain variance.  None (default) gives equal weights.
    """
    console = console or Console()
    kernel_tag = f"_k{kernel_half_width}" if kernel_half_width > 0 else ""
    trim_tag = f"_trim{trim_quantile}" if trim_quantile > 0 else ""
    frames = []
    for w_idx, (w_start, w_end) in enumerate(window_list):
        fname = (
            f"km_{w_start.strftime('%Y-%m-%d')}_to_"
            f"{w_end.strftime('%Y-%m-%d')}_{phase_a_si}s{kernel_tag}{trim_tag}.csv"
        )
        path = os.path.join(km_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path).dropna(subset=["drift"])
            if not df.empty:
                df["_w_idx"] = w_idx
                frames.append(df)

    if not frames:
        console.print(
            f"[yellow]compute_km_spatial_var: no KM CSVs in {km_dir}; "
            "using fallback.[/yellow]"
        )
        return None

    km_all = pd.concat(frames, ignore_index=True)
    km_all["drift_ann"] = km_all["drift"] * _SEC_PER_YEAR

    if x_range is not None:
        x_lo_km, x_hi_km = x_range
        in_range = (km_all["bin_center"] >= x_lo_km) & (km_all["bin_center"] <= x_hi_km)
        working = km_all[in_range].copy() if in_range.any() else km_all.copy()
    else:
        working = km_all.copy()

    d_all = working["drift_ann"].values

    # EMA weights: newest window → weight 1, older windows → exponential decay
    if ema_halflife_windows is not None:
        n_total = len(window_list)
        ages = (n_total - 1 - working["_w_idx"].values).astype(float)
        w_all = np.exp(-ages * np.log(2.0) / ema_halflife_windows)
    else:
        w_all = np.ones(len(working), dtype=float)

    # Trim by unweighted quantile to remove sparse-bin outliers
    if len(d_all) >= 10 and drift_trim_pct > 0:
        lo = float(np.quantile(d_all, drift_trim_pct))
        hi = float(np.quantile(d_all, 1.0 - drift_trim_pct))
        mask = (d_all >= lo) & (d_all <= hi)
        d, w = d_all[mask], w_all[mask]
    else:
        d, w = d_all, w_all

    if len(d) < 3:
        d, w = d_all, w_all

    w_sum = w.sum()
    mu_w = float((w * d).sum() / w_sum)
    sp_var = float((w * (d - mu_w) ** 2).sum() / w_sum)

    ema_tag = (
        f"  ema_hl={int(round(ema_halflife_windows * 30.44))}d"
        if ema_halflife_windows is not None
        else ""
    )
    console.print(
        f"  KM spatial Var={sp_var:.4g} /yr^2  "
        f"range=[{d.min():.1f}, {d.max():.1f}]/yr  "
        f"n_bins={len(d)}{ema_tag}"
    )
    return sp_var


# =============================================================================
# SHARED PIPELINE INIT  (Steps 1–6, importable by backtest_jumps etc.)
# =============================================================================


def init_gp_pipeline(
    start_date,
    end_date,
    km_init_months=0,
    km_ema_halflife_windows=None,
    phase_a_si=30,
    kernel_hw_phase_a=3,
    phase_gp_si=900,
    kernel_hw=50,
    trim_quantile=0.01,
    n_bins=200,
    weight_threshold=5,
    n_inducing=10,
    temporal_ls_days=5.0,
    spatial_lengthscale_init=None,
    spatial_var_mode="km",
    spatial_var_fixed=300.0,
    sigma2_override=None,
    sigma2_stable_days=4,
    sigma2_stable_thr=0.05,
    ema_halflife_days=0.0,
    regime_output_dir="regime_results",
    console=None,
) -> dict:
    """Steps 1–6: download, aggregate, KM estimation, GP series load,
    spatial-var determination, and Kalman-GP model initialisation.

    Parameters
    ----------
    km_init_months : int
        Months of history before ``start_date`` to use for KM estimation.
        0 (default) uses the same ``[start_date, end_date]`` period as the GP
        run — identical to the old ``prepare_phase_a`` behaviour.
        When > 0, raw data for the pre-period is downloaded and aggregated,
        KM estimation covers the pre-period only, and the GP series is loaded
        strictly from ``[start_date, end_date]``.

    Returns
    -------
    dict with keys: model, x_all, dx_all, r_hat_all, dt_step, dt_t, dt_t_pd,
        x_range_global, sp_var, sigma2, raw_sigma2, snapped_start, snapped_end,
        gp_windows, km_windows, regime_output_dir, n_inducing, sl_init.
    """
    console = console or Console()

    # --- Snap GP window boundaries ---
    snapped_start, snapped_end = normalize_window_boundaries(
        start_date, end_date, "monthly", console=console
    )
    gp_windows = list(iter_windows(snapped_start, snapped_end, "monthly"))

    # --- KM pre-period (causal: only history before snapped_start) ---
    if km_init_months > 0:
        km_start_ts = pd.Timestamp(snapped_start) - pd.DateOffset(months=km_init_months)
        km_end_ts = pd.Timestamp(snapped_start) - pd.Timedelta(days=1)
        km_start, km_end = normalize_window_boundaries(
            km_start_ts.to_pydatetime(),
            km_end_ts.to_pydatetime(),
            "monthly",
            console=console,
        )
        km_windows = list(iter_windows(km_start, km_end, "monthly"))
        data_start = km_start
    else:
        km_windows = gp_windows
        km_start = snapped_start
        km_end = snapped_end
        data_start = snapped_start

    # --- Step 1: Download raw data ---
    console.rule("[bold cyan]Step 1 — Download raw data")
    dc.ensure_data(data_start, snapped_end)

    # --- Step 2: Aggregate log-returns ---
    console.rule("[bold cyan]Step 2 — Aggregate log-returns")
    for w_start, w_end in km_windows:
        dc.aggregate_log_returns_range(
            w_start,
            w_end,
            phase_a_si,
            kernel_half_width=kernel_hw_phase_a,
            trim_quantile=trim_quantile,
            ema_halflife_days=0.0,
        )
    for w_start, w_end in gp_windows:
        dc.aggregate_log_returns_range(
            w_start,
            w_end,
            phase_gp_si,
            kernel_half_width=kernel_hw,
            trim_quantile=trim_quantile,
            ema_halflife_days=ema_halflife_days,
        )

    # --- Step 3: Phase A KM estimation ---
    console.rule("[bold cyan]Step 3 — Phase A (KM estimation)")
    run_phase_a(
        km_start,
        km_end,
        phase_a_si,
        kernel_half_width=kernel_hw_phase_a,
        trim_quantile=trim_quantile,
        ema_halflife_days=0.0,
        n_bins=n_bins,
        weight_threshold=weight_threshold,
        window_type="monthly",
        output_dir=regime_output_dir,
        console=console,
    )

    # --- Step 4: Load Phase GP series ---
    console.rule("[bold cyan]Step 4 — Load Phase GP series")
    x_all, dx_all, dt_step, dt_t = load_series(
        snapped_start,
        snapped_end,
        phase_gp_si,
        kernel_half_width=kernel_hw,
        trim_quantile=trim_quantile,
        ema_halflife_days=ema_halflife_days,
        window_type="monthly",
    )
    dt_t_pd = pd.to_datetime(dt_t)
    N = len(dx_all)
    console.print(f"  N = {N} increments,  dt = {dt_step:.0f}s")

    r_hat_all = (dx_all / dt_step) * _SEC_PER_YEAR
    console.print(
        f"  r_hat: mean={r_hat_all.mean():+.3e}/yr  std={r_hat_all.std():.3e}/yr  "
        f"var={r_hat_all.var():.3e} /yr^2"
    )

    x_lo = float(np.percentile(x_all, 1))
    x_hi = float(np.percentile(x_all, 99))
    x_range_global = (x_lo, x_hi)

    if sigma2_override is not None:
        raw_sigma2 = sigma2_override
    else:
        raw_sigma2 = _estimate_calm_sigma2(
            x_all,
            dt_t_pd,
            r_hat_all,
            dt_step,
            stable_days=sigma2_stable_days,
            stable_thr=sigma2_stable_thr,
            console=console,
        )

    # --- Step 5: Determine spatial_var ---
    console.rule("[bold cyan]Step 5 — Determine spatial_var")
    if spatial_var_mode == "km":
        km_dir = os.path.join(regime_output_dir, "km")
        # When km_init_months > 0 the KM pre-period covers a different price
        # range than the GP period, so the GP x_range filter would incorrectly
        # cut out most KM bins.  Only apply x_range filtering when both periods
        # are the same (km_init_months == 0 → km_windows == gp_windows).
        km_x_range = x_range_global if km_init_months == 0 else None
        sp_var = compute_km_spatial_var(
            km_dir,
            km_windows,
            phase_a_si,
            x_range=km_x_range,
            kernel_half_width=kernel_hw_phase_a,
            trim_quantile=trim_quantile,
            ema_halflife_windows=km_ema_halflife_windows,
            console=console,
        )
        if sp_var is None or sp_var < 1.0:
            sp_var = spatial_var_fixed
            console.print(
                f"[yellow]KM var unusable; using fallback spatial_var={sp_var}[/yellow]"
            )
    else:
        sp_var = spatial_var_fixed
        console.print(f"  spatial_var={sp_var:.4g} [fixed]")

    if sigma2_override is None:
        sigma2 = max(raw_sigma2 - sp_var, 0.01 * raw_sigma2)
        console.print(
            f"  sigma2 estimation: raw={raw_sigma2:.4g}  "
            f"sp_var={sp_var:.4g}  corrected={sigma2:.4g}"
        )
    else:
        sigma2 = raw_sigma2

    obs_noise = sigma2 / dt_step
    gp_snr = sp_var / obs_noise
    snr_msg = (
        f"  spatial_var={sp_var:.4g}  obs_noise={obs_noise:.4g}  "
        f"GP SNR/obs={gp_snr:.4g}"
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

    # --- Step 6: Initialise Kalman-GP model ---
    console.rule("[bold cyan]Step 6 — Initialise Kalman-GP model")
    sl_init = (
        spatial_lengthscale_init
        if spatial_lengthscale_init is not None
        else (x_hi - x_lo) / (3 * n_inducing)
    )

    model = KalmanGPDriftModel(
        spatial_lengthscale=sl_init,
        temporal_lengthscale_days=temporal_ls_days,
        spatial_variance=sp_var,
        sigma2=sigma2,
        dt=float(dt_step),
    )
    model.initialise(x_range=x_range_global, n_inducing=n_inducing, data_x=x_all)
    console.print(
        f"  inducing M={model.M}  state_dim={2 * model.M}  "
        f"x_range=[{x_lo:.4f},{x_hi:.4f}]  "
        f"spatial_ls={model.spatial_ls:.4f}  "
        f"temporal_ls={model.temporal_ls:.2f}d  "
        f"spatial_var={model.spatial_var:.4g}"
    )

    return {
        "model": model,
        "x_all": x_all,
        "dx_all": dx_all,
        "r_hat_all": r_hat_all,
        "dt_step": float(dt_step),
        "dt_t": dt_t,
        "dt_t_pd": dt_t_pd,
        "x_range_global": x_range_global,
        "sp_var": sp_var,
        "sigma2": sigma2,
        "raw_sigma2": raw_sigma2,
        "snapped_start": snapped_start,
        "snapped_end": snapped_end,
        "gp_windows": gp_windows,
        "km_windows": km_windows,
        "regime_output_dir": regime_output_dir,
        "n_inducing": n_inducing,
        "sl_init": sl_init,
    }


# =============================================================================
# PIPELINE HELPERS
# =============================================================================


def prepare_phase_a(snapped_start, snapped_end, window_list, console):
    """Steps 1–3: download, aggregate, and run Phase A KM estimation.

    Uses the module-level config variables so callers don't need to thread
    every parameter through.
    """
    console.rule("[bold cyan]Step 1 — Download raw data")
    dc.ensure_data(snapped_start, snapped_end)

    console.rule("[bold cyan]Step 2 — Aggregate log-returns")
    for si in sorted({phase_a_seconds_interval, phase_gp_seconds_interval}):
        for w_start, w_end in window_list:
            dc.aggregate_log_returns_range(
                w_start,
                w_end,
                si,
                kernel_half_width=kernel_half_width,
                trim_quantile=trim_quantile,
                ema_halflife_days=EMA_HALFLIFE_DAYS
                if si == phase_gp_seconds_interval
                else 0.0,
            )

    console.rule("[bold cyan]Step 3 — Phase A (KM estimation)")
    run_phase_a(
        snapped_start,
        snapped_end,
        phase_a_seconds_interval,
        kernel_half_width=kernel_half_width_phase_a,
        trim_quantile=trim_quantile,
        ema_halflife_days=0.0,
        n_bins=n_bins,
        weight_threshold=weight_threshold,
        window_type="monthly",
        output_dir=OUTPUT_DIR,
        console=console,
    )


# =============================================================================
# PIPELINE
# =============================================================================


def main() -> None:
    import sys

    console = Console(
        file=open(
            sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False
        )
    )
    rng = np.random.default_rng(SEED)

    # ---- Steps 1–6: data collection + model initialisation ------------------
    _init = init_gp_pipeline(
        start_date,
        end_date,
        km_init_months=0,
        phase_a_si=phase_a_seconds_interval,
        kernel_hw_phase_a=kernel_half_width_phase_a,
        phase_gp_si=phase_gp_seconds_interval,
        kernel_hw=kernel_half_width,
        trim_quantile=trim_quantile,
        n_bins=n_bins,
        weight_threshold=weight_threshold,
        n_inducing=N_INDUCING,
        temporal_ls_days=TEMPORAL_LENGTHSCALE_DAYS_INIT,
        spatial_lengthscale_init=SPATIAL_LENGTHSCALE_INIT,
        spatial_var_mode=SPATIAL_VAR_SOURCE,
        spatial_var_fixed=SPATIAL_VAR_FIXED,
        sigma2_override=SIGMA2,
        sigma2_stable_days=SIGMA2_STABLE_DAYS,
        sigma2_stable_thr=SIGMA2_STABLE_THR,
        ema_halflife_days=EMA_HALFLIFE_DAYS,
        regime_output_dir=OUTPUT_DIR,
        console=console,
    )
    model = _init["model"]
    x_prev = _init["x_all"]
    dx = _init["dx_all"]
    r_hat = _init["r_hat_all"]
    dt_t = _init["dt_t"]
    dt_t_pd = _init["dt_t_pd"]
    x_range_global = _init["x_range_global"]
    snapped_start = _init["snapped_start"]
    snapped_end = _init["snapped_end"]
    window_list = _init["gp_windows"]
    _n_inducing_eff = _init["n_inducing"]

    stage_tag = (
        f"{SPATIAL_VAR_SOURCE}var_{'reproject' if USE_REPROJECT else 'noreproject'}"
    )
    gp_dir = gp_output_dir(phase_gp_seconds_interval, root=GP_OUTPUT_DIR_ROOT)
    os.makedirs(gp_dir, exist_ok=True)

    N = len(dx)
    t_seconds = (dt_t_pd.astype(np.int64) / 1e9).values.astype(float)
    dt_series = pd.Series(dt_t_pd)

    # Pre-compute window→observation index map (reused in Kalman loop)
    window_idx_arr = np.full(N, -1, dtype=int)
    for w_idx, (w_start, w_end) in enumerate(window_list):
        mask = (dt_series >= pd.Timestamp(w_start)) & (
            dt_series < pd.Timestamp(w_end) + pd.Timedelta(days=1)
        )
        window_idx_arr[mask.values] = w_idx

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    console.rule("[bold cyan]Step 7 — Sequential Kalman updates + topology")

    topology_rows = []
    snapshots = []  # (datetime, state_mean, state_cov, x_range_for_topo, inducing_x)
    snapshot_every = 1
    snapshot_counter = 0

    for w_idx, (w_start, w_end) in enumerate(window_list):
        obs_mask = window_idx_arr == w_idx
        if not obs_mask.any():
            continue

        x_w = x_prev[obs_mask]
        r_hat_w = r_hat[obs_mask]
        t_w = t_seconds[obs_mask]
        dt_w = dt_t[obs_mask]

        # --- Weekly rolling-window reproject ------------------------------------
        # Split this window's observations into ISO-weekly sub-batches.
        # Each sub-batch starts with a reproject to the 30-day trailing x-range,
        # so the inducing grid shifts smoothly with the price instead of jumping
        # at calendar-month boundaries.
        dt_w_pd = pd.to_datetime(dt_w)
        dt_w_series = pd.Series(dt_w_pd)
        week_periods = dt_w_series.dt.to_period("W").values
        topo_range = x_range_global  # fallback if USE_REPROJECT=False

        if USE_REPROJECT:
            for wk in sorted(set(week_periods)):
                wk_mask = week_periods == wk
                x_wk = x_w[wk_mask]
                r_hat_wk = r_hat_w[wk_mask]
                t_wk = t_w[wk_mask]
                dt_wk_first = dt_w_pd[wk_mask].min()

                # 30-day rolling x-range strictly before this week's first obs
                # — causal: no current-week data enters the inducing grid placement.
                roll_cutoff = dt_wk_first - pd.Timedelta(days=1)
                roll_lo = roll_cutoff - pd.Timedelta(days=REPROJECT_WINDOW_DAYS - 1)
                roll_mask_all = (dt_t_pd >= roll_lo) & (dt_t_pd <= roll_cutoff)
                x_roll = x_prev[roll_mask_all]
                if len(x_roll) < 2:
                    x_roll = x_prev[dt_t_pd < dt_wk_first]
                if len(x_roll) < 2:
                    x_roll = x_wk

                topo_range = model.reproject_to_range(
                    float(x_roll.min()), float(x_roll.max()), n_inducing=_n_inducing_eff
                )

                model.update(x_wk, r_hat_wk, t_wk)
        else:
            model.update(x_w, r_hat_w, t_w)

        # --- Topology at end of window ---
        topo = topology_from_gp(
            model,
            topo_range,
            n_grid=N_GRID,
            n_samples=N_SAMPLES,
            min_crossing_sep=MIN_CROSSING_SEP,
            min_barrier_fraction=min_barrier_fraction,
            rng=rng,
        )
        dt_query = pd.Timestamp(dt_w[-1])
        topology_rows.append(
            {
                "datetime": dt_query,
                "window_start": str(pd.Timestamp(w_start).date()),
                "p_multiwell_gp": topo["p_multiwell"],
                "mean_n_wells": topo["mean_n_wells"],
                "barrier_mean": topo["barrier_mean"],
                "barrier_std": topo["barrier_std"],
                "u_range": topo["u_range"],
                "mu_std_to_mean": topo["mu_std_to_mean"],
            }
        )

        snapshot_counter += 1
        if snapshot_counter % snapshot_every == 0:
            snapshots.append(
                (
                    dt_query,
                    model.state_mean.copy(),
                    model.state_cov.copy(),
                    topo_range,
                    model.inducing_x.copy(),  # needed for correct predict() when reprojecting
                    pd.Timestamp(w_start),  # for log-price slice in topology plot
                )
            )

        console.print(
            f"  window {w_idx}  "
            f"p_multi_gp={topo['p_multiwell']:.2f}  "
            f"barrier={topo['barrier_mean']:.2f}  "
            f"sigma/mu={topo['mu_std_to_mean']:.2f}"
        )

    df_topology = pd.DataFrame(topology_rows)
    df_params = pd.DataFrame([model.get_params()])

    # -------------------------------------------------------------------------
    console.rule(
        f"[bold cyan]Step 7b — Daily topology (last {RECENT_DAILY_DAYS} days "
        "for RunGP_update bootstrap)"
    )
    # Replay the last RECENT_DAILY_DAYS calendar days day-by-day using a
    # separate model restored from the nearest preceding monthly snapshot.
    # This gives RunGP_update daily p_multiwell/barrier rows immediately so
    # slope signals (slope_p_multiwell, slope_z_p_multiwell) are available
    # without an expensive 90-day state rebuild on the first update run.
    recent_start_dt = (
        pd.Timestamp(snapped_end) - pd.Timedelta(days=RECENT_DAILY_DAYS - 1)
    ).normalize()
    best_snap_rd, best_snap_rd_dt = None, None
    for snap in snapshots:
        snap_dt = pd.Timestamp(snap[0])
        if snap_dt < recent_start_dt:
            if best_snap_rd_dt is None or snap_dt > best_snap_rd_dt:
                best_snap_rd, best_snap_rd_dt = snap, snap_dt

    recent_daily_topo: list[dict] = []
    if best_snap_rd is not None:
        p_rd = model.get_params()
        rdm = KalmanGPDriftModel(
            spatial_lengthscale=p_rd["spatial_lengthscale"],
            temporal_lengthscale_days=p_rd["temporal_lengthscale_days"],
            spatial_variance=p_rd["spatial_variance"],
            sigma2=float(model.sigma2),
            dt=float(model.dt),
        )
        rdm.inducing_x = np.asarray(best_snap_rd[4]).copy()
        rdm.M = len(rdm.inducing_x)
        rdm._recompute_hp_dependent()
        rdm._I_2M = np.eye(2 * rdm.M)
        rdm.state_mean = np.asarray(best_snap_rd[1]).copy()
        rdm.state_cov = np.asarray(best_snap_rd[2]).copy()

        rd_replay_start = (
            (best_snap_rd_dt + pd.Timedelta(days=1)).normalize().to_pydatetime()
        )
        x_rd, dx_rd, dt_rd, dt_t_rd = load_series(
            rd_replay_start,
            pd.Timestamp(snapped_end).to_pydatetime(),
            phase_gp_seconds_interval,
            kernel_half_width=kernel_half_width,
            trim_quantile=trim_quantile,
            ema_halflife_days=EMA_HALFLIFE_DAYS,
            window_type="monthly",
        )
        dt_t_rd_pd = pd.to_datetime(dt_t_rd)
        r_hat_rd = (dx_rd / dt_rd) * _SEC_PER_YEAR
        topo_range_rd = tuple(best_snap_rd[3])
        last_repr_rd: pd.Timestamp | None = None

        for d in sorted(dt_t_rd_pd.normalize().unique()):
            if last_repr_rd is None or (d - last_repr_rd).days >= 7:
                roll_lo_rd = d - pd.Timedelta(days=REPROJECT_WINDOW_DAYS - 1)
                rmask = (dt_t_rd_pd.normalize() >= roll_lo_rd) & (
                    dt_t_rd_pd.normalize() <= d
                )
                xr = x_rd[rmask]
                if len(xr) >= 2:
                    topo_range_rd = rdm.reproject_to_range(
                        float(xr.min()),
                        float(xr.max()),
                        n_inducing=_n_inducing_eff,
                    )
                    last_repr_rd = d

            dmask = dt_t_rd_pd.normalize() == d
            t_d_rd = (dt_t_rd_pd[dmask].astype(np.int64) / 1e9).values.astype(float)
            rdm.update(x_rd[dmask], r_hat_rd[dmask], t_d_rd)

            if d < recent_start_dt:
                continue

            topo_d = topology_from_gp(
                rdm,
                topo_range_rd,
                n_grid=N_GRID,
                n_samples=N_SAMPLES,
                min_crossing_sep=MIN_CROSSING_SEP,
                min_barrier_fraction=min_barrier_fraction,
                rng=rng,
            )
            recent_daily_topo.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "p_multiwell": topo_d["p_multiwell"],
                    "barrier_mean": topo_d["barrier_mean"],
                    "barrier_std": topo_d["barrier_std"],
                    "barrier_snr": (
                        topo_d["barrier_mean"] / topo_d["barrier_std"]
                        if topo_d["barrier_std"] > 0
                        else 0.0
                    ),
                    "mean_n_wells": topo_d["mean_n_wells"],
                }
            )
            console.print(
                f"  {d.date()}  p_multi={topo_d['p_multiwell']:.3f}"
                f"  barrier={topo_d['barrier_mean']:.4f}±{topo_d['barrier_std']:.4f}"
            )
        del rdm
        console.print(
            f"  [green]{len(recent_daily_topo)} daily rows will be saved to pickle[/green]"
        )
    else:
        console.print(
            f"  [yellow]No monthly snapshot before {recent_start_dt.date()}; "
            "recent_daily_topo omitted (RunGP_update will rebuild from scratch).[/yellow]"
        )

    stem = (
        f"gp_{pd.Timestamp(snapped_start).strftime('%Y-%m-%d')}_to_"
        f"{pd.Timestamp(snapped_end).strftime('%Y-%m-%d')}"
        f"_{phase_gp_seconds_interval}s_{stage_tag}"
    )
    df_topology.to_csv(os.path.join(gp_dir, f"{stem}_topology.csv"), index=False)
    df_params.to_csv(os.path.join(gp_dir, f"{stem}_params.csv"), index=False)
    console.print(f"[green]Wrote[/green] {gp_dir}/{stem}_topology.csv")

    # -------------------------------------------------------------------------
    console.rule("[bold cyan]Step 8 — GP potential U(x) topology snapshots")
    plot_topology_snapshots(
        model,
        x_range_global,
        snapshots,
        snapped_start,
        snapped_end,
        phase_gp_seconds_interval,
        os.path.join(gp_dir, f"{stem}_topology_snapshots.png"),
        n_grid=N_GRID,
        n_samples=N_SAMPLES,
        rng=rng,
    )

    console.rule("[bold cyan]Step 9 \u2014 log-price vs topology plot")
    # Build a daily log-price series over the full GP period and detect jumps.
    _daily_lp = (
        pd.Series(x_prev, index=dt_t_pd).groupby(pd.Grouper(freq="D")).median().dropna()
    )
    events_df = detect_price_jumps(
        _daily_lp,
        stable_days=SIGMA2_STABLE_DAYS,
        stable_thr=SIGMA2_STABLE_THR,
        jump_thr=0.1,
        settle_days=SIGMA2_STABLE_DAYS,
    )
    console.print(f"  {len(events_df)} jump events detected for plot")
    plot_logprice_topology(
        model,
        snapshots,
        x_prev,
        dt_t,
        snapped_start,
        snapped_end,
        phase_gp_seconds_interval,
        os.path.join(gp_dir, f"{stem}_logprice_topology.png"),
        spatial_var_source=SPATIAL_VAR_SOURCE,
        use_reproject=USE_REPROJECT,
        n_grid=N_GRID,
        events_df=events_df,
    )

    console.rule("[bold cyan]Step 10 \u2014 GP drift + KM overlay plot")
    plot_drift_with_km(
        model,
        snapshots,
        snapped_start,
        snapped_end,
        phase_a_seconds_interval,
        phase_gp_seconds_interval,
        OUTPUT_DIR,
        os.path.join(gp_dir, f"{stem}_drift_km.png"),
        spatial_var_source=SPATIAL_VAR_SOURCE,
        use_reproject=USE_REPROJECT,
        n_grid=N_GRID,
        rng=rng,
        kernel_hw_phase_a=kernel_half_width_phase_a,
        trim_quantile=trim_quantile,
    )

    console.rule("[bold cyan]Step 11 — Persist model state")
    state_path = os.path.join(gp_dir, f"{stem}_state.pkl")
    state_blob = {
        "schema_version": 1,
        "pipeline": "gp",
        # config snapshot
        "config": {
            "start_date": pd.Timestamp(snapped_start),
            "end_date": pd.Timestamp(snapped_end),
            "phase_a_seconds_interval": phase_a_seconds_interval,
            "phase_gp_seconds_interval": phase_gp_seconds_interval,
            "kernel_half_width": kernel_half_width,
            "kernel_half_width_phase_a": kernel_half_width_phase_a,
            "trim_quantile": trim_quantile,
            "ema_halflife_days": EMA_HALFLIFE_DAYS,
            "window_type": "monthly",
            "n_bins": n_bins,
            "weight_threshold": weight_threshold,
            "min_barrier_fraction": min_barrier_fraction,
            "spatial_var_source": SPATIAL_VAR_SOURCE,
            "use_reproject": USE_REPROJECT,
            "reproject_margin": REPROJECT_MARGIN,
            "n_inducing": _n_inducing_eff,
            "n_grid": N_GRID,
            "n_samples": N_SAMPLES,
            "min_crossing_sep": MIN_CROSSING_SEP,
            "seed": SEED,
            "output_dir": OUTPUT_DIR,
            "gp_dir": gp_dir,
            "stem": stem,
        },
        # Kalman-GP state
        "model_params": model.get_params(),
        "dt": float(model.dt),
        "sigma2": float(model.sigma2),
        "inducing_x": model.inducing_x.copy(),
        "state_mean": model.state_mean.copy(),
        "state_cov": model.state_cov.copy(),
        "x_range_global": x_range_global,
        # bookkeeping
        "last_dt": pd.Timestamp(dt_t[-1]),
        "snapshots": snapshots,
        "topology_history": df_topology.copy(),
        "recent_daily_topo": recent_daily_topo,
    }
    with open(state_path, "wb") as fh:
        pickle.dump(state_blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
    console.print(f"[green]Wrote[/green] {state_path}")

    console.rule("[bold green]Done")


if __name__ == "__main__":
    main()
