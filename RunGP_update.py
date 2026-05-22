"""
RunGP_update — incremental Kalman-GP update with new data.

Loads the persisted state pickle from `RunGP_simple.py`, ingests new daily
data appended after the saved `last_dt`, runs a Kalman update on the new
window, and produces diagnostics + forecasts in:

    GP_updates/{NEW_END_DATE}/

WHAT THIS SCRIPT DOES
---------------------
1.  Loads the most recent `*_state.pkl` from `gp_results/{si}s_simple/{no_hp|hp}/`.
2.  Ensures + aggregates new data from the day after `state['last_dt']` through
    NEW_END_DATE (today by default).
3.  Reprojects the GP to the new window's x-range (mirrors RunGP_simple.py).
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


# =============================================================================
# CONFIGURATION
# =============================================================================

# Source of the previous Kalman-GP state.
# - None: auto-pick the newest `*_state.pkl` under STATE_SEARCH_DIR.
# - explicit path: load exactly this pickle.
STATE_PATH: str | None = None
STATE_SEARCH_DIR: str = os.path.join("gp_results", "900s_simple", "no_hp")

# End of the update window. None ⇒ today (UTC date).
# NB the start of the update window is auto-derived from the loaded state.
NEW_END_DATE: datetime | None = None

# Reprojection margin (same convention as RunGP_simple.py).
REPROJECT_MARGIN = 0.1

# KM bins for the "observed drift red dots" overlay.
KM_N_BINS = 80
KM_WEIGHT_THRESHOLD = 5

# Forecast horizons in days.
FORECAST_HORIZONS = [1, 3, 7, 14, 30]

# Trend window: number of trailing topology rows used for slope tests.
TREND_LOOKBACK = 6

# Topology + sampling.
N_GRID = 200
N_SAMPLES = 200
MIN_CROSSING_SEP = 10
MIN_BARRIER_FRACTION = 0.1
SEED = 42

# Output root.
OUTPUT_ROOT = "GP_updates"


# =============================================================================
# HELPERS
# =============================================================================


def _autopick_state(search_dir: str) -> str:
    cands = sorted(glob.glob(os.path.join(search_dir, "*_state.pkl")))
    if not cands:
        raise FileNotFoundError(
            f"No `*_state.pkl` found in {search_dir}. "
            "Run RunGP_simple.py first to persist a base state."
        )
    return max(cands, key=os.path.getmtime)


def _restore_model(blob: dict) -> KalmanGPDriftModel:
    """Rebuild a KalmanGPDriftModel from a state pickle."""
    p = blob["model_params"]
    m = KalmanGPDriftModel(
        spatial_lengthscale=p["spatial_lengthscale"],
        temporal_lengthscale_days=p["temporal_lengthscale_days"],
        spatial_variance=p["spatial_variance"],
        sigma2=blob["sigma2"],
        dt=blob["dt"],
    )
    # Lay down inducing grid + HP caches without resetting state to prior.
    m.inducing_x = np.asarray(blob["inducing_x"]).copy()
    m.M = len(m.inducing_x)
    m._recompute_hp_dependent()
    m._I_2M = np.eye(2 * m.M)
    m.state_mean = np.asarray(blob["state_mean"]).copy()
    m.state_cov = np.asarray(blob["state_cov"]).copy()
    return m


def _predict_at(model: KalmanGPDriftModel, x_scalar: float) -> tuple[float, float]:
    """Posterior mean and variance of mu at a single x location (current state)."""
    mu_mean, mu_var = model.predict(np.array([x_scalar]), full_cov=False)
    return float(mu_mean[0]), float(mu_var[0])


def _km_bins_for_overlay(
    x_log_prices: np.ndarray,
    seconds_interval: float,
    n_bins: int,
    weight_threshold: int,
    x_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """KM drift bins (annualised) over the new window for the red-dot overlay."""
    df = estimate_km(
        x_log_prices,
        seconds_interval=seconds_interval,
        n_bins=n_bins,
        weight_threshold=weight_threshold,
    ).dropna(subset=["drift"])
    if df.empty:
        return df
    df = df.copy()
    df["drift_ann"] = df["drift"] * _SEC_PER_YEAR
    if x_range is not None:
        m = (df["bin_center"] >= x_range[0]) & (df["bin_center"] <= x_range[1])
        df = df[m].copy()
    return df.sort_values("bin_center").reset_index(drop=True)


def _is_last_day_of_month(d: datetime) -> bool:
    """Return True if *d* is the last calendar day of its month."""
    return (d + timedelta(days=1)).month != d.month


def _recalibrate_from_km(
    model: KalmanGPDriftModel,
    new_start_date: datetime,
    new_end_date: datetime,
    si_a: int,
    khw: int,
    trim: float,
    cfg: dict,
    console: Console,
) -> dict:
    """
    Run full Phase-A KM on the completed month, save the canonical
    ``km_{start}_to_{end}_{si_a}s.csv`` next to the existing Phase-A KM
    files, derive a fresh ``spatial_var`` from the bins, and patch the model
    in-place.

    This mirrors what ``run_phase_a`` does at the end of each calendar
    month so that the GP amplitude stays calibrated to the new month's
    volatility.  Called only when ``new_end_date`` is the last day of its
    month.

    Returns a dict of recalibration metadata written into diagnostics.json.
    """
    km_dir = os.path.join(cfg.get("output_dir", "regime_results"), "km")
    os.makedirs(km_dir, exist_ok=True)

    # Load phase-A series for the new window.
    x_prev_a, dx_a, _, _ = load_series(
        new_start_date, new_end_date, si_a,
        kernel_half_width=khw, trim_quantile=trim,
        window_type=None,
    )
    if len(x_prev_a) == 0:
        console.print("[yellow]  recalibrate: no phase-A data; skipping.[/yellow]")
        return {"recalibrated": False, "reason": "no_phase_a_data"}

    log_prices_a = np.concatenate(
        [x_prev_a, x_prev_a[-1:] + dx_a[-1:]]
    ) if len(dx_a) else x_prev_a

    n_bins = int(cfg.get("n_bins", 200))
    wt = int(cfg.get("weight_threshold", 5))
    km_full = estimate_km(
        log_prices_a, seconds_interval=si_a,
        n_bins=n_bins, weight_threshold=wt,
    )

    # Save canonical KM CSV so ``plot_drift_with_km`` will find it next run.
    km_csv = os.path.join(
        km_dir,
        f"km_{new_start_date.strftime('%Y-%m-%d')}_to_"
        f"{new_end_date.strftime('%Y-%m-%d')}_{si_a}s.csv",
    )
    km_full.to_csv(km_csv, index=False)
    console.print(f"  [green]KM CSV saved:[/green] {km_csv}  ({n_bins} bins)")

    # Derive spatial_var from the trimmed, in-range KM drift bins.
    km_valid = km_full.dropna(subset=["drift"])
    if km_valid.empty:
        console.print("[yellow]  recalibrate: all KM bins below weight threshold; skipping var update.[/yellow]")
        return {"recalibrated": False, "reason": "all_bins_below_threshold",
                "km_csv": km_csv}

    d_ann = km_valid["drift"].values * _SEC_PER_YEAR
    if len(d_ann) >= 10:
        lo = float(np.quantile(d_ann, 0.02))
        hi = float(np.quantile(d_ann, 0.98))
        d_trim = d_ann[(d_ann >= lo) & (d_ann <= hi)]
        if len(d_trim) >= 3:
            d_ann = d_trim
    new_sp_var = float(np.var(d_ann))

    if new_sp_var < 1.0:
        console.print(
            f"[yellow]  recalibrate: new spatial_var={new_sp_var:.4g} too small; skipping.[/yellow]"
        )
        return {"recalibrated": False, "reason": "spatial_var_too_small",
                "km_csv": km_csv, "new_spatial_var": new_sp_var}

    old_sp_var = float(model.spatial_var)
    model.spatial_var = new_sp_var
    model._recompute_hp_dependent()
    console.print(
        f"  spatial_var:  {old_sp_var:.4g} → {new_sp_var:.4g}  "
        f"(ratio={new_sp_var/old_sp_var:.3f})"
    )
    return {
        "recalibrated": True,
        "km_csv": km_csv,
        "km_n_bins": n_bins,
        "old_spatial_var": old_sp_var,
        "new_spatial_var": new_sp_var,
        "spatial_var_ratio": float(new_sp_var / old_sp_var),
    }


def _trend_slope(y: np.ndarray) -> tuple[float, float]:
    """OLS slope of y vs index, plus a crude two-sided z-stat for sign-of-slope.

    Returns (slope_per_step, z_stat). z_stat is meaningful only for length >= 4.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 2:
        return float("nan"), float("nan")
    x = np.arange(n, dtype=float)
    x_c = x - x.mean()
    y_c = y - y.mean()
    sxx = float(np.sum(x_c * x_c))
    if sxx <= 0:
        return float("nan"), float("nan")
    slope = float(np.sum(x_c * y_c) / sxx)
    if n < 4:
        return slope, float("nan")
    resid = y_c - slope * x_c
    s2 = float(np.sum(resid * resid) / (n - 2))
    se = float(np.sqrt(s2 / sxx)) if s2 > 0 else 0.0
    z = slope / se if se > 0 else float("nan")
    return slope, z


# =============================================================================
# PLOTS
# =============================================================================


def _plot_drift_snapshot(model, x_range, km_df, topo, out_path, dt_query):
    x_grid = np.linspace(x_range[0], x_range[1], N_GRID)
    mu_mean, mu_var = model.predict(x_grid, full_cov=False)
    mu_std = np.sqrt(np.maximum(mu_var, 0.0))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.fill_between(
        x_grid, mu_mean - 2 * mu_std, mu_mean + 2 * mu_std,
        color="steelblue", alpha=0.2, label="GP ±2σ",
    )
    ax.plot(x_grid, mu_mean, color="steelblue", linewidth=1.5, label="GP drift")
    if not km_df.empty:
        w = km_df["weight"].values.astype(float)
        sz = 12 + 60 * w / max(w.max(), 1.0)
        ax.scatter(
            km_df["bin_center"], km_df["drift_ann"],
            s=sz, c="crimson", alpha=0.7, edgecolors="none",
            zorder=5, label="observed KM (new data)",
        )
    ax.scatter(
        model.inducing_x, np.zeros(model.M),
        marker="|", color="darkgreen", s=80, zorder=6, label="inducing",
    )
    ax.set_xlabel("log-price")
    ax.set_ylabel("drift [/yr]")
    ax.set_title(
        f"Drift snapshot @ {pd.Timestamp(dt_query).date()}  "
        f"p_multi={topo['p_multiwell']:.2f}  σ/μ={topo['mu_std_to_mean']:.1f}"
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_potential_snapshot(model, x_range, km_df, topo, out_path, dt_query, rng):
    x_grid = np.linspace(x_range[0], x_range[1], N_GRID)
    mu_mean, _ = model.predict(x_grid, full_cov=False)
    U_mean = -cumulative_trapezoid(mu_mean, x_grid, initial=0.0)
    U_mean -= U_mean.min()

    f_samples = model.sample_drift(x_grid, n_samples=N_SAMPLES, rng=rng)
    U_samples = -cumulative_trapezoid(f_samples, x_grid, axis=0, initial=0.0)
    U_samples -= U_samples.min(axis=0, keepdims=True)
    U_std = U_samples.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(
        x_grid, U_mean - 2 * U_std, U_mean + 2 * U_std,
        color="steelblue", alpha=0.2, label="±2σ",
    )
    ax.plot(x_grid, U_mean, color="steelblue", linewidth=1.5, label="GP U(x)")

    # Observed potential from KM bins: U_obs = -∫(km_drift) dx on bin grid.
    if not km_df.empty and len(km_df) >= 3:
        xb = km_df["bin_center"].values
        db = km_df["drift_ann"].values
        U_obs = -cumulative_trapezoid(db, xb, initial=0.0)
        # baseline-align to GP curve at the closest grid point
        gp_at_first = float(np.interp(xb[0], x_grid, U_mean))
        U_obs = U_obs - U_obs[0] + gp_at_first
        ax.plot(xb, U_obs, color="crimson", linewidth=1.0, alpha=0.6,
                linestyle="--", label="observed U_KM (aligned)")
        ax.scatter(xb, U_obs, color="crimson", s=22, alpha=0.85,
                   zorder=5, edgecolors="none")

    ax.scatter(
        model.inducing_x, np.full(model.M, ax.get_ylim()[0]),
        marker="|", color="darkgreen", s=60, zorder=4,
    )
    ax.set_xlabel("log-price")
    ax.set_ylabel("U(x) = −∫μ dx")
    ax.set_title(
        f"Potential snapshot @ {pd.Timestamp(dt_query).date()}  "
        f"barrier={topo['barrier_mean']:.2f}±{topo['barrier_std']:.2f}  "
        f"p_multi={topo['p_multiwell']:.2f}"
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_history_with_forecast(
    history: pd.DataFrame,
    new_row: dict,
    forecasts: pd.DataFrame,
    fields: list[str],
    out_path: str,
    title_suffix: str,
):
    """Stacked time series of (history → new point → forecast horizons)."""
    fig, axes = plt.subplots(
        len(fields), 1, figsize=(10, 2.6 * len(fields)),
        sharex=True, squeeze=False,
    )
    axes = axes.flatten()
    hist = history.copy()
    hist["datetime"] = pd.to_datetime(hist["datetime"])
    new_dt = pd.Timestamp(new_row["datetime"])

    for k, field in enumerate(fields):
        ax = axes[k]
        # Map the canonical field name → the column actually present in `hist`.
        # History was saved by RunGP_simple with `p_multiwell_gp` and `kramers`,
        # while forecasts/new_row use `p_multiwell` and `kramers_mean`.
        hist_field = field
        if field == "p_multiwell" and "p_multiwell_gp" in hist.columns:
            hist_field = "p_multiwell_gp"
        elif field == "kramers_mean" and "kramers" in hist.columns:
            hist_field = "kramers"
        if hist_field in hist.columns:
            ax.plot(hist["datetime"], hist[hist_field], "-o",
                    color="steelblue", markersize=3, label="history")
            # std band if available (barrier)
            std_col = f"{field}_std" if f"{field}_std" in hist.columns else None
            if std_col is None and field == "barrier_mean" and "barrier_std" in hist.columns:
                std_col = "barrier_std"
            if std_col:
                lo = hist[hist_field].values - hist[std_col].values
                hi = hist[hist_field].values + hist[std_col].values
                ax.fill_between(hist["datetime"], lo, hi,
                                color="steelblue", alpha=0.15)
        # New point
        nv = new_row.get(hist_field, new_row.get(field))
        if nv is not None and np.isfinite(nv):
            ax.scatter([new_dt], [nv], color="crimson", s=60,
                       zorder=5, label="new update", edgecolors="black", linewidths=0.6)
        # Forecast
        if not forecasts.empty and field in forecasts.columns:
            fc_dt = [new_dt + pd.Timedelta(days=int(h)) for h in forecasts["horizon_days"]]
            ax.plot(fc_dt, forecasts[field].values, "--^",
                    color="darkorange", markersize=5, label="forecast")
            # forecast std band
            std_field = f"{field.replace('_mean','')}_std" if field.endswith("_mean") else None
            if std_field and std_field in forecasts.columns:
                fcv = forecasts[field].values
                fcs = forecasts[std_field].values
                ax.fill_between(fc_dt, fcv - fcs, fcv + fcs,
                                color="darkorange", alpha=0.15)
        ax.set_ylabel(field, fontsize=9)
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("date")
    fig.suptitle(f"Topology trajectory  |  {title_suffix}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_innovations(
    dt_seq: np.ndarray, innov_z: np.ndarray, out_path: str, dt_query
):
    if len(innov_z) == 0:
        return
    abs_z = np.abs(innov_z)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(dt_seq, innov_z, color="steelblue", linewidth=0.6, alpha=0.7)
    axes[0].axhline(0, color="grey", linewidth=0.5)
    for thr in (2, 3):
        axes[0].axhline(+thr, color="crimson", linewidth=0.4, linestyle=":")
        axes[0].axhline(-thr, color="crimson", linewidth=0.4, linestyle=":")
    axes[0].set_ylabel("innovation z")
    axes[0].set_title(
        f"Per-observation surprise  |  new window ending {pd.Timestamp(dt_query).date()}  "
        f"mean|z|={abs_z.mean():.2f}  frac|z|>2={float(np.mean(abs_z>2)):.2f}"
    )
    # rolling mean |z|
    w = max(10, len(abs_z) // 50)
    if len(abs_z) >= w:
        kernel = np.ones(w) / w
        rm = np.convolve(abs_z, kernel, mode="valid")
        axes[1].plot(dt_seq[w - 1:], rm, color="crimson", linewidth=1.0,
                     label=f"rolling |z|, w={w}")
    axes[1].axhline(np.sqrt(2 / np.pi), color="grey", linestyle="--",
                    linewidth=0.6, label="E[|z|] under H0")
    axes[1].set_ylabel("rolling |z|")
    axes[1].set_xlabel("datetime")
    axes[1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_fragility(history: pd.DataFrame, new_row: dict, out_path: str):
    hist = history.copy()
    hist["datetime"] = pd.to_datetime(hist["datetime"])
    bm = hist["barrier_mean"].replace(0, np.nan)
    fragility = hist["barrier_std"] / bm
    new_dt = pd.Timestamp(new_row["datetime"])
    new_bm = float(new_row.get("barrier_mean", np.nan))
    new_bs = float(new_row.get("barrier_std", np.nan))
    new_frag = new_bs / new_bm if new_bm and np.isfinite(new_bm) and new_bm > 0 else np.nan

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    axes[0].plot(hist["datetime"], hist["mu_std_to_mean"], "-o",
                 color="steelblue", markersize=3, label="history")
    nv = float(new_row.get("mu_std_to_mean", np.nan))
    if np.isfinite(nv):
        axes[0].scatter([new_dt], [nv], color="crimson", s=60, zorder=5,
                        edgecolors="black", linewidths=0.6, label="new")
    axes[0].axhline(1.0, color="grey", linestyle="--", linewidth=0.6,
                    label="σ/μ = 1 (noise-dominated)")
    axes[0].set_ylabel("σ(μ) / |μ|")
    axes[0].set_title("Posterior identifiability + barrier fragility")
    axes[0].legend(fontsize=8, loc="best")
    axes[0].grid(alpha=0.3)

    axes[1].plot(hist["datetime"], fragility, "-o",
                 color="darkorange", markersize=3, label="history")
    if np.isfinite(new_frag):
        axes[1].scatter([new_dt], [new_frag], color="crimson", s=60, zorder=5,
                        edgecolors="black", linewidths=0.6, label="new")
    axes[1].axhline(0.5, color="grey", linestyle="--", linewidth=0.6,
                    label="fragility = 0.5")
    axes[1].set_ylabel("barrier_std / barrier_mean")
    axes[1].set_xlabel("date")
    axes[1].legend(fontsize=8, loc="best")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    console = Console(
        file=open(sys.stdout.fileno(), mode="w",
                  encoding="utf-8", buffering=1, closefd=False)
    )
    rng = np.random.default_rng(SEED)

    # ---------------- Load previous state -------------------------------------
    console.rule("[bold cyan]Step 1 — Load previous Kalman-GP state")
    state_path = STATE_PATH or _autopick_state(STATE_SEARCH_DIR)
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

    # ---------------- Determine new date range --------------------------------
    new_start_date = (last_dt + pd.Timedelta(days=1)).floor("D").to_pydatetime()
    end_arg = NEW_END_DATE or datetime.utcnow().replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    # Binance data lags ~1 day → cap at yesterday.
    yesterday = (datetime.utcnow() - timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
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
            new_start_date, new_end_date, s,
            kernel_half_width=khw, trim_quantile=trim,
        )

    # ---------------- Load Phase GP series for new window ---------------------
    console.rule("[bold cyan]Step 4 — Load new-window Phase GP series")
    x_prev, dx, dt_step, dt_t = load_series(
        new_start_date, new_end_date, si,
        kernel_half_width=khw, trim_quantile=trim,
        window_type=None,
    )
    N = len(dx)
    if N == 0:
        console.print("[yellow]No usable new-window increments. Abort.[/yellow]")
        return
    r_hat = (dx / dt_step) * _SEC_PER_YEAR
    t_seconds = (pd.to_datetime(dt_t).astype(np.int64) / 1e9).values.astype(float)
    console.print(
        f"  N={N}  dt={dt_step:.0f}s  "
        f"x_window=[{x_prev.min():.4f},{x_prev.max():.4f}]  "
        f"r_hat std={r_hat.std():.2e}/yr"
    )

    # ---------------- Reproject + optional month-end KM recalibration ---------
    console.rule("[bold cyan]Step 5 — Reproject + Kalman update")
    x_w_width = max(float(x_prev.max() - x_prev.min()), 1e-4)
    margin = REPROJECT_MARGIN * x_w_width
    new_x_lo = float(x_prev.min()) - margin
    new_x_hi = float(x_prev.max()) + margin
    model.reproject_to_range(new_x_lo, new_x_hi, n_inducing=n_ind)
    topo_range = (new_x_lo, new_x_hi)

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
            model, new_start_date, new_end_date,
            si_a=si_a, khw=khw, trim=trim,
            cfg=cfg, console=console,
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
    console.print(
        f"  mean|z|={mean_abs_z:.2f}  (E[|z|]≈{np.sqrt(2/np.pi):.2f} under H0)  "
        f"frac|z|>2={frac_z_gt2:.2f}"
    )

    # ---------------- Topology for the new window -----------------------------
    console.rule("[bold cyan]Step 6 — Topology on updated state")
    topo = topology_from_gp(
        model, topo_range,
        n_grid=N_GRID, n_samples=N_SAMPLES,
        min_crossing_sep=MIN_CROSSING_SEP,
        min_barrier_fraction=MIN_BARRIER_FRACTION,
        rng=rng,
    )
    dt_query = pd.Timestamp(dt_t[-1])
    new_row = {
        "datetime": dt_query,
        "window_start": str(pd.Timestamp(new_start_date).date()),
        "p_multiwell_gp": topo["p_multiwell"],
        "mean_n_wells": topo["mean_n_wells"],
        "barrier_mean": topo["barrier_mean"],
        "barrier_std": topo["barrier_std"],
        "kramers_mean": topo["kramers_mean"],
        "u_range": topo["u_range"],
        "mu_std_to_mean": topo["mu_std_to_mean"],
    }
    console.print(
        f"  p_multi={topo['p_multiwell']:.2f}  "
        f"barrier={topo['barrier_mean']:.3f}±{topo['barrier_std']:.3f}  "
        f"σ/μ={topo['mu_std_to_mean']:.2f}  "
        f"kramers={topo['kramers_mean']:.2e}"
    )

    # ---------------- KM bins for red-dot overlay -----------------------------
    console.rule("[bold cyan]Step 7 — Empirical KM bins (red-dot overlay)")
    # Use phase-A resolution data for the overlay (denser; matches existing plots).
    x_prev_km, _dx_km, _dt_km, _ = load_series(
        new_start_date, new_end_date, si_a,
        kernel_half_width=khw, trim_quantile=trim,
        window_type=None,
    )
    log_prices_km = np.concatenate([x_prev_km, x_prev_km[-1:] + _dx_km[-1:]]) \
        if len(_dx_km) else x_prev_km
    km_df = _km_bins_for_overlay(
        log_prices_km, seconds_interval=si_a,
        n_bins=KM_N_BINS, weight_threshold=KM_WEIGHT_THRESHOLD,
        x_range=topo_range,
    )
    console.print(f"  KM bins kept: {len(km_df)}")

    # ---------------- Forecasts ----------------------------------------------
    console.rule("[bold cyan]Step 8 — Forecast topology at horizons")
    forecasts = forecast_topology(
        model, FORECAST_HORIZONS, topo_range,
        n_grid=N_GRID, n_samples=N_SAMPLES,
        min_crossing_sep=MIN_CROSSING_SEP,
        min_barrier_fraction=MIN_BARRIER_FRACTION,
        rng=rng,
    )
    # Add fragility/derived columns for convenience.
    fc = forecasts.copy()
    fc["fragility"] = fc["barrier_std"] / fc["barrier_mean"].replace(0, np.nan)
    fc.to_csv(os.path.join(out_dir, "forecast.csv"), index=False)
    console.print(fc.to_string(index=False))

    # ---------------- Trend tests --------------------------------------------
    console.rule("[bold cyan]Step 9 — Trend tests over recent history")
    history = blob["topology_history"].copy()
    history["datetime"] = pd.to_datetime(history["datetime"])
    history = history.sort_values("datetime").reset_index(drop=True)

    tail = history.tail(TREND_LOOKBACK)
    slope_pm, z_pm = _trend_slope(tail["p_multiwell_gp"].values)
    slope_bm, z_bm = _trend_slope(tail["barrier_mean"].values)
    slope_kr, _ = _trend_slope(np.log(np.maximum(tail["kramers"].values, 1e-30))
                               if "kramers" in tail.columns else np.array([]))
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
            if topo["barrier_mean"] > 0 else None
        ),
        "new_mu_std_to_mean": topo["mu_std_to_mean"],
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
        sig = "" if not np.isfinite(z) else (
            "  [bold]SIG[/bold]" if abs(z) > 2 else
            ("  weak" if abs(z) > 1 else "")
        )
        return f"  {label}: slope/window={slope:+.4g} {sign}  z={z:+.2f}{sig}"

    console.print(_verdict(slope_pm, z_pm, "p_multiwell"))
    console.print(_verdict(slope_bm, z_bm, "barrier_mean"))
    console.print(
        f"  fragility (barrier_std/barrier_mean): {diagnostics['new_fragility']}"
        if diagnostics["new_fragility"] is not None else
        "  fragility: undefined (barrier_mean=0)"
    )

    # ---------------- Plots ---------------------------------------------------
    console.rule("[bold cyan]Step 10 — Plots")
    _plot_drift_snapshot(
        model, topo_range, km_df, topo,
        os.path.join(out_dir, "drift_snapshot.png"), dt_query,
    )
    _plot_potential_snapshot(
        model, topo_range, km_df, topo,
        os.path.join(out_dir, "potential_snapshot.png"), dt_query, rng,
    )
    _plot_history_with_forecast(
        history, new_row, fc,
        fields=["p_multiwell", "barrier_mean", "kramers_mean"],
        out_path=os.path.join(out_dir, "topology_trajectory.png"),
        title_suffix=f"updated through {new_end_date.date()}",
    )
    _plot_innovations(
        pd.to_datetime(dt_t), z_innov,
        os.path.join(out_dir, "innovations.png"), dt_query,
    )
    _plot_fragility(
        history, new_row,
        os.path.join(out_dir, "fragility.png"),
    )

    # ---------------- Chained state pickle ------------------------------------
    console.rule("[bold cyan]Step 11 — Save chained state")
    updated_history = pd.concat(
        [history, pd.DataFrame([{
            "datetime": new_row["datetime"],
            "window_start": new_row["window_start"],
            "p_multiwell_gp": new_row["p_multiwell_gp"],
            "p_multiwell_a": np.nan,
            "mean_n_wells": new_row["mean_n_wells"],
            "barrier_mean": new_row["barrier_mean"],
            "barrier_std": new_row["barrier_std"],
            "kramers": new_row["kramers_mean"],
            "u_range": new_row["u_range"],
            "mu_std_to_mean": new_row["mu_std_to_mean"],
        }])],
        ignore_index=True,
    )
    updated_history.to_csv(
        os.path.join(out_dir, "topology_history.csv"), index=False,
    )

    new_snapshots = list(blob.get("snapshots", []))
    new_snapshots.append((
        dt_query,
        model.state_mean.copy(),
        model.state_cov.copy(),
        topo_range,
        model.inducing_x.copy(),
        pd.Timestamp(new_start_date),
    ))
    new_blob = dict(blob)
    new_blob.update({
        "model_params": model.get_params(),
        "inducing_x": model.inducing_x.copy(),
        "state_mean": model.state_mean.copy(),
        "state_cov": model.state_cov.copy(),
        "last_dt": dt_query,
        "snapshots": new_snapshots,
        "topology_history": updated_history,
    })
    chained_path = os.path.join(out_dir, "update_state.pkl")
    with open(chained_path, "wb") as fh:
        pickle.dump(new_blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
    console.print(f"  wrote {chained_path}")

    console.rule("[bold green]Done")


if __name__ == "__main__":
    main()
