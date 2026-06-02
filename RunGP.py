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
         For each window: optional reproject_to_range \u2192 Kalman update \u2192 topology_from_gp
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
from paths import gp_output_dir, gp_state_stem
from plots import plot_topology_snapshots, plot_logprice_topology, plot_drift_with_km


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
# 'km'    — compute var(KM annualised drift bins) from Phase A output
# 'fixed' — use SPATIAL_VAR_FIXED directly
SPATIAL_VAR_SOURCE = "km"
SPATIAL_VAR_FIXED = 300.0  # used when SPATIAL_VAR_SOURCE='fixed' or as fallback

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
TEMPORAL_LENGTHSCALE_DAYS_INIT = 5.0  # days

# --- Observation noise -------------------------------------------------------
SIGMA2 = None  # None -> var(r_hat * dt) / dt  [/year]^2 * sec

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


def compute_km_spatial_var(
    km_dir,
    window_list,
    phase_a_si,
    x_range=None,
    drift_trim_pct=0.02,
    kernel_half_width=0,
    trim_quantile=0.0,
    console=None,
):
    """Spatial variance Var[mu(x)] in /year^2, estimated from KM drift bins.

    Reads all per-window KM CSVs, concatenates the annualised drift values,
    applies a light trim to remove sparse-bin outliers, and returns the
    variance.  Used as spatial_var in the GP prior so that the GP amplitude
    is calibrated to the observed drift variability in x.
    """
    console = console or Console()
    kernel_tag = f"_k{kernel_half_width}" if kernel_half_width > 0 else ""
    trim_tag = f"_trim{trim_quantile}" if trim_quantile > 0 else ""
    frames = []
    for w_start, w_end in window_list:
        fname = (
            f"km_{w_start.strftime('%Y-%m-%d')}_to_"
            f"{w_end.strftime('%Y-%m-%d')}_{phase_a_si}s{kernel_tag}{trim_tag}.csv"
        )
        path = os.path.join(km_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path).dropna(subset=["drift"])
            if not df.empty:
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
    if len(d_all) >= 10 and drift_trim_pct > 0:
        lo = float(np.quantile(d_all, drift_trim_pct))
        hi = float(np.quantile(d_all, 1.0 - drift_trim_pct))
        d = d_all[(d_all >= lo) & (d_all <= hi)]
    else:
        d = d_all

    if len(d) < 3:
        d = d_all

    sp_var = float(np.var(d))
    console.print(
        f"  KM spatial Var={sp_var:.4g} /yr^2  "
        f"range=[{d.min():.1f}, {d.max():.1f}]/yr  "
        f"n_bins={len(d)}"
    )
    return sp_var


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
            )

    console.rule("[bold cyan]Step 3 — Phase A (KM estimation)")
    run_phase_a(
        snapped_start,
        snapped_end,
        phase_a_seconds_interval,
        kernel_half_width=kernel_half_width_phase_a,
        trim_quantile=trim_quantile,
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

    snapped_start, snapped_end = normalize_window_boundaries(
        start_date,
        end_date,
        "monthly",
        console=console,
    )
    stage_tag = (
        f"{SPATIAL_VAR_SOURCE}var_{'reproject' if USE_REPROJECT else 'noreproject'}"
    )
    gp_dir = gp_output_dir(phase_gp_seconds_interval, root=GP_OUTPUT_DIR_ROOT)
    os.makedirs(gp_dir, exist_ok=True)

    window_list = list(iter_windows(snapped_start, snapped_end, "monthly"))

    prepare_phase_a(snapped_start, snapped_end, window_list, console)

    # -------------------------------------------------------------------------
    console.rule("[bold cyan]Step 4 — Load Phase GP series")
    x_prev, dx, dt, dt_t = load_series(
        snapped_start,
        snapped_end,
        phase_gp_seconds_interval,
        kernel_half_width=kernel_half_width,
        trim_quantile=trim_quantile,
        window_type="monthly",
    )
    N = len(dx)
    console.print(f"  N = {N} increments,  dt = {dt:.0f}s")

    r = (dx / dt) * _SEC_PER_YEAR
    r_hat = r.copy()
    console.print(
        f"  r_hat: mean={r_hat.mean():+.3e}/yr  std={r_hat.std():.3e}/yr  "
        f"var={r_hat.var():.3e} /yr^2"
    )

    t_seconds = (pd.to_datetime(dt_t).astype(np.int64) / 1e9).values.astype(float)

    x_lo = float(np.percentile(x_prev, 1))
    x_hi = float(np.percentile(x_prev, 99))
    x_range_global = (x_lo, x_hi)

    # Raw estimate — will be corrected for spatial drift variance after sp_var
    # is resolved in Step 5.  Keep raw_sigma2 separate so the correction can
    # be applied regardless of SPATIAL_VAR_SOURCE.
    raw_sigma2 = SIGMA2 if SIGMA2 is not None else float(np.var(r_hat * dt) / dt)
    sigma2 = raw_sigma2  # placeholder; updated below after sp_var is known

    # Pre-compute window→observation index map (reused in Kalman loop)
    dt_series = pd.Series(pd.to_datetime(dt_t))

    window_idx_arr = np.full(N, -1, dtype=int)
    for w_idx, (w_start, w_end) in enumerate(window_list):
        mask = (dt_series >= pd.Timestamp(w_start)) & (
            dt_series < pd.Timestamp(w_end) + pd.Timedelta(days=1)
        )
        window_idx_arr[mask.values] = w_idx

    _n_inducing_eff = N_INDUCING
    _temporal_ls_eff = TEMPORAL_LENGTHSCALE_DAYS_INIT

    # Spatial lengthscale initial guess: smaller than the inducing spacing
    # so the posterior can resolve well/barrier features.
    sl_init = (
        SPATIAL_LENGTHSCALE_INIT
        if SPATIAL_LENGTHSCALE_INIT is not None
        else (x_hi - x_lo) / (3 * max(_n_inducing_eff, 1))
    )

    # -------------------------------------------------------------------------
    console.rule("[bold cyan]Step 5 — Determine spatial_var")

    if SPATIAL_VAR_SOURCE == "km":
        km_dir = os.path.join(OUTPUT_DIR, "km")
        sp_var = compute_km_spatial_var(
            km_dir,
            window_list,
            phase_a_seconds_interval,
            x_range=x_range_global,
            console=console,
        )
        if sp_var is None or sp_var < 1.0:
            sp_var = SPATIAL_VAR_FIXED
            console.print(
                f"[yellow]KM var unusable; using fallback spatial_var={sp_var}[/yellow]"
            )
    else:
        sp_var = SPATIAL_VAR_FIXED
        console.print(f"  spatial_var={sp_var:.4g} [fixed]")

    # Correct sigma2: Var[r_hat*dt]/dt = sigma2 + Var[mu(x,t)] ≈ sigma2 + sp_var.
    # Subtracting sp_var removes the drift-variation component so sigma2 reflects
    # pure observation noise.  Clamp to 1% of raw to avoid sigma2 ≤ 0.
    if SIGMA2 is None:
        sigma2 = max(raw_sigma2 - sp_var, 0.01 * raw_sigma2)
        console.print(
            f"  sigma2 estimation: raw={raw_sigma2:.4g}  "
            f"sp_var={sp_var:.4g}  corrected={sigma2:.4g}"
        )
    obs_noise = sigma2 / dt
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

    # -------------------------------------------------------------------------
    console.rule("[bold cyan]Step 6 — Initialise Kalman-GP model")
    model = KalmanGPDriftModel(
        spatial_lengthscale=sl_init,
        temporal_lengthscale_days=_temporal_ls_eff,
        spatial_variance=sp_var,
        sigma2=sigma2,
        dt=float(dt),
    )
    model.initialise(x_range=x_range_global, n_inducing=_n_inducing_eff, data_x=x_prev)
    console.print(
        f"  inducing M={model.M}  state_dim={2 * model.M}  "
        f"x_range=[{x_lo:.4f},{x_hi:.4f}]  "
        f"spatial_ls={model.spatial_ls:.4f}  "
        f"temporal_ls={model.temporal_ls:.2f}d  "
        f"spatial_var={model.spatial_var:.4g}"
    )

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    console.rule("[bold cyan]Step 7 — Sequential Kalman updates + topology")

    topology_rows = []
    snapshots = []  # (datetime, state_mean, state_cov, x_range_for_topo, inducing_x)
    snapshot_every = 1
    snapshot_counter = 0
    dt_t_pd = pd.to_datetime(dt_t)  # full datetime array, used for rolling x-range

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
