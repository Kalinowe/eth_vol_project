"""
RunGP_simple — stationary Kalman-GP pipeline.

Main execution script for the simple (stationary) sequential GP pipeline.
All configuration is in the CONFIGURATION block below.  Run with:

    $env:PYTHONIOENCODING='utf-8'; & ".venv/Scripts/python.exe" RunGP_simple.py

Pipeline steps
--------------
Stage 1  upfront 3-param HP-opt (spatial_ls, temporal_ls, spatial_var)
         optional 2-param HP-opt for spatial_ls and temporal_ls only;
         spatial_var is held fixed at the KM-derived value (HP_OPT_MODE='ls_only').

    1  Download raw data
    2  Aggregate log-returns
    3  Phase A (KM bins + regime labels)
    4  Load Phase GP series
    5  Determine spatial_var from KM or fixed value
    6  Initialise Kalman-GP model
    7  HP optimisation (if HP_OPT_MODE != 'none')
    8  Sequential Kalman updates + topology per window
         For each window: optional reproject_to_range → Kalman update → topology_from_gp
    9  GP potential U(x) topology snapshots
   10  Log-price vs topology plot
   11  GP drift + KM overlay plot
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
    print_regime_table,
)
from phase_GP import KalmanGPDriftModel, topology_from_gp, _SEC_PER_YEAR
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
window_type = "monthly"  # 'weekly' | 'biweekly' | 'monthly'
n_bins = 200
weight_threshold = 5
min_barrier_fraction = 0.1
min_well_separation = 0.0


# =============================================================================
# MODEL CONFIG
# =============================================================================

# --- Spatial variance source --------------------------------------------------
# 'km'    — compute var(KM annualised drift bins) from Phase A output
# 'fixed' — use SPATIAL_VAR_FIXED directly
SPATIAL_VAR_SOURCE = "km"
SPATIAL_VAR_FIXED = 300.0  # used when SPATIAL_VAR_SOURCE='fixed' or as fallback

# --- HP optimisation ----------------------------------------------------------
# 'none'    — use initial-guess lengthscales throughout (fastest)
# 'ls_only' — optimise spatial_ls and temporal_ls; hold spatial_var fixed
#              spatial_var is set from KM data; MLL is nearly flat in it
# 'all'     — jointly optimise spatial_ls, temporal_ls, spatial_var (not recommended)
HP_OPT_MODE = "none"
HP_OPT_N_RESTARTS = 3
HP_OPT_MAX_SAMPLES = 5000  # subsample observations fed to MLL optimiser

# --- Reprojection (move inducing points into each window's observed x-range) --
USE_REPROJECT = True
REPROJECT_MARGIN = 0.1  # fraction of window x-width added on each side

# --- Lengthscales (initial values; starting points for HP-opt or final values
#     when HP_OPT_MODE='none') -------------------------------------------------
# spatial_ls: None -> (x_hi - x_lo) / (N_INDUCING)  — smaller than
#   the inducing spacing, matching the dynamic pipeline (N=30) resolution.
#   Smaller ls lets the posterior capture sharper well/barrier features in drift.
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

# --- Output ------------------------------------------------------------------
OUTPUT_DIR = "regime_results"
GP_OUTPUT_DIR_ROOT = "gp_results"


# =============================================================================
# HELPERS
# =============================================================================


def compute_km_spatial_var(
    km_dir, window_list, phase_a_si, x_range=None, drift_trim_pct=0.02, console=None
):
    """Spatial variance Var[mu(x)] in /year^2, estimated from KM drift bins.

    Reads all per-window KM CSVs, concatenates the annualised drift values,
    applies a light trim to remove sparse-bin outliers, and returns the
    variance.  Used as spatial_var in the GP prior so that the GP amplitude
    is calibrated to the observed drift variability in x.
    """
    console = console or Console()
    frames = []
    for w_start, w_end in window_list:
        fname = (
            f"km_{w_start.strftime('%Y-%m-%d')}_to_"
            f"{w_end.strftime('%Y-%m-%d')}_{phase_a_si}s.csv"
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
        window_type,
        console=console,
    )
    stage_tag = (
        f"{SPATIAL_VAR_SOURCE}var_{HP_OPT_MODE}hp_"
        f"{'reproject' if USE_REPROJECT else 'noreproject'}"
    )
    hp_subdir = "no_hp" if HP_OPT_MODE == "none" else "hp"
    gp_output_dir = os.path.join(
        GP_OUTPUT_DIR_ROOT,
        f"{phase_gp_seconds_interval}s_simple",
        hp_subdir,
    )
    os.makedirs(gp_output_dir, exist_ok=True)

    window_list = list(iter_windows(snapped_start, snapped_end, window_type))

    # -------------------------------------------------------------------------
    console.rule("[bold cyan]Step 1 — Download raw data")
    dc.ensure_data(snapped_start, snapped_end)

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    console.rule("[bold cyan]Step 3 — Phase A (KM bins + regime labels)")
    labels_df = run_phase_a(
        snapped_start,
        snapped_end,
        phase_a_seconds_interval,
        kernel_half_width=kernel_half_width_phase_a,
        trim_quantile=trim_quantile,
        n_bins=n_bins,
        weight_threshold=weight_threshold,
        min_barrier_fraction=min_barrier_fraction,
        min_well_separation=min_well_separation,
        window_type=window_type,
        output_dir=OUTPUT_DIR,
        console=console,
    )
    print_regime_table(labels_df, console=console)

    # -------------------------------------------------------------------------
    console.rule("[bold cyan]Step 4 — Load Phase GP series")
    x_prev, dx, dt, dt_t = load_series(
        snapped_start,
        snapped_end,
        phase_gp_seconds_interval,
        kernel_half_width=kernel_half_width,
        trim_quantile=trim_quantile,
        window_type=window_type,
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

    sigma2 = SIGMA2 if SIGMA2 is not None else float(np.var(r_hat * dt) / dt)
    obs_noise = sigma2 / dt

    # Pre-compute window→observation index map (reused in HP opt + Kalman loop)
    labels_ts = labels_df.copy()
    labels_ts["window_start"] = pd.to_datetime(labels_ts["window_start"])
    labels_ts["window_end"] = pd.to_datetime(labels_ts["window_end"])
    dt_series = pd.Series(pd.to_datetime(dt_t))

    window_idx_arr = np.full(N, -1, dtype=int)
    for i, row in labels_ts.iterrows():
        mask = (dt_series >= row["window_start"]) & (
            dt_series < row["window_end"] + pd.Timedelta(days=1)
        )
        window_idx_arr[mask.values] = i

    # Per-window-type overrides
    _n_inducing_eff = N_INDUCING
    _temporal_ls_eff = TEMPORAL_LENGTHSCALE_DAYS_INIT
    _hp_opt_mode_eff = HP_OPT_MODE
    if window_type == "weekly":
        _n_inducing_eff = min(N_INDUCING, 6)
        _temporal_ls_eff = min(
            TEMPORAL_LENGTHSCALE_DAYS_INIT, 4.0
        )  # forget after ~4 days
        _hp_opt_mode_eff = "none"  # MLL is too flat at weekly SNR
        console.print(
            f"[yellow]weekly overrides:[/yellow] "
            f"N_INDUCING {N_INDUCING}->{_n_inducing_eff}  "
            f"temporal_ls {TEMPORAL_LENGTHSCALE_DAYS_INIT}->{_temporal_ls_eff}d  "
            f"HP_OPT_MODE {HP_OPT_MODE!r}->'none'"
        )

    # Spatial lengthscale initial guess.
    # Rule: smaller than the inducing spacing matches the dynamic pipeline
    # (N=30 / range) resolution without tripling the state dimension.
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

    console.print(
        f"  spatial_var={sp_var:.4g}  obs_noise={obs_noise:.4g}  "
        f"GP SNR/obs={sp_var / obs_noise:.4g}"
    )

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
    if _hp_opt_mode_eff != "none":
        console.rule(f"[bold cyan]Step 7 — HP optimisation ({_hp_opt_mode_eff})")

        # Subsample observations for MLL
        x_hp = x_prev.copy()
        r_hp = r_hat.copy()
        t_hp = t_seconds.copy()
        if len(x_hp) > HP_OPT_MAX_SAMPLES:
            idx_sub = np.linspace(0, len(x_hp) - 1, HP_OPT_MAX_SAMPLES, dtype=int)
            x_hp, r_hp, t_hp = x_hp[idx_sub], r_hp[idx_sub], t_hp[idx_sub]

        # bounds_range = median per-window x-width so ls bounds are calibrated
        # to the window scale (where topology is evaluated), not the full history.
        win_widths = []
        for w_idx in labels_ts.index:
            xw = x_prev[window_idx_arr == w_idx]
            if len(xw) > 1:
                win_widths.append(float(xw.max() - xw.min()))
        if win_widths:
            med_width = float(np.median(win_widths))
            bounds_range = (x_lo, x_lo + med_width)
        else:
            bounds_range = x_range_global

        model.optimise_hp(
            x_hp,
            r_hp,
            t_hp,
            n_restarts=HP_OPT_N_RESTARTS,
            x_range=x_range_global,
            bounds_range=bounds_range,
            fix_spatial_var=(_hp_opt_mode_eff == "ls_only"),
            console=console,
        )
        # Re-initialise state with optimised HPs (don't carry stale prior cov).
        model.initialise(
            x_range=x_range_global, n_inducing=_n_inducing_eff, data_x=x_prev
        )
        console.print(
            f"  Post-opt:  spatial_ls={model.spatial_ls:.4f}  "
            f"temporal_ls={model.temporal_ls:.2f}d  "
            f"spatial_var={model.spatial_var:.4g}"
        )
    else:
        console.rule("[bold cyan]Step 7 — HP optimisation SKIPPED (HP_OPT_MODE=none)")

    # -------------------------------------------------------------------------
    console.rule("[bold cyan]Step 8 — Sequential Kalman updates + topology")

    topology_rows = []
    snapshots = []  # (datetime, state_mean, state_cov, x_range_for_topo, inducing_x)
    snapshot_every = 4 if window_type == "weekly" else 1
    snapshot_counter = 0

    for w_idx in labels_ts.index:
        obs_mask = window_idx_arr == w_idx
        if not obs_mask.any():
            continue

        x_w = x_prev[obs_mask]
        r_hat_w = r_hat[obs_mask]
        t_w = t_seconds[obs_mask]
        dt_w = dt_t[obs_mask]
        row = labels_ts.loc[w_idx]

        # --- Reproject inducing points to this window's observed x-range ---
        if USE_REPROJECT:
            x_w_width = max(float(x_w.max() - x_w.min()), 1e-4)
            margin = REPROJECT_MARGIN * x_w_width
            x_w_lo = float(x_w.min()) - margin
            x_w_hi = float(x_w.max()) + margin
            model.reproject_to_range(x_w_lo, x_w_hi, n_inducing=_n_inducing_eff)
            topo_range = (x_w_lo, x_w_hi)
        else:
            topo_range = x_range_global

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
                "window_start": str(pd.Timestamp(row["window_start"]).date()),
                "p_multiwell_gp": topo["p_multiwell"],
                "p_multiwell_a": float(row.get("p_multiwell", np.nan)),
                "mean_n_wells": topo["mean_n_wells"],
                "barrier_mean": topo["barrier_mean"],
                "barrier_std": topo["barrier_std"],
                "kramers": topo["kramers_mean"],
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
                    pd.Timestamp(
                        row["window_start"]
                    ),  # for log-price slice in topology plot
                )
            )

        p_multi_a = float(row.get("p_multiwell", np.nan))
        p_a_str = f"{p_multi_a:.2f}" if np.isfinite(p_multi_a) else "n/a"
        console.print(
            f"  window {w_idx}  "
            f"p_multi_gp={topo['p_multiwell']:.2f}  "
            f"p_multi_a={p_a_str}  "
            f"barrier={topo['barrier_mean']:.2f}  "
            f"sigma/mu={topo['mu_std_to_mean']:.2f}"
        )

    df_topology = pd.DataFrame(topology_rows)
    df_params = pd.DataFrame([model.get_params()])

    stem = (
        f"simple_{pd.Timestamp(snapped_start).strftime('%Y-%m-%d')}_to_"
        f"{pd.Timestamp(snapped_end).strftime('%Y-%m-%d')}"
        f"_{phase_gp_seconds_interval}s_{stage_tag}"
    )
    df_topology.to_csv(os.path.join(gp_output_dir, f"{stem}_topology.csv"), index=False)
    df_params.to_csv(os.path.join(gp_output_dir, f"{stem}_params.csv"), index=False)
    console.print(f"[green]Wrote[/green] {gp_output_dir}/{stem}_topology.csv")

    # -------------------------------------------------------------------------
    console.rule("[bold cyan]Step 9 — GP potential U(x) topology snapshots")
    plot_topology_snapshots(
        model,
        x_range_global,
        snapshots,
        snapped_start,
        snapped_end,
        phase_gp_seconds_interval,
        os.path.join(gp_output_dir, f"{stem}_topology_snapshots.png"),
        n_grid=N_GRID,
        n_samples=N_SAMPLES,
        rng=rng,
    )

    console.rule("[bold cyan]Step 10 — log-price vs topology plot")
    plot_logprice_topology(
        model,
        snapshots,
        x_prev,
        dt_t,
        snapped_start,
        snapped_end,
        phase_gp_seconds_interval,
        os.path.join(gp_output_dir, f"{stem}_logprice_topology.png"),
        spatial_var_source=SPATIAL_VAR_SOURCE,
        hp_opt_mode=HP_OPT_MODE,
        use_reproject=USE_REPROJECT,
        n_grid=N_GRID,
    )

    console.rule("[bold cyan]Step 11 — GP drift + KM overlay plot")
    plot_drift_with_km(
        model,
        snapshots,
        snapped_start,
        snapped_end,
        phase_a_seconds_interval,
        phase_gp_seconds_interval,
        OUTPUT_DIR,
        os.path.join(gp_output_dir, f"{stem}_drift_km.png"),
        spatial_var_source=SPATIAL_VAR_SOURCE,
        hp_opt_mode=HP_OPT_MODE,
        use_reproject=USE_REPROJECT,
        n_grid=N_GRID,
        rng=rng,
    )

    console.rule("[bold cyan]Step 12 — Persist model state")
    state_path = os.path.join(gp_output_dir, f"{stem}_state.pkl")
    state_blob = {
        "schema_version": 1,
        "pipeline": "simple",
        # config snapshot
        "config": {
            "start_date": pd.Timestamp(snapped_start),
            "end_date": pd.Timestamp(snapped_end),
            "phase_a_seconds_interval": phase_a_seconds_interval,
            "phase_gp_seconds_interval": phase_gp_seconds_interval,
            "kernel_half_width": kernel_half_width,
            "kernel_half_width_phase_a": kernel_half_width_phase_a,
            "trim_quantile": trim_quantile,
            "window_type": window_type,
            "n_bins": n_bins,
            "weight_threshold": weight_threshold,
            "min_barrier_fraction": min_barrier_fraction,
            "min_well_separation": min_well_separation,
            "spatial_var_source": SPATIAL_VAR_SOURCE,
            "hp_opt_mode": HP_OPT_MODE,
            "use_reproject": USE_REPROJECT,
            "reproject_margin": REPROJECT_MARGIN,
            "n_inducing": _n_inducing_eff,
            "n_grid": N_GRID,
            "n_samples": N_SAMPLES,
            "min_crossing_sep": MIN_CROSSING_SEP,
            "seed": SEED,
            "output_dir": OUTPUT_DIR,
            "gp_output_dir": gp_output_dir,
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
    }
    with open(state_path, "wb") as fh:
        pickle.dump(state_blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
    console.print(f"[green]Wrote[/green] {state_path}")

    console.rule("[bold green]Done")


if __name__ == "__main__":
    main()
