"""
RunGP — non-stationary Kalman-GP pipeline with explicit trend tracking.

This is the trend-augmented sibling of RunGP_simple.py.  It addresses the
"downward-sloping U(x)" problem seen in trending periods by adding a scalar
Kalman-tracked trend state beta(t):

    r_hat_t = beta(t) + mu(x_{t-1}, t) + eps_t

beta absorbs the slow, x-independent drift; mu(x) captures only the residual
x-dependent structure.  Topology is computed from mu only, so trends no
longer disguise themselves as one-sided wells.

Reproject_to_range is dropped — the trend term handles the bulk translation
that reprojecting was trying to follow.  Inducing points are placed once over
the global (p1, p99) x-range; increase N_INDUCING for per-window resolution.

Run:
    python RunGP.py

Key knobs vs RunGP_simple:
    TREND_LENGTHSCALE_DAYS  default 30 — temporal lengthscale of beta;
                            must be > TEMPORAL_LENGTHSCALE_DAYS_INIT (GP)
    TREND_VAR_FRACTION      default 0.7 — share of KM total variance allocated
                            to the trend (vs GP).  Higher = more aggressive
                            trend absorption.
    SPATIAL_VAR_FRACTION    default 0.3 — share allocated to spatial GP.

Calibration check on a trending month:
    - beta should track the per-month average annualised log-return
    - sigma/mu should mostly land below 2.0 once beta has stabilised
    - GP wells should appear as clean zero crossings of mu(x), not slope
"""

import os
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
from phase_GP import topology_from_gp, _SEC_PER_YEAR
from phase_GP_trend import KalmanGPDriftWithTrendModel
from plots import plot_topology_snapshots, plot_logprice_topology, plot_drift_with_km


# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Date range ---------------------------------------------------------------
start_date = datetime(2025, 1, 1)
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
min_barrier_fraction = 0.15
min_well_separation = 0.0

# =============================================================================
# MODEL CONFIG
# =============================================================================

# --- KM-derived total drift variance ----------------------------------------
# The Phase-A KM bin variance estimates Var[beta + mu(x)].  We split it
# between the trend and the spatial GP via these fractions.
# Raising SPATIAL_VAR_FRACTION from 0.3 to 0.45 gives the spatial GP more
# signal budget so that x-dependent mean-reversion is not absorbed by beta.
SPATIAL_VAR_FRACTION = 0.45
TREND_VAR_FRACTION = 1 - SPATIAL_VAR_FRACTION
SPATIAL_VAR_FALLBACK = SPATIAL_VAR_FRACTION * 1000.0
TREND_VAR_FALLBACK = TREND_VAR_FRACTION * 1000.0

# --- Lengthscales (initial values; no HP-opt by default) --------------------
# Defaults tuned empirically on ETH 2024 monthly @900s:
#   spatial_ls = (x_hi - x_lo) / N_INDUCING  (one corr. length per inducing
#   interval; just smooth enough that mu(x) shows clear oscillations,
#   not so smooth that beta absorbs all structure).
#   temporal_ls = 21d: GP shape memory spans ~3 weeks so observations from
#   across the full monthly window all contribute to the end-of-month state.
#   (With 7d temporal_ls observations from 3 weeks ago had only ~0.6% weight.)
#   trend_ls = 90d: 4.3x separation from GP temporal_ls preserves beta/mu
#   identifiability while letting beta evolve on a quarterly timescale.
# None -> (x_hi - x_lo) / (2 * N_INDUCING): half the inducing spacing,
# matching the window-scale resolution without over-smoothing.
SPATIAL_LENGTHSCALE_INIT = None  # None -> (x_hi - x_lo) / (2 * N_INDUCING)
TEMPORAL_LENGTHSCALE_DAYS_INIT = 21.0
TREND_LENGTHSCALE_DAYS = 90.0

# --- Observation noise -------------------------------------------------------
SIGMA2 = None  # None -> var(r_hat * dt) / dt  [/year]^2 * sec

# --- Model size --------------------------------------------------------------
# N=80: covers (x_hi - x_lo) ~ 0.6 with spacing ~0.0075 so monthly windows
# of width ~0.1 see ~13 effective inducing points.  No reproject needed.
# Increasing from 30 gives denser per-window coverage, reducing the
# variance-dominated posterior that comes from only ~5 inducing pts/window.
N_INDUCING = 20

# --- Topology ----------------------------------------------------------------
N_GRID = 200
N_SAMPLES = 200
MIN_CROSSING_SEP = 10

# Fraction of prior variance below which an inducing point is considered
# "active" (meaningfully updated by observations).  topo_range is clipped to
# the span of active inducing points + 1 spatial_ls margin, preventing
# stale/unobserved inducing points from inflating posterior variance.
# 0.95 = 5% variance reduction required; lower values are stricter.
ACTIVE_VAR_THRESHOLD = 0.85

# At the START of each window, inflate the f-block of P_ff this fraction
# toward the prior covariance K_zz.  Prevents Kalman-gain collapse: after
# many monthly windows the posterior P_ff shrinks to a small steady-state
# value and new observations can barely move f_inducing.  0.3 = inject 30%
# of prior variance back each window so the GP can re-learn spatial structure.
# Set to 0.0 to disable (pure sequential Kalman with no per-window reset).
WINDOW_RESET_FRACTION = 0.3

# --- Reproducibility ---------------------------------------------------------
SEED = 42

# --- Output ------------------------------------------------------------------
OUTPUT_DIR = "regime_results"
GP_OUTPUT_DIR_ROOT = "gp_results_dynamic"


# =============================================================================
# HELPERS
# =============================================================================


def compute_km_total_var(
    km_dir, window_list, phase_a_si, x_range=None, drift_trim_pct=0.02, console=None
):
    """Total KM variance Var[beta + mu(x)] in /year^2."""
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
            f"[yellow]compute_km_total_var: no KM CSVs in {km_dir}; "
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

    total_var = float(np.var(d))
    console.print(
        f"  KM total Var[beta+mu]={total_var:.4g} /yr^2  "
        f"range=[{d.min():.1f}, {d.max():.1f}]/yr  "
        f"n_bins={len(d)}"
    )
    return total_var


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
    stage_tag = f"trend_ls{int(TREND_LENGTHSCALE_DAYS)}d_N{N_INDUCING}"
    gp_output_dir = os.path.join(
        GP_OUTPUT_DIR_ROOT,
        f"{phase_gp_seconds_interval}s_trend",
        "no_hp",
    )
    os.makedirs(gp_output_dir, exist_ok=True)

    window_list = list(iter_windows(snapped_start, snapped_end, window_type))

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

    # Per-window-type overrides
    _n_inducing_eff = N_INDUCING
    _temporal_ls_eff = TEMPORAL_LENGTHSCALE_DAYS_INIT
    _trend_ls_eff = TREND_LENGTHSCALE_DAYS
    if window_type == "weekly":
        _n_inducing_eff = min(N_INDUCING, 12)
        _temporal_ls_eff = min(TEMPORAL_LENGTHSCALE_DAYS_INIT, 4.0)
        _trend_ls_eff = min(TREND_LENGTHSCALE_DAYS, 10.0)
        console.print(
            f"[yellow]weekly overrides:[/yellow] "
            f"N_INDUCING {N_INDUCING}->{_n_inducing_eff}  "
            f"temporal_ls {TEMPORAL_LENGTHSCALE_DAYS_INIT}->{_temporal_ls_eff}d  "
            f"trend_ls {TREND_LENGTHSCALE_DAYS}->{_trend_ls_eff}d"
        )

    sl_init = (
        SPATIAL_LENGTHSCALE_INIT
        if SPATIAL_LENGTHSCALE_INIT is not None
        else (x_hi - x_lo) / (2 * max(_n_inducing_eff, 1))
    )

    console.rule("[bold cyan]Step 5 — Determine spatial_var and trend_var")
    km_dir = os.path.join(OUTPUT_DIR, "km")
    total_var = compute_km_total_var(
        km_dir,
        window_list,
        phase_a_seconds_interval,
        x_range=x_range_global,
        console=console,
    )
    if total_var is None or total_var < 1.0:
        sp_var = SPATIAL_VAR_FALLBACK
        trend_var = TREND_VAR_FALLBACK
        console.print(
            f"[yellow]Using fallback: spatial_var={sp_var}  "
            f"trend_var={trend_var}[/yellow]"
        )
    else:
        sp_var = SPATIAL_VAR_FRACTION * total_var
        trend_var = TREND_VAR_FRACTION * total_var

    console.print(
        f"  spatial_var={sp_var:.4g}  trend_var={trend_var:.4g}  "
        f"obs_noise={obs_noise:.4g}  "
        f"GP SNR/obs={sp_var / obs_noise:.4g}  "
        f"beta SNR/obs={trend_var / obs_noise:.4g}"
    )

    console.rule("[bold cyan]Step 6 — Initialise trend-augmented Kalman-GP")
    model = KalmanGPDriftWithTrendModel(
        spatial_lengthscale=sl_init,
        temporal_lengthscale_days=_temporal_ls_eff,
        spatial_variance=sp_var,
        trend_lengthscale_days=_trend_ls_eff,
        trend_variance=trend_var,
        sigma2=sigma2,
        dt=float(dt),
    )
    model.initialise(x_range=x_range_global, n_inducing=_n_inducing_eff, data_x=x_prev)
    console.print(
        f"  inducing M={model.M}  state_dim={2 + 2 * model.M}  "
        f"x_range=[{x_lo:.4f},{x_hi:.4f}]  "
        f"spatial_ls={model.spatial_ls:.4f}  "
        f"temporal_ls={model.temporal_ls:.2f}d  "
        f"trend_ls={model.trend_ls:.2f}d  "
        f"spatial_var={model.spatial_var:.4g}  "
        f"trend_var={model.trend_var:.4g}"
    )

    console.rule("[bold cyan]Step 7 — Sequential Kalman updates + topology")
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

    topology_rows = []
    snapshots = []
    snapshot_every = 4 if window_type == "weekly" else 1
    snapshot_counter = 0

    # Indices into state vector for the f (GP value) components.
    # State layout: [beta, beta', f_1, f'_1, ..., f_M, f'_M]
    idx_f_state = 2 + np.arange(0, 2 * model.M, 2)

    for w_idx in labels_ts.index:
        obs_mask = window_idx_arr == w_idx
        if not obs_mask.any():
            continue

        x_w = x_prev[obs_mask]
        r_hat_w = r_hat[obs_mask]
        t_w = t_seconds[obs_mask]
        dt_w = dt_t[obs_mask]
        row = labels_ts.loc[w_idx]

        # Per-window P_ff inflation: blend current posterior covariance toward
        # the prior K_zz so that Kalman gain doesn't collapse over many months.
        if WINDOW_RESET_FRACTION > 0.0:
            P_f = model.state_cov[np.ix_(idx_f_state, idx_f_state)]
            model.state_cov[np.ix_(idx_f_state, idx_f_state)] = (
                1.0 - WINDOW_RESET_FRACTION
            ) * P_f + WINDOW_RESET_FRACTION * model._K_zz_jit

        # Update after the P_ff inflation so the inflated covariance drives
        # a meaningful Kalman gain within this window.
        model.update(x_w, r_hat_w, t_w)

        # Clip topo_range to inducing points with meaningfully reduced posterior
        # variance (P_ff_diag < ACTIVE_VAR_THRESHOLD * spatial_var).  Stale
        # inducing points at prior inflate variance and produce flat potentials.
        p_ff_diag = np.diag(model.state_cov)[idx_f_state]
        active_mask = p_ff_diag < ACTIVE_VAR_THRESHOLD * model.spatial_var
        ls_margin = 1.0 * model.spatial_ls
        if active_mask.sum() >= 2:
            active_inducing = model.inducing_x[active_mask]
            topo_range = (
                max(x_lo, float(active_inducing.min()) - ls_margin),
                min(x_hi, float(active_inducing.max()) + ls_margin),
            )
        else:
            topo_range = (
                max(x_lo, float(x_w.min()) - ls_margin),
                min(x_hi, float(x_w.max()) + ls_margin),
            )

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
                "beta": model.beta_mean,
                "beta_std": model.beta_std,
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
                    model.inducing_x.copy(),
                    pd.Timestamp(row["window_start"]),
                )
            )

        p_multi_a = float(row.get("p_multiwell", np.nan))
        p_a_str = f"{p_multi_a:.2f}" if np.isfinite(p_multi_a) else "n/a"
        console.print(
            f"  window {w_idx}  "
            f"p_multi_gp={topo['p_multiwell']:.2f}  "
            f"p_multi_a={p_a_str}  "
            f"barrier={topo['barrier_mean']:.2f}  "
            f"sigma/mu={topo['mu_std_to_mean']:.2f}  "
            f"beta={model.beta_mean:+.2f}\u00b1{model.beta_std:.2f}/yr"
        )

    df_topology = pd.DataFrame(topology_rows)
    df_params = pd.DataFrame([model.get_params()])

    stem = (
        f"trend_{pd.Timestamp(snapped_start).strftime('%Y-%m-%d')}_to_"
        f"{pd.Timestamp(snapped_end).strftime('%Y-%m-%d')}"
        f"_{phase_gp_seconds_interval}s_{stage_tag}"
    )
    df_topology.to_csv(os.path.join(gp_output_dir, f"{stem}_topology.csv"), index=False)
    df_params.to_csv(os.path.join(gp_output_dir, f"{stem}_params.csv"), index=False)
    console.print(f"[green]Wrote[/green] {gp_output_dir}/{stem}_topology.csv")

    console.rule("[bold cyan]Step 8 — GP potential U(x) snapshots")
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

    console.rule("[bold cyan]Step 9 — log-price vs topology plot")
    plot_logprice_topology(
        model,
        snapshots,
        x_prev,
        dt_t,
        snapped_start,
        snapped_end,
        phase_gp_seconds_interval,
        os.path.join(gp_output_dir, f"{stem}_logprice_topology.png"),
        spatial_var_source="km_split",
        hp_opt_mode="none",
        use_reproject=False,
        n_grid=N_GRID,
    )

    console.rule("[bold cyan]Step 10 — GP drift + KM overlay plot")
    plot_drift_with_km(
        model,
        snapshots,
        snapped_start,
        snapped_end,
        phase_a_seconds_interval,
        phase_gp_seconds_interval,
        OUTPUT_DIR,
        os.path.join(gp_output_dir, f"{stem}_drift_km.png"),
        spatial_var_source="km_split",
        hp_opt_mode="none",
        use_reproject=False,
        n_grid=N_GRID,
        rng=rng,
    )

    console.rule("[bold green]Done")


if __name__ == "__main__":
    main()
