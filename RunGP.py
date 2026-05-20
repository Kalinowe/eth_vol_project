"""
Phase GP execution button.

Runs the full pipeline that feeds the sequential Kalman-GP drift model:
  1. Download Binance daily dumps (if missing) and aggregate log-prices.
  2. Phase A — regime_estimation: per-window Kramers–Moyal drift,
     GP-based potential, p_multiwell.
  3. Phase B — markov_chain: empirical regime chain; mean dwell time sets
     the Matern temporal lengthscale and the EMA demean halflife.
  4. Phase GP — phase_GP: sequential Kalman-filter GP over the EMA-demeaned
     drift field mu(x, t), with topology extraction and forecasting.

All knobs are declared in the CONFIGURATION block below.
Edit them and re-run this file.
"""

import os
from datetime import datetime

from rich.console import Console

import data_collection as dc
from regime_estimation import (
    iter_windows,
    normalize_window_boundaries,
    run_phase_a,
    print_regime_table,
)
from phase_GP import run_phase_gp


# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Date range ---------------------------------------------------------------
start_date = datetime(2025, 1, 1)
end_date   = datetime(2025, 5, 25)

# --- Data aggregation ---------------------------------------------------------
# Phase A (Kramers–Moyal) prefers small dt (30–300 s): KM is O(dt)-biased and
# large dt blurs well structure via inter-basin jumps.
# Phase GP (Kalman filter) prefers large dt (1800–3600 s): same SNR but fewer
# iterations, smaller per-step obs_noise, and better numerical stability.
phase_a_seconds_interval  = 300    # Phase A sampling interval (seconds)
phase_gp_seconds_interval = 1800   # Phase GP sampling interval (seconds)
kernel_half_width = 50             # smoothing half-width for the log-price aggregator
trim_quantile     = 0.01           # symmetric tail trim on log-prices (0 to disable)

# --- Phase A topology classifier ---------------------------------------------
window_type          = 'monthly'   # 'weekly' | 'biweekly' | 'monthly'
n_bins               = 200        # number of KM bins per window
weight_threshold     = 5          # minimum bin occupancy for a usable KM cell
min_barrier_fraction = 0.1        # keep a well only if barrier >= this * U_range
min_well_separation  = 0.0        # minimum separation between wells (log-price units)

# --- Phase GP — spatial kernel -----------------------------------------------
# None -> derived from data as 0.15 × std(x_prev) (half of Phase A's 0.3×std heuristic)
spatial_lengthscale = None
spatial_variance    = 1.0    # GP prior variance (initial; refined by HP opt)

# --- Phase GP — temporal kernel (Matern 3/2) ----------------------------------
# None -> half of Phase B mean dwell time (Phase A regimes resolve faster than dwell)
temporal_lengthscale_days = None

# --- Phase GP — EMA drift demeaning ------------------------------------------
# None -> derived from Phase B mean dwell time (same value as temporal_lengthscale_days)
# 0    -> EMA disabled; r_hat = r. Use this when the dwell-scale EMA would
#         smear the spatial drift structure the GP is supposed to detect.
ema_halflife_days = 0.0

# --- Phase GP — observation noise --------------------------------------------
# None -> estimated as var(dx) / dt from the data
sigma2 = None

# --- Phase GP — Kalman state --------------------------------------------------
n_inducing = 8      # number of inducing points; Kalman state dim = 2 * n_inducing

# --- Phase GP — HP optimisation ----------------------------------------------
hp_opt_after_n_windows = 2        # optimise HPs after this many Phase-A windows
hp_opt_n_restarts      = 3        # L-BFGS-B restarts ('scipy' method only)
hp_opt_method          = 'gpflow'  # 'scipy' (multi-restart L-BFGS-B) or 'gpflow' (SGPR)

# --- Phase GP — topology extraction ------------------------------------------
n_grid           = 200   # grid points for the posterior drift field
n_samples        = 200   # Monte-Carlo samples for topology statistics
min_crossing_sep = 10    # minimum grid-point separation between zero crossings

# --- Phase GP — topology forecasting -----------------------------------------
forecast_horizons_days = (1.0, 3.0, 7.0, 14.0)   # Kalman rollout horizons (days)

# --- Reproducibility ---------------------------------------------------------
seed = 42

# --- Output ------------------------------------------------------------------
output_dir = 'regime_results'   # Phase A / Phase B artefacts
gp_output_dir_root = 'gp_results'   # Phase GP artefacts; each phase_gp_seconds_interval gets its own subfolder


# =============================================================================
# PIPELINE
# =============================================================================

def main() -> None:
    console = Console()

    snapped_start, snapped_end = normalize_window_boundaries(
        start_date, end_date, window_type, console=console,
    )

    # -------------------------------------------------------------------------
    console.rule('[bold cyan]Step 1 — Download raw data')
    dc.ensure_data(snapped_start, snapped_end)

    # -------------------------------------------------------------------------
    console.rule('[bold cyan]Step 2 — Aggregate log-returns (per window)')
    window_list = list(iter_windows(snapped_start, snapped_end, window_type))
    for si in sorted({phase_a_seconds_interval, phase_gp_seconds_interval}):
        for window_start, window_end in window_list:
            dc.aggregate_log_returns_range(
                window_start, window_end, si,
                kernel_half_width=kernel_half_width,
                trim_quantile=trim_quantile,
            )

    # -------------------------------------------------------------------------
    console.rule('[bold cyan]Step 3 — Phase A: regime topology classification')
    labels_df = run_phase_a(
        snapped_start, snapped_end,
        phase_a_seconds_interval,
        kernel_half_width=kernel_half_width,
        trim_quantile=trim_quantile,
        n_bins=n_bins,
        weight_threshold=weight_threshold,
        min_barrier_fraction=min_barrier_fraction,
        min_well_separation=min_well_separation,
        window_type=window_type,
        output_dir=output_dir,
        console=console,
    )
    print_regime_table(labels_df, console=console)

    labels_csv = os.path.join(
        output_dir,
        f"regime_labels_{snapped_start.strftime('%Y-%m-%d')}_to_"
        f"{snapped_end.strftime('%Y-%m-%d')}_{phase_a_seconds_interval}s_{window_type}.csv",
    )

    # -------------------------------------------------------------------------
    # Phase B and Phase GP run inside run_phase_gp. Phase B is called via
    # _phase_b_dwell_days; because labels_csv is provided it is never re-run.
    topology_every_n_obs = 7 * 24 * 3600 // phase_gp_seconds_interval  # one topology snapshot per week

    console.rule('[bold cyan]Step 4–5 — Phase B (dwell times) → Phase GP')
    run_phase_gp(
        start_date=snapped_start,
        end_date=snapped_end,
        seconds_interval=phase_gp_seconds_interval,
        phase_a_seconds_interval=phase_a_seconds_interval,
        window_type=window_type,
        labels_csv=labels_csv,
        sigma2=sigma2,
        spatial_lengthscale=spatial_lengthscale,
        temporal_lengthscale_days=temporal_lengthscale_days,
        spatial_variance=spatial_variance,
        ema_halflife_days=ema_halflife_days,
        n_inducing=n_inducing,
        n_grid=n_grid,
        n_samples=n_samples,
        hp_opt_after_n_windows=hp_opt_after_n_windows,
        hp_opt_n_restarts=hp_opt_n_restarts,
        hp_opt_method=hp_opt_method,
        forecast_horizons_days=forecast_horizons_days,
        topology_every_n_obs=topology_every_n_obs,
        kernel_half_width=kernel_half_width,
        trim_quantile=trim_quantile,
        min_crossing_sep=min_crossing_sep,
        min_barrier_fraction=min_barrier_fraction,
        output_dir=output_dir,
        gp_output_dir=os.path.join(gp_output_dir_root, f'{phase_gp_seconds_interval}s'),
        seed=seed,
        console=console,
    )


if __name__ == '__main__':
    main()
