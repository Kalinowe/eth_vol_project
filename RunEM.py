"""
Phase C execution button.

Runs the full pipeline that feeds Expectation–Maximisation:
  1. Download Binance daily dumps (if missing) and aggregate log-prices.
  2. Phase A — regime_estimation: per-window Kramers–Moyal drift,
     GP-based potential, p_multiwell.
  3. Phase B — markov_chain: empirical (MLE) regime chain; the Dirichlet
     posterior is what Phase C consumes as the pi_init.
  4. Phase C — ExpectationMaximisation.run_phase_c: EM for the
     Markov-switching SDE with a week-scale Dirichlet prior on the
     transition matrix and the GP p_multiwell tilt on the emission
     log-likelihoods.

All knobs that affect the Phase C outcome are declared in the
CONFIGURATION block below. Edit them and re-run this file.
"""

import os
from datetime import datetime

from rich.console import Console

import data_collection as dc
from regime_estimation import (
    normalize_window_boundaries,
    run_phase_a,
    print_regime_table,
)
from markov_chain import run_markov_chain
from ExpectationMaximisation import run_phase_c


# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Date range ---------------------------------------------------------------
start_date = datetime(2025, 1, 1)
end_date   = datetime(2025, 12, 31)

# --- Data aggregation / KM estimation ----------------------------------------
# seconds_interval: the bar size used inside Phase C. Phase A scans all the
# intervals listed in seconds_intervals_phase_a so the GP topology can be
# cross-checked across resolutions.
seconds_interval         = 300
seconds_intervals_phase_a = [300]
kernel_half_width        = 20       # smoothing half-width for log-price aggregator
trim_quantile            = 0.01    # symmetric tail trim on log-prices
n_bins                   = 100     # number of bins for KM drift / diffusion
weight_threshold         = 5       # minimum bin count for a usable KM cell
detrend                  = 1       # 1 -> linear-detrend log-prices before KM

# --- Phase A topology classifier ---------------------------------------------
window_type           = 'weekly'   # 'weekly' | 'biweekly' | 'monthly'
min_barrier_fraction  = 0.1        # keep a well only if barrier >= this * U_range
min_well_separation   = 0.0

# --- Phase B empirical Markov chain ------------------------------------------
prior_lambda      = 0.1            # strength of Phase-B-informed Dirichlet prior
use_soft_counts   = True           # weight transitions by p_multiwell

# --- Phase C EM convergence --------------------------------------------------
max_iter   = 30
tol        = 1e-4
epsilon_P  = 1e-3                  # near-identity init for the transition matrix
sigma2     = None                  # None -> estimate from var(dx)/dt
regime_specific_sigma2 = False     # if True, re-estimate sigma2 per state each M-step
theta_init = 'km'                  # 'km' (Phase-A-pooled curves) or 'moments' (data-only)

# --- Phase C priors ----------------------------------------------------------
# Dwell-time enforcement: pull P[i,i] toward 1 - dt / target_dwell_seconds.
target_dwell_seconds   = 7 * 86400.0   # one week
dwell_prior_strength   = 10          # prior mass = strength * len(dx); 0 disables
# GP label tilt: add eta_label * log p_multiwell_window(t) to emission log-b.
eta_label              = 0.5          # 0 disables

# --- Output ------------------------------------------------------------------
regime_dir   = 'regime_results'        # Phase A + Phase B artefacts
phase_c_dir  = 'phase_c_results'       # Phase C EM artefacts


# =============================================================================
# PIPELINE
# =============================================================================

def main() -> None:
    console = Console()

    snapped_start, snapped_end = normalize_window_boundaries(
        start_date, end_date, window_type, console=console,
    )

    console.rule('[bold cyan]Step 1 — Download + aggregate raw data')
    dc.ensure_data(snapped_start, snapped_end)
    for interval in sorted(set(seconds_intervals_phase_a) | {seconds_interval}):
        dc.aggregate_log_returns_range(
            snapped_start, snapped_end, interval,
            kernel_half_width=kernel_half_width,
            trim_quantile=trim_quantile,
            detrend=detrend,
        )

    console.rule('[bold cyan]Step 2 — Phase A: regime estimation')
    labels_df = run_phase_a(
        snapped_start, snapped_end,
        seconds_intervals_phase_a,
        kernel_half_width=kernel_half_width,
        trim_quantile=trim_quantile,
        n_bins=n_bins,
        weight_threshold=weight_threshold,
        detrend=detrend,
        min_barrier_fraction=min_barrier_fraction,
        min_well_separation=min_well_separation,
        window_type=window_type,
        output_dir=regime_dir,
        console=console,
    )
    print_regime_table(labels_df, console=console)

    labels_csv = os.path.join(
        regime_dir,
        f"regime_labels_{snapped_start.strftime('%Y-%m-%d')}_to_"
        f"{snapped_end.strftime('%Y-%m-%d')}_{window_type}.csv",
    )

    console.rule('[bold cyan]Step 3 — Phase B: empirical Markov chain')
    mc = run_markov_chain(
        labels_csv,
        window_type=window_type,
        seconds_interval=seconds_interval,
        output_dir=regime_dir,
        prior_lambda=prior_lambda,
        use_soft_counts=use_soft_counts,
        console=console,
    )
    pi_init = mc['pi'] if mc is not None else None

    console.rule('[bold cyan]Step 4 — Phase C: Expectation–Maximisation')
    run_phase_c(
        snapped_start, snapped_end, seconds_interval,
        labels_csv=labels_csv,
        pi_init=pi_init,
        epsilon_P=epsilon_P,
        sigma2=sigma2,
        regime_specific_sigma2=regime_specific_sigma2,
        kernel_half_width=kernel_half_width,
        trim_quantile=trim_quantile,
        detrend=detrend,
        max_iter=max_iter,
        tol=tol,
        output_dir=phase_c_dir,
        target_dwell_seconds=target_dwell_seconds,
        dwell_prior_strength=dwell_prior_strength,
        eta_label=eta_label,
        theta_init=theta_init,
        console=console,
    )


if __name__ == '__main__':
    main()
