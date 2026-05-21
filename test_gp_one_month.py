"""
Diagnostic: can the Kalman GP recover non-flat drift from one month of data?

Compares Phase A KM bins (the empirical "ground truth" for conditional drift)
against the Kalman GP posterior at several spatial_var settings. If the GP
posterior is flat for the default spatial_var but tracks KM at a higher value,
the prior is the bottleneck and HP-opt bounds need to be relaxed.

Hypothesis being tested:
    Annualised r_hat has var(r_hat) ~ 30,000 [/year]^2 per observation.
    With spatial_var = 1.0, Kalman gain per obs ~= 3e-5, and N=2900 obs/month
    barely shifts the posterior from the zero prior. GP samples are effectively
    prior draws -> topology metrics are noise.

Run:
    python test_gp_one_month.py
"""

from datetime import datetime
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data_collection as dc
from data_collection import load_series, ema_demean_drift
from phase_GP import KalmanGPDriftModel, topology_from_gp, _SEC_PER_YEAR


warnings.filterwarnings('ignore', category=RuntimeWarning)


# --- Configuration ------------------------------------------------------------
START          = datetime(2025, 2, 1)
END            = datetime(2025, 3, 31)
SECONDS        = 900                  # same as RunGP.py phase_gp_seconds_interval
KERNEL_HW      = 50
TRIM_QUANTILE  = 0.01
N_INDUCING     = 8

SPATIAL_VARS   = [1.0, 1e2, 1e4, 'hp_opt']   # 'hp_opt' = start at 1e4 and run scipy HP opt
TEMPORAL_LS_D  = 5.0                  # days
N_GRID         = 200
N_SAMPLES      = 200

KM_CSV         = 'regime_results/km/km_2025-03-01_to_2025-03-31_900s.csv'
OUT_PNG        = 'gp_results/test_gp_one_month.png'


# --- Load one month of returns -----------------------------------------------
print(f'Loading {START.date()} -> {END.date()} at {SECONDS}s ...')
dc.ensure_data(START, END)
dc.aggregate_log_returns_range(
    START, END, SECONDS,
    kernel_half_width=KERNEL_HW,
    trim_quantile=TRIM_QUANTILE,
)
x_prev, dx, dt, dt_t = load_series(
    START, END, SECONDS,
    kernel_half_width=KERNEL_HW,
    trim_quantile=TRIM_QUANTILE,
)
N = len(dx)
print(f'  N = {N} increments, dt = {dt:.0f}s')

# Annualised drift observations
r = (dx / dt) * _SEC_PER_YEAR
r_hat = r.copy()                      # EMA disabled, mirroring run_phase_gp
print(f'  r_hat: mean = {r_hat.mean():.3e}/yr  std = {r_hat.std():.3e}/yr')
print(f'  var(r_hat) = {r_hat.var():.3e}  (this is approx obs_noise per observation)')

t_seconds = (pd.to_datetime(dt_t).astype(np.int64) / 1e9).values.astype(float)

x_lo = float(np.percentile(x_prev, 1))
x_hi = float(np.percentile(x_prev, 99))
spatial_ls = 0.15 * float(np.std(x_prev))
print(f'  x range = [{x_lo:.4f}, {x_hi:.4f}]  spatial_ls = {spatial_ls:.4f}')

sigma2 = float(np.var(r_hat * dt) / dt)
obs_noise = sigma2 / dt
print(f'  sigma2 = {sigma2:.3e}    obs_noise = sigma2/dt = {obs_noise:.3e}')
print()

# --- KM ground truth ---------------------------------------------------------
if os.path.exists(KM_CSV):
    km = pd.read_csv(KM_CSV)
    km = km.dropna(subset=['drift']).reset_index(drop=True)
    km_drift_ann = km['drift'].values * _SEC_PER_YEAR
    print(f'KM: {len(km)} bins; drift range = [{km_drift_ann.min():.1f}, {km_drift_ann.max():.1f}] /year')
else:
    print(f'WARNING: KM CSV not found at {KM_CSV}; KM scatter will be empty.')
    km = None

# --- Sweep over spatial_var --------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes = axes.flatten()

for ax, sp_var_setting in zip(axes, SPATIAL_VARS):
    do_hp_opt = (sp_var_setting == 'hp_opt')
    sp_var_init = 1e4 if do_hp_opt else float(sp_var_setting)

    model = KalmanGPDriftModel(
        spatial_lengthscale=spatial_ls,
        temporal_lengthscale_days=TEMPORAL_LS_D,
        spatial_variance=sp_var_init,
        sigma2=sigma2,
        dt=float(dt),
    )
    model.initialise(x_range=(x_lo, x_hi), n_inducing=N_INDUCING, data_x=x_prev)

    if do_hp_opt:
        from rich.console import Console
        model.optimise_hp(
            x_prev, r_hat, t_seconds,
            n_restarts=2, method='scipy',
            x_range=(x_lo, x_hi), console=Console(quiet=True),
        )
        print(f'  HP-opt result: spatial_ls={model.spatial_ls:.4f}  '
              f'temporal_ls={model.temporal_ls:.2f}d  '
              f'spatial_var={model.spatial_var:.4g}')

    model.update(x_prev, r_hat, t_seconds)
    sp_var = model.spatial_var

    x_grid = np.linspace(x_lo, x_hi, N_GRID)
    mu_mean, mu_var = model.predict(x_grid, full_cov=False)
    mu_std = np.sqrt(np.maximum(mu_var, 0.0))

    topo = topology_from_gp(
        model, x_range=(x_lo, x_hi),
        n_grid=N_GRID, n_samples=N_SAMPLES,
        rng=np.random.default_rng(0),
    )

    # Effective Kalman gain heuristic for label
    k_per_obs = sp_var / (sp_var + obs_noise)
    n_per_inducing = N / N_INDUCING
    posterior_var_naive = (1.0 / sp_var + n_per_inducing / obs_noise) ** -1

    tag = 'HP-opt' if do_hp_opt else 'fixed'
    ax.set_title(
        f'[{tag}] spatial_var={sp_var:g}   K/obs≈{k_per_obs:.2e}\n'
        f'p_multiwell={topo["p_multiwell"]:.2f}  '
        f'barrier={topo["barrier_mean"]:.3f}  '
        f'σ/μ={topo["mu_std_to_mean"]:.2f}',
        fontsize=10,
    )

    if km is not None:
        ax.scatter(km['bin_center'], km_drift_ann,
                   s=2 + 30 * km['weight'] / km['weight'].max(),
                   c='crimson', alpha=0.5, label='KM bins', zorder=2)

    ax.fill_between(x_grid, mu_mean - 2 * mu_std, mu_mean + 2 * mu_std,
                    color='steelblue', alpha=0.2, label='GP ±2σ', zorder=1)
    ax.plot(x_grid, mu_mean, color='steelblue', linewidth=1.5,
            label='GP mean', zorder=3)
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax.scatter(model.inducing_x, np.zeros(model.M),
               marker='|', color='darkgreen', s=80, label='inducing', zorder=4)
    ax.set_ylabel('drift [/year]')
    ax.legend(fontsize=7, loc='best')

    print(
        f'spatial_var={sp_var:>8g}  '
        f'p_multi={topo["p_multiwell"]:.2f}  '
        f'barrier={topo["barrier_mean"]:.3f}  '
        f'σ/μ={topo["mu_std_to_mean"]:.2f}  '
        f'GP-mean range=[{mu_mean.min():+.2f}, {mu_mean.max():+.2f}]  '
        f'GP-std mean={mu_std.mean():.2f}'
    )

axes[-1].set_xlabel('log(price)')
axes[-2].set_xlabel('log(price)')
fig.suptitle(
    f'Kalman GP one-month diagnostic — {START.date()} to {END.date()}, '
    f'{SECONDS}s, N={N}',
    fontsize=11,
)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
fig.savefig(OUT_PNG, dpi=130)
print(f'\nSaved: {OUT_PNG}')
