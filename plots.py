import os

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

from force_field_estimation import integrate_drift_to_potential


# ---------------------------------------------------------------------------
# Phase A / KM plots
# ---------------------------------------------------------------------------

def plot_potentials_from_km_results(results_by_interval, output_dir='plots', range_label=None, detrend=False):
    """
    Plot integrated potential functions for precomputed KM results.
    """
    os.makedirs(output_dir, exist_ok=True)

    detrend_label = ' [detrended]' if detrend else ''
    detrend_suffix = '_detrended' if detrend else ''
    sec_per_year = 365.25 * 24 * 3600

    plt.figure(figsize=(12, 7))
    for x_seconds, result_df in results_by_interval.items():
        potential_df = integrate_drift_to_potential(result_df[['bin_center', 'drift']])
        plt.plot(potential_df['bin_center'], potential_df['potential'], label=f'{x_seconds}s interval')
    plt.title(f'Integrated Potentials Comparison - {range_label}{detrend_label}')
    plt.xlabel('Log price bin center')
    plt.ylabel('Potential U(x)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_path = os.path.join(output_dir, f'combined_potentials_{range_label}{detrend_suffix}.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved {output_path}')

    plt.figure(figsize=(12, 7))
    for x_seconds, result_df in results_by_interval.items():
        ann_drift = result_df['drift'] * sec_per_year
        plt.plot(result_df['bin_center'], ann_drift, label=f'{x_seconds}s interval')
    plt.title(f'Annualized Drift ($D_1$) Comparison - {range_label}')
    plt.xlabel('Log price bin center')
    plt.ylabel('Annualized Expected Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_path_drift = os.path.join(output_dir, f'drift_comparison_{range_label}.png')
    plt.savefig(output_path_drift, dpi=150)
    plt.close()
    print(f'Saved {output_path_drift}')

    plt.figure(figsize=(12, 7))
    for x_seconds, result_df in results_by_interval.items():
        ann_vol = np.sqrt(2 * result_df['diffusion'] * sec_per_year)
        plt.plot(result_df['bin_center'], ann_vol, label=f'{x_seconds}s interval')
    plt.title(f'Annualized Volatility ($\\sigma$) Comparison - {range_label}')
    plt.xlabel('Log price bin center')
    plt.ylabel('Annualized Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_path_vol = os.path.join(output_dir, f'volatility_comparison_{range_label}.png')
    plt.savefig(output_path_vol, dpi=150)
    plt.close()
    print(f'Saved {output_path_vol}')


def plot_price_series(df, output_dir='plots', range_label=None, smooth_window=20, detrend=False):
    """ETH price over time with a rolling-mean smoothing overlay."""
    os.makedirs(output_dir, exist_ok=True)
    prices = np.exp(df['log_first_price'])
    times = pd.to_datetime(df['datetime'])
    smoothed = prices.rolling(window=smooth_window, center=True, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(times, prices, color='steelblue', alpha=0.35, linewidth=0.6, label='Price')
    ax.plot(times, smoothed, color='firebrick', linewidth=1.4,
            label=f'Rolling mean (window={smooth_window})')
    ax.set_title(f'ETH/USDT Price - {range_label}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USDT)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    output_path = os.path.join(output_dir, f'price_series_{range_label}.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output_path}')


def plot_log_price_series(df, output_dir='plots', range_label=None, smooth_window=20):
    """Log-price over time with a rolling-mean smoothing overlay."""
    os.makedirs(output_dir, exist_ok=True)
    log_prices = df['log_first_price']
    times = pd.to_datetime(df['datetime'])
    smoothed = log_prices.rolling(window=smooth_window, center=True, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(times, log_prices, color='steelblue', alpha=0.35, linewidth=0.6, label='Log price')
    ax.plot(times, smoothed, color='firebrick', linewidth=1.4,
            label=f'Rolling mean (window={smooth_window})')
    ax.set_title(f'ETH/USDT Log Price - {range_label}')
    ax.set_xlabel('Date')
    ax.set_ylabel('log(Price / USDT)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    output_path = os.path.join(output_dir, f'log_price_series_{range_label}.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output_path}')


# ---------------------------------------------------------------------------
# Phase B — Markov chain plots
# ---------------------------------------------------------------------------

def plot_mc_transition_heatmap(A, states, output_path):
    """Annotated heatmap of the MLE transition matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(A, vmin=0, vmax=1, cmap='Blues')
    fig.colorbar(im, ax=ax, label='Transition probability')
    ax.set_xticks(range(len(states)))
    ax.set_yticks(range(len(states)))
    ax.set_xticklabels(states, rotation=15, ha='right')
    ax.set_yticklabels(states)
    ax.set_xlabel('To state')
    ax.set_ylabel('From state')
    ax.set_title('MLE Transition Matrix')
    for i in range(len(states)):
        for j in range(len(states)):
            text_color = 'white' if A[i, j] > 0.6 else 'black'
            ax.text(j, i, f'{A[i, j]:.3f}', ha='center', va='center',
                    color=text_color, fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved {output_path}')


def plot_mc_stationary(pi, dwells, states, colors, output_path):
    """Bar chart of stationary distribution with mean-dwell annotations."""
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(states, pi, color=colors, edgecolor='black', linewidth=0.8)
    for bar, p, d in zip(bars, pi, dwells):
        dwell_str = (
            f'π={p:.3f}\ndwell={d:.1f}w' if np.isfinite(d) else f'π={p:.3f}\ndwell=∞'
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            dwell_str,
            ha='center', va='bottom', fontsize=9,
        )
    ax.set_ylim(0, 1.25)
    ax.set_ylabel('Stationary probability')
    ax.set_title('Stationary Distribution & Mean Dwell')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved {output_path}')


def plot_mc_timeline(
    seq_df, states, colors, state_idx, output_path,
    date_start=None, date_end=None,
):
    """Horizontal date-axis timeline coloured by regime."""
    fig, ax = plt.subplots(figsize=(13, 2.2))
    for _, row in seq_df.iterrows():
        color = colors[state_idx[row['regime']]]
        ax.axvspan(row['window_start'], row['window_end'], color=color, alpha=0.85)

    x_min = date_start if date_start is not None else seq_df['window_start'].min()
    x_max = date_end if date_end is not None else seq_df['window_end'].max()
    ax.set_xlim(x_min, x_max)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45, ha='right')
    ax.set_yticks([])
    ax.set_xlabel('Date')
    ax.set_title('Regime Timeline')

    patches = [mpatches.Patch(color=colors[i], label=states[i]) for i in range(len(states))]
    ax.legend(handles=patches, loc='upper right', fontsize=9)
    ax.grid(axis='x', alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved {output_path}')


# ---------------------------------------------------------------------------
# Phase GP — sequential Kalman-GP plots
# ---------------------------------------------------------------------------

def plot_gp_topology_series(df_topology, out_path):
    """
    Four-panel time series:
      1. p_multiwell_gp (sequential GP) vs p_multiwell_a (Phase A weekly)
      2. barrier_mean +/- 1 std shading
      3. kramers on log scale
      4. mean_n_wells
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    dt = pd.to_datetime(df_topology['datetime'])
    fmt = mdates.AutoDateFormatter(mdates.AutoDateLocator())

    axes[0].plot(dt, df_topology['p_multiwell_gp'],
                 color='steelblue', linewidth=0.8, label='GP (sequential)')
    if 'p_multiwell_a' in df_topology.columns:
        axes[0].step(dt, df_topology['p_multiwell_a'],
                     color='tomato', linewidth=0.8,
                     alpha=0.6, label='Phase A (weekly)')
    axes[0].axhline(0.5, color='black', linewidth=0.5, linestyle=':')
    axes[0].set_ylabel('P(multi-well)')
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=7)

    axes[1].plot(dt, df_topology['barrier_mean'],
                 color='darkorange', linewidth=0.8)
    axes[1].fill_between(
        dt,
        df_topology['barrier_mean'] - df_topology['barrier_std'],
        df_topology['barrier_mean'] + df_topology['barrier_std'],
        alpha=0.3, color='darkorange',
    )
    axes[1].set_ylabel(r'Barrier height $\Delta U$')

    axes[2].semilogy(dt, np.maximum(df_topology['kramers'], 1e-30),
                     color='purple', linewidth=0.8)
    axes[2].set_ylabel(r'Kramers rate $\Gamma$')

    axes[3].plot(dt, df_topology['mean_n_wells'],
                 color='green', linewidth=0.8)
    axes[3].axhline(1.5, color='black', linewidth=0.5, linestyle=':')
    axes[3].set_ylabel('Mean well count')
    axes[3].xaxis.set_major_formatter(fmt)

    fig.suptitle('Sequential Kalman-GP topology signal', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved {out_path}')


def plot_gp_potential_snapshots(model, x_range, snapshots, out_path, n_snapshots=6):
    """
    Grid of U(x) snapshots reconstructed from saved Kalman states.

    snapshots is a list of (datetime, state_mean, state_cov) tuples appended
    during run_phase_gp's main loop. Each panel restores the saved state on
    the supplied model, predicts the posterior-mean drift, and integrates
    to U(x).
    """
    if not snapshots:
        print(f'[plot_gp_potential_snapshots] no snapshots; skipping {out_path}')
        return

    n_snapshots = min(n_snapshots, len(snapshots))
    idx = np.linspace(0, len(snapshots) - 1, n_snapshots, dtype=int)

    rows = max(1, n_snapshots // 3) if n_snapshots > 2 else 1
    cols = max(1, int(np.ceil(n_snapshots / rows)))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows),
                             squeeze=False)
    axes = axes.flatten()

    x_grid = np.linspace(x_range[0], x_range[1], 200)
    saved_mean = model.state_mean.copy()
    saved_cov = model.state_cov.copy()

    for k, ax in enumerate(axes):
        if k >= n_snapshots:
            ax.axis('off')
            continue
        dt_q, sm, sc = snapshots[idx[k]]
        model.state_mean = sm
        model.state_cov = sc
        mu_mean, _ = model.predict(x_grid, full_cov=False)
        U = -cumulative_trapezoid(mu_mean, x_grid, initial=0.0)
        U -= U.min()
        ax.plot(x_grid, U, color='steelblue', linewidth=1.2)
        ax.set_title(str(pd.Timestamp(dt_q).date()), fontsize=8)
        ax.set_xlabel('log-price', fontsize=7)
        ax.set_ylabel('U(x)', fontsize=7)
        ax.tick_params(labelsize=6)

    model.state_mean = saved_mean
    model.state_cov = saved_cov

    fig.suptitle('GP posterior mean potential U(x) — snapshots', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    print('This module exposes plotting helpers; import its functions from the pipeline scripts.')
