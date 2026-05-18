import os

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from force_field_estimation import integrate_drift_to_potential


def plot_potentials_from_km_results(results_by_interval, output_dir='plots', range_label=None, detrend=False):
    """
    Plot integrated potential functions for precomputed KM results.

    Args:
        results_by_interval: dict mapping interval length to KM result DataFrame with
            columns ['bin_center', 'drift', 'diffusion', 'weight']
        output_dir: Directory where PNG plots are saved
        range_label: Optional label used for plot titles and filenames
        detrend: If True, annotates plots and filenames to indicate log-prices were detrended
    """
    os.makedirs(output_dir, exist_ok=True)

    detrend_label = ' [detrended]' if detrend else ''
    detrend_suffix = '_detrended' if detrend else ''
    sec_per_year = 365.25 * 24 * 3600

    # 1. Potential Plot
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

    # 2. Annualized Drift Plot
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

    # 3. Annualized Volatility Plot
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
    """
    Plot ETH price (not log-price) over time with a rolling-mean smoothing overlay.

    Args:
        df: DataFrame with columns ['datetime', 'log_first_price']
        output_dir: Directory where the PNG is saved
        range_label: Label used in the title and filename
        smooth_window: Rolling mean window in number of bars
        detrend: Unused; kept for call-site compatibility
    """
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
    """
    Plot log-price over time with a rolling-mean smoothing overlay.

    Args:
        df: DataFrame with columns ['datetime', 'log_first_price']
        output_dir: Directory where the PNG is saved
        range_label: Label used in the title and filename
        smooth_window: Rolling mean window in number of bars
    """
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
# Markov-chain plots (called from markov_chain.py)
# ---------------------------------------------------------------------------

def plot_mc_transition_heatmap(
    A: np.ndarray,
    states: list[str],
    output_path: str,
) -> None:
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


def plot_mc_stationary(
    pi: np.ndarray,
    dwells: np.ndarray,
    states: list[str],
    colors: list[str],
    output_path: str,
) -> None:
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
    seq_df: pd.DataFrame,
    states: list[str],
    colors: list[str],
    state_idx: dict,
    output_path: str,
    date_start: pd.Timestamp | None = None,
    date_end: pd.Timestamp | None = None,
) -> None:
    """
    Horizontal date-axis timeline coloured by regime.
    Gaps where windows were dropped appear as white space.
    """
    fig, ax = plt.subplots(figsize=(13, 2.2))

    for _, row in seq_df.iterrows():
        color = colors[state_idx[row['regime']]]
        ax.axvspan(row['window_start'], row['window_end'], color=color, alpha=0.85)

    x_min = date_start if date_start is not None else seq_df['window_start'].min()
    x_max = date_end   if date_end   is not None else seq_df['window_end'].max()
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


def plot_phase_c_gamma_timeline(gamma, output_path, datetimes=None):
    """
    Smoothed multi-well probability gamma_t over the full series.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    if datetimes is not None:
        ax.plot(datetimes, gamma[:, 1], color='#F44336', lw=0.6)
        ax.set_xlabel('time')
    else:
        ax.plot(gamma[:, 1], color='#F44336', lw=0.6)
        ax.set_xlabel('step index')
    ax.set_ylim(0, 1)
    ax.set_ylabel(r'$\gamma_t$(multi-well)')
    ax.set_title('Phase C — smoothed regime probability')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f'Saved {output_path}')

def plot_phase_c_drifts(x_prev, theta, output_path):
    """
    Fitted parametric drifts mu_1 (OU) and mu_2 (cubic) on a common x grid,
    plus a histogram of x_prev for context. Drifts are plotted **annualised**
    (multiplied by SEC_PER_YEAR) to match the project convention.
    """
    sec_per_year = 365.25 * 24 * 3600
    x_lo, x_hi = float(np.min(x_prev)), float(np.max(x_prev))
    x_grid = np.linspace(x_lo, x_hi, 400)
    mu1 = (-theta['kappa'] * (x_grid - theta['m'])) * sec_per_year
    u = x_grid - theta['c']
    mu2 = (-theta['alpha'] * u ** 3 + theta['beta'] * u) * sec_per_year

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]},
    )
    ax1.plot(x_grid, mu1, color='#2196F3',
             label=r'state 0  $-\kappa(x-m)$  (single-well)')
    ax1.plot(x_grid, mu2, color='#F44336',
             label=r'state 1  $-\alpha(x-c)^3+\beta(x-c)$  (multi-well)')
    ax1.axhline(0, color='gray', alpha=0.4, lw=0.5)
    ax1.set_ylabel('drift (yr$^{-1}$)')
    ax1.set_title('Phase C — fitted parametric drifts (annualised)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.hist(x_prev, bins=80, color='gray', alpha=0.6)
    ax2.set_ylabel('counts')
    ax2.set_xlabel('log-price x')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f'Saved {output_path}')

def plot_gamma_kappa_overlay(
    gamma: np.ndarray,
    df_kappa: pd.DataFrame,
    output_path: str,
    datetimes=None,
) -> None:
    """
    Overlay: filled area = smoothed multi-well probability γ_t (right axis,
    [0, 1]); line on top = annualised rolling κ(t) (left axis). Designed to
    answer 'does a fall in κ precede a multi-well regime?'.
    """
    if datetimes is not None:
        x_gamma = pd.to_datetime(datetimes)
    else:
        x_gamma = np.arange(len(gamma))

    fig, ax_k = plt.subplots(figsize=(13, 4.5))
    ax_g = ax_k.twinx()

    # Right axis: filled regime probability (drawn first, lower z-order).
    ax_g.fill_between(
        x_gamma, 0.0, gamma[:, 1],
        color='#F44336', alpha=0.22,
        linewidth=0,
        label=r'$\gamma_t$(multi-well)',
    )
    ax_g.plot(x_gamma, gamma[:, 1], color='#F44336', lw=0.6, alpha=0.55)
    ax_g.set_ylim(0.0, 1.0)
    ax_g.set_ylabel(r'$\gamma_t$ (multi-well)', color='#F44336')
    ax_g.tick_params(axis='y', labelcolor='#F44336')

    # Left axis: kappa line (drawn on top via zorder shuffle).
    if df_kappa is not None and not df_kappa.empty:
        col = (
            'kappa_ann_smoothed'
            if 'kappa_ann_smoothed' in df_kappa.columns
            else ('kappa_ann' if 'kappa_ann' in df_kappa.columns else 'kappa')
        )
        ax_k.plot(
            df_kappa['datetime'], df_kappa[col],
            color='#1565C0', lw=1.6,
            label=r'$\kappa_{\mathrm{ann}}(t)$ (smoothed)',
        )
        median_k = float(np.nanmedian(df_kappa[col]))
        ax_k.axhline(
            median_k, color='gray', ls='--', lw=0.9, alpha=0.7,
            label=f'median κ = {median_k:.3g}',
        )
    ax_k.set_ylabel(r'$\kappa_{\mathrm{ann}}$ (yr$^{-1}$)', color='#1565C0')
    ax_k.tick_params(axis='y', labelcolor='#1565C0')
    ax_k.set_xlabel('Date')
    ax_k.set_title('Phase C — κ(t) over smoothed multi-well probability')
    ax_k.grid(True, alpha=0.3)

    # Put the kappa axis (and line) above the filled area.
    ax_k.set_zorder(ax_g.get_zorder() + 1)
    ax_k.patch.set_visible(False)

    # Combined legend across both axes.
    h_k, l_k = ax_k.get_legend_handles_labels()
    h_g, l_g = ax_g.get_legend_handles_labels()
    ax_k.legend(h_k + h_g, l_k + l_g, loc='upper left', fontsize=9, framealpha=0.85)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    print(f'Saved {output_path}')


def plot_kappa_series(df_kappa: pd.DataFrame, output_path: str) -> None:
    """
    Line chart of annualised kappa vs datetime with a horizontal dashed
    reference at the median across the series. Expects the column
    'kappa_ann' (kappa multiplied by SEC_PER_YEAR) — see
    ExpectationMaximisation.estimate_kappa_series.
    """
    if df_kappa.empty:
        print(f'[plot_kappa_series] empty DataFrame; skipping {output_path}')
        return

    col = (
        'kappa_ann_smoothed'
        if 'kappa_ann_smoothed' in df_kappa.columns
        else ('kappa_ann' if 'kappa_ann' in df_kappa.columns else 'kappa')
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_kappa['datetime'], df_kappa[col],
            color='#2196F3', lw=1.0, label=r'$\kappa_{\mathrm{ann}}(t)$')
    median_k = float(np.nanmedian(df_kappa[col]))
    ax.axhline(median_k, color='gray', ls='--', lw=0.9,
               label=f'median = {median_k:.3g}')
    ax.set_xlabel('Date')
    ax.set_ylabel(r'$\kappa_{\mathrm{ann}}$ (yr$^{-1}$)')
    ax.set_title(r'Phase C — rolling $\kappa_{\mathrm{ann}}(t)$ (critical-slowing-down signal)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f'Saved {output_path}')



def _plot_extreme_window_overlays(
    labels_csv,
    start_date,
    end_date,
    seconds_interval,
    kernel_half_width,
    trim_quantile,
    detrend,
    window_type,
    output_dir,
    console,
    n_per_class=2,
):
    """
    Pick the n_per_class windows with the lowest p_multiwell (most clearly
    single-well) and the n_per_class with the highest (most clearly
    multi-well), and produce a price + rolling-κ overlay for each.

    EM imports are done lazily so plots.py stays import-free of
    ExpectationMaximisation (which itself imports plots).
    """
    from ExpectationMaximisation import estimate_kappa_series, load_series

    if not os.path.exists(labels_csv):
        console.print(
            f'[yellow]Skipping window overlays: labels CSV missing ({labels_csv}).[/yellow]'
        )
        return

    labels_df = pd.read_csv(labels_csv, parse_dates=['window_start', 'window_end'])
    labels_df['seconds_interval'] = pd.to_numeric(
        labels_df['seconds_interval'], errors='coerce'
    ).astype('Int64')
    sub = labels_df[labels_df['seconds_interval'] == int(seconds_interval)].copy()
    sub = sub.dropna(subset=['p_multiwell'])
    if sub.empty:
        console.print(
            '[yellow]No labelled windows with p_multiwell — skipping overlays.[/yellow]'
        )
        return

    sub = sub.sort_values('p_multiwell')
    single_pick = sub.head(n_per_class)
    multi_pick = sub.tail(n_per_class).iloc[::-1]
    picks = [('single-well', row) for _, row in single_pick.iterrows()] + \
            [('multi-well',  row) for _, row in multi_pick.iterrows()]

    interval_dir = os.path.join(output_dir, str(int(seconds_interval)))
    plot_dir = os.path.join(interval_dir, 'window_overlays')
    os.makedirs(plot_dir, exist_ok=True)

    x_prev, dx, dt, dt_t = load_series(
        start_date, end_date, seconds_interval,
        kernel_half_width=kernel_half_width,
        trim_quantile=trim_quantile,
        detrend=detrend,
        window_type=window_type,
    )
    df_kappa_full = estimate_kappa_series(x_prev, dx, float(dt), dt_t)
    dt_t_series = pd.to_datetime(dt_t)

    for label, row in picks:
        w_start = pd.Timestamp(row['window_start'])
        w_end   = pd.Timestamp(row['window_end'])
        p_mw    = float(row['p_multiwell'])

        mask = (dt_t_series >= w_start) & (dt_t_series <= w_end + pd.Timedelta(days=1))
        if int(mask.sum()) < 5:
            console.print(
                f'[yellow]Window {w_start.date()}..{w_end.date()} too sparse — skipping.[/yellow]'
            )
            continue
        x_win = x_prev[mask.values]
        t_win = dt_t_series[mask.values]

        if df_kappa_full is not None and not df_kappa_full.empty:
            kappa_mask = (
                (df_kappa_full['datetime'] >= w_start)
                & (df_kappa_full['datetime'] <= w_end + pd.Timedelta(days=1))
            )
            df_kappa_win = df_kappa_full[kappa_mask].reset_index(drop=True)
        else:
            df_kappa_win = df_kappa_full

        out_path = os.path.join(
            plot_dir,
            f'window_{label}_{w_start.strftime("%Y-%m-%d")}_pmw{p_mw:.2f}.png',
        )
        plot_window_price_kappa(
            datetimes=t_win,
            log_prices=x_win,
            df_kappa=df_kappa_win,
            output_path=out_path,
            title=(
                f'{label}  |  {w_start.date()} → {w_end.date()}  |  '
                f'p_multiwell={p_mw:.2f}'
            ),
        )


def plot_window_price_kappa(
    datetimes,
    log_prices,
    df_kappa: pd.DataFrame,
    output_path: str,
    title: str | None = None,
) -> None:
    """
    Per-window diagnostic: log-price (left axis, blue) overlaid with the
    rolling annualised κ(t) restricted to the same window (right axis, red).

    Inputs come from ``ExpectationMaximisation.estimate_kappa_series``
    (rolling-OLS κ(t)) and ``ExpectationMaximisation.load_series``
    (per-window concatenated log_first_price). ``datetimes`` and
    ``log_prices`` must already be sliced to the window of interest.
    """
    times = pd.to_datetime(datetimes)

    fig, ax_p = plt.subplots(figsize=(11, 4))
    ax_p.plot(times, log_prices, color='#1565C0', lw=0.9, label='log-price')
    ax_p.set_xlabel('Date')
    ax_p.set_ylabel('log-price', color='#1565C0')
    ax_p.tick_params(axis='y', labelcolor='#1565C0')
    ax_p.grid(True, alpha=0.3)

    ax_k = ax_p.twinx()
    if df_kappa is not None and not df_kappa.empty:
        col = (
            'kappa_ann_smoothed'
            if 'kappa_ann_smoothed' in df_kappa.columns
            else ('kappa_ann' if 'kappa_ann' in df_kappa.columns else 'kappa')
        )
        ax_k.plot(
            pd.to_datetime(df_kappa['datetime']), df_kappa[col],
            color='#F44336', lw=1.4,
            label=r'$\kappa_{\mathrm{ann}}(t)$ (smoothed)',
        )
    else:
        ax_k.text(
            0.5, 0.5, 'no κ samples in this window',
            transform=ax_k.transAxes, ha='center', va='center',
            color='gray', fontsize=10,
        )
    ax_k.set_ylabel(r'$\kappa_{\mathrm{ann}}$ (yr$^{-1}$)', color='#F44336')
    ax_k.tick_params(axis='y', labelcolor='#F44336')

    if title:
        ax_p.set_title(title)

    h_p, l_p = ax_p.get_legend_handles_labels()
    h_k, l_k = ax_k.get_legend_handles_labels()
    if l_p or l_k:
        ax_p.legend(h_p + h_k, l_p + l_k, loc='upper left', fontsize=9, framealpha=0.85)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    print(f'Saved {output_path}')


def plot_mcmc_kappa_posterior(kappa_samples, kappa_em, output_path):
    """
    Histogram of posterior kappa samples with a vertical line at the EM
    point estimate for comparison.
    """
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(kappa_samples, bins=40, density=True, alpha=0.7,
            color='#2196F3', label='Posterior')
    ax.axvline(kappa_em, color='#F44336', linewidth=1.5,
               linestyle='--', label=f'EM estimate ({kappa_em:.4g})')
    ax.set_xlabel(r'$\kappa$ (mean-reversion rate)')
    ax.set_ylabel('Density')
    ax.set_title(r'Posterior distribution of $\kappa$')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Saved {output_path}')


def plot_mcmc_loglik_trace(chain_loglik, n_burn, output_path):
    """
    Log-likelihood trace across all MCMC sweeps with burn-in shaded separately.
    """
    fig, ax = plt.subplots(figsize=(9, 3))
    iters = np.arange(len(chain_loglik))
    ax.plot(iters[:n_burn], chain_loglik[:n_burn],
            color='gray', alpha=0.6, linewidth=0.8, label='Burn-in')
    ax.plot(iters[n_burn:], chain_loglik[n_burn:],
            color='#2196F3', linewidth=0.8, label='Sampling')
    ax.axvline(n_burn, color='black', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Sweep')
    ax.set_ylabel('Log-likelihood')
    ax.set_title('MCMC log-likelihood trace')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Saved {output_path}')


def _plot_extreme_window_overlays(
    labels_csv,
    start_date,
    end_date,
    seconds_interval,
    kernel_half_width,
    trim_quantile,
    detrend,
    window_type,
    output_dir,
    console,
    n_per_class=2,
):
    """
    Pick the n_per_class windows with the lowest p_multiwell (most clearly
    single-well) and the n_per_class with the highest (most clearly
    multi-well), and produce a price + rolling-κ overlay for each.

    Lives next to main() so it only runs under the RunEM entry-point — the
    MCMC EM warm-start path imports run_phase_c directly and never touches
    this function.
    """
    if not os.path.exists(labels_csv):
        console.print(f'[yellow]Skipping window overlays: labels CSV missing ({labels_csv}).[/yellow]')
        return

    labels_df = pd.read_csv(labels_csv, parse_dates=['window_start', 'window_end'])
    labels_df['seconds_interval'] = pd.to_numeric(
        labels_df['seconds_interval'], errors='coerce'
    ).astype('Int64')
    sub = labels_df[labels_df['seconds_interval'] == int(seconds_interval)].copy()
    sub = sub.dropna(subset=['p_multiwell'])
    if sub.empty:
        console.print('[yellow]No labelled windows with p_multiwell — skipping overlays.[/yellow]')
        return

    sub = sub.sort_values('p_multiwell')
    single_pick = sub.head(n_per_class)
    multi_pick = sub.tail(n_per_class).iloc[::-1]
    picks = [('single-well', row) for _, row in single_pick.iterrows()] + \
            [('multi-well',  row) for _, row in multi_pick.iterrows()]

    plot_dir = os.path.join(output_dir, 'window_overlays')
    os.makedirs(plot_dir, exist_ok=True)

    # Load the full concatenated series once so we can slice per window.
    x_prev, dx, dt, dt_t = load_series(
        start_date, end_date, seconds_interval,
        kernel_half_width=kernel_half_width,
        trim_quantile=trim_quantile,
        detrend=detrend,
        window_type=window_type,
    )
    df_kappa_full = estimate_kappa_series(x_prev, dx, float(dt), dt_t)
    dt_t_series = pd.to_datetime(dt_t)

    for label, row in picks:
        w_start = pd.Timestamp(row['window_start'])
        w_end   = pd.Timestamp(row['window_end'])
        p_mw    = float(row['p_multiwell'])

        mask = (dt_t_series >= w_start) & (dt_t_series <= w_end + pd.Timedelta(days=1))
        if mask.sum() < 5:
            console.print(f'[yellow]Window {w_start.date()}..{w_end.date()} too sparse — skipping.[/yellow]')
            continue
        x_win = x_prev[mask]
        t_win = dt_t_series[mask]

        if not df_kappa_full.empty:
            kappa_mask = (
                (df_kappa_full['datetime'] >= w_start)
                & (df_kappa_full['datetime'] <= w_end + pd.Timedelta(days=1))
            )
            df_kappa_win = df_kappa_full[kappa_mask].reset_index(drop=True)
        else:
            df_kappa_win = df_kappa_full

        out_path = os.path.join(
            plot_dir,
            f'window_{label}_{w_start.strftime("%Y-%m-%d")}_pmw{p_mw:.2f}.png',
        )
        plot_window_price_kappa(
            datetimes=t_win,
            log_prices=x_win,
            df_kappa=df_kappa_win,
            output_path=out_path,
            title=(
                f'{label}  |  {w_start.date()} → {w_end.date()}  |  '
                f'p_multiwell={p_mw:.2f}'
            ),
        )

if __name__ == '__main__':
    print('This module expects precomputed KM results to be passed in from main.')
