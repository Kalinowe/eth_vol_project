import os

import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

_SEC_PER_YEAR = 365.25 * 24 * 3600  # same constant as phase_GP; inlined to avoid circular import


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


def plot_gp_potential_snapshots(model, x_range, snapshots, out_path,
                                n_snapshots=6, n_samples=200):
    """
    Grid of U(x) snapshots reconstructed from saved Kalman states, overlaid
    with the ±2σ posterior envelope from joint drift samples.

    snapshots is a list of (datetime, state_mean, state_cov) tuples appended
    during run_phase_gp's main loop. Each panel restores the saved state on
    the supplied model, predicts the posterior-mean drift, integrates to
    U(x), and draws posterior samples to expose the band that drives
    p_multiwell.
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

    saved_mean = model.state_mean.copy()
    saved_cov = model.state_cov.copy()

    for k, ax in enumerate(axes):
        if k >= n_snapshots:
            ax.axis('off')
            continue
        snap = snapshots[idx[k]]
        dt_q, sm, sc = snap[0], snap[1], snap[2]
        snap_x_range = snap[3] if len(snap) > 3 else x_range
        x_grid = np.linspace(snap_x_range[0], snap_x_range[1], 200)

        model.state_mean = sm
        model.state_cov = sc

        mu_mean, _ = model.predict(x_grid, full_cov=False)
        U_mean = -cumulative_trapezoid(mu_mean, x_grid, initial=0.0)
        U_mean -= U_mean.min()

        # Posterior samples → integrate → per-sample baseline-shift → std.
        # Same construction the topology stats use, so the envelope reflects
        # exactly the uncertainty that p_multiwell is integrating over.
        rng = np.random.default_rng(42)
        f_samples = model.sample_drift(x_grid, n_samples=n_samples, rng=rng)
        U_samples = -cumulative_trapezoid(f_samples, x_grid, axis=0, initial=0.0)
        U_samples = U_samples - U_samples.min(axis=0, keepdims=True)
        U_std = U_samples.std(axis=1)

        ax.fill_between(
            x_grid, U_mean - 2 * U_std, U_mean + 2 * U_std,
            color='steelblue', alpha=0.2, linewidth=0, label='±2σ',
        )
        ax.plot(x_grid, U_mean, color='steelblue', linewidth=1.2, label='mean')
        ax.set_title(str(pd.Timestamp(dt_q).date()), fontsize=8)
        ax.set_xlabel('log-price', fontsize=7)
        ax.set_ylabel('U(x)', fontsize=7)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.legend(fontsize=6, loc='upper right')

    model.state_mean = saved_mean
    model.state_cov = saved_cov

    fig.suptitle('GP posterior mean potential U(x) with ±2σ envelope', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved {out_path}')


def plot_km_vs_gp_overlay(model, x_range, snapshots, labels_df, km_dir,
                          phase_a_seconds_interval, out_path,
                          n_snapshots=6, n_grid=200):
    """
    Per-snapshot overlay of Phase A KM drift (scatter, annualised) on top of
    the GP posterior drift (mean ± 2σ, annualised) at the snapshot state.

    Diagnostic for the KM-vs-GP shape disagreement: if the GP mean tracks the
    KM scatter, the two estimators agree and any remaining mismatch is
    posterior uncertainty. If GP is flat while KM is multi-well, the GP isn't
    seeing the spatial signal (typically because preprocessing has removed it).
    """
    if not snapshots:
        print(f'[plot_km_vs_gp_overlay] no snapshots; skipping {out_path}')
        return

    sec_per_year = 365.25 * 24 * 3600
    n = min(n_snapshots, len(snapshots))
    idx = np.linspace(0, len(snapshots) - 1, n, dtype=int)

    rows = max(1, n // 3) if n > 2 else 1
    cols = max(1, int(np.ceil(n / rows)))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows),
                             squeeze=False)
    axes = axes.flatten()

    saved_mean = model.state_mean.copy()
    saved_cov = model.state_cov.copy()

    df_lbl = labels_df.copy()
    df_lbl['window_start'] = pd.to_datetime(df_lbl['window_start'])
    df_lbl['window_end']   = pd.to_datetime(df_lbl['window_end'])

    for k, ax in enumerate(axes):
        if k >= n:
            ax.axis('off')
            continue
        snap = snapshots[idx[k]]
        dt_q, sm, sc = snap[0], snap[1], snap[2]
        snap_x_range = snap[3] if len(snap) > 3 else x_range
        x_grid = np.linspace(snap_x_range[0], snap_x_range[1], n_grid)

        model.state_mean = sm
        model.state_cov  = sc

        mu_mean, mu_var = model.predict(x_grid, full_cov=False)
        # GP state is already stored in [/year]; no further scaling needed.
        mu_std  = np.sqrt(np.maximum(mu_var, 0.0))

        dt_q_pd = pd.Timestamp(dt_q)
        mask = (df_lbl['window_start'] <= dt_q_pd) & (
            df_lbl['window_end'] + pd.Timedelta(days=1) > dt_q_pd
        )
        match = df_lbl[mask]

        km_df = None
        if not match.empty:
            row = match.iloc[0]
            ws = row['window_start'].strftime('%Y-%m-%d')
            we = row['window_end'].strftime('%Y-%m-%d')
            km_path = os.path.join(
                km_dir, f'km_{ws}_to_{we}_{phase_a_seconds_interval}s.csv',
            )
            if os.path.exists(km_path):
                km_df = pd.read_csv(km_path).dropna(subset=['drift'])

        ax.axhline(0.0, color='grey', linewidth=0.5)
        ax.fill_between(x_grid, mu_mean - 2 * mu_std, mu_mean + 2 * mu_std,
                        color='steelblue', alpha=0.2, linewidth=0,
                        label='GP ±2σ')
        ax.plot(x_grid, mu_mean, color='steelblue', linewidth=1.2,
                label='GP mean')

        if km_df is not None and not km_df.empty:
            km_drift = km_df['drift'].values * sec_per_year
            w = km_df['weight'].values.astype(float)
            w_norm = w / w.max() if w.max() > 0 else w
            sizes = 5 + 25 * w_norm
            ax.scatter(km_df['bin_center'].values, km_drift,
                       s=sizes, color='crimson', alpha=0.55,
                       edgecolors='none', label='Phase A KM')

        ax.set_title(str(dt_q_pd.date()), fontsize=8)
        ax.set_xlabel('log-price', fontsize=7)
        ax.set_ylabel('drift (ann.)', fontsize=7)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.legend(fontsize=6, loc='best')

    model.state_mean = saved_mean
    model.state_cov  = saved_cov

    fig.suptitle('Phase A KM drift vs Phase GP posterior drift (annualised)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved {out_path}')


# ---------------------------------------------------------------------------
# RunGP_simple — KM vs GP overlay
# ---------------------------------------------------------------------------

def plot_km_vs_gp_simple(
    model, x_range_global, snapshots,
    snapped_start, snapped_end,
    phase_a_si, phase_gp_si,
    output_dir, out_path,
    spatial_var_source, hp_opt_mode, use_reproject,
    n_grid=200, n_samples=200, rng=None,
):
    from phase_GP import topology_from_gp  # lazy import to break circular dependency
    """
    Grid of panels (one per snapshot) showing the GP posterior drift only.
    Y-axis is derived from the GP posterior across all snapshots.
    """
    rng = rng or np.random.default_rng(0)
    if not snapshots:
        print(f'[plot_km_vs_gp_simple] no snapshots; skipping {out_path}')
        return

    n    = len(snapshots)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows), squeeze=False)
    axes_flat  = axes.flatten()

    # Preserve model state across all panels.
    saved_mean     = model.state_mean.copy()
    saved_cov      = model.state_cov.copy()
    saved_inducing = model.inducing_x.copy()

    # Y-axis bounds: 2nd/98th percentile of (mu_mean ± 2σ) across all snapshots.
    all_lo, all_hi = [], []
    for snap in snapshots:
        dt_q, sm, sc, snap_x_range, snap_ind = snap[:5]
        model.state_mean = sm
        model.state_cov  = sc
        if not np.array_equal(model.inducing_x, snap_ind):
            model.inducing_x = snap_ind
            model._recompute_hp_dependent()
        xg = np.linspace(snap_x_range[0], snap_x_range[1], n_grid)
        mu, var = model.predict(xg, full_cov=False)
        s = np.sqrt(np.maximum(var, 0.0))
        all_lo.append((mu - 2 * s).min())
        all_hi.append((mu + 2 * s).max())
    y_lo = float(np.percentile(all_lo, 2))
    y_hi = float(np.percentile(all_hi, 98))
    pad  = 0.1 * max(abs(y_lo), abs(y_hi), 1.0)
    y_lo -= pad
    y_hi += pad

    for k, ax in enumerate(axes_flat):
        if k >= n:
            ax.axis('off')
            continue

        dt_query, sm, sc, snap_x_range, snap_inducing_x = snapshots[k][:5]
        model.state_mean = sm
        model.state_cov  = sc
        if not np.array_equal(model.inducing_x, snap_inducing_x):
            model.inducing_x = snap_inducing_x
            model._recompute_hp_dependent()

        x_grid  = np.linspace(snap_x_range[0], snap_x_range[1], n_grid)
        mu_mean, mu_var = model.predict(x_grid, full_cov=False)
        mu_std  = np.sqrt(np.maximum(mu_var, 0.0))

        ax.axhline(0, color='grey', linewidth=0.5)
        ax.fill_between(
            x_grid, mu_mean - 2 * mu_std, mu_mean + 2 * mu_std,
            color='steelblue', alpha=0.2, label='GP ±2σ',
        )
        ax.plot(x_grid, mu_mean, color='steelblue', linewidth=1.4, label='GP mean')
        ax.scatter(
            model.inducing_x, np.zeros(model.M),
            marker='|', color='darkgreen', s=60, zorder=5, label='inducing',
        )
        ax.set_ylim(y_lo, y_hi)

        topo_title = topology_from_gp(
            model, snap_x_range, n_grid=50, n_samples=50, rng=rng,
        )
        beta_val = getattr(model, 'beta_mean', None)
        beta_str = f'  β={beta_val:+.1f}' if beta_val is not None else ''
        ax.set_title(
            f'{pd.Timestamp(dt_query).date()}  '
            f'p={topo_title["p_multiwell"]:.2f}  '
            f'σ/μ={topo_title["mu_std_to_mean"]:.1f}{beta_str}',
            fontsize=8,
        )
        ax.set_xlabel('log-price', fontsize=7)
        ax.set_ylabel('drift [/yr]', fontsize=7)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.legend(fontsize=6, loc='best')

    model.state_mean = saved_mean
    model.state_cov  = saved_cov
    if not np.array_equal(model.inducing_x, saved_inducing):
        model.inducing_x = saved_inducing
        model._recompute_hp_dependent()

    fig.suptitle(
        f'GP drift topology  |  {snapped_start.date()} – {snapped_end.date()}, '
        f'{phase_gp_si}s  |  sp_var={spatial_var_source}  '
        f'HP={hp_opt_mode}  reproject={use_reproject}',
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_topology_snapshots(
    model, x_range_global, snapshots,
    snapped_start, snapped_end,
    phase_gp_si,
    out_path,
    n_grid=200, n_samples=200, rng=None,
):
    """
    Grid of panels (one per snapshot) showing the potential U(x) = -∫μ(x)dx,
    the integral of the GP posterior drift.  Each panel title shows the date,
    p_multiwell, σ/μ, and β (when available).  A ±2σ posterior envelope is
    drawn from drift samples integrated in the same way.
    """
    from phase_GP import topology_from_gp  # lazy import to break circular dependency
    rng = rng or np.random.default_rng(0)
    if not snapshots:
        print(f'[plot_topology_snapshots] no snapshots; skipping {out_path}')
        return

    n    = len(snapshots)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows), squeeze=False)
    axes_flat = axes.flatten()

    saved_mean     = model.state_mean.copy()
    saved_cov      = model.state_cov.copy()
    saved_inducing = model.inducing_x.copy()

    for k, ax in enumerate(axes_flat):
        if k >= n:
            ax.axis('off')
            continue

        snap = snapshots[k]
        dt_query, sm, sc, snap_x_range, snap_inducing_x = snap[:5]
        model.state_mean = sm
        model.state_cov  = sc
        if not np.array_equal(model.inducing_x, snap_inducing_x):
            model.inducing_x = snap_inducing_x
            model._recompute_hp_dependent()

        x_grid = np.linspace(snap_x_range[0], snap_x_range[1], n_grid)

        # Posterior mean drift → integrate to potential
        mu_mean, mu_var = model.predict(x_grid, full_cov=False)
        U_mean = -cumulative_trapezoid(mu_mean, x_grid, initial=0.0)
        U_mean -= U_mean.min()

        # Drift samples → integrate → baseline-shift → std band
        f_samples = model.sample_drift(x_grid, n_samples=n_samples, rng=rng)
        U_samples = -cumulative_trapezoid(f_samples, x_grid, axis=0, initial=0.0)
        U_samples -= U_samples.min(axis=0, keepdims=True)
        U_std = U_samples.std(axis=1)

        ax.fill_between(
            x_grid, U_mean - 2 * U_std, U_mean + 2 * U_std,
            color='steelblue', alpha=0.2, linewidth=0, label='±2σ',
        )
        ax.plot(x_grid, U_mean, color='steelblue', linewidth=1.4, label='mean')
        # Mark inducing points on x-axis
        ax.scatter(
            snap_inducing_x, np.zeros(len(snap_inducing_x)),
            marker='|', color='darkgreen', s=60, zorder=5, label='inducing',
        )

        topo = topology_from_gp(model, snap_x_range, n_grid=50, n_samples=50, rng=rng)
        beta_val = getattr(model, 'beta_mean', None)
        beta_str = f'  β={beta_val:+.1f}' if beta_val is not None else ''
        ax.set_title(
            f'{pd.Timestamp(dt_query).date()}  '
            f'p={topo["p_multiwell"]:.2f}  '
            f'σ/μ={topo["mu_std_to_mean"]:.1f}{beta_str}',
            fontsize=8,
        )
        ax.set_xlabel('log-price', fontsize=7)
        ax.set_ylabel('U(x) = −∫μ dx', fontsize=7)
        ax.set_xlim(snap_x_range[0], snap_x_range[1])
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.legend(fontsize=6, loc='best')

    model.state_mean = saved_mean
    model.state_cov  = saved_cov
    if not np.array_equal(model.inducing_x, saved_inducing):
        model.inducing_x = saved_inducing
        model._recompute_hp_dependent()

    fig.suptitle(
        f'GP potential U(x) snapshots  |  '
        f'{snapped_start.date()} – {snapped_end.date()}, {phase_gp_si}s',
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_drift_with_km(
    model, snapshots,
    snapped_start, snapped_end,
    phase_a_si, phase_gp_si,
    output_dir, out_path,
    spatial_var_source, hp_opt_mode, use_reproject,
    n_grid=200, rng=None,
):
    """
    Grid of panels (one per snapshot) showing GP posterior drift overlaid with
    Phase A KM bin estimates.

    Y-axis uses a weighted percentile of KM drift values so sparse boundary
    bins with extreme drift do not inflate the scale.
    """
    rng = rng or np.random.default_rng(0)
    if not snapshots:
        print(f'[plot_drift_with_km] no snapshots; skipping {out_path}')
        return

    km_dir = os.path.join(output_dir, 'km')

    # ---- Y-axis bounds: weighted percentile of KM drift across the full period ----
    all_drifts, all_weights = [], []
    for fname in sorted(os.listdir(km_dir)):
        if not fname.endswith(f'_{phase_a_si}s.csv'):
            continue
        parts = fname.replace('.csv', '').split('_')
        try:
            km_start = pd.Timestamp(parts[1])
            km_end   = pd.Timestamp(parts[3])
        except Exception:
            continue
        if km_start < pd.Timestamp(snapped_start) or km_end > pd.Timestamp(snapped_end) + pd.Timedelta(days=2):
            continue
        df = pd.read_csv(os.path.join(km_dir, fname)).dropna(subset=['drift'])
        if df.empty:
            continue
        all_drifts.extend((df['drift'].values * _SEC_PER_YEAR).tolist())
        all_weights.extend(df['weight'].values.tolist())

    if all_drifts:
        arr = np.asarray(all_drifts)
        wts = np.asarray(all_weights, dtype=float)
        wts = wts / wts.sum()
        order = np.argsort(arr)
        cdf   = np.cumsum(wts[order])
        q2  = arr[order[np.searchsorted(cdf, 0.02)]]
        q98 = arr[order[np.searchsorted(cdf, 0.98)]]
        pad = 0.25 * max(abs(q2), abs(q98), 1.0)
        y_lo, y_hi = q2 - pad, q98 + pad
    else:
        y_lo, y_hi = -200.0, 200.0

    n    = len(snapshots)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows), squeeze=False)
    axes_flat  = axes.flatten()

    # Preserve model state across all panels.
    saved_mean     = model.state_mean.copy()
    saved_cov      = model.state_cov.copy()
    saved_inducing = model.inducing_x.copy()

    for k, ax in enumerate(axes_flat):
        if k >= n:
            ax.axis('off')
            continue

        snap = snapshots[k]
        dt_query, sm, sc, snap_x_range, snap_inducing_x = snap[:5]
        win_start = snap[5] if len(snap) > 5 else pd.Timestamp(snapped_start)

        model.state_mean = sm
        model.state_cov  = sc
        if not np.array_equal(model.inducing_x, snap_inducing_x):
            model.inducing_x = snap_inducing_x
            model._recompute_hp_dependent()

        x_grid  = np.linspace(snap_x_range[0], snap_x_range[1], n_grid)
        # Use total drift (mu + beta) for comparison with KM, which estimates
        # total drift.  Falls back to mu-only for the stationary model.
        _predict_total = getattr(model, 'predict_total', model.predict)
        mu_mean, mu_var = _predict_total(x_grid, full_cov=False)
        mu_std  = np.sqrt(np.maximum(mu_var, 0.0))
        beta_val = getattr(model, 'beta_mean', None)

        ax.axhline(0, color='grey', linewidth=0.5)
        if beta_val is not None:
            ax.axhline(
                beta_val, color='orange', linewidth=0.9, linestyle='--',
                alpha=0.8, label=f'β={beta_val:+.1f}',
            )
        ax.fill_between(
            x_grid, mu_mean - 2 * mu_std, mu_mean + 2 * mu_std,
            color='steelblue', alpha=0.2, label='GP ±2σ',
        )
        ax.plot(x_grid, mu_mean, color='steelblue', linewidth=1.4, label='GP total')

        # Find the KM CSV whose window contains dt_query.
        km_df = None
        for fname in sorted(os.listdir(km_dir)):
            if not fname.endswith(f'_{phase_a_si}s.csv'):
                continue
            parts = fname.replace('.csv', '').split('_')
            try:
                km_start = pd.Timestamp(parts[1])
                km_end   = pd.Timestamp(parts[3])
            except Exception:
                continue
            if km_start <= pd.Timestamp(dt_query) <= km_end + pd.Timedelta(days=1):
                cand = pd.read_csv(os.path.join(km_dir, fname)).dropna(subset=['drift'])
                if not cand.empty:
                    km_df = cand
                    break

        if km_df is not None:
            km_sorted = km_df.sort_values('bin_center')
            drift_ann = km_sorted['drift'].values * _SEC_PER_YEAR
            w         = km_sorted['weight'].values.astype(float)
            sz        = 5 + 25 * w / max(w.max(), 1.0)
            ax.scatter(
                km_sorted['bin_center'].values, drift_ann,
                s=sz, c='crimson', alpha=0.45, edgecolors='none',
                zorder=4, label='KM bins',
            )

        ax.scatter(
            model.inducing_x, np.zeros(model.M),
            marker='|', color='darkgreen', s=60, zorder=5, label='inducing',
        )
        ax.set_xlim(snap_x_range[0], snap_x_range[1])
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(
            f'{pd.Timestamp(win_start).date()} – {pd.Timestamp(dt_query).date()}',
            fontsize=8,
        )
        ax.set_xlabel('log-price', fontsize=7)
        ax.set_ylabel('drift [/yr]', fontsize=7)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.legend(fontsize=6, loc='best')

    model.state_mean = saved_mean
    model.state_cov  = saved_cov
    if not np.array_equal(model.inducing_x, saved_inducing):
        model.inducing_x = saved_inducing
        model._recompute_hp_dependent()

    fig.suptitle(
        f'GP drift + KM estimates  |  {snapped_start.date()} – {snapped_end.date()}, '
        f'{phase_gp_si}s  |  sp_var={spatial_var_source}  '
        f'HP={hp_opt_mode}  reproject={use_reproject}',
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_logprice_topology(
    model, snapshots,
    x_prev_all, dt_t_all,
    snapped_start, snapped_end,
    phase_gp_si,
    out_path,
    spatial_var_source, hp_opt_mode, use_reproject,
    n_grid=200,
):
    """
    Grid of panels (one per snapshot) showing:
      - log-price time series for that window (blue line)
      - GP posterior mean zero-crossings as horizontal lines:
          green dashed  = stable well  (drift: + → −, price attracted here)
          red dotted    = unstable fix (drift: − → +, price repelled)
      - p_multiwell and well count in the panel title.
    """
    if not snapshots:
        print(f'[plot_logprice_topology] no snapshots; skipping {out_path}')
        return

    dt_arr = pd.to_datetime(dt_t_all)

    n    = len(snapshots)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows), squeeze=False)
    axes_flat = axes.flatten()

    # Preserve model state.
    saved_mean     = model.state_mean.copy()
    saved_cov      = model.state_cov.copy()
    saved_inducing = model.inducing_x.copy()

    for k, ax in enumerate(axes_flat):
        if k >= n:
            ax.axis('off')
            continue

        snap = snapshots[k]
        dt_query, sm, sc, snap_x_range, snap_inducing_x = snap[:5]
        win_start = snap[5] if len(snap) > 5 else (
            pd.Timestamp(snapped_start) if k == 0 else pd.Timestamp(snapshots[k - 1][0])
        )
        win_end = pd.Timestamp(dt_query)

        # Slice log-price for this window.
        mask = (dt_arr >= win_start) & (dt_arr <= win_end)
        times_w  = dt_arr[mask]
        x_w      = np.asarray(x_prev_all)[mask]

        ax.plot(times_w, x_w, color='steelblue', linewidth=0.7, alpha=0.8, label='log-price')

        # Restore GP state for this snapshot.
        model.state_mean = sm
        model.state_cov  = sc
        if not np.array_equal(model.inducing_x, snap_inducing_x):
            model.inducing_x = snap_inducing_x
            model._recompute_hp_dependent()

        # Find zero-crossings of the GP posterior mean drift.
        x_grid   = np.linspace(snap_x_range[0], snap_x_range[1], n_grid)
        mu_mean, _ = model.predict(x_grid, full_cov=False)
        sign_arr = np.sign(mu_mean)
        cross_idx = np.where(np.diff(sign_arr) != 0)[0]

        for ci in cross_idx:
            # Linear interpolation to the exact zero.
            dx   = x_grid[ci + 1] - x_grid[ci]
            frac = -mu_mean[ci] / (mu_mean[ci + 1] - mu_mean[ci] + 1e-30)
            x_cross = x_grid[ci] + frac * dx
            # Stable well: drift slope negative at crossing (+ → −).
            is_stable = mu_mean[ci + 1] < mu_mean[ci]
            ax.axhline(
                x_cross,
                color='green' if is_stable else 'crimson',
                linestyle='--' if is_stable else ':',
                linewidth=1.1 if is_stable else 0.8,
                alpha=0.85,
                label=('well' if is_stable else 'saddle') if ci == cross_idx[0] else '_',
            )

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.set_ylabel('log-price', fontsize=7)

        # p_multiwell from the GP state (reuse stored topo info via quick predict).
        n_stable = sum(
            1 for ci in cross_idx
            if mu_mean[ci + 1] < mu_mean[ci]
        )
        ax.set_title(
            f'{win_start.date()} – {win_end.date()}\n'
            f'wells≈{n_stable}',
            fontsize=7,
        )
        if k == 0:
            ax.legend(fontsize=6, loc='best')

    model.state_mean = saved_mean
    model.state_cov  = saved_cov
    if not np.array_equal(model.inducing_x, saved_inducing):
        model.inducing_x = saved_inducing
        model._recompute_hp_dependent()

    fig.suptitle(
        f'Log-price & GP topology  |  {pd.Timestamp(snapped_start).date()} – {pd.Timestamp(snapped_end).date()}, '
        f'{phase_gp_si}s  |  sp_var={spatial_var_source}  '
        f'HP={hp_opt_mode}  reproject={use_reproject}',
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    print('This module exposes plotting helpers; import its functions from the pipeline scripts.')
