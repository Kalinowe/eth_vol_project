"""
Phase A of the Halperin-extension regime model.

Pipeline per calendar-month window:
  1. Make sure the daily Binance dumps are downloaded and unzipped.
  2. Aggregate into log-return bars at one or more sampling intervals.
  3. Run Kramers-Moyal on the log-prices to recover non-parametric drift mu(x).
  4. Classify the topology of mu(x) -- count stable equilibria, infer regime.

Each (window, sampling-interval) row produces a regime label that can be
sanity-checked by re-running RunEM.py on that exact window and visually
inspecting the static potential plot.
"""

import glob
import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy.integrate import cumulative_trapezoid
from scipy.signal import argrelmin
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

import data_collection as dc
from force_field_estimation import estimate_km


# ---------------------------------------------------------------------------
# Window iteration
# ---------------------------------------------------------------------------

def iter_month_windows(start_date, end_date):
    """
    Yield (window_start, window_end) for each calendar month overlapping
    [start_date, end_date]. Partial months at either end are clipped.
    """
    current = datetime(start_date.year, start_date.month, 1)
    while current <= end_date:
        if current.month == 12:
            next_month_start = datetime(current.year + 1, 1, 1)
        else:
            next_month_start = datetime(current.year, current.month + 1, 1)
        month_end = next_month_start - timedelta(days=1)

        window_start = max(current, start_date)
        window_end = min(month_end, end_date)
        yield window_start, window_end

        current = next_month_start


def iter_fixed_windows(start_date, end_date, days):
    """
    Yield (window_start, window_end) for non-overlapping fixed-length windows of
    `days` days starting from start_date. The last window is clipped to end_date.
    """
    current = start_date
    while current <= end_date:
        window_end = min(current + timedelta(days=days - 1), end_date)
        yield current, window_end
        current = window_end + timedelta(days=1)


def iter_windows(start_date, end_date, window_type='monthly'):
    """
    Dispatch to the appropriate window iterator.

    Args:
        window_type: 'weekly' (7 days), 'biweekly' (14 days), or 'monthly'
                     (calendar-aligned).
    """
    if window_type == 'weekly':
        return iter_fixed_windows(start_date, end_date, 7)
    elif window_type == 'biweekly':
        return iter_fixed_windows(start_date, end_date, 14)
    elif window_type == 'monthly':
        return iter_month_windows(start_date, end_date)
    else:
        raise ValueError(f"Unknown window_type '{window_type}'. Use 'weekly', 'biweekly', or 'monthly'.")


def normalize_window_boundaries(
    start_date: datetime,
    end_date: datetime,
    window_type: str,
    console=None,
) -> tuple[datetime, datetime]:
    """
    Snap start_date and end_date to exact window boundaries so that every
    window yielded by iter_windows is complete.

      weekly   : start → preceding Monday (inclusive), end → following Sunday
      biweekly : start → preceding Monday, end → the Sunday that closes the
                 biweekly cycle containing end_date (counting from snapped start)
      monthly  : start → 1st of the month, end → last day of the month

    Prints a notice when either date is adjusted.
    """
    orig_start, orig_end = start_date, end_date

    if window_type in ('weekly', 'biweekly'):
        # floor start to Monday (weekday() == 0)
        start_date = start_date - timedelta(days=start_date.weekday())
        if window_type == 'weekly':
            # ceil end to Sunday (weekday() == 6)
            end_date = end_date + timedelta(days=(6 - end_date.weekday()) % 7)
        else:
            # ceil end to the Sunday that closes the biweekly period from snapped start
            days_span = (end_date - start_date).days + 1
            n_windows = (days_span + 13) // 14   # ceiling division
            end_date = start_date + timedelta(days=14 * n_windows - 1)
    elif window_type == 'monthly':
        start_date = start_date.replace(day=1)
        if end_date.month == 12:
            end_date = datetime(end_date.year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(end_date.year, end_date.month + 1, 1) - timedelta(days=1)
    else:
        raise ValueError(
            f"Unknown window_type '{window_type}'. Use 'weekly', 'biweekly', or 'monthly'."
        )

    if start_date != orig_start or end_date != orig_end:
        _console = console or Console()
        parts = []
        if start_date != orig_start:
            parts.append(f"start [cyan]{orig_start.date()}[/cyan] → [cyan]{start_date.date()}[/cyan]")
        if end_date != orig_end:
            parts.append(f"end [cyan]{orig_end.date()}[/cyan] → [cyan]{end_date.date()}[/cyan]")
        _console.print(
            f"[yellow]Window boundary correction ({window_type}):[/yellow] " + "  |  ".join(parts)
        )

    return start_date, end_date


# ---------------------------------------------------------------------------
# Topological classifier
# ---------------------------------------------------------------------------

def _greedy_well_filter(x, U, threshold, min_well_separation):
    """
    Apply the legacy argrelmin + barrier-height greedy filter to a single
    potential curve. Returns (kept_idx, barriers) — barriers measured between
    consecutive kept minima.
    """
    min_idx = argrelmin(U, order=5)[0]
    if len(min_idx) == 0:
        return [], []

    def barrier_between_indices(ia, ib):
        lo, hi = min(ia, ib), max(ia, ib)
        U_peak = U[lo:hi + 1].max()
        return float(U_peak - max(U[ia], U[ib]))

    kept_idx = [min_idx[0]]
    for idx in min_idx[1:]:
        bh = barrier_between_indices(kept_idx[-1], idx)
        sep = abs(x[idx] - x[kept_idx[-1]])
        if bh >= threshold and sep >= min_well_separation:
            kept_idx.append(idx)

    barriers = [
        barrier_between_indices(kept_idx[i], kept_idx[i + 1])
        for i in range(len(kept_idx) - 1)
    ]
    return kept_idx, barriers


def classify_potential_topology(
    km_result_df,
    min_barrier_fraction=0.1,
    min_well_separation=0.0,
    annualize_drift=True,
    n_restarts=3,
    weight_threshold=5,
    n_samples=200,
    n_grid=300,
    random_state=0,
):
    """
    Classify the topology of U(x) by fitting a Gaussian process to the
    Kramers–Moyal drift estimate and integrating posterior drift samples
    into potential samples. Counting wells per posterior sample yields
    P(n_wells >= 2 | data), so windows near the topological transition
    receive a continuous probability rather than a hard label.

    Returns the same keys as the previous version plus:
        p_multiwell  -- P(n_wells >= 2 | data), used downstream as eta weight.
    well_locations and barriers are computed from the *posterior mean*
    potential rather than a single noisy estimate. n_wells is the posterior
    mean well count (a float).

    Args:
        km_result_df: output of estimate_km — columns
            ['bin_center', 'drift', 'diffusion', 'weight'].
        weight_threshold: bin count threshold matching estimate_km; used to
            scale per-bin GP noise as alpha = weight_threshold / weights
            (capped at 10.0 for numerical stability in tail bins).
    """
    df = km_result_df.dropna(subset=['drift']).reset_index(drop=True)
    if len(df) < 5:
        return {
            'n_wells': 0.0, 'p_multiwell': 0.0,
            'well_locations': [], 'barriers': [],
            'u_range': 0.0, 'regime': 'no-equilibrium',
        }

    sec_per_year = 365.25 * 24 * 3600
    x = df['bin_center'].values.astype(float)
    f = df['drift'].values.astype(float)
    if annualize_drift:
        f = f * sec_per_year

    # Per-bin noise variance: bins with low counts get larger alpha. The
    # weight column survives the dropna above because we only filtered on
    # 'drift' (NaN where weight <= weight_threshold in estimate_km).
    weights = df['weight'].values.astype(float)
    weights = np.maximum(weights, 1.0)
    alpha = np.minimum(weight_threshold / weights, 10.0)

    x_std = float(np.std(x)) if np.std(x) > 0 else 1.0
    kernel = (
        RBF(length_scale=x_std * 0.3, length_scale_bounds=(1e-3, 10.0))
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e3))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        n_restarts_optimizer=n_restarts,
        normalize_y=True,
    )
    # Widening the WhiteKernel bound above lets the optimiser drive
    # noise_level very low when the per-bin alpha already absorbs the noise;
    # any residual convergence warning is benign and suppressed here so the
    # per-window console output stays clean.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        gp.fit(x.reshape(-1, 1), f)

    x_grid = np.linspace(x.min(), x.max(), n_grid)
    f_samples = gp.sample_y(
        x_grid.reshape(-1, 1), n_samples=n_samples, random_state=random_state,
    )  # shape (n_grid, n_samples)

    # Integrate each drift sample into a potential and count wells.
    # cumulative_trapezoid returns (n_grid - 1,); prepend 0 to align with x_grid.
    U_samples = -cumulative_trapezoid(f_samples, x_grid, axis=0, initial=0.0)
    U_samples = U_samples - U_samples.min(axis=0, keepdims=True)

    # Posterior mean potential, used for representative well_locations / barriers.
    U_mean = U_samples.mean(axis=1)
    U_mean = U_mean - U_mean.min()
    u_range_mean = float(U_mean.max())

    # Per-sample well counts using the legacy filter.
    n_wells_samples = np.empty(n_samples, dtype=int)
    for j in range(n_samples):
        U_j = U_samples[:, j]
        u_range_j = float(U_j.max())
        if u_range_j == 0.0:
            n_wells_samples[j] = 0
            continue
        thr_j = min_barrier_fraction * u_range_j
        kept_idx, _ = _greedy_well_filter(x_grid, U_j, thr_j, min_well_separation)
        n_wells_samples[j] = len(kept_idx)

    p_multiwell = float(np.mean(n_wells_samples >= 2))
    mean_n_wells = float(np.mean(n_wells_samples))

    if mean_n_wells < 1.3:
        regime = 'single-well'
    elif mean_n_wells < 1.7:
        regime = 'uncertain'
    else:
        regime = 'multi-well'

    if u_range_mean == 0.0:
        return {
            'n_wells': round(mean_n_wells, 2),
            'p_multiwell': round(p_multiwell, 4),
            'well_locations': [], 'barriers': [],
            'u_range': 0.0, 'regime': regime,
        }

    threshold_mean = min_barrier_fraction * u_range_mean
    kept_idx_mean, barriers_mean = _greedy_well_filter(
        x_grid, U_mean, threshold_mean, min_well_separation,
    )
    well_locations = [float(x_grid[i]) for i in kept_idx_mean]
    barriers = [round(b, 6) for b in barriers_mean]

    return {
        'n_wells': round(mean_n_wells, 2),
        'p_multiwell': round(p_multiwell, 4),
        'well_locations': well_locations,
        'barriers': barriers,
        'u_range': round(u_range_mean, 4),
        'regime': regime,
    }


# ---------------------------------------------------------------------------
# Per-window driver
# ---------------------------------------------------------------------------

def analyze_window(
    window_start,
    window_end,
    seconds_interval,
    kernel_half_width=5,
    trim_quantile=0.01,
    n_bins=200,
    weight_threshold=5,
    detrend=0,
    min_barrier_fraction=0.1,
    min_well_separation=0.0,
    min_observations=100,
):
    """
    Run the KM + topology pipeline on one (window, interval) pair.

    Returns a dict on success, or None if the window is unusable (data missing,
    too few observations, etc.).
    """
    # If the aggregated CSV for this (window, interval) already exists we can
    # skip the download and unzip entirely — aggregate_log_returns_range will
    # load it from disk.
    trim_tag = f'_trim{trim_quantile}' if trim_quantile > 0 else ''
    kernel_tag = f'_k{kernel_half_width}' if kernel_half_width > 0 else ''
    detrend_tag = '_detrended' if detrend else ''
    agg_file = os.path.join(
        'data',
        f'ETHUSDT-aggReturns-{window_start.strftime("%Y-%m-%d")}'
        f'_to_{window_end.strftime("%Y-%m-%d")}'
        f'-{seconds_interval}sec{kernel_tag}{trim_tag}{detrend_tag}.csv',
    )
    if not os.path.exists(agg_file):
        try:
            dc.ensure_data(window_start, window_end)
        except FileNotFoundError as exc:
            print(f"[skip] {window_start.date()}..{window_end.date()}: {exc}")
            return None

    try:
        df = dc.aggregate_log_returns_range(
            window_start,
            window_end,
            seconds_interval,
            kernel_half_width=kernel_half_width,
            trim_quantile=trim_quantile,
            detrend=detrend,
        )
    except Exception as exc:
        print(f"[skip] aggregation failed for {window_start.date()}..{window_end.date()}: {exc}")
        return None

    if df.empty:
        return None

    log_prices = df['log_first_price'].dropna().values
    if len(log_prices) < min_observations:
        return None

    km_df = estimate_km(log_prices, seconds_interval, n_bins=n_bins, weight_threshold=weight_threshold)
    topo = classify_potential_topology(
        km_df,
        min_barrier_fraction=min_barrier_fraction,
        min_well_separation=min_well_separation,
        weight_threshold=weight_threshold,
    )

    return {
        'window_start': window_start,
        'window_end': window_end,
        'seconds_interval': seconds_interval,
        'n_observations': int(len(log_prices)),
        'km_df': km_df,
        **topo,
    }


# ---------------------------------------------------------------------------
# Partial-cache helper
# ---------------------------------------------------------------------------

def _load_cached_windows(
    output_dir: str,
    window_type: str,
    start_date: datetime,
    end_date: datetime,
    seconds_intervals: list,
) -> tuple[pd.DataFrame, set]:
    """
    Scan output_dir for regime_labels_*_{window_type}.csv files.

    Returns
    -------
    cached_df  : DataFrame of rows whose window falls entirely within
                 [start_date, end_date]; dates stored as strings (YYYY-MM-DD).
    covered    : set of (window_start_str, window_end_str) pairs where every
                 requested seconds_interval is already present in cached_df.
    """
    pattern = os.path.join(output_dir, f'regime_labels_*_{window_type}.csv')
    csv_files = glob.glob(pattern)
    if not csv_files:
        return pd.DataFrame(), set()

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    frames = []
    for path in csv_files:
        try:
            df = pd.read_csv(path, dtype={'window_start': str, 'window_end': str})
            mask = (df['window_start'] >= start_str) & (df['window_end'] <= end_str)
            sub = df[mask]
            if not sub.empty:
                frames.append(sub)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(), set()

    cached_df = pd.concat(frames, ignore_index=True)
    cached_df = cached_df.drop_duplicates(
        subset=['window_start', 'window_end', 'seconds_interval'], keep='first'
    )

    # Rows from before p_multiwell was added are missing or NaN there —
    # drop them so freshly computed rows can take their place.
    if 'p_multiwell' not in cached_df.columns:
        cached_df['p_multiwell'] = np.nan
    valid_rows = cached_df[cached_df['p_multiwell'].notna()
                           | (cached_df['regime'] == 'unavailable')].copy()

    intervals_set = set(seconds_intervals)
    covered = set()
    for (ws, we), grp in valid_rows.groupby(['window_start', 'window_end']):
        if intervals_set.issubset(set(grp['seconds_interval'].values)):
            covered.add((ws, we))

    # Return only valid rows; otherwise the downstream drop_duplicates
    # (keep='first') would shadow freshly computed p_multiwell values with
    # stale NaN ones from older CSVs.
    return valid_rows, covered


# ---------------------------------------------------------------------------
# Phase A driver
# ---------------------------------------------------------------------------

def run_phase_a(
    start_date,
    end_date,
    seconds_intervals,
    kernel_half_width=5,
    trim_quantile=0.01,
    n_bins=200,
    weight_threshold=5,
    detrend=0,
    min_barrier_fraction=0.1,
    min_well_separation=0.0,
    output_dir='regime_results',
    window_type='monthly',
    console=None,
):
    """
    Loop over windows x sampling intervals and assemble a label table.

    start_date and end_date are snapped to exact window boundaries before
    execution (see normalize_window_boundaries).  A notice is printed when
    either date is adjusted.

    Args:
        window_type: 'weekly', 'biweekly', or 'monthly' (default).

    Returns the assembled DataFrame and writes it to
    `<output_dir>/regime_labels_<snapped_start>_to_<snapped_end>_<window_type>.csv`.
    """
    os.makedirs(output_dir, exist_ok=True)
    console = console or Console()

    start_date, end_date = normalize_window_boundaries(
        start_date, end_date, window_type, console=console
    )

    fname = (
        f"regime_labels_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}"
        f"_{window_type}.csv"
    )
    out_path = os.path.join(output_dir, fname)

    # Fast path: exact output file already on disk — but only if it carries
    # p_multiwell (older CSVs written before that column existed force a rerun).
    if os.path.exists(out_path):
        cached = pd.read_csv(out_path)
        has_pmulti = (
            'p_multiwell' in cached.columns
            and cached[cached['regime'].isin(['single-well', 'multi-well',
                                              'uncertain', 'no-equilibrium'])]
                  ['p_multiwell'].notna().all()
        )
        if has_pmulti:
            console.print(f'[green]Labels CSV already exists:[/green] {fname} — loading from disk.')
            return cached
        console.print(
            f'[yellow]Cached labels CSV {fname} predates p_multiwell — '
            'recomputing missing windows.[/yellow]'
        )

    # Partial cache: collect rows already computed in other files
    cached_df, covered_windows = _load_cached_windows(
        output_dir, window_type, start_date, end_date, seconds_intervals
    )

    all_windows = list(iter_windows(start_date, end_date, window_type))
    missing_windows = [
        (ws, we) for ws, we in all_windows
        if (ws.strftime('%Y-%m-%d'), we.strftime('%Y-%m-%d')) not in covered_windows
    ]

    if covered_windows:
        console.print(
            f'[green]Partial cache:[/green] {len(covered_windows)} window(s) reused from existing files, '
            f'[yellow]{len(missing_windows)}[/yellow] window(s) to compute.'
        )

    km_dir = os.path.join(output_dir, 'km')
    os.makedirs(km_dir, exist_ok=True)

    rows = []
    for window_start, window_end in missing_windows:
        for seconds_interval in seconds_intervals:
            result = analyze_window(
                window_start,
                window_end,
                seconds_interval,
                kernel_half_width=kernel_half_width,
                trim_quantile=trim_quantile,
                n_bins=n_bins,
                weight_threshold=weight_threshold,
                detrend=detrend,
                min_barrier_fraction=min_barrier_fraction,
                min_well_separation=min_well_separation,
            )
            if result is None:
                rows.append({
                    'window_start': window_start.strftime('%Y-%m-%d'),
                    'window_end': window_end.strftime('%Y-%m-%d'),
                    'seconds_interval': seconds_interval,
                    'regime': 'unavailable',
                    'n_wells': None,
                    'p_multiwell': None,
                    'well_locations': None,
                    'barriers': None,
                    'n_observations': 0,
                })
                continue
            km_path = os.path.join(
                km_dir,
                f"km_{result['window_start'].strftime('%Y-%m-%d')}_to_"
                f"{result['window_end'].strftime('%Y-%m-%d')}_{seconds_interval}s.csv",
            )
            result['km_df'].to_csv(km_path, index=False)
            rows.append({
                'window_start': result['window_start'].strftime('%Y-%m-%d'),
                'window_end': result['window_end'].strftime('%Y-%m-%d'),
                'seconds_interval': result['seconds_interval'],
                'regime': result['regime'],
                'n_wells': result['n_wells'],
                'p_multiwell': result.get('p_multiwell', np.nan),
                'well_locations': [round(x, 6) for x in result['well_locations']],
                'barriers': [round(b, 6) for b in result['barriers']],
                'u_range': result['u_range'],
                'n_observations': result['n_observations'],
            })

    # Merge cached and newly computed rows
    parts = []
    if not cached_df.empty:
        parts.append(cached_df)
    if rows:
        parts.append(pd.DataFrame(rows))

    out_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    # Deduplicate; prefer freshly computed rows (keep='last') so a stale cache
    # entry cannot shadow an updated p_multiwell or regime label.
    out_df = out_df.drop_duplicates(
        subset=['window_start', 'window_end', 'seconds_interval'], keep='last'
    )
    out_df = out_df.sort_values(
        ['window_start', 'seconds_interval']
    ).reset_index(drop=True)
    out_df.to_csv(out_path, index=False)
    return out_df


def print_regime_table(labels_df, console=None):
    console = console or Console()
    table = Table(title='Phase A: Monthly Regime Labels')
    table.add_column('Start', style='cyan')
    table.add_column('End', style='cyan')
    table.add_column(u'Δt (s)', justify='right')
    table.add_column('Regime', style='green')
    table.add_column('Wells', justify='right', style='magenta')
    table.add_column('p_multiwell', justify='right', style='magenta')
    table.add_column('U range (ann.)', justify='right', style='yellow')
    table.add_column('Barriers', style='yellow')
    table.add_column('# obs', justify='right')

    has_pmulti = 'p_multiwell' in labels_df.columns

    for _, r in labels_df.iterrows():
        u_range_str = f"{r['u_range']:.2f}" if pd.notna(r.get('u_range')) else ''
        n_wells_val = r['n_wells']
        if pd.isna(n_wells_val):
            n_wells_str = ''
        else:
            # n_wells is now a float (posterior mean); show one decimal.
            n_wells_str = f'{float(n_wells_val):.1f}'
        if has_pmulti and pd.notna(r.get('p_multiwell')):
            pmulti_str = f"{float(r['p_multiwell']):.2f}"
        else:
            pmulti_str = ''
        table.add_row(
            str(r['window_start']),
            str(r['window_end']),
            str(r['seconds_interval']),
            str(r['regime']),
            n_wells_str,
            pmulti_str,
            u_range_str,
            str(r['barriers']) if r['barriers'] is not None else '',
            str(r['n_observations']),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Standalone execution is intentionally disabled: this module is part of
    # the pipeline and its parameters live in RunEM.py. Running it directly
    # would silently fall back to the old hardcoded config and quietly
    # diverge from whatever RunEM is currently configured to do.
    Console().print(
        '[yellow]regime_estimation.py is part of the pipeline; '
        'run [bold]python RunEM.py[/bold] to execute Phase A with the '
        'centrally-configured parameters.[/yellow]'
    )
    raise SystemExit(0)
