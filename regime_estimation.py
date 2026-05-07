"""
Phase A of the Halperin-extension regime model.

Pipeline per calendar-month window:
  1. Make sure the daily Binance dumps are downloaded and unzipped.
  2. Aggregate into log-return bars at one or more sampling intervals.
  3. Run Kramers-Moyal on the log-prices to recover non-parametric drift mu(x).
  4. Classify the topology of mu(x) -- count stable equilibria, infer regime.

Each (window, sampling-interval) row produces a regime label that can be
sanity-checked by setting main.py to that exact window and visually inspecting
the static potential plot.
"""

import os
from datetime import datetime, timedelta

import pandas as pd
from rich.console import Console
from rich.table import Table

import data_collection as dc
from force_field_estimation import estimate_km, integrate_drift_to_potential


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

def classify_potential_topology(
    km_result_df,
    min_barrier_fraction=0.1,
    min_well_separation=0.0,
    annualize_drift=True,
):
    """
    Classify the topology of the potential U(x) estimated from KM coefficients.

    Strategy: integrate the drift to get U(x), find all local minima, then keep
    only those whose barrier height is >= min_barrier_fraction × total U range.
    Working on the integrated potential is more robust than detecting zero
    crossings of the raw drift, because integration smooths bin-level noise.

    Args:
        km_result_df: output of estimate_km — columns
            ['bin_center', 'drift', 'diffusion', 'weight'].
        min_barrier_fraction: a candidate well is kept only if the U-barrier
            separating it from the nearest kept well is at least this fraction
            of the full potential range. Default 0.1 (10%). Increase toward
            0.2–0.3 to suppress minor wiggles; set to 0 to keep every minimum.
        min_well_separation: additional filter — drop a well if it is closer
            than this many log-price units to the previously kept well.
        annualize_drift: if True, multiply drift by seconds-per-year before
            integrating. The topology is unchanged, but barrier heights and
            potential range are in annualised units that are easier to reason
            about (same scale as the drift plots in main.py).

    Returns:
        dict with keys:
            n_wells          -- count of stable wells after filtering
            well_locations   -- list of x positions of local U minima
            barriers         -- list of U-barriers between adjacent wells
                                (length n_wells - 1)
            u_range          -- total potential range (max - min of U)
            regime           -- 'no-equilibrium' | 'single-well' |
                                'double-well' | '<n>-well'
    """
    from scipy.signal import argrelmin

    df = km_result_df.dropna(subset=['drift']).reset_index(drop=True)
    if len(df) < 5:
        return {
            'n_wells': 0, 'well_locations': [], 'barriers': [],
            'u_range': 0.0, 'regime': 'no-equilibrium',
        }

    km_for_pot = df[['bin_center', 'drift']].copy()
    if annualize_drift:
        sec_per_year = 365.25 * 24 * 3600
        km_for_pot['drift'] = km_for_pot['drift'] * sec_per_year

    pot_df = integrate_drift_to_potential(km_for_pot)
    x = pot_df['bin_center'].values
    U = pot_df['potential'].values
    # Shift so global minimum = 0
    U = U - U.min()
    u_range = float(U.max())

    if u_range == 0:
        return {
            'n_wells': 0, 'well_locations': [], 'barriers': [],
            'u_range': 0.0, 'regime': 'no-equilibrium',
        }

    threshold = min_barrier_fraction * u_range

    # Local minima of U: point must be smaller than neighbours within `order` bins.
    # order=5 means a minimum must be lower than 5 bins on each side; this already
    # rejects very narrow wiggles without needing extra smoothing.
    min_idx = argrelmin(U, order=5)[0]

    if len(min_idx) == 0:
        return {
            'n_wells': 0, 'well_locations': [], 'barriers': [],
            'u_range': round(u_range, 4), 'regime': 'no-equilibrium',
        }

    # For each pair of consecutive candidate minima, compute the barrier height
    # (max U between them minus the higher of the two minima values).
    def barrier_between_indices(ia, ib):
        lo, hi = min(ia, ib), max(ia, ib)
        U_peak = U[lo:hi + 1].max()
        return float(U_peak - max(U[ia], U[ib]))

    # Greedy filter: accept the first minimum, then accept subsequent ones
    # only if they clear both the barrier and separation thresholds.
    kept_idx = [min_idx[0]]
    for idx in min_idx[1:]:
        bh = barrier_between_indices(kept_idx[-1], idx)
        sep = abs(x[idx] - x[kept_idx[-1]])
        if bh >= threshold and sep >= min_well_separation:
            kept_idx.append(idx)

    barriers = [
        round(barrier_between_indices(kept_idx[i], kept_idx[i + 1]), 6)
        for i in range(len(kept_idx) - 1)
    ]
    well_locations = [float(x[i]) for i in kept_idx]

    n = len(kept_idx)
    if n == 0:
        regime = 'no-equilibrium'
    elif n == 1:
        regime = 'single-well'
    else:
        regime = 'multi-well'

    return {
        'n_wells': n,
        'well_locations': well_locations,
        'barriers': barriers,
        'u_range': round(u_range, 4),
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

    rows = []
    for window_start, window_end in iter_windows(start_date, end_date, window_type):
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
                    'well_locations': None,
                    'barriers': None,
                    'n_observations': 0,
                })
                continue
            rows.append({
                'window_start': result['window_start'].strftime('%Y-%m-%d'),
                'window_end': result['window_end'].strftime('%Y-%m-%d'),
                'seconds_interval': result['seconds_interval'],
                'regime': result['regime'],
                'n_wells': result['n_wells'],
                'well_locations': [round(x, 6) for x in result['well_locations']],
                'barriers': [round(b, 6) for b in result['barriers']],
                'u_range': result['u_range'],
                'n_observations': result['n_observations'],
            })

    out_df = pd.DataFrame(rows)
    fname = (
        f"regime_labels_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}"
        f"_{window_type}.csv"
    )
    out_df.to_csv(os.path.join(output_dir, fname), index=False)
    return out_df


def print_regime_table(labels_df, console=None):
    console = console or Console()
    table = Table(title='Phase A: Monthly Regime Labels')
    table.add_column('Start', style='cyan')
    table.add_column('End', style='cyan')
    table.add_column(u'Δt (s)', justify='right')
    table.add_column('Regime', style='green')
    table.add_column('Wells', justify='right', style='magenta')
    table.add_column('U range (ann.)', justify='right', style='yellow')
    table.add_column('Barriers', style='yellow')
    table.add_column('# obs', justify='right')

    for _, r in labels_df.iterrows():
        u_range_str = f"{r['u_range']:.2f}" if pd.notna(r.get('u_range')) else ''
        table.add_row(
            str(r['window_start']),
            str(r['window_end']),
            str(r['seconds_interval']),
            str(r['regime']),
            '' if pd.isna(r['n_wells']) else str(int(r['n_wells'])),
            u_range_str,
            str(r['barriers']) if r['barriers'] is not None else '',
            str(r['n_observations']),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # --- Configuration (mirrors main.py defaults so windows can be cross-checked) ---
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)

    seconds_intervals = [30, 60, 120]
    kernel_half_width = 5
    trim_quantile = 0.01
    n_bins = 100
    weight_threshold = 5
    detrend = 1  # leave the trend in; let mu(x) absorb it

    # Window granularity: 'weekly', 'biweekly', or 'monthly'
    window_type = 'weekly'

    # Topology filters. min_barrier_fraction: keep a well only if its barrier
    # height is >= this fraction of the total U range. 0.1 = 10%.
    # Increase toward 0.2-0.3 to suppress minor wiggles.
    min_barrier_fraction = 0.1
    min_well_separation = 0.0

    _console = Console()
    labels = run_phase_a(
        start_date,
        end_date,
        seconds_intervals,
        kernel_half_width=kernel_half_width,
        trim_quantile=trim_quantile,
        n_bins=n_bins,
        weight_threshold=weight_threshold,
        detrend=detrend,
        min_barrier_fraction=min_barrier_fraction,
        min_well_separation=min_well_separation,
        window_type=window_type,
        console=_console,
    )

    print_regime_table(labels)
