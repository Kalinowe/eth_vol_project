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

import numpy as np
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
    Yield (window_start, window_end) tuples for each calendar month overlapping
    [start_date, end_date]. Windows are clipped to the user-supplied range, so
    partial months at either end are returned as partial windows.
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


# ---------------------------------------------------------------------------
# Topological classifier
# ---------------------------------------------------------------------------

def classify_potential_topology(km_result_df, min_well_separation=0.0, min_barrier_height=0.0):
    """
    Identify stable equilibria from a non-parametric drift estimate.

    A stable equilibrium is a zero-crossing of mu(x) with negative slope:
    mu = -dU/dx implies dmu/dx = -d^2U/dx^2, so dmu/dx < 0 <=> d^2U/dx^2 > 0
    which is exactly the condition for a local minimum of the potential.

    Args:
        km_result_df: output of estimate_km, with columns
            ['bin_center', 'drift', 'diffusion', 'weight'].
        min_well_separation: drop a candidate well if it is closer than this
            (in log-price units) to the previously kept well.
        min_barrier_height: drop a candidate well if the U-barrier separating
            it from the previously kept well is below this threshold. Useful
            for filtering numerical wiggles.

    Returns:
        dict with keys:
            n_wells          -- count of stable equilibria after filtering
            well_locations   -- list of x positions
            barriers         -- list of U-barriers between adjacent wells
                                (length n_wells - 1, may be empty)
            regime           -- 'no-equilibrium' | 'single-well' |
                                'double-well' | '<n>-well'
    """
    df = km_result_df.dropna(subset=['drift']).reset_index(drop=True)
    if len(df) < 3:
        return {'n_wells': 0, 'well_locations': [], 'barriers': [], 'regime': 'no-equilibrium'}

    x = df['bin_center'].values
    mu = df['drift'].values

    # zero crossings of mu
    sign = np.sign(mu)
    sign_changes = np.where(np.diff(sign) != 0)[0]

    candidate_xs = []
    for idx in sign_changes:
        x0, x1 = x[idx], x[idx + 1]
        m0, m1 = mu[idx], mu[idx + 1]
        if m1 != m0:
            x_cross = x0 - m0 * (x1 - x0) / (m1 - m0)  # linear interp
        else:
            x_cross = 0.5 * (x0 + x1)
        slope = (m1 - m0) / (x1 - x0) if x1 != x0 else 0.0
        if slope < 0:  # stable
            candidate_xs.append(x_cross)

    # Compute the potential once, used for barrier heights
    pot_df = integrate_drift_to_potential(km_result_df[['bin_center', 'drift']])
    x_pot = pot_df['bin_center'].values
    U_pot = pot_df['potential'].values

    def barrier_between(xa, xb):
        if xa == xb:
            return 0.0
        lo, hi = sorted([xa, xb])
        mask = (x_pot >= lo) & (x_pot <= hi)
        if not mask.any():
            return 0.0
        U_max = U_pot[mask].max()
        Ua = float(np.interp(xa, x_pot, U_pot))
        Ub = float(np.interp(xb, x_pot, U_pot))
        return float(U_max - min(Ua, Ub))

    # Apply separation/barrier filters greedily, keeping the leftmost in any group
    kept = []
    for cand in candidate_xs:
        if not kept:
            kept.append(cand)
            continue
        sep = abs(cand - kept[-1])
        bh = barrier_between(kept[-1], cand)
        if sep >= min_well_separation and bh >= min_barrier_height:
            kept.append(cand)
        # else: merge by skipping this candidate

    barriers = [barrier_between(kept[i], kept[i + 1]) for i in range(len(kept) - 1)]

    n = len(kept)
    if n == 0:
        regime = 'no-equilibrium'
    elif n == 1:
        regime = 'single-well'
    elif n == 2:
        regime = 'double-well'
    else:
        regime = f'{n}-well'

    return {
        'n_wells': n,
        'well_locations': kept,
        'barriers': barriers,
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
    min_well_separation=0.0,
    min_barrier_height=0.0,
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
    agg_file = os.path.join(
        'data',
        f'ETHUSDT-aggReturns-{window_start.strftime("%Y-%m-%d")}'
        f'_to_{window_end.strftime("%Y-%m-%d")}-{seconds_interval}sec.csv',
    )
    if not os.path.exists(agg_file):
        dc.download_data(window_start, window_end)
        try:
            dc.unzip_data(window_start, window_end)
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
        min_well_separation=min_well_separation,
        min_barrier_height=min_barrier_height,
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
    min_well_separation=0.0,
    min_barrier_height=0.0,
    output_dir='regime_results',
):
    """
    Loop over month windows x sampling intervals and assemble a label table.

    Returns the assembled DataFrame and writes it to
    `<output_dir>/regime_labels_<start>_to_<end>.csv`.
    """
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for window_start, window_end in iter_month_windows(start_date, end_date):
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
                min_well_separation=min_well_separation,
                min_barrier_height=min_barrier_height,
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
                'n_observations': result['n_observations'],
            })

    out_df = pd.DataFrame(rows)
    fname = f"regime_labels_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.csv"
    out_df.to_csv(os.path.join(output_dir, fname), index=False)
    return out_df


def print_regime_table(labels_df, console=None):
    console = console or Console()
    table = Table(title='Phase A: Monthly Regime Labels')
    table.add_column('Start', style='cyan')
    table.add_column('End', style='cyan')
    table.add_column(u'Δt (s)', justify='right')
    table.add_column('Wells', justify='right', style='magenta')
    table.add_column('Regime', style='green')
    table.add_column('Well locations', style='yellow')
    table.add_column('Barriers', style='yellow')
    table.add_column('# obs', justify='right')

    for _, r in labels_df.iterrows():
        table.add_row(
            str(r['window_start']),
            str(r['window_end']),
            str(r['seconds_interval']),
            '' if pd.isna(r['n_wells']) else str(int(r['n_wells'])),
            str(r['regime']),
            str(r['well_locations']) if r['well_locations'] is not None else '',
            str(r['barriers']) if r['barriers'] is not None else '',
            str(r['n_observations']),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # --- Configuration (mirrors main.py defaults so windows can be cross-checked) ---
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 6, 30)

    seconds_intervals = [30, 60, 240]
    kernel_half_width = 5
    trim_quantile = 0.01
    n_bins = 200
    weight_threshold = 5
    detrend = 0  # leave the trend in; let mu(x) absorb it

    # Topology filters: 0 = no filter. Tune after eyeballing the first run.
    min_well_separation = 0.0
    min_barrier_height = 0.0

    labels = run_phase_a(
        start_date,
        end_date,
        seconds_intervals,
        kernel_half_width=kernel_half_width,
        trim_quantile=trim_quantile,
        n_bins=n_bins,
        weight_threshold=weight_threshold,
        detrend=detrend,
        min_well_separation=min_well_separation,
        min_barrier_height=min_barrier_height,
    )

    print_regime_table(labels)
