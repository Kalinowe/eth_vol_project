"""
Kramers-Moyal estimation of non-parametric drift mu(x) from log-price data.

Per calendar-month window: aggregate log-return bars, run KM, and write
drift/diffusion bins to {output_dir}/km/.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from kramersmoyal import km as kmc_lib
from rich.console import Console

import data_collection as dc


# ---------------------------------------------------------------------------
# KM drift estimator
# ---------------------------------------------------------------------------


def estimate_km(log_prices, seconds_interval, n_bins=200, weight_threshold=5):
    """
    Run Kramers-Moyal estimation on a 1D log-price series.

    Args:
        log_prices: 1D array-like of log-prices, NaNs already dropped.
        seconds_interval: dt between consecutive observations, in seconds.
        n_bins: number of bins used by the KM histogram.
        weight_threshold: minimum bin weight (count) to keep; below this drift
            and diffusion are set to NaN to avoid divergent estimates in the tails.

    Returns:
        DataFrame with columns ['bin_center', 'drift', 'diffusion', 'weight'].
        Drift is annualizable as drift * sec_per_year. Diffusion is sigma^2 / 2.
    """
    x_series = np.asarray(log_prices, dtype=np.float64).reshape(-1, 1)
    kmc, edges = kmc_lib(x_series, bins=n_bins, full=False)

    weights = kmc[0, :]
    drift = np.where(weights > weight_threshold, kmc[1, :] / seconds_interval, np.nan)
    diffusion = np.where(
        weights > weight_threshold, kmc[2, :] / (2 * seconds_interval), np.nan
    )
    centers = edges[0]

    return pd.DataFrame(
        {
            "bin_center": centers,
            "drift": drift,
            "diffusion": diffusion,
            "weight": weights,
        }
    )


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


def iter_windows(start_date, end_date, window_type="monthly"):
    """
    Dispatch to the appropriate window iterator.

    Args:
        window_type: 'weekly' (7 days), 'biweekly' (14 days), or 'monthly'
                     (calendar-aligned).
    """
    if window_type == "weekly":
        return iter_fixed_windows(start_date, end_date, 7)
    elif window_type == "biweekly":
        return iter_fixed_windows(start_date, end_date, 14)
    elif window_type == "monthly":
        return iter_month_windows(start_date, end_date)
    else:
        raise ValueError(
            f"Unknown window_type '{window_type}'. Use 'weekly', 'biweekly', or 'monthly'."
        )


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

    if window_type in ("weekly", "biweekly"):
        # floor start to Monday (weekday() == 0)
        start_date = start_date - timedelta(days=start_date.weekday())
        if window_type == "weekly":
            # ceil end to Sunday (weekday() == 6)
            end_date = end_date + timedelta(days=(6 - end_date.weekday()) % 7)
        else:
            # ceil end to the Sunday that closes the biweekly period from snapped start
            days_span = (end_date - start_date).days + 1
            n_windows = (days_span + 13) // 14  # ceiling division
            end_date = start_date + timedelta(days=14 * n_windows - 1)
    elif window_type == "monthly":
        start_date = start_date.replace(day=1)
        if end_date.month == 12:
            end_date = datetime(end_date.year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(end_date.year, end_date.month + 1, 1) - timedelta(
                days=1
            )
    else:
        raise ValueError(
            f"Unknown window_type '{window_type}'. Use 'weekly', 'biweekly', or 'monthly'."
        )

    if start_date != orig_start or end_date != orig_end:
        _console = console or Console()
        parts = []
        if start_date != orig_start:
            parts.append(
                f"start [cyan]{orig_start.date()}[/cyan] → [cyan]{start_date.date()}[/cyan]"
            )
        if end_date != orig_end:
            parts.append(
                f"end [cyan]{orig_end.date()}[/cyan] → [cyan]{end_date.date()}[/cyan]"
            )
        _console.print(
            f"[yellow]Window boundary correction ({window_type}):[/yellow] "
            + "  |  ".join(parts)
        )

    return start_date, end_date


# ---------------------------------------------------------------------------
# Per-window driver
# ---------------------------------------------------------------------------


def analyze_window(
    window_start,
    window_end,
    seconds_interval,
    kernel_half_width=5,
    trim_quantile=0.01,
    ema_halflife_days=0.0,
    n_bins=200,
    weight_threshold=5,
    min_observations=100,
):
    """
    Ensure data exists, aggregate log-returns, and run KM estimation for
    one (window, interval) pair.

    Returns a dict with keys [window_start, window_end, seconds_interval,
    n_observations, km_df], or None if the window is unusable.
    """
    # If the aggregated CSV for this (window, interval) already exists we can
    # skip the download and unzip entirely — aggregate_log_returns_range will
    # load it from disk.
    kernel_tag = f"_k{kernel_half_width}" if kernel_half_width > 0 else ""
    ema_tag = f"_emahlf{int(ema_halflife_days)}d"
    agg_file = os.path.join(
        "data",
        f"ETHUSDT-aggReturns-{window_start.strftime('%Y-%m-%d')}"
        f"_to_{window_end.strftime('%Y-%m-%d')}"
        f"-{seconds_interval}sec{kernel_tag}{ema_tag}.csv",
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
            ema_halflife_days=ema_halflife_days,
        )
    except Exception as exc:
        print(
            f"[skip] aggregation failed for {window_start.date()}..{window_end.date()}: {exc}"
        )
        return None

    if df.empty:
        return None

    log_prices = df["log_first_price"].dropna().values
    if len(log_prices) < min_observations:
        return None

    km_df = estimate_km(
        log_prices, seconds_interval, n_bins=n_bins, weight_threshold=weight_threshold
    )
    return {
        "window_start": window_start,
        "window_end": window_end,
        "seconds_interval": seconds_interval,
        "n_observations": int(len(log_prices)),
        "km_df": km_df,
    }


# ---------------------------------------------------------------------------
# KM estimation driver
# ---------------------------------------------------------------------------


def run_km_estimation(
    start_date,
    end_date,
    seconds_interval,
    kernel_half_width=5,
    trim_quantile=0.01,
    ema_halflife_days=0.0,
    n_bins=200,
    weight_threshold=5,
    output_dir="regime_results",
    window_type="monthly",
    console=None,
):
    """
    Iterate over windows, run KM estimation on each, and write per-window KM
    CSV files to {output_dir}/km/. Windows whose KM CSV already exists on disk
    are skipped (cache hit).

    Returns a DataFrame with columns [window_start, window_end, n_observations].
    """
    os.makedirs(output_dir, exist_ok=True)
    console = console or Console()
    seconds_interval = int(seconds_interval)

    start_date, end_date = normalize_window_boundaries(
        start_date, end_date, window_type, console=console
    )

    kernel_tag = f"_k{kernel_half_width}" if kernel_half_width > 0 else ""
    km_dir = os.path.join(output_dir, "km")
    os.makedirs(km_dir, exist_ok=True)

    all_windows = list(iter_windows(start_date, end_date, window_type))
    rows = []
    n_cached = 0
    for window_start, window_end in all_windows:
        km_path = os.path.join(
            km_dir,
            f"km_{window_start.strftime('%Y-%m-%d')}_to_"
            f"{window_end.strftime('%Y-%m-%d')}_{seconds_interval}s"
            f"{kernel_tag}.csv",
        )
        if os.path.exists(km_path):
            n_cached += 1
            rows.append(
                {
                    "window_start": window_start.strftime("%Y-%m-%d"),
                    "window_end": window_end.strftime("%Y-%m-%d"),
                    "n_observations": None,
                }
            )
            continue

        result = analyze_window(
            window_start,
            window_end,
            seconds_interval,
            kernel_half_width=kernel_half_width,
            trim_quantile=trim_quantile,
            ema_halflife_days=ema_halflife_days,
            n_bins=n_bins,
            weight_threshold=weight_threshold,
        )
        if result is None:
            rows.append(
                {
                    "window_start": window_start.strftime("%Y-%m-%d"),
                    "window_end": window_end.strftime("%Y-%m-%d"),
                    "n_observations": 0,
                }
            )
            continue

        result["km_df"].to_csv(km_path, index=False)
        rows.append(
            {
                "window_start": result["window_start"].strftime("%Y-%m-%d"),
                "window_end": result["window_end"].strftime("%Y-%m-%d"),
                "n_observations": result["n_observations"],
            }
        )

    if n_cached:
        console.print(
            f"[green]KM cache:[/green] {n_cached} window(s) reused, "
            f"{len(all_windows) - n_cached} computed."
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    Console().print(
        "[yellow]kramers_moyal.py is a library module; import it from RunGP.py.[/yellow]"
    )
    raise SystemExit(0)
