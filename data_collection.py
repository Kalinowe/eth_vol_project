import os
import requests
from datetime import datetime, timedelta, timezone
import zipfile
import pandas as pd
import numpy as np

def ensure_data(start_date, end_date):
    """
    Ensure that extracted CSV files exist for every day in [start_date, end_date].

    For each day:
      - If the CSV already exists, nothing is done.
      - If a zip exists but no CSV, the zip is extracted.
      - If neither exists, the zip is downloaded then extracted.
      - After extraction the zip is deleted regardless.

    Args:
        start_date: datetime object for start date
        end_date: datetime object for end date

    Raises:
        FileNotFoundError: If a day's CSV cannot be obtained (download failed
            and no existing zip or CSV was found).
    """
    os.makedirs('data', exist_ok=True)
    current = start_date
    while current <= end_date:
        date_str = current.strftime('%Y-%m-%d')
        zip_path = f"data/ETHUSDT-aggTrades-{date_str}.zip"
        csv_path = f"data/ETHUSDT-aggTrades-{date_str}.csv"

        if not os.path.exists(csv_path):
            if not os.path.exists(zip_path):
                url = (
                    f"https://data.binance.vision/data/spot/daily/aggTrades/ETHUSDT/"
                    f"ETHUSDT-aggTrades-{date_str}.zip"
                )
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        with open(zip_path, 'wb') as f:
                            f.write(response.content)
                        print(f"Downloaded {date_str}")
                    else:
                        print(f"Failed to download {date_str}: {response.status_code}")
                except Exception as e:
                    print(f"Error downloading {date_str}: {e}")

            if os.path.exists(zip_path):
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall('data')
                    print(f"Extracted {date_str}")
                except Exception as e:
                    raise Exception(f"Error unzipping {date_str}: {e}")

            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"Missing data for {date_str}: download failed and no CSV found"
                )

        if os.path.exists(zip_path):
            os.remove(zip_path)

        current += timedelta(days=1)

def aggregate_log_returns(csv_path, x_seconds, prev_log_last=None, kernel_half_width=0):
    """
    Aggregate log returns over x_seconds intervals for one day's Binance aggTrades CSV.

    Each bar boundary t_k = midnight + k * x_seconds gets a backward moving average
    price: BMA(t_k) = mean of all raw trades in [t_k - 2*kernel_half_width, t_k].
    Log-return for bar k = log(BMA(t_{k+1})) - log(BMA(t_k)).

    Avoids materialising a per-second timeseries: uses searchsorted + cumsum so
    only the bar-boundary points are evaluated regardless of interval length.

    Args:
        csv_path:           Path to a daily Binance aggTrades CSV.
        x_seconds:          Bar length in seconds.
        prev_log_last:      log(BMA) carried from the previous day; used to seed
                            the midnight boundary when no trades fall in its window.
        kernel_half_width:  Backward smoothing half-width in seconds. BMA window
                            spans [t - 2*kernel_half_width, t]. 0 = single second.

    Returns:
        (DataFrame, float|None): columns datetime/log_return/log_first_price/
            log_last_price, and log(BMA) at the last boundary of the day.
    """
    filename = os.path.basename(csv_path)
    date_parts = filename.split('-')
    date_str = f"{date_parts[-3]}-{date_parts[-2]}-{date_parts[-1].split('.')[0]}"
    dt = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    midnight = int(dt.timestamp())

    day_seconds = 24 * 3600
    num_intervals = int(day_seconds // x_seconds)

    df = pd.read_csv(csv_path, header=None, usecols=[1, 5])
    df = df.rename(columns={1: 'price', 5: 'timestamp_us'})
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['timestamp_us'] = pd.to_numeric(df['timestamp_us'], errors='coerce')
    df = df.dropna(subset=['price', 'timestamp_us'])

    if df.empty:
        bar_starts = np.arange(midnight, midnight + day_seconds, x_seconds, dtype=np.int64)
        return pd.DataFrame({
            'datetime':        pd.to_datetime(bar_starts, unit='s'),
            'log_return':      np.nan,
            'log_first_price': np.nan,
            'log_last_price':  np.nan,
        }), prev_log_last

    df['timestamp_us'] = df['timestamp_us'].astype(np.int64)
    # Binance uses μs in recent data (~1.7e15) and ms in older data (~1.7e12).
    sample_ts = int(df['timestamp_us'].iloc[0])
    if sample_ts > 1e15:
        df['timestamp_s'] = df['timestamp_us'] // 1_000_000
    else:
        df['timestamp_s'] = df['timestamp_us'] // 1_000
    df = df.sort_values('timestamp_s')

    ts_arr  = df['timestamp_s'].values.astype(np.int64)
    px_arr  = df['price'].values.astype(np.float64)

    # Bar boundaries: t_k = midnight + k * x_seconds, k = 0 .. num_intervals
    boundaries = (midnight + np.arange(num_intervals + 1, dtype=np.int64) * x_seconds)
    window_secs = int(kernel_half_width)  # backward window width in seconds
    # Apply kernel rolling mean (Uniform Kernel / SMA)
    if kernel_half_width > 0:
        window_size = kernel_half_width + 1
        per_second = per_second.rolling(window=window_size, center=False, min_periods=1).mean()

    # Vectorised BMA computation via cumsum + searchsorted.
    # cum_sum[i] = sum(px_arr[:i]) so sum(px_arr[a:b]) = cum_sum[b] - cum_sum[a].
    cum_sum = np.empty(len(px_arr) + 1, dtype=np.float64)
    cum_sum[0] = 0.0
    np.cumsum(px_arr, out=cum_sum[1:])

    idx_lo = np.searchsorted(ts_arr, boundaries - window_secs, side='left')
    idx_hi = np.searchsorted(ts_arr, boundaries,               side='right')
    count  = idx_hi - idx_lo
    valid  = count > 0
    bma    = np.full(len(boundaries), np.nan)
    bma[valid] = (cum_sum[idx_hi[valid]] - cum_sum[idx_lo[valid]]) / count[valid]

    # Seed the midnight boundary with the previous day's BMA when no trades fall
    # in its kernel window (window spans across midnight into the prior day's data).
    if np.isnan(bma[0]) and prev_log_last is not None:
        bma[0] = np.exp(prev_log_last)

    # Forward-fill price gaps (market halts, low-liquidity seconds), then
    # backward-fill the leading edge if midnight was empty and prev_log_last absent.
    bma = pd.Series(bma).ffill().bfill().values

    if np.any(~np.isfinite(bma)) or np.all(bma <= 0):
        bar_starts = boundaries[:-1]
        return pd.DataFrame({
            'datetime':        pd.to_datetime(bar_starts, unit='s'),
            'log_return':      np.nan,
            'log_first_price': np.nan,
            'log_last_price':  np.nan,
        }), prev_log_last

    log_bma = np.log(bma)
    log_first = log_bma[:-1]
    log_last  = log_bma[1:]

    return pd.DataFrame({
        'datetime':        pd.to_datetime(boundaries[:-1], unit='s'),
        'log_return':      log_last - log_first,
        'log_first_price': log_first,
        'log_last_price':  log_last,
    }), float(log_bma[-1])


def aggregate_log_returns_range(start_date, end_date, x_seconds, output_dir='data', kernel_half_width=0, trim_quantile=0.0, detrend=1):
    """
    Aggregate log returns for each CSV in a date range and export the combined results.

    Args:
        start_date: datetime or string in YYYY-MM-DD format
        end_date: datetime or string in YYYY-MM-DD format
        x_seconds: Interval length in seconds
        output_dir: Directory to write the combined CSV file
        kernel_half_width: Kernel width parameter
        trim_quantile: Fraction of data to trim from extremes (e.g., 0.01 for 1%)
        detrend: 1 to remove a linear trend from log-prices before returning, 0 to leave raw

    Returns:
        The combined DataFrame of aggregated log returns for the range.
    """
    def _to_datetime(dt_obj):
        if isinstance(dt_obj, str):
            return datetime.strptime(dt_obj, '%Y-%m-%d')
        if isinstance(dt_obj, datetime):
            return dt_obj
        raise TypeError('start_date and end_date must be datetime or YYYY-MM-DD string')

    start_dt = _to_datetime(start_date)
    end_dt = _to_datetime(end_date)

    if start_dt > end_dt:
        raise ValueError('start_date must be less than or equal to end_date')

    os.makedirs(output_dir, exist_ok=True)

    trim_tag = f'_trim{trim_quantile}' if trim_quantile > 0 else ''
    kernel_tag = f'_k{kernel_half_width}' if kernel_half_width > 0 else ''
    detrend_tag = '_detrended' if detrend else ''
    output_file = os.path.join(
        output_dir,
        f'ETHUSDT-aggReturns-{start_dt.strftime("%Y-%m-%d")}_to_{end_dt.strftime("%Y-%m-%d")}'
        f'-{x_seconds}sec{kernel_tag}{trim_tag}{detrend_tag}.csv',
    )
    if os.path.exists(output_file):
        cached = pd.read_csv(output_file, parse_dates=['datetime'])
        return cached
    else:

        current = start_dt
        combined_frames = []

        prev_log_last = None
        while current <= end_dt:
            date_str = current.strftime('%Y-%m-%d')
            csv_path = os.path.join(output_dir, f'ETHUSDT-aggTrades-{date_str}.csv')
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f'Missing required CSV file: {csv_path}')

            daily_df, prev_log_last = aggregate_log_returns(
                csv_path, 
                x_seconds, 
                prev_log_last=prev_log_last, 
                kernel_half_width=kernel_half_width
            )
            combined_frames.append(daily_df)
            current += timedelta(days=1)

        if combined_frames:
            combined_df = pd.concat(combined_frames, ignore_index=True)
        else:
            combined_df = pd.DataFrame(columns=['datetime', 'log_return'])

        # Validate evenly spaced intervals across the full stacked range
        if not combined_df.empty:
            if not combined_df['datetime'].is_monotonic_increasing:
                raise ValueError('Combined datetime values are not strictly increasing.')

            interval_seconds = combined_df['datetime'].diff().dt.total_seconds().iloc[1:]
            if not np.allclose(interval_seconds, x_seconds, atol=1e-6):
                bad_index = interval_seconds[~np.isclose(interval_seconds, x_seconds, atol=1e-6)].index[0]
                bad_start = combined_df.loc[bad_index - 1, 'datetime']
                bad_end = combined_df.loc[bad_index, 'datetime']
                raise ValueError(
                    f'Uneven interval spacing detected between {bad_start} and {bad_end}: '
                    f'{interval_seconds.iloc[bad_index-1]} seconds (expected {x_seconds})'
                )
            
        # Trim extreme log prices if requested
        if trim_quantile > 0 and not combined_df.empty:
            lower_q = combined_df['log_first_price'].quantile(trim_quantile / 2)
            upper_q = combined_df['log_first_price'].quantile(1 - trim_quantile / 2)
            combined_df = combined_df[(combined_df['log_first_price'] >= lower_q) & 
                                    (combined_df['log_first_price'] <= upper_q)].copy()    
                
        # Detrend the *drift*, not the price level. We compute the per-bar
        # drift mu_w = mean(diff(log_first_price)) within the window and
        # subtract mu_w * (k - k_mean) from both log-price columns. That
        # operation:
        #   - leaves the within-window mean log-price intact (well positions
        #     in absolute log-price space are preserved),
        #   - zeroes the mean of dx = diff(log_first_price) within the
        #     window (the drift is demeaned),
        #   - keeps log_return = log_last_price - log_first_price unchanged
        #     (the same shift is applied to both columns).
        if detrend and not combined_df.empty:
            valid_first = combined_df['log_first_price'].notna()
            if int(valid_first.sum()) >= 2:
                x_first = combined_df.loc[valid_first, 'log_first_price'].to_numpy()
                mean_dx = float(np.mean(np.diff(x_first)))   # per-bar drift
                t = np.arange(len(combined_df), dtype=np.float64)
                k_mean = float(t[valid_first].mean())
                adjustment = mean_dx * (t - k_mean)
                for col in ('log_first_price', 'log_last_price'):
                    combined_df[col] = combined_df[col] - adjustment
                combined_df['log_return'] = (
                    combined_df['log_last_price'] - combined_df['log_first_price']
                )


        combined_df.to_csv(output_file, index=False)
        return combined_df

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _aggregated_returns_path(
    start_date, end_date, seconds_interval,
    kernel_half_width, trim_quantile, detrend,
    data_dir='data',
):
    """Path of the cached aggregated-returns CSV per data_collection naming."""
    def _to_dt(d):
        if isinstance(d, str):
            return datetime.strptime(d, '%Y-%m-%d')
        return d

    start_dt = _to_dt(start_date)
    end_dt = _to_dt(end_date)
    kernel_tag = f'_k{kernel_half_width}' if kernel_half_width > 0 else ''
    trim_tag = f'_trim{trim_quantile}' if trim_quantile > 0 else ''
    detrend_tag = '_detrended' if detrend else ''
    return os.path.join(
        data_dir,
        f'ETHUSDT-aggReturns-{start_dt.strftime("%Y-%m-%d")}_to_'
        f'{end_dt.strftime("%Y-%m-%d")}-{seconds_interval}sec'
        f'{kernel_tag}{trim_tag}{detrend_tag}.csv',
    )


def load_series(
    start_date,
    end_date,
    seconds_interval,
    kernel_half_width=5,
    trim_quantile=0.01,
    detrend=0,
    window_type=None,
):
    """
    Load aggregated log-prices over [start_date, end_date] at one sampling
    interval and emit (x_prev, dx, dt, datetimes) with cross-gap increments
    dropped.

    Two operating modes:

    - ``window_type is None`` (legacy): a single whole-range aggregated CSV
      produced by ``dc.aggregate_log_returns_range(start, end, ...)`` is read.
      Detrending (when enabled at aggregation time) is applied globally.

    - ``window_type='weekly'|'biweekly'|'monthly'``: the per-window CSVs
      produced by Phase A's per-window aggregation are read and concatenated
      in time order. With ``detrend=1`` each window's linear trend has been
      removed *within* that window only, so the concatenated log-price series
      is locally detrended. Cross-window dx are dropped via a window_idx tag
      so a jump between two independently-detrended windows cannot leak into
      the SDE M-step.

    Returns
    -------
    x_prev    : (N,) log-price at t-1
    dx        : (N,) increment x_t - x_{t-1}
    dt        : float, nominal step in seconds
    dt_t      : (N,) datetime of x_t (for plotting), aligned with dx
    """
    if window_type is None:
        cached_path = _aggregated_returns_path(
            start_date, end_date, seconds_interval,
            kernel_half_width, trim_quantile, detrend,
        )
        if os.path.exists(cached_path):
            df = pd.read_csv(cached_path, parse_dates=['datetime'])
            if df.empty:
                df = aggregate_log_returns_range(
                    start_date, end_date, seconds_interval,
                    kernel_half_width=kernel_half_width,
                    trim_quantile=trim_quantile,
                    detrend=detrend,
                )
        else:
            df = aggregate_log_returns_range(
                start_date, end_date, seconds_interval,
                kernel_half_width=kernel_half_width,
                trim_quantile=trim_quantile,
                detrend=detrend,
            )
        if df.empty:
            raise ValueError('Empty aggregated returns dataframe.')
        df = df.sort_values('datetime').reset_index(drop=True)
        df['__window_idx'] = 0
    else:
        from regime_estimation import iter_windows
        frames = []
        for w_idx, (window_start, window_end) in enumerate(
            iter_windows(start_date, end_date, window_type)
        ):
            cached_path = _aggregated_returns_path(
                window_start, window_end, seconds_interval,
                kernel_half_width, trim_quantile, detrend,
            )
            if os.path.exists(cached_path):
                df_w = pd.read_csv(cached_path, parse_dates=['datetime'])
                if df_w.empty:
                    df_w = aggregate_log_returns_range(
                        window_start, window_end, seconds_interval,
                        kernel_half_width=kernel_half_width,
                        trim_quantile=trim_quantile,
                        detrend=detrend,
                    )
            else:
                df_w = aggregate_log_returns_range(
                    window_start, window_end, seconds_interval,
                    kernel_half_width=kernel_half_width,
                    trim_quantile=trim_quantile,
                    detrend=detrend,
                )
            if df_w is None or df_w.empty:
                continue
            df_w = df_w.copy()
            df_w['__window_idx'] = w_idx
            frames.append(df_w)
        if not frames:
            raise ValueError(
                f'No per-window aggregated CSVs available for '
                f'{start_date}..{end_date} at {seconds_interval}s '
                f'({window_type}).'
            )
        df = pd.concat(frames, ignore_index=True).sort_values('datetime')
        df = df.reset_index(drop=True)

    x = df['log_first_price'].values.astype(float)
    ts = pd.to_datetime(df['datetime']).values
    win_idx = df['__window_idx'].values.astype(int)

    dt_arr = (np.diff(ts) / np.timedelta64(1, 's')).astype(float)
    finite = np.isfinite(np.diff(x)) & np.isfinite(x[:-1])
    same_window = (win_idx[:-1] == win_idx[1:])
    valid = (dt_arr > 0) & (dt_arr <= 1.5 * seconds_interval) & finite & same_window

    x_prev = x[:-1][valid]
    dx = np.diff(x)[valid]
    dt_t = ts[1:][valid]

    return x_prev, dx, float(seconds_interval), dt_t


## Change 4 — EMA drift demeaning

def ema_demean_drift(r, dt_t, halflife_days=14.0):
    """
    Remove the slow-moving drift trend from scaled increments r = dx/dt
    using a causal exponential moving average (EMA).

    The EMA captures the price trend component of the drift. Subtracting
    it leaves only the restoring force structure (wells, barriers) that
    the GP should model.

    The halflife should be several times the OU mean-reversion timescale
    (1/kappa) so the EMA tracks genuine trend without absorbing within-
    regime mean-reversion. Two to four weeks is appropriate for ETH at
    30-second intervals.

    Parameters
    ----------
    r             : (N,) scaled increment array, r_t = dx_t / dt
    dt_t          : (N,) observation datetimes (numpy datetime64 or
                    anything pd.to_datetime accepts)
    halflife_days : float, EMA halflife in days. Default 14.

    Pass halflife_days = 0 (or None, or non-finite) to disable demeaning —
    used by Phase GP when the spatial signal is fragile against an EMA
    whose halflife is comparable to the regime dwell time.

    Returns
    -------
    r_hat : (N,) EMA-demeaned drift — what the GP observes as mu(x, t)
    r_bar : (N,) EMA trend estimate — the removed component

    Notes
    -----
    Uses pd.Series.ewm with times= and halflife= in time-unit strings
    so the halflife is wall-clock time, not observation count. This is
    critical at 30-second intervals where a '14-day halflife' should
    mean 14 days of real time, not 14 observations.
    """
    r = np.asarray(r, dtype=float)
    if halflife_days is None or not np.isfinite(halflife_days) or halflife_days <= 0:
        return r.copy(), np.zeros_like(r)
    halflife_str = f'{halflife_days * 24 * 3600:.0f}s'
    r_series = pd.Series(r, index=pd.to_datetime(dt_t))
    r_bar_series = r_series.ewm(
        halflife=halflife_str,
        times=r_series.index,
        adjust=False,   # causal: no future data used
    ).mean()

    r_bar = r_bar_series.values
    r_hat = r - r_bar
    return r_hat, r_bar

# ---------------------------------------------------------------------------
# Useful tool
# ---------------------------------------------------------------------------

def window_seconds(window_type):
    if window_type == 'weekly':
        return 7 * 24 * 3600
    elif window_type == 'biweekly':
        return 14 * 24 * 3600
    elif window_type == 'monthly':
        return 30 * 24 * 3600
    else:
        raise ValueError(f"Unknown window_type '{window_type}'. Use 'weekly', 'biweekly', or 'monthly'.")