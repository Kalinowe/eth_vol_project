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
    Read a CSV file and aggregate log returns over X second intervals starting from midnight.

    Args:
        csv_path: Path to the CSV file
        x_seconds: Interval length in seconds
        prev_log_last: Previous interval log price carried from an earlier day
        kernel_half_width: Number of seconds to each side for the rolling mean kernel

    Returns:
        Tuple[DataFrame, float|None]: DataFrame with 'datetime', 'log_return',
            'log_first_price', 'log_last_price' and the last log price from the file.
    """
    # Extract date from filename
    filename = os.path.basename(csv_path)
    date_parts = filename.split('-')
    date_str = f"{date_parts[-3]}-{date_parts[-2]}-{date_parts[-1].split('.')[0]}"
    dt = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    midnight = int(dt.timestamp())

    df = pd.read_csv(csv_path, header=None, usecols=[1, 5])
    df = df.rename(columns={1: 'price', 5: 'timestamp_us'})
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['timestamp_us'] = pd.to_numeric(df['timestamp_us'], errors='coerce')
    df = df.dropna(subset=['price', 'timestamp_us'])

    if df.empty:
        day_seconds = 24 * 3600
        num_intervals = int(day_seconds // x_seconds)
        intervals = np.arange(midnight, midnight + day_seconds, x_seconds, dtype=np.int64)
        result_df = pd.DataFrame({
            'datetime': pd.to_datetime(intervals, unit='s'),
            'log_return': np.nan,
            'log_first_price': np.nan,
            'log_last_price': np.nan
        })
        return result_df, prev_log_last

    df['timestamp_us'] = df['timestamp_us'].astype(np.int64)
    # Binance uses μs in recent data (~1.7e15) and ms in older data (~1.7e12).
    # Divide accordingly so timestamp_s is always Unix seconds.
    sample_ts = int(df['timestamp_us'].iloc[0])
    if sample_ts > 1e15:
        df['timestamp_s'] = df['timestamp_us'] // 1_000_000  # microseconds
    else:
        df['timestamp_s'] = df['timestamp_us'] // 1_000      # milliseconds
    df = df.sort_values('timestamp_s')

    day_seconds = 24 * 3600
    full_day_range = np.arange(midnight, midnight + day_seconds, dtype=np.int64)
    
    # Group by second and average
    per_second_raw = df.groupby('timestamp_s')['price'].mean()
    
    # Reindex to include every second of the day and fill NaNs
    per_second = per_second_raw.reindex(full_day_range)
    
    # Initial fill: if the very first second is NaN, use the previous day's last price
    if pd.isna(per_second.iloc[0]) and prev_log_last is not None:
        per_second.iloc[0] = np.exp(prev_log_last)
        
    per_second = per_second.ffill().bfill() # bfill handles cases with no prev_log_last

    # Apply kernel rolling mean (Uniform Kernel / SMA)
    if kernel_half_width > 0:
        window_size = 2 * kernel_half_width + 1
        per_second = per_second.rolling(window=window_size, center=False, min_periods=1).mean()

    prices_array = per_second.values
    
    num_intervals = int(day_seconds // x_seconds)
    intervals = []
    log_returns = []
    log_first_prices = []
    log_last_prices = []

    for i in range(num_intervals):
        start_idx = i * x_seconds
        end_idx = min((i + 1) * x_seconds, day_seconds - 1)

        first_price = prices_array[start_idx]
        last_price = prices_array[end_idx]

        log_first = np.log(first_price)
        log_last = np.log(last_price)
        log_ret = log_last - log_first
        prev_log_last = log_last

        intervals.append(midnight + start_idx)
        log_returns.append(log_ret)
        log_first_prices.append(log_first)
        log_last_prices.append(log_last)

    result_df = pd.DataFrame({
        'datetime': pd.to_datetime(intervals, unit='s'),
        'log_return': log_returns,
        'log_first_price': log_first_prices,
        'log_last_price': log_last_prices
    })

    return result_df, prev_log_last


def aggregate_log_returns_range(start_date, end_date, x_seconds, output_dir='data', kernel_half_width=0, trim_quantile=0.0):
    """
    Aggregate log returns for each CSV in a date range and export the combined results.

    Args:
        start_date: datetime or string in YYYY-MM-DD format
        end_date: datetime or string in YYYY-MM-DD format
        x_seconds: Interval length in seconds
        output_dir: Directory to write the combined CSV file
        kernel_half_width: Kernel width parameter
        trim_quantile: Fraction of data to trim from extremes (e.g., 0.01 for 1%)

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

    output_file = _aggregated_returns_path(
        start_dt, end_dt, x_seconds, kernel_half_width, trim_quantile, data_dir=output_dir,
    )
    if os.path.exists(output_file):
        return pd.read_csv(output_file, parse_dates=['datetime'])

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
            kernel_half_width=kernel_half_width,
        )
        combined_frames.append(daily_df)
        current += timedelta(days=1)

    if combined_frames:
        combined_df = pd.concat(combined_frames, ignore_index=True)
    else:
        combined_df = pd.DataFrame(columns=['datetime', 'log_return'])

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

    if trim_quantile > 0 and not combined_df.empty:
        lower_q = combined_df['log_first_price'].quantile(trim_quantile / 2)
        upper_q = combined_df['log_first_price'].quantile(1 - trim_quantile / 2)
        combined_df = combined_df[
            (combined_df['log_first_price'] >= lower_q) &
            (combined_df['log_first_price'] <= upper_q)
        ].copy()

    combined_df.to_csv(output_file, index=False)
    return combined_df


# ---------------------------------------------------------------------------
# Series loader (used by phase_GP)
# ---------------------------------------------------------------------------

def _aggregated_returns_path(
    start_date, end_date, seconds_interval,
    kernel_half_width, trim_quantile,
    data_dir='data',
):
    """Path of the cached aggregated-returns CSV."""
    def _to_dt(d):
        if isinstance(d, str):
            return datetime.strptime(d, '%Y-%m-%d')
        return d

    start_dt = _to_dt(start_date)
    end_dt = _to_dt(end_date)
    kernel_tag = f'_k{kernel_half_width}' if kernel_half_width > 0 else ''
    trim_tag = f'_trim{trim_quantile}' if trim_quantile > 0 else ''
    return os.path.join(
        data_dir,
        f'ETHUSDT-aggReturns-{start_dt.strftime("%Y-%m-%d")}_to_'
        f'{end_dt.strftime("%Y-%m-%d")}-{seconds_interval}sec'
        f'{kernel_tag}{trim_tag}.csv',
    )


def load_series(
    start_date,
    end_date,
    seconds_interval,
    kernel_half_width=5,
    trim_quantile=0.01,
    window_type=None,
):
    """
    Load aggregated log-prices over [start_date, end_date] at one sampling
    interval and emit (x_prev, dx, dt, dt_t) with cross-gap increments
    dropped.

    Two modes:
      - window_type is None : read a single whole-range aggregated CSV.
      - window_type set     : read per-window CSVs (matching Phase A's window
        type) and concatenate. Increments that span two windows are dropped.

    Returns
    -------
    x_prev : (N,) log-price at t-1
    dx     : (N,) increment x_t - x_{t-1}
    dt     : float, nominal step in seconds (seconds_interval)
    dt_t   : (N,) datetime of x_t aligned with dx
    """
    if window_type is None:
        cached_path = _aggregated_returns_path(
            start_date, end_date, seconds_interval,
            kernel_half_width, trim_quantile,
        )
        if os.path.exists(cached_path):
            df = pd.read_csv(cached_path, parse_dates=['datetime'])
            if df.empty:
                df = aggregate_log_returns_range(
                    start_date, end_date, seconds_interval,
                    kernel_half_width=kernel_half_width,
                    trim_quantile=trim_quantile,
                )
        else:
            df = aggregate_log_returns_range(
                start_date, end_date, seconds_interval,
                kernel_half_width=kernel_half_width,
                trim_quantile=trim_quantile,
            )
        if df.empty:
            raise ValueError('Empty aggregated returns dataframe.')
        df = df.sort_values('datetime').reset_index(drop=True)
        df['__window_idx'] = 0
    else:
        # Imported lazily to avoid a circular import (regime_estimation imports
        # data_collection at module load).
        from regime_estimation import iter_windows
        frames = []
        for w_idx, (window_start, window_end) in enumerate(
            iter_windows(start_date, end_date, window_type)
        ):
            cached_path = _aggregated_returns_path(
                window_start, window_end, seconds_interval,
                kernel_half_width, trim_quantile,
            )
            if os.path.exists(cached_path):
                df_w = pd.read_csv(cached_path, parse_dates=['datetime'])
                if df_w.empty:
                    df_w = aggregate_log_returns_range(
                        window_start, window_end, seconds_interval,
                        kernel_half_width=kernel_half_width,
                        trim_quantile=trim_quantile,
                    )
            else:
                df_w = aggregate_log_returns_range(
                    window_start, window_end, seconds_interval,
                    kernel_half_width=kernel_half_width,
                    trim_quantile=trim_quantile,
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


def ema_demean_drift(r, dt_t, halflife_days=14.0):
    """
    Causal EMA demeaning of the scaled increment r = dx/dt. Halflife is
    interpreted in wall-clock time (essential at 30-second sampling).

    Returns
    -------
    r_hat : (N,) demeaned increment (r - EMA(r))
    r_bar : (N,) the EMA trend that was removed
    """
    halflife_str = f'{halflife_days * 24 * 3600:.0f}s'
    r_series = pd.Series(r, index=pd.to_datetime(dt_t))
    r_bar_series = r_series.ewm(
        halflife=halflife_str,
        times=r_series.index,
        adjust=False,
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