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


def aggregate_log_returns_range(start_date, end_date, x_seconds, output_dir='data', kernel_half_width=0, trim_quantile=0.0, detrend=0):
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
                
        # Remove linear trend from log-prices if requested
        if detrend and not combined_df.empty:
            t = np.arange(len(combined_df), dtype=np.float64)
            for col in ('log_first_price', 'log_last_price'):
                valid = combined_df[col].notna()
                coeffs = np.polyfit(t[valid], combined_df.loc[valid, col], 1)
                trend = np.polyval(coeffs, t)
                combined_df[col] = combined_df[col] - trend
            combined_df['log_return'] = combined_df['log_last_price'] - combined_df['log_first_price']

        # Trim extreme log prices if requested
        if trim_quantile > 0 and not combined_df.empty:
            lower_q = combined_df['log_first_price'].quantile(trim_quantile / 2)
            upper_q = combined_df['log_first_price'].quantile(1 - trim_quantile / 2)
            combined_df = combined_df[(combined_df['log_first_price'] >= lower_q) & 
                                    (combined_df['log_first_price'] <= upper_q)].copy()

        combined_df.to_csv(output_file, index=False)
        return combined_df
