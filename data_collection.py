import os
import requests
from datetime import datetime, timedelta, timezone
import zipfile
import pandas as pd
import numpy as np

def download_data(start_date, end_date):
    os.makedirs('data', exist_ok=True)
    current = start_date
    while current <= end_date:
        date_str = current.strftime('%Y-%m-%d')
        file_path = f"data/ETHUSDT-aggTrades-{date_str}.zip"
        
        # Skip if file already exists
        if os.path.exists(file_path):
            print(f"File already exists for {date_str}, skipping")
            current += timedelta(days=1)
            continue
        
        url = f"https://data.binance.vision/data/spot/daily/aggTrades/ETHUSDT/ETHUSDT-aggTrades-{date_str}.zip"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {date_str}")
            else:
                print(f"Failed to download {date_str}: {response.status_code}")
        except Exception as e:
            print(f"Error downloading {date_str}: {e}")
        current += timedelta(days=1)

def unzip_data(start_date, end_date):
    """
    Unzip aggregated trades data for a date range.
    
    Args:
        start_date: datetime object for start date
        end_date: datetime object for end date
    
    Raises:
        FileNotFoundError: If any required zip file is missing for dates in range
    """
    # First, check that all required files exist
    current = start_date
    date_list = []
    while current <= end_date:
        date_str = current.strftime('%Y-%m-%d')
        zip_file = f"data/ETHUSDT-aggTrades-{date_str}.zip"
        if not os.path.exists(zip_file):
            raise FileNotFoundError(f"Missing required file: {zip_file}")
        date_list.append((current, date_str, zip_file))
        current += timedelta(days=1)
    
    # Unzip files
    for date_obj, date_str, zip_file in date_list:
        csv_file = f"data/ETHUSDT-aggTrades-{date_str}.csv"
        
        # Unzip if CSV doesn't already exist
        if not os.path.exists(csv_file):
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall('data')
                print(f"Unzipped {date_str}")
            except Exception as e:
                raise Exception(f"Error unzipping {date_str}: {e}")

def aggregate_log_returns(csv_path, x_seconds, prev_log_last=None):
    """
    Read a CSV file and aggregate log returns over X second intervals starting from midnight.

    Args:
        csv_path: Path to the CSV file
        x_seconds: Interval length in seconds
        prev_log_last: Previous interval log price carried from an earlier day

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
    df['timestamp_s'] = df['timestamp_us'] // 1_000_000
    df = df.sort_values('timestamp_s')

    per_second = df.groupby('timestamp_s', sort=True)['price'].mean()
    timestamps_sec = per_second.index.to_numpy(dtype=np.int64)
    prices = per_second.to_numpy(dtype=np.float64)

    day_seconds = 24 * 3600
    num_intervals = int(day_seconds // x_seconds)

    intervals = []
    log_returns = []
    log_first_prices = []
    log_last_prices = []

    for i in range(num_intervals):
        start_sec = midnight + i * x_seconds
        end_sec = start_sec + x_seconds

        left = np.searchsorted(timestamps_sec, start_sec, side='left')
        right = np.searchsorted(timestamps_sec, end_sec, side='left')

        if left < right:
            first_price = prices[left]
            last_price = prices[right - 1]
            if first_price > 0 and last_price > 0:
                log_first = np.log(first_price)
                log_last = np.log(last_price)
                log_ret = log_last - log_first
                prev_log_last = log_last
            else:
                log_first = np.nan
                log_last = np.nan
                log_ret = np.nan
        else:
            if prev_log_last is not None:
                log_first = prev_log_last
                log_last = prev_log_last
                log_ret = 0.0
            else:
                log_first = np.nan
                log_last = np.nan
                log_ret = np.nan

        intervals.append(start_sec)
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


def aggregate_log_returns_range(start_date, end_date, x_seconds, output_dir='data'):
    """
    Aggregate log returns for each CSV in a date range and export the combined results.

    Args:
        start_date: datetime or string in YYYY-MM-DD format
        end_date: datetime or string in YYYY-MM-DD format
        x_seconds: Interval length in seconds
        output_dir: Directory to write the combined CSV file

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
    current = start_dt
    combined_frames = []

    prev_log_last = None
    while current <= end_dt:
        date_str = current.strftime('%Y-%m-%d')
        csv_path = os.path.join(output_dir, f'ETHUSDT-aggTrades-{date_str}.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'Missing required CSV file: {csv_path}')

        daily_df, prev_log_last = aggregate_log_returns(csv_path, x_seconds, prev_log_last=prev_log_last)
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

    output_file = os.path.join(
        output_dir,
        f'ETHUSDT-aggReturns-{start_dt.strftime("%Y-%m-%d")}_to_{end_dt.strftime("%Y-%m-%d")}-{x_seconds}sec.csv'
    )
    combined_df.to_csv(output_file, index=False)

    return combined_df



