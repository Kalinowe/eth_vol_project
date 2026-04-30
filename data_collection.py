import os
import requests
from datetime import datetime, timedelta, timezone
import argparse
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

def aggregate_log_returns(csv_path, x_seconds):
    """
    Read a CSV file and aggregate log returns over X second intervals starting from midnight.
    
    Args:
        csv_path: Path to the CSV file
        x_seconds: Interval length in seconds
    
    Returns:
        DataFrame with 'timestamp' (start of interval in seconds) and 'log_return'
    """
    # Extract date from filename
    filename = os.path.basename(csv_path)
    date_parts = filename.split('-')
    date_str = f"{date_parts[-3]}-{date_parts[-2]}-{date_parts[-1].split('.')[0]}"
    dt = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    midnight_microseconds = int(dt.timestamp() * 1e6)
    midnight = midnight_microseconds / 1e6  # in seconds
    
    df = pd.read_csv(csv_path, header=None)
    prices = df.iloc[:, 1].values  # second column, price
    timestamps = df.iloc[:, 5].values  # sixth column, timestamp in microseconds
    
    # Convert timestamps to seconds
    timestamps_sec = timestamps / 1e6
    
    # Number of intervals in a day
    day_seconds = 24 * 3600
    num_intervals = int(day_seconds // x_seconds)
    
    intervals = []
    log_returns = []
    
    for i in range(num_intervals):
        start_sec = midnight + i * x_seconds
        end_sec = start_sec + x_seconds
        
        mask = (timestamps_sec >= start_sec) & (timestamps_sec < end_sec)
        
        if np.any(mask):
            first_price = prices[mask][0]
            last_price = prices[mask][-1]
            if first_price > 0:
                log_ret = np.log(last_price / first_price)
            else:
                log_ret = 0.0
        else:
            log_ret = 0.0
        
        intervals.append(start_sec)
        log_returns.append(log_ret)
    
    result_df = pd.DataFrame({
        'datetime': pd.to_datetime(intervals, unit='s'),
        'log_return': log_returns
    })
    
    return result_df


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

    while current <= end_dt:
        date_str = current.strftime('%Y-%m-%d')
        csv_path = os.path.join(output_dir, f'ETHUSDT-aggTrades-{date_str}.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'Missing required CSV file: {csv_path}')

        daily_df = aggregate_log_returns(csv_path, x_seconds)
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



