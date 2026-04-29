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



