import os
import requests
from datetime import datetime, timedelta
import argparse
import zipfile
import pandas as pd

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
    Unzip and combine aggregated trades data for a date range.
    
    Args:
        start_date: datetime object for start date
        end_date: datetime object for end date
    
    Returns:
        DataFrame with combined data (earlier dates on top)
    
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
    
    # Unzip files and read CSVs
    dfs = []
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
        
        # Read CSV
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
            print(f"Loaded {date_str}")
        except Exception as e:
            raise Exception(f"Error reading CSV {date_str}: {e}")
    
    # Combine dataframes (earlier dates on top)
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Add placeholder column names if CSV doesn't have expected columns
    # Placeholder columns: trade_id, price, quantity, trade_time, is_buyer_maker
    if len(combined_df.columns) == 0:
        combined_df.columns = ['trade_id', 'price', 'quantity', 'quote_asset_quantity', 'trade_time', 'is_buyer_maker']
    
    return combined_df



