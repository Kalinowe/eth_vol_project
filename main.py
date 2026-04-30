import data_collection as dt
import pandas as pd
from datetime import datetime

start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 1, 31)  # Just one day for example
seconds_interval = 5

dt.download_data(start_date, end_date)

dt.unzip_data(start_date, end_date)

# Example: aggregate log returns for 5 second intervals
#csv_path = "data/ETHUSDT-aggTrades-2025-01-01.csv"
#result = dt.aggregate_log_returns(csv_path, seconds_interval)

df = dt.aggregate_log_returns_range(start_date, end_date, seconds_interval)
print(df.head())