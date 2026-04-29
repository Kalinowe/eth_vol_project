import data_collection as dt
import pandas as pd
from datetime import datetime

start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 12, 31)

dt.download_data(start_date, end_date)

df = dt.unzip_data(start_date, end_date)
print(df.head())

pd.read_csv()