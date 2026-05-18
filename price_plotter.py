from datetime import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd

import data_collection as dc
from regime_estimation import iter_windows

# Make sure raw dumps + per-window aggregated CSVs for 2025 exist
start = datetime(2025, 1, 1)
end   = datetime(2025, 12, 31)
dc.ensure_data(start, end)

frames = []
for ws, we in iter_windows(start, end, 'weekly'):
    df_w = dc.aggregate_log_returns_range(
        ws, we, x_seconds=600,
        kernel_half_width=50, trim_quantile=0.01, detrend=0,
    )
    if df_w is not None and not df_w.empty:
        frames.append(df_w[['datetime', 'log_first_price']])

df = pd.concat(frames, ignore_index=True).sort_values('datetime')

os.makedirs('plots', exist_ok=True)
out = 'plots/log_price_2025.png'

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(pd.to_datetime(df['datetime']), df['log_first_price'],
        color='#1565C0', lw=0.7)
ax.set_xlabel('Date'); ax.set_ylabel('log-price')
ax.set_title('ETH/USDT log-price — 2025')
ax.grid(True, alpha=0.3)
fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(out, dpi=150); plt.close(fig)
print('Saved', out)
