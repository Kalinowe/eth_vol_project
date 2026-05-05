import os
import numpy as np
import data_collection as dt
import pandas as pd
import plots
from datetime import datetime
from kramersmoyal import km as kmc_lib
from rich.console import Console
from rich.table import Table

console = Console()

start_date = datetime(2025, 4, 1)
end_date = datetime(2025, 4, 30)
dt.download_data(start_date, end_date)
dt.unzip_data(start_date, end_date)

# Test with 3 different time intervals
time_intervals = [30, 60, 240]
results_by_interval = {}
# Kernel half-width: e.g., 5 means a 11-second window (5+1+5)
kernel_half_width = 5 
# Trim the top and bottom 0.5% of prices to avoid edge noise (total 1%)
trim_quantile = 0.01
# Number of bins for KMC estimation. Adjust based on data size and desired resolution.
n_bins = 200  

for seconds_interval in time_intervals:
    df = dt.aggregate_log_returns_range(
        start_date, 
        end_date, 
        seconds_interval, 
        kernel_half_width=kernel_half_width,
        trim_quantile=trim_quantile
    )
    print(f"\n{'='*60}")
    print(f"Processing interval: {seconds_interval} seconds")
    print(f"{'='*60}")
    print(f"Aggregated data shape: {df.shape}")
    print(df.head())
    
    # Library implementation
    x_series = np.asarray(df['log_first_price'].dropna().values, dtype=np.float64).reshape(-1, 1)
    lib_kmc, lib_edges = kmc_lib(x_series, bins= n_bins, full=False)
    
    # Extract coefficients
    lib_weights = lib_kmc[0, :]
    # Threshold > 5 or 10 is recommended to filter out statistically insignificant noise
    # D1 = M1 / dt. Divergence is often fixed by higher weight thresholds.
    lib_drift = np.where(lib_weights > 5, lib_kmc[1, :] / seconds_interval, np.nan)
    # D2 = M2 / (2 * dt). 
    lib_diffusion = np.where(lib_weights > 5, lib_kmc[2, :] / (2 * seconds_interval), np.nan)
    lib_centers = lib_edges[0]
    
    # Store results in DataFrame
    result_df = pd.DataFrame({
        'bin_center': lib_centers,
        'drift': lib_drift,
        'diffusion': lib_diffusion,
        'weight': lib_weights
    })
    
    results_by_interval[seconds_interval] = result_df
    
    print(f"\nDrift mean: {np.nanmean(lib_drift):.8g}")
    print(f"Diffusion mean: {np.nanmean(lib_diffusion):.8g}")


# Generate potential plots from the precomputed KM results
print(f"\n{'='*60}")
print('Generating potential plots from precomputed KM results')
print(f"{'='*60}\n")

range_label = f"{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}"
output_dir = os.path.join('plots', range_label)
plots.plot_potentials_from_km_results(results_by_interval, output_dir=output_dir, range_label=range_label)

# Diffusion comparison table
print("\n")
table2 = Table(title="Kramers-Moyal Diffusion: 3 Time Intervals")
table2.add_column("Bin", style="cyan")

for interval in time_intervals:
    table2.add_column(f"{interval}s Diffusion", justify="right", style="magenta")

for i in range(10):
    row = [str(i)]
    for interval in time_intervals:
        diff_val = results_by_interval[interval]['diffusion'].iloc[i]
        row.append(f"{diff_val:.8g}")
    table2.add_row(*row)

console.print(table2)