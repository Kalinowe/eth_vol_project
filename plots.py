import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from force_field_estimation import integrate_drift_to_potential


def plot_potentials_from_km_results(results_by_interval, output_dir='plots', range_label=None):
    """
    Plot integrated potential functions for precomputed KM results.

    Args:
        results_by_interval: dict mapping interval length to KM result DataFrame with
            columns ['bin_center', 'drift', 'diffusion', 'weight']
        output_dir: Directory where PNG plots are saved
        range_label: Optional label used for plot titles and filenames
    """
    os.makedirs(output_dir, exist_ok=True)

    # Annualization factor
    sec_per_year = 365.25 * 24 * 3600

    # 1. Potential Plot
    plt.figure(figsize=(12, 7))
    for x_seconds, result_df in results_by_interval.items():
        potential_df = integrate_drift_to_potential(result_df[['bin_center', 'drift']])
        plt.plot(potential_df['bin_center'], potential_df['potential'], label=f'{x_seconds}s interval')

    plt.title(f'Integrated Potentials Comparison - {range_label}')
    plt.xlabel('Log price bin center')
    plt.ylabel('Potential U(x)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, f'combined_potentials_{range_label}.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved {output_path}')

    # 2. Annualized Drift Plot
    plt.figure(figsize=(12, 7))
    for x_seconds, result_df in results_by_interval.items():
        ann_drift = result_df['drift'] * sec_per_year
        plt.plot(result_df['bin_center'], ann_drift, label=f'{x_seconds}s interval')

    plt.title(f'Annualized Drift ($D_1$) Comparison - {range_label}')
    plt.xlabel('Log price bin center')
    plt.ylabel('Annualized Expected Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path_drift = os.path.join(output_dir, f'drift_comparison_{range_label}.png')
    plt.savefig(output_path_drift, dpi=150)
    plt.close()
    print(f'Saved {output_path_drift}')

    # 3. Annualized Volatility Plot
    plt.figure(figsize=(12, 7))
    for x_seconds, result_df in results_by_interval.items():
        # sigma = sqrt(2 * D2)
        ann_vol = np.sqrt(2 * result_df['diffusion'] * sec_per_year)
        plt.plot(result_df['bin_center'], ann_vol, label=f'{x_seconds}s interval')

    plt.title(f'Annualized Volatility ($\sigma$) Comparison - {range_label}')
    plt.xlabel('Log price bin center')
    plt.ylabel('Annualized Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path_vol = os.path.join(output_dir, f'volatility_comparison_{range_label}.png')
    plt.savefig(output_path_vol, dpi=150)
    plt.close()
    print(f'Saved {output_path_vol}')
    

def plot_price_series(df, output_dir='plots', range_label=None, smooth_window=20):
    """
    Plot ETH price (not log-price) over time with a rolling-mean smoothing overlay.

    Args:
        df: DataFrame with columns ['datetime', 'log_first_price']
        output_dir: Directory where the PNG is saved
        range_label: Label used in the title and filename
        smooth_window: Rolling mean window in number of bars
    """
    os.makedirs(output_dir, exist_ok=True)

    prices = np.exp(df['log_first_price'])
    times = pd.to_datetime(df['datetime'])

    smoothed = prices.rolling(window=smooth_window, center=True, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(times, prices, color='steelblue', alpha=0.35, linewidth=0.6, label='Price')
    ax.plot(times, smoothed, color='firebrick', linewidth=1.4,
            label=f'Rolling mean (window={smooth_window})')

    ax.set_title(f'ETH/USDT Price - {range_label}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USDT)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    output_path = os.path.join(output_dir, f'price_series_{range_label}.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output_path}')


if __name__ == '__main__':
    print('This module expects precomputed KM results to be passed in from main.')
