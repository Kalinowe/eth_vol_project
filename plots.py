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
    

if __name__ == '__main__':
    print('This module expects precomputed KM results to be passed in from main.')
