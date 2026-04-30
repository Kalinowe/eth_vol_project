import os

import matplotlib.pyplot as plt
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

    safe_label = f"_{range_label}" if range_label else ""

    for x_seconds, result_df in results_by_interval.items():
        potential_df = integrate_drift_to_potential(result_df[['bin_center', 'drift']])

        plt.figure(figsize=(10, 6))
        plt.plot(potential_df['bin_center'], potential_df['potential'], marker='o', linestyle='-')
        label_text = f' {range_label}' if range_label else ''
        plt.title(f'Integrated potential from KM drift ({x_seconds}s interval){label_text}')
        plt.xlabel('Log price bin center')
        plt.ylabel('Potential')
        plt.grid(True)
        plt.tight_layout()

        output_path = os.path.join(output_dir, f'potential_{x_seconds}s{safe_label}.png')
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f'Saved {output_path}')


if __name__ == '__main__':
    print('This module expects precomputed KM results to be passed in from main.')
