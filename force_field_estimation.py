import numpy as np
import pandas as pd
from kramersmoyal import km as kmc_lib


def estimate_km(log_prices, seconds_interval, n_bins=200, weight_threshold=5):
    """
    Run Kramers-Moyal estimation on a 1D log-price series.

    Args:
        log_prices: 1D array-like of log-prices, NaNs already dropped.
        seconds_interval: dt between consecutive observations, in seconds.
        n_bins: number of bins used by the KM histogram.
        weight_threshold: minimum bin weight (count) to keep; below this drift
            and diffusion are set to NaN to avoid divergent estimates in the tails.

    Returns:
        DataFrame with columns ['bin_center', 'drift', 'diffusion', 'weight'].
        Drift is annualizable as drift * sec_per_year. Diffusion is sigma^2 / 2.
    """
    x_series = np.asarray(log_prices, dtype=np.float64).reshape(-1, 1)
    kmc, edges = kmc_lib(x_series, bins=n_bins, full=False)

    weights = kmc[0, :]
    drift = np.where(weights > weight_threshold, kmc[1, :] / seconds_interval, np.nan)
    diffusion = np.where(weights > weight_threshold, kmc[2, :] / (2 * seconds_interval), np.nan)
    centers = edges[0]

    return pd.DataFrame({
        'bin_center': centers,
        'drift': drift,
        'diffusion': diffusion,
        'weight': weights,
    })
