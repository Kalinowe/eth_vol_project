import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

def integrate_drift_to_potential(drift_df):
    """
    Numerically integrate the Kramers-Moyal drift to obtain the potential function.
    
    Given drift = -dU/dx (force field), we compute the potential U(x) by integrating
    the drift over x, assuming U(x_min) = 0.
    
    Args:
        drift_df: DataFrame with columns 'bin_center' and 'drift'
    
    Returns:
        DataFrame with columns 'bin_center', 'drift', and 'potential'
    """
    x = drift_df['bin_center'].values
    f = drift_df['drift'].values
    
    # Use cumulative trapezoidal integration
    # cumulative_trapezoid returns values at the bin centers, with the first value implicitly 0
    potential = np.concatenate([[0.0], cumulative_trapezoid(f, x)])
    
    result_df = pd.DataFrame({
        'bin_center': x,
        'drift': f,
        'potential': potential
    })
    
    return result_df
