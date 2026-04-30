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
    # Filter out bins where drift is NaN (low weight bins).
    # By removing the rows, we allow the integrator and plotter to 
    # interpolate across the gap rather than leaving a "hole".
    valid_df = drift_df.dropna(subset=['drift']).copy()
    
    x = valid_df['bin_center'].values
    f = valid_df['drift'].values
    
    # Potential U(x) is defined such that drift = -dU/dx
    # Therefore U(x) = - integral(drift dx)
    potential = -np.concatenate([[0.0], cumulative_trapezoid(f, x)])
    
    valid_df.loc[:, 'potential'] = potential
    
    return valid_df
