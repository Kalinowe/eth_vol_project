"""Path helpers shared by RunGP.py, RunGP_update.py, and find_well_jumps.py.

Functions
---------
gp_output_dir  – build the output directory path for a given run configuration.
gp_state_stem  – build the base filename stem for a GP state file.
"""

import os

import pandas as pd


def gp_output_dir(seconds_interval: int, root: str = "gp_results") -> str:
    """Return the output directory for a GP run.

    Parameters
    ----------
    seconds_interval : int
        Aggregation interval in seconds (e.g. 900).
    root : str
        Repository-relative root for GP results.

    Examples
    --------
    >>> gp_output_dir(900)
    'gp_results/900s'
    """
    return os.path.join(root, f"{seconds_interval}s")


def gp_state_stem(start, end, seconds_interval: int, stage_tag: str) -> str:
    """Return the filename stem for a GP state pickle.

    Parameters
    ----------
    start, end :
        Anything parseable by ``pd.Timestamp``.
    seconds_interval : int
        Aggregation interval in seconds.
    stage_tag : str
        Descriptive tag appended to the stem (e.g. ``"kmvar_reproject"``).

    Examples
    --------
    >>> gp_state_stem("2024-01-01", "2025-12-31", 900, "kmvar_reproject")
    'gp_2024-01-01_to_2025-12-31_900s_kmvar_reproject'
    """
    return (
        f"gp_{pd.Timestamp(start).strftime('%Y-%m-%d')}_to_"
        f"{pd.Timestamp(end).strftime('%Y-%m-%d')}"
        f"_{seconds_interval}s_{stage_tag}"
    )
