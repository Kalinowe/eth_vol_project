"""Path helpers shared by RunGP.py, RunGP_update.py, and find_well_jumps.py.

Functions
---------
gp_output_dir  – build the output directory path for a given run configuration.
"""

import os


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
