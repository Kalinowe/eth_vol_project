"""
Empirical discrete-time Markov chain fitted on the regime label sequence
produced by regime_estimation.run_phase_a().

Label mapping
-------------
Only two states are modelled:
    'single-well'  ->  state 0
    'double-well'  ->  state 1

Any other label (no-equilibrium, n-well for n>2, unavailable) is dropped
with a warning.  Transitions that cross a dropped window are NOT counted
(gap-aware: two windows must be temporally adjacent to contribute a
transition).

Ergodicity check
----------------
After computing the stationary distribution pi, we verify:

    np.linalg.norm(pi @ A - pi, 1) < STATIONARITY_TOL

and that the chain is irreducible (every off-diagonal entry of N is > 0).
A MarkovChainError is raised before any further computation if either check
fails, so no compute is wasted on a degenerate result.

Outputs (written to output_dir)
--------------------------------
    mc_<stem>_<interval>s_transition.png
    mc_<stem>_<interval>s_stationary.png
    mc_<stem>_<interval>s_timeline.png
    mc_<stem>_<interval>s_results.csv
"""

import glob
import os
import warnings

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

import plots

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATES = ['single-well', 'double-well']
STATE_IDX = {s: i for i, s in enumerate(STATES)}
COLORS = ['#2196F3', '#F44336']    # blue = single-well, red = double-well
STATIONARITY_TOL = 1e-6


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class MarkovChainError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_label_sequence(labels_csv: str, seconds_interval: int) -> pd.DataFrame:
    """
    Load the regime labels CSV and return only the rows for `seconds_interval`
    whose regime is in STATES, sorted chronologically.

    Rows with any other regime are dropped and a warning is emitted.
    Returns DataFrame with columns ['window_start', 'window_end', 'regime'].
    """
    df = pd.read_csv(labels_csv, parse_dates=['window_start', 'window_end'])
    df = df[df['seconds_interval'] == seconds_interval].copy()
    df = df.sort_values('window_start').reset_index(drop=True)

    mask_valid = df['regime'].isin(STATES)
    n_dropped = int((~mask_valid).sum())
    if n_dropped > 0:
        dropped_counts = df.loc[~mask_valid, 'regime'].value_counts().to_dict()
        warnings.warn(
            f"{n_dropped} window(s) dropped (non-binary regime): {dropped_counts}",
            stacklevel=2,
        )

    return (
        df[mask_valid][['window_start', 'window_end', 'regime']]
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def _adjacent(end_of_prev: pd.Timestamp, start_of_next: pd.Timestamp) -> bool:
    """True when the two windows share a boundary (gap ≤ 1 day)."""
    return (start_of_next - end_of_prev).days <= 1


def build_transition_counts(seq_df: pd.DataFrame) -> np.ndarray:
    """
    N[i,j] = # times state i was immediately followed by state j.
    Pairs separated by a dropped window (non-adjacent) are skipped.
    """
    K = len(STATES)
    N = np.zeros((K, K), dtype=int)
    for t in range(len(seq_df) - 1):
        cur  = seq_df.iloc[t]
        nxt  = seq_df.iloc[t + 1]
        if not _adjacent(cur['window_end'], nxt['window_start']):
            continue
        N[STATE_IDX[cur['regime']], STATE_IDX[nxt['regime']]] += 1
    return N


def mle_transition_matrix(N: np.ndarray) -> np.ndarray:
    """
    Row-normalise N.  A row with zero total count (state never left) is left
    as uniform so A stays row-stochastic, but this case triggers the
    irreducibility check downstream.
    """
    row_sums = N.sum(axis=1, keepdims=True).astype(float)
    safe = np.where(row_sums == 0, 1.0, row_sums)
    A = N / safe
    A[row_sums.flatten() == 0] = 1.0 / len(STATES)
    return A


def stationary_distribution(A: np.ndarray) -> np.ndarray:
    """Left eigenvector of A for eigenvalue 1, normalised to sum to 1."""
    eigenvalues, eigenvectors = np.linalg.eig(A.T)
    idx = int(np.argmin(np.abs(eigenvalues - 1.0)))
    pi = np.real(eigenvectors[:, idx])
    pi = np.abs(pi)
    pi /= pi.sum()
    return pi


def mean_dwell_times(A: np.ndarray) -> np.ndarray:
    """E[dwell | state i] = 1 / (1 - A[i,i]) in units of windows."""
    diag = np.diag(A)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(diag < 1.0, 1.0 / (1.0 - diag), np.inf)


# ---------------------------------------------------------------------------
# Ergodicity / stationarity guard
# ---------------------------------------------------------------------------

def check_chain(N: np.ndarray, A: np.ndarray, pi: np.ndarray) -> None:
    """
    Raise MarkovChainError if:
      1. The chain is not irreducible (an off-diagonal count is zero), OR
      2. pi is not a genuine stationary distribution of A within tolerance.

    Call this before plotting or saving.
    """
    K = len(STATES)

    # -- irreducibility: every state must be reachable from every other --
    off_diag_zero = [
        (STATES[i], STATES[j])
        for i in range(K) for j in range(K)
        if i != j and N[i, j] == 0
    ]
    if off_diag_zero:
        pairs = ', '.join(f'{a}→{b}' for a, b in off_diag_zero)
        raise MarkovChainError(
            f'Chain is not irreducible: no observed transitions for [{pairs}]. '
            'Collect more windows or broaden the date range.'
        )

    # -- stationarity: pi @ A == pi --
    residual = float(np.linalg.norm(pi @ A - pi, 1))
    if residual > STATIONARITY_TOL:
        raise MarkovChainError(
            f'Stationary-distribution check failed: ||pi @ A - pi||_1 = {residual:.2e} '
            f'(tolerance {STATIONARITY_TOL:.0e}). '
            'This may indicate a numerical issue or a non-ergodic chain.'
        )


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_results(
    N: np.ndarray,
    A: np.ndarray,
    pi: np.ndarray,
    dwells: np.ndarray,
    residual: float,
    n_transitions: int,
    console: Console | None = None,
) -> None:
    console = console or Console()

    console.print(
        f"\n[bold cyan]Empirical Markov Chain  |  "
        f"{n_transitions} transitions  |  "
        f"||πA−π||₁ = {residual:.2e}[/bold cyan]\n"
    )

    count_table = Table(title='Transition Count Matrix  N[i→j]')
    count_table.add_column('From \\ To', style='cyan')
    for s in STATES:
        count_table.add_column(s, justify='right')
    for i, s_from in enumerate(STATES):
        count_table.add_row(s_from, *[str(N[i, j]) for j in range(len(STATES))])
    console.print(count_table)

    prob_table = Table(title='MLE Transition Matrix  A[i→j]')
    prob_table.add_column('From \\ To', style='cyan')
    for s in STATES:
        prob_table.add_column(s, justify='right', style='green')
    for i, s_from in enumerate(STATES):
        prob_table.add_row(s_from, *[f'{A[i, j]:.4f}' for j in range(len(STATES))])
    console.print(prob_table)

    summary = Table(title='Stationary Distribution & Mean Dwell Times')
    summary.add_column('State', style='cyan')
    summary.add_column('π (stationary)', justify='right', style='green')
    summary.add_column('Mean dwell (windows)', justify='right', style='yellow')
    for i, s in enumerate(STATES):
        dwell_str = f'{dwells[i]:.2f}' if np.isfinite(dwells[i]) else '∞'
        summary.add_row(s, f'{pi[i]:.4f}', dwell_str)
    console.print(summary)


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def save_results_csv(
    A: np.ndarray,
    pi: np.ndarray,
    dwells: np.ndarray,
    output_path: str,
) -> None:
    rows = [
        {'from_state': STATES[i], 'to_state': STATES[j], 'A': A[i, j]}
        for i in range(len(STATES))
        for j in range(len(STATES))
    ]
    df_summary = pd.DataFrame({
        'state': STATES,
        'pi': pi,
        'mean_dwell_windows': dwells,
    })
    with open(output_path, 'w') as f:
        f.write('# transition_matrix\n')
        pd.DataFrame(rows).to_csv(f, index=False)
        f.write('\n# stationary_distribution\n')
        df_summary.to_csv(f, index=False)
    print(f'Saved {output_path}')


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_markov_chain(
    labels_csv: str,
    seconds_interval: int = 30,
    output_dir: str = 'regime_results',
    console: Console | None = None,
) -> dict | None:
    """
    Fit and report an empirical Markov chain from the regime label sequence.

    Parameters
    ----------
    labels_csv       : path to CSV produced by regime_estimation.run_phase_a()
    seconds_interval : which sampling interval row to use (must exist in CSV)
    output_dir       : directory for plots and CSV results

    Returns
    -------
    dict with keys {N, A, pi, dwells, seq_df} or None if data is insufficient.
    Raises MarkovChainError if ergodicity / stationarity checks fail.
    """
    os.makedirs(output_dir, exist_ok=True)
    console = console or Console()

    seq_df = load_label_sequence(labels_csv, seconds_interval)

    if len(seq_df) < 2:
        console.print(
            f'[red]Only {len(seq_df)} valid window(s) at {seconds_interval}s — '
            'need at least 2 to estimate transitions.[/red]'
        )
        return None

    N = build_transition_counts(seq_df)
    A = mle_transition_matrix(N)
    pi = stationary_distribution(A)
    dwells = mean_dwell_times(A)
    residual = float(np.linalg.norm(pi @ A - pi, 1))

    # Guard: raises MarkovChainError on failure — do this before any output
    check_chain(N, A, pi)

    n_transitions = int(N.sum())
    print_results(N, A, pi, dwells, residual, n_transitions, console=console)

    stem = os.path.splitext(os.path.basename(labels_csv))[0]
    prefix = os.path.join(output_dir, f'mc_{stem}_{seconds_interval}s')

    plots.plot_mc_transition_heatmap(A, STATES, f'{prefix}_transition.png')
    plots.plot_mc_stationary(pi, dwells, STATES, COLORS, f'{prefix}_stationary.png')
    plots.plot_mc_timeline(seq_df, STATES, COLORS, STATE_IDX, f'{prefix}_timeline.png')
    save_results_csv(A, pi, dwells, f'{prefix}_results.csv')

    return {'N': N, 'A': A, 'pi': pi, 'dwells': dwells, 'seq_df': seq_df}


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    csvs = sorted(glob.glob(os.path.join('regime_results', 'regime_labels_*.csv')))
    if not csvs:
        raise FileNotFoundError(
            'No regime_labels_*.csv found in regime_results/. '
            'Run regime_estimation.py first.'
        )
    labels_csv = csvs[-1]
    seconds_interval = 30

    _console = Console()
    _console.print(f'[cyan]Labels file : {labels_csv}[/cyan]')
    _console.print(f'[cyan]Interval    : {seconds_interval}s[/cyan]')

    run_markov_chain(labels_csv, seconds_interval=seconds_interval)
