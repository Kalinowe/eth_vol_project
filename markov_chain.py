"""
Empirical discrete-time Markov chain fitted on the regime label sequence
produced by regime_estimation.run_phase_a().

Label mapping
-------------
Only two states are modelled:
    'single-well'  ->  state 0
    'double-well'  ->  state 1

Vote aggregation
----------------
The regime labels CSV contains one row per (window, seconds_interval).
Before chain estimation, all intervals for each window are collapsed into a
single regime via majority vote:

    - 'X/N' confidence when a STATES regime wins the plurality
      (e.g. '2/3' or '3/3' with three intervals)
    - 'invalid regimes' when the plurality winner is not in STATES or when
      there is a tie

Windows whose aggregated regime is not in STATES are dropped with a warning;
transitions that cross such gaps are NOT counted (gap-aware adjacency check).

The seconds_interval argument selects which interval's metadata columns
(n_wells, well_locations, barriers, u_range, n_observations) are attached
to each window row and included in the return dict.

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
# Data loading and aggregation
# ---------------------------------------------------------------------------

def load_labels_csv(labels_csv: str) -> pd.DataFrame:
    """Load the full regime labels CSV (all intervals), sorted by window then interval."""
    df = pd.read_csv(labels_csv, parse_dates=['window_start', 'window_end'])
    return df.sort_values(['window_start', 'seconds_interval']).reset_index(drop=True)


def aggregate_label_votes(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse multiple seconds_interval rows per window into a single row via
    majority vote.

    Assumes an odd number of intervals so ties cannot occur among valid states,
    but a tie guard is included regardless.

    Returns DataFrame with columns ['window_start', 'window_end', 'regime',
    'confidence'] where confidence is 'X/N' when a STATES regime wins (X votes
    out of N intervals) or 'invalid regimes' otherwise.
    """
    rows = []
    for (ws, we), grp in full_df.groupby(['window_start', 'window_end'], sort=True):
        counts = grp['regime'].value_counts()
        total = int(counts.sum())
        top_regime = counts.index[0]
        top_count = int(counts.iloc[0])

        tie = len(counts) > 1 and counts.iloc[0] == counts.iloc[1]
        if tie or top_regime not in STATE_IDX:
            regime = top_regime
            confidence = 'invalid regimes'
        else:
            regime = top_regime
            confidence = f'{top_count}/{total}'

        rows.append({'window_start': ws, 'window_end': we,
                     'regime': regime, 'confidence': confidence})

    return pd.DataFrame(rows)


def attach_interval_metadata(
    agg_df: pd.DataFrame,
    full_df: pd.DataFrame,
    seconds_interval: int,
) -> pd.DataFrame:
    """
    Left-join per-window metadata from the specified seconds_interval onto
    the aggregated DataFrame.

    Metadata columns joined: n_wells, well_locations, barriers, u_range,
    n_observations (whichever are present in full_df).
    """
    meta_cols = ['window_start', 'window_end', 'n_wells', 'well_locations',
                 'barriers', 'u_range', 'n_observations']
    available = [c for c in meta_cols if c in full_df.columns]
    sub = full_df[full_df['seconds_interval'] == seconds_interval][available].copy()
    if sub.empty:
        warnings.warn(
            f'No rows found for seconds_interval={seconds_interval}; '
            'metadata columns will be NaN.',
            stacklevel=3,
        )
    return agg_df.merge(sub, on=['window_start', 'window_end'], how='left')


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
    seconds_interval : interval whose metadata (n_wells, barriers, etc.) is
                       attached to each window row; does not filter the vote
    output_dir       : directory for plots and CSV results

    Returns
    -------
    dict with keys {N, A, pi, dwells, seq_df} or None if data is insufficient.
    seq_df has one row per window with regime, confidence, and metadata columns
    from seconds_interval.
    Raises MarkovChainError if ergodicity / stationarity checks fail.
    """
    os.makedirs(output_dir, exist_ok=True)
    console = console or Console()

    # --- Aggregate all intervals into one regime per window ---
    full_df = load_labels_csv(labels_csv)
    agg_df = aggregate_label_votes(full_df)
    seq_df = attach_interval_metadata(agg_df, full_df, seconds_interval)

    # Drop windows where the majority vote did not resolve to a modelled state
    mask_valid = seq_df['regime'].isin(STATES)
    n_dropped = int((~mask_valid).sum())
    if n_dropped > 0:
        dropped = seq_df.loc[~mask_valid, ['window_start', 'window_end', 'regime', 'confidence']]
        warnings.warn(
            f"{n_dropped} window(s) dropped (no valid majority): "
            f"{dropped[['regime', 'confidence']].value_counts().to_dict()}",
            stacklevel=2,
        )
    seq_df = seq_df[mask_valid].reset_index(drop=True)

    if len(seq_df) < 2:
        console.print(
            f'[red]Only {len(seq_df)} valid window(s) after vote aggregation — '
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
