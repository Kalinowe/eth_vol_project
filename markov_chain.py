"""
Empirical discrete-time Markov chain fitted on the regime label sequence
produced by regime_estimation.run_phase_a().

Label mapping
-------------
Only two states are modelled:
    'single-well'  ->  state 0
    'multi-well'  ->  state 1

Input
-----
A single-interval labels CSV with one row per window (see
``regime_estimation.run_phase_a``). The interval lives in the filename;
this module does not select among intervals.

Each row provides ``regime`` (hard label from Phase A) and ``p_multiwell``
(GP posterior over multi-well topology). The Markov-chain transitions can be
counted either as hard one-hot vectors (``use_soft_counts=False``) or as
soft counts ``outer([1-p_mw, p_mw], [1-p_mw, p_mw])`` (``use_soft_counts=True``,
default).

Ergodicity check
----------------
After computing pi we verify

    np.linalg.norm(pi @ A - pi, 1) < STATIONARITY_TOL

and that every off-diagonal of N exceeds 0.5 (soft counts are non-integer).
A MarkovChainError is raised before any further computation if either check
fails, so no compute is wasted on a degenerate result.

Outputs (written to output_dir)
--------------------------------
    mc_<labels_stem>_transition.png
    mc_<labels_stem>_stationary.png
    mc_<labels_stem>_timeline.png
    mc_<labels_stem>_results.csv
"""

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

STATES = ['single-well', 'multi-well']
STATE_IDX = {s: i for i, s in enumerate(STATES)}
COLORS = ['#2196F3', '#F44336']    # blue = single-well, red = multi-well
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
    """
    Load the single-interval regime labels CSV, sorted by window. Adds a
    ``confidence`` diagnostic column from ``p_multiwell`` (distance from 0.5
    in the direction of the assigned regime) for the timeline plot.
    """
    df = pd.read_csv(labels_csv, parse_dates=['window_start', 'window_end'])
    print(f'Loaded {len(df)} rows from {labels_csv}')
    df = df.sort_values('window_start').reset_index(drop=True)
    if 'p_multiwell' in df.columns and 'regime' in df.columns:
        df['confidence'] = np.where(
            df['regime'] == 'multi-well',
            df['p_multiwell'],
            1.0 - df['p_multiwell'].fillna(0.5),
        )
    return df


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def _adjacent(end_of_prev: pd.Timestamp, start_of_next: pd.Timestamp) -> bool:
    """True when the two windows share a boundary (gap ≤ 1 day)."""
    return (start_of_next - end_of_prev).days <= 1


def build_transition_counts(seq_df: pd.DataFrame) -> np.ndarray:
    """
    N[i,j] = # times state i was immediately followed by state j (hard labels).
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


def build_transition_counts_soft(seq_df: pd.DataFrame) -> np.ndarray:
    """
    Soft transition counts using p_multiwell as the per-window posterior over
    states. For each consecutive adjacent pair (t, t+1):

        p_t = [1 - p_mw_t, p_mw_t]
        N  += outer(p_t, p_{t+1})

    A row whose p_multiwell is missing falls back to its hard label
    (one-hot vector), so windows that predate p_multiwell still contribute.
    Pairs separated by a dropped window are skipped.
    """
    K = len(STATES)
    N = np.zeros((K, K), dtype=float)

    def _row_prob(row):
        p_mw = row.get('p_multiwell', np.nan)
        if pd.notna(p_mw):
            p = float(np.clip(p_mw, 0.0, 1.0))
            return np.array([1.0 - p, p])
        v = np.zeros(K)
        v[STATE_IDX[row['regime']]] = 1.0
        return v

    for t in range(len(seq_df) - 1):
        cur  = seq_df.iloc[t]
        nxt  = seq_df.iloc[t + 1]
        if not _adjacent(cur['window_end'], nxt['window_start']):
            continue
        p_cur = _row_prob(cur)
        p_nxt = _row_prob(nxt)
        N += np.outer(p_cur, p_nxt)
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


def map_transition_matrix(N: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Posterior mean of each row of the transition matrix under a
    Dirichlet(alpha[s,:]) prior.

        A_post[s, s'] = (N[s, s'] + alpha[s, s'])
                        / sum_{s''} (N[s, s''] + alpha[s, s''])

    With alpha = np.ones((K,K)) this is a uniform (add-one) prior.
    With alpha = 1 + lambda_ * N_hist it is a Phase-B-informed prior.
    Always produces a strictly positive, row-stochastic matrix.
    """
    numerator = N.astype(float) + np.asarray(alpha, dtype=float)
    denominator = numerator.sum(axis=1, keepdims=True)
    return numerator / denominator


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

    # -- irreducibility: every state must be reachable from every other.
    # Soft counts are non-integer; require at least 0.5 effective transitions
    # so a single fractional whisper does not pass the check.
    off_diag_zero = [
        (STATES[i], STATES[j])
        for i in range(K) for j in range(K)
        if i != j and N[i, j] < 0.5
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
    A_mle: np.ndarray,
    pi_mle: np.ndarray,
    dwells_mle: np.ndarray,
    residual: float,
    n_transitions: float,
    window_type: str = 'weekly',
    console: Console | None = None,
    alpha: np.ndarray | None = None,
    prior_lambda: float | None = None,
) -> None:
    """
    Print the *empirical* (MLE) transition matrix, stationary distribution and
    dwell times. The Dirichlet-smoothed posterior is kept inside
    run_markov_chain's return dict for downstream Phase C use but is not
    displayed here so the console reflects pure data.
    """
    console = console or Console()

    n_str = f'{n_transitions:.2f}' if isinstance(n_transitions, float) else str(n_transitions)
    console.print(
        f"\n[bold cyan]Empirical Markov Chain (MLE)  |  "
        f"{n_str} transitions  |  "
        f"||πA−π||₁ = {residual:.2e}  |  "
        f"Window type: {window_type}[/bold cyan]\n"
    )

    count_table = Table(title='Transition Count Matrix  N[i→j]')
    count_table.add_column('From \\ To', style='cyan')
    for s in STATES:
        count_table.add_column(s, justify='right')
    for i, s_from in enumerate(STATES):
        cells = [
            f'{N[i, j]:.2f}' if N.dtype.kind == 'f' else str(int(N[i, j]))
            for j in range(len(STATES))
        ]
        count_table.add_row(s_from, *cells)
    console.print(count_table)

    prob_table = Table(title='Empirical Transition Matrix  A_MLE[i→j]')
    prob_table.add_column('From \\ To', style='cyan')
    for s in STATES:
        prob_table.add_column(s, justify='right', style='green')
    for i, s_from in enumerate(STATES):
        prob_table.add_row(s_from, *[f'{A_mle[i, j]:.4f}' for j in range(len(STATES))])
    console.print(prob_table)

    summary = Table(title='Empirical Stationary Distribution & Mean Dwell Times')
    summary.add_column('State', style='cyan')
    summary.add_column('π_MLE (stationary)', justify='right', style='green')
    summary.add_column('Mean dwell (windows)', justify='right', style='yellow')
    for i, s in enumerate(STATES):
        dwell_str = f'{dwells_mle[i]:.2f}' if np.isfinite(dwells_mle[i]) else '∞'
        summary.add_row(s, f'{pi_mle[i]:.4f}', dwell_str)
    console.print(summary)

    if alpha is not None and not np.all(np.asarray(alpha) == 1.0):
        lam_str = f'λ={prior_lambda:.2f}' if prior_lambda is not None else ''
        console.print(
            f'[dim]Note: a Dirichlet-smoothed posterior ({lam_str}) is also '
            'computed and returned for Phase C; it is hidden here so the '
            'console reflects the raw empirical estimate.[/dim]'
        )


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def save_results_csv(
    A: np.ndarray,
    pi: np.ndarray,
    dwells: np.ndarray,
    output_path: str,
    alpha: np.ndarray | None = None,
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
        if alpha is not None:
            prior_rows = [
                {'from_state': STATES[i], 'to_state': STATES[j],
                 'alpha': float(alpha[i, j])}
                for i in range(len(STATES))
                for j in range(len(STATES))
            ]
            f.write('\n# prior_counts\n')
            pd.DataFrame(prior_rows).to_csv(f, index=False)
    print(f'Saved {output_path}')


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_markov_chain(
    labels_csv: str,
    window_type: str = 'weekly',
    output_dir: str = 'regime_results',
    console: Console | None = None,
    prior_counts: np.ndarray | None = None,
    prior_lambda: float = 0.3,
    use_soft_counts: bool = True,
) -> dict | None:
    """
    Fit and report an empirical Markov chain from the single-interval regime
    label sequence.

    Parameters
    ----------
    labels_csv     : path to CSV produced by regime_estimation.run_phase_a()
                     for one ``seconds_interval``. The interval is encoded
                     in the filename.
    output_dir     : directory for plots and CSV results.
    prior_counts   : optional (K,K) historical count matrix used to build a
                     Phase-B-informed Dirichlet prior alpha = 1 + lambda*N_hist.
                     When None, a uniform add-one prior is used.
    prior_lambda   : strength of the historical prior. Ignored if
                     prior_counts is None.
    use_soft_counts: if True, transitions are counted by outer products of
                     ``[1-p_mw, p_mw]`` per window.

    Returns
    -------
    dict with keys
        N, alpha, seq_df
        A, pi, dwells              -- Dirichlet posterior mean (for Phase C)
        A_mle, pi_mle, dwells_mle  -- pure empirical MLE (matches console output)
    or None if data is insufficient.
    Raises MarkovChainError if irreducibility (on N) or stationarity checks fail.

    Console output and persisted plots/CSV reflect the MLE view so that the
    standalone report is purely empirical; the Dirichlet posterior is kept in
    the return dict only as a prior for downstream Phase C consumers.
    """
    os.makedirs(output_dir, exist_ok=True)
    console = console or Console()

    seq_df = load_labels_csv(labels_csv)

    # Drop windows whose Phase A label did not resolve to a modelled state.
    mask_valid = seq_df['regime'].isin(STATES)
    n_dropped = int((~mask_valid).sum())
    if n_dropped > 0:
        dropped = seq_df.loc[~mask_valid, ['window_start', 'window_end', 'regime']]
        warnings.warn(
            f"{n_dropped} window(s) dropped (regime not in STATES): "
            f"{dropped['regime'].value_counts().to_dict()}",
            stacklevel=2,
        )
    seq_df = seq_df[mask_valid].reset_index(drop=True)

    if len(seq_df) < 2:
        console.print(
            f'[red]Only {len(seq_df)} valid window(s) — '
            'need at least 2 to estimate transitions.[/red]'
        )
        return None

    if use_soft_counts:
        N = build_transition_counts_soft(seq_df)
        console.print(
            '[cyan]Soft transition counts weighted by p_multiwell '
            f'(total mass = {N.sum():.2f}).[/cyan]'
        )
    else:
        N = build_transition_counts(seq_df).astype(float)

    # Irreducibility is a property of the empirical sequence; check before
    # smoothing so that a degenerate run halts even with a Dirichlet prior.
    K = len(STATES)
    if prior_counts is not None:
        alpha = np.ones((K, K)) + prior_lambda * np.asarray(prior_counts, dtype=float)
    else:
        alpha = np.ones((K, K))   # uniform add-one prior — eliminates zero entries

    # Empirical (MLE) view — shown in the console.
    A_mle = mle_transition_matrix(N)
    pi_mle = stationary_distribution(A_mle)
    dwells_mle = mean_dwell_times(A_mle)
    residual_mle = float(np.linalg.norm(pi_mle @ A_mle - pi_mle, 1))

    # Dirichlet-smoothed posterior — returned for downstream Phase C consumers
    # so they can still benefit from the regularisation.
    A = map_transition_matrix(N, alpha)
    pi = stationary_distribution(A)
    dwells = mean_dwell_times(A)

    # Guard: raises MarkovChainError on failure — do this before any output.
    # check_chain inspects N (not A) so the irreducibility test still bites
    # even though A itself is regularised away from zero.
    check_chain(N, A, pi)

    n_transitions = float(N.sum()) if N.dtype.kind == 'f' else int(N.sum())
    print_results(
        N, A_mle, pi_mle, dwells_mle, residual_mle, n_transitions,
        window_type=window_type,
        console=console,
        alpha=alpha, prior_lambda=prior_lambda if prior_counts is not None else None,
    )

    stem = os.path.splitext(os.path.basename(labels_csv))[0]
    prefix = os.path.join(output_dir, f'mc_{stem}')

    # Plots and persisted CSV reflect the MLE view; the smoothed posterior is
    # only used as a prior downstream and is recoverable from alpha + N.
    plots.plot_mc_transition_heatmap(A_mle, STATES, f'{prefix}_transition.png')
    plots.plot_mc_stationary(pi_mle, dwells_mle, STATES, COLORS, f'{prefix}_stationary.png')
    plots.plot_mc_timeline(seq_df, STATES, COLORS, STATE_IDX, f'{prefix}_timeline.png')
    save_results_csv(A_mle, pi_mle, dwells_mle, f'{prefix}_results.csv', alpha=alpha)

    return {
        'N': N,
        'A': A, 'pi': pi, 'dwells': dwells,
        'A_mle': A_mle, 'pi_mle': pi_mle, 'dwells_mle': dwells_mle,
        'seq_df': seq_df, 'alpha': alpha,
    }


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Standalone execution is intentionally disabled: this module is part of
    # the pipeline and its parameters live in RunEM.py. Running it directly
    # would silently fall back to hardcoded config and quietly diverge from
    # whatever RunEM is currently configured to do.
    Console().print(
        '[yellow]markov_chain.py is part of the pipeline; '
        'run [bold]python RunEM.py[/bold] to execute Phase B with the '
        'centrally-configured parameters.[/yellow]'
    )
    raise SystemExit(0)
