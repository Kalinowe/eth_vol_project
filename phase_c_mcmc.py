"""
Phase C (MCMC) — FFBS Gibbs sampler for the Markov-switching SDE.

Targets the full joint posterior p(theta, P, S_{1:T} | x_{1:T}) via a Gibbs
sampler with three blocks per sweep:

  Block 1  Draw S_{1:T} | theta, P, x   (Forward Filter Backward Sample)
  Block 2  Draw theta   | S_{1:T}, x    (conjugate OU + MH for cubic)
  Block 3  Draw P       | S_{1:T}       (Dirichlet conjugate, exact)

Warm-started from the EM solution produced by phase_c.run_phase_c. Heavy
upstream work (label generation, Phase B chain, EM warm start, MCMC chain)
is cached by checking for the corresponding output file before recomputing.

Outputs (under output_dir):
  phase_c_mcmc_<from>_to_<to>_<int>s_chain.npz        raw chain arrays (thinned)
  phase_c_mcmc_<from>_to_<to>_<int>s_summary.csv      posterior means + 95% CIs
  phase_c_mcmc_<from>_to_<to>_<int>s_kappa_post.png   posterior of kappa
  phase_c_mcmc_<from>_to_<to>_<int>s_gamma_post.png   posterior mean smoothed probs
  phase_c_mcmc_<from>_to_<to>_<int>s_loglik_trace.png log-lik trace
  phase_c_mcmc_<from>_to_<to>_<int>s_forecast.csv     posterior predictive forecasts
"""

import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd
from numpy.linalg import matrix_power
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from scipy.stats import norm

# Reuse from existing modules — do not duplicate
from phase_c import (
    EPS,
    STATES,
    emission_log_b,
    load_series,
    mu_multi,
)
import plots


# ---------------------------------------------------------------------------
# Block 1 — Forward Filter Backward Sample
# ---------------------------------------------------------------------------

def ffbs(log_b: np.ndarray, P: np.ndarray, pi0: np.ndarray,
         rng: np.random.Generator):
    """
    Forward Filter Backward Sample.

    Runs the Hamilton forward filter to get p_filt[t] then draws a single
    state sequence S_{1:T} by sampling backward.

    Returns
    -------
    S        : (T,) int array, values in {0, ..., K-1}
    p_filt   : (T, K) filtered state probabilities
    log_lik  : float, sum of log normalising constants
    """
    T, K = log_b.shape

    p_filt = np.empty((T, K))
    log_c = np.empty(T)

    log_pi = np.log(np.clip(pi0, EPS, None))
    z = log_pi + log_b[0]
    mz = float(np.max(z))
    log_c[0] = mz + np.log(np.exp(z - mz).sum())
    p_filt[0] = np.exp(z - log_c[0])

    for t in range(1, T):
        p_pred = p_filt[t - 1] @ P
        log_p = np.log(np.clip(p_pred, EPS, None))
        z = log_p + log_b[t]
        mz = float(np.max(z))
        log_c[t] = mz + np.log(np.exp(z - mz).sum())
        p_filt[t] = np.exp(z - log_c[t])

    log_lik = float(log_c.sum())

    S = np.empty(T, dtype=np.int8)
    last = p_filt[T - 1] / p_filt[T - 1].sum()
    S[T - 1] = rng.choice(K, p=last)

    for t in range(T - 2, -1, -1):
        weights = p_filt[t] * P[:, S[t + 1]]
        s = weights.sum()
        if s <= 0:
            weights = np.full(K, 1.0 / K)
        else:
            weights = weights / s
        S[t] = rng.choice(K, p=weights)

    return S, p_filt, log_lik


# ---------------------------------------------------------------------------
# Block 2 — Draw theta | S, x
# ---------------------------------------------------------------------------

def draw_ou_params(x_prev, dx, dt, S, sigma2, prior_ou, rng):
    """
    Conjugate Normal draw of (kappa, m) given S_{1:T}.

    With S known, dx_t/dt | x_{t-1}, S_t=0 ~ N(a + b*x_{t-1}, sigma2/dt)
    where a = kappa*m, b = -kappa. Linear Gaussian regression with a
    Normal prior → Normal posterior, drawn jointly.
    """
    mask = (S == 0)
    if mask.sum() < 4:
        return {'kappa': prior_ou['kappa'], 'm': prior_ou['m']}

    x0 = x_prev[mask]
    r0 = (dx / dt)[mask]
    n = int(mask.sum())
    w = dt / sigma2  # precision per observation

    X = np.column_stack([np.ones(n), x0])
    XtX = (X.T * w) @ X
    Xtr = (X.T * w) @ r0

    Lambda_prior = np.diag([
        1.0 / prior_ou.get('a_var', 1e4),
        1.0 / prior_ou.get('b_var', 1e4),
    ])
    mu_prior = np.array([
        prior_ou.get('a_mean', 0.0),
        prior_ou.get('b_mean', 0.0),
    ])

    Lambda_post = XtX + Lambda_prior
    mu_post = np.linalg.solve(Lambda_post, Xtr + Lambda_prior @ mu_prior)
    Sigma_post = np.linalg.inv(Lambda_post)

    ab = rng.multivariate_normal(mu_post, Sigma_post)
    a_draw, b_draw = float(ab[0]), float(ab[1])

    kappa = max(-b_draw, 1e-6)
    m = a_draw / kappa if kappa > 1e-9 else float(np.mean(x0))
    return {'kappa': kappa, 'm': m}


def draw_cubic_params(x_prev, dx, dt, S, sigma2, theta_cur,
                      proposal_scale, rng):
    """
    Random-walk Metropolis-Hastings step for (alpha, beta, c) given S_{1:T}.
    Symmetric Gaussian proposal centred on the current value; alpha is
    reflected to the positive half-line via the absolute value.
    """
    mask = (S == 1)
    if mask.sum() < 4:
        return theta_cur, False

    x1 = x_prev[mask]
    r1 = (dx / dt)[mask]
    var = sigma2 / dt
    sd = float(np.sqrt(var))

    def log_lik_cubic(alpha, beta, c):
        mu = mu_multi(x1, alpha, beta, c)
        return float(norm.logpdf(r1, loc=mu, scale=sd).sum())

    alpha_prop = abs(theta_cur['alpha']
                     + rng.normal(0, proposal_scale['alpha']))
    beta_prop = theta_cur['beta'] + rng.normal(0, proposal_scale['beta'])
    c_prop = theta_cur['c'] + rng.normal(0, proposal_scale['c'])

    ll_cur = log_lik_cubic(theta_cur['alpha'], theta_cur['beta'],
                           theta_cur['c'])
    ll_prop = log_lik_cubic(alpha_prop, beta_prop, c_prop)

    log_accept = ll_prop - ll_cur
    accepted = bool(np.log(rng.uniform()) < log_accept)

    if accepted:
        return ({'alpha': float(alpha_prop), 'beta': float(beta_prop),
                 'c': float(c_prop)}, True)
    return theta_cur, False


# TODO: a conjugate Inverse-Gamma update for sigma2 is the natural next
# extension. Currently sigma2 is held fixed at var(dx)/dt from the EM run.


# ---------------------------------------------------------------------------
# Block 3 — Draw P | S
# ---------------------------------------------------------------------------

def draw_P(S: np.ndarray, alpha_prior: np.ndarray,
           rng: np.random.Generator) -> np.ndarray:
    """
    Exact Dirichlet draw of P given the sampled state sequence S.
    alpha_prior[s, s'] are Dirichlet prior pseudo-counts (from Phase B N).
    """
    K = alpha_prior.shape[0]
    N = np.zeros((K, K), dtype=float)
    # Vectorised transition counting
    s_from = S[:-1].astype(int)
    s_to = S[1:].astype(int)
    np.add.at(N, (s_from, s_to), 1.0)

    P = np.empty((K, K))
    for s in range(K):
        alpha_post = alpha_prior[s] + N[s]
        P[s] = rng.dirichlet(alpha_post)
    return P


# ---------------------------------------------------------------------------
# Label-augmented emission
# ---------------------------------------------------------------------------

def augmented_log_b(x_prev, dx, dt, theta, sigma2,
                    window_assignments, p_label_pairs, eta=0.3):
    """
    Log emission probabilities augmented by Phase A label likelihood:
        b_tilde_s(t) = b_s(t) * p_label(s | w_t)^eta

    p_label_pairs : (W, 2) array of [p_single, p_multi] per window index.
                    Row -1 (default) is uniform [0.5, 0.5].
    """
    log_b = emission_log_b(x_prev, dx, dt, theta, sigma2)

    if eta == 0.0 or p_label_pairs is None:
        return log_b

    # Map -1 sentinel to the last row (uniform), looked up by gather
    idx = window_assignments.copy()
    n_w = p_label_pairs.shape[0] - 1   # last row holds the uniform default
    idx = np.where(idx < 0, n_w, idx)
    log_b = log_b + eta * np.log(np.clip(p_label_pairs[idx], 1e-6, 1.0 - 1e-6))
    return log_b


def build_window_assignments(dt_t, labels_df):
    """
    Assign each observation timestamp to the index of the Phase A window it
    falls in. Observations outside all windows get -1 (uniform p_label).

    Returns
    -------
    window_assignments : (T,) int array
    p_label_pairs      : (W+1, 2) array; rows 0..W-1 are [1-p_mw, p_mw] per
                         window, last row is the uniform default.
    """
    dt_series = pd.Series(pd.to_datetime(dt_t))
    W = len(labels_df)
    window_assignments = np.full(len(dt_t), -1, dtype=int)
    for i, row in labels_df.iterrows():
        mask = ((dt_series >= row['window_start'])
                & (dt_series < row['window_end']))
        window_assignments[mask.values] = i

    p_mw = labels_df['p_multiwell'].fillna(0.5).to_numpy(dtype=float)
    p_pairs = np.empty((W + 1, 2))
    p_pairs[:W, 0] = 1.0 - p_mw
    p_pairs[:W, 1] = p_mw
    p_pairs[W] = [0.5, 0.5]
    return window_assignments, p_pairs


# ---------------------------------------------------------------------------
# Gibbs loop
# ---------------------------------------------------------------------------

def run_gibbs(
    x_prev, dx, dt,
    theta_init, P_init, S_init,
    sigma2, alpha_prior,
    window_assignments, p_label_pairs,
    prior_ou, proposal_scale,
    n_burn=500, n_sample=1000, thin=5,
    eta=0.3,
    rng=None,
    console=None,
):
    rng = rng if rng is not None else np.random.default_rng(42)
    console = console or Console()

    T = len(dx)
    K = 2
    n_keep = n_sample // thin

    chain_theta = []
    chain_P = np.empty((n_keep, K, K))
    chain_S = np.empty((n_keep, T), dtype=np.int8)
    chain_loglik = np.empty(n_burn + n_sample)
    p_filt_accum = np.zeros((T, K))

    theta = dict(theta_init)
    P = P_init.copy()
    S = S_init.astype(np.int8).copy()

    n_accepted_cubic = 0
    n_proposed_cubic = 0
    accept_window = []

    total = n_burn + n_sample
    with Progress(SpinnerColumn(),
                  '[progress.description]{task.description}',
                  TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task('Gibbs sampler', total=total)

        for it in range(total):

            # Block 1: Draw S | theta, P, x
            log_b = augmented_log_b(
                x_prev, dx, dt, theta, sigma2,
                window_assignments, p_label_pairs, eta=eta,
            )
            S, p_filt, log_lik = ffbs(log_b, P, P[0], rng)
            chain_loglik[it] = log_lik

            # Block 2a: OU conjugate draw
            ou_draw = draw_ou_params(
                x_prev, dx, dt, S, sigma2, prior_ou, rng,
            )
            theta['kappa'] = ou_draw['kappa']
            theta['m'] = ou_draw['m']

            # Block 2b: cubic MH step
            cubic_draw, accepted = draw_cubic_params(
                x_prev, dx, dt, S, sigma2, theta, proposal_scale, rng,
            )
            theta['alpha'] = cubic_draw['alpha']
            theta['beta'] = cubic_draw['beta']
            theta['c'] = cubic_draw['c']

            n_proposed_cubic += 1
            if accepted:
                n_accepted_cubic += 1

            # Dual-averaging adaptation during burn-in (every 50 sweeps)
            if it < n_burn and (it + 1) % 50 == 0:
                accept_window.append(
                    n_accepted_cubic / max(n_proposed_cubic, 1)
                )
                rate = float(np.mean(accept_window[-5:]))
                if rate < 0.15:
                    for k in proposal_scale:
                        proposal_scale[k] *= 0.7
                elif rate > 0.45:
                    for k in proposal_scale:
                        proposal_scale[k] *= 1.3

            # Block 3: Draw P | S (exact Dirichlet)
            P = draw_P(S, alpha_prior, rng)

            # Storage: post-burn-in, thinned
            if it >= n_burn and (it - n_burn) % thin == 0:
                idx = (it - n_burn) // thin
                chain_theta.append(dict(theta))
                chain_P[idx] = P
                chain_S[idx] = S
                p_filt_accum += p_filt

            prog.advance(task)

    acceptance_rate_cubic = n_accepted_cubic / max(n_proposed_cubic, 1)
    p_filt_mean = p_filt_accum / max(n_keep, 1)

    if acceptance_rate_cubic < 0.15:
        console.print(
            f'[yellow]Warning: cubic MH acceptance rate = '
            f'{acceptance_rate_cubic:.2f}. Inspect cubic parametrisation '
            f'or narrow proposal_scale.[/yellow]'
        )

    return {
        'chain_theta':           chain_theta,
        'chain_P':               chain_P,
        'chain_S':               chain_S,
        'chain_loglik':          chain_loglik,
        'acceptance_rate_cubic': acceptance_rate_cubic,
        'p_filt_mean':           p_filt_mean,
    }


# ---------------------------------------------------------------------------
# Posterior summaries
# ---------------------------------------------------------------------------

def summarise_chain(chain_theta, chain_P, chain_loglik, n_burn,
                    acceptance_rate_cubic=None, console=None):
    """
    Posterior mean, sd, and 95% credible intervals for theta and P.
    """
    console = console or Console()
    rows = []

    for name in ['kappa', 'm', 'alpha', 'beta', 'c']:
        vals = np.array([t[name] for t in chain_theta])
        rows.append({
            'param': name,
            'mean':  float(vals.mean()),
            'sd':    float(vals.std()),
            'q025':  float(np.percentile(vals, 2.5)),
            'q975':  float(np.percentile(vals, 97.5)),
        })

    K = chain_P.shape[1]
    for i in range(K):
        for j in range(K):
            vals = chain_P[:, i, j]
            rows.append({
                'param': f'P_{i}{j}',
                'mean':  float(vals.mean()),
                'sd':    float(vals.std()),
                'q025':  float(np.percentile(vals, 2.5)),
                'q975':  float(np.percentile(vals, 97.5)),
            })

    if acceptance_rate_cubic is not None:
        rows.append({
            'param': 'mh_accept_cubic',
            'mean':  float(acceptance_rate_cubic),
            'sd':    float('nan'),
            'q025':  float('nan'),
            'q975':  float('nan'),
        })

    post_ll = chain_loglik[n_burn:]
    console.print(
        f'[cyan]Post-burn-in log-lik: mean={post_ll.mean():.4e}  '
        f'sd={post_ll.std():.4e}[/cyan]'
    )
    return pd.DataFrame(rows)


def posterior_predictive_forecast(p_filt_mean, chain_P,
                                  k_steps=(1, 5, 20, 60)):
    """
    Posterior predictive k-step ahead regime probabilities, averaging A^k
    across posterior samples to propagate uncertainty in P.
    """
    rows = []
    n_samples = chain_P.shape[0]

    for k in k_steps:
        Ak_mean = np.mean(
            np.stack([matrix_power(chain_P[i], int(k))
                      for i in range(n_samples)]),
            axis=0,
        )
        p_ahead = p_filt_mean @ Ak_mean
        T = p_filt_mean.shape[0]
        for t in range(T):
            rows.append({
                't':              int(t),
                'k':              int(k),
                'p_single_ahead': float(p_ahead[t, 0]),
                'p_multi_ahead': float(p_ahead[t, 1]),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cached EM warm-start loader
# ---------------------------------------------------------------------------

def _phase_c_em_paths(stem_em, output_dir):
    return {
        'theta':  os.path.join(output_dir, f'{stem_em}_theta.csv'),
        'probs':  os.path.join(output_dir, f'{stem_em}_probs.csv'),
    }


def _try_load_em_result(stem_em, output_dir, console):
    """
    Reconstruct a minimal EM warm-start dict from cached phase_c outputs.
    Returns None if the cache is incomplete.

    The cache exposes theta, sigma2, P, and gamma (the Kim smoother output).
    These are exactly what run_phase_c_mcmc needs from EM.
    """
    paths = _phase_c_em_paths(stem_em, output_dir)
    if not all(os.path.exists(p) for p in paths.values()):
        return None

    try:
        theta_df = pd.read_csv(paths['theta'])
        params = dict(zip(theta_df['param'], theta_df['value']))
        theta = {k: float(params[k]) for k in
                 ('kappa', 'm', 'alpha', 'beta', 'c')}
        sigma2 = float(params['sigma2'])
        P = np.array([
            [params['P_00'], params['P_01']],
            [params['P_10'], params['P_11']],
        ], dtype=float)

        probs_df = pd.read_csv(paths['probs'])
        gamma = probs_df[['gamma_single', 'gamma_multi']].to_numpy(
            dtype=float
        )
    except (KeyError, ValueError) as exc:
        console.print(
            f'[yellow]Found EM cache but could not parse it ({exc}); '
            f'will rerun EM.[/yellow]'
        )
        return None

    console.print(
        f'[green]Loaded EM warm start from cache:[/green] {paths["theta"]}'
    )
    return {'theta': theta, 'P': P, 'gamma': gamma, 'sigma2': sigma2}


# ---------------------------------------------------------------------------
# Phase B count-matrix loader (avoid recomputing if results CSV exists)
# ---------------------------------------------------------------------------

def _try_load_phase_b_counts(labels_csv, seconds_interval, output_dir,
                             prior_lambda, console):
    """
    Recover the Phase B N count matrix without re-running run_markov_chain.

    The Phase B results CSV stores the prior pseudo-counts alpha = 1 + lambda*N.
    Inverting that recovers N exactly. Returns None if the file is missing
    or malformed.
    """
    stem = os.path.splitext(os.path.basename(labels_csv))[0]
    mc_csv = os.path.join(output_dir, f'mc_{stem}_{seconds_interval}s_results.csv')
    if not os.path.exists(mc_csv):
        return None

    try:
        with open(mc_csv) as f:
            text = f.read()
        if '# prior_counts' not in text:
            return None

        sections = text.split('# prior_counts')
        prior_block = sections[1].strip()
        from io import StringIO
        prior_df = pd.read_csv(StringIO(prior_block))
        if prior_lambda <= 0:
            return None

        # alpha = 1 + lambda * N  ->  N = (alpha - 1) / lambda
        K = 2
        N = np.zeros((K, K), dtype=float)
        # Match by row order from STATES of markov_chain (single-well, multi-well)
        from markov_chain import STATES as MC_STATES
        idx_of = {s: i for i, s in enumerate(MC_STATES)}
        for _, row in prior_df.iterrows():
            i = idx_of.get(row['from_state'])
            j = idx_of.get(row['to_state'])
            if i is None or j is None:
                return None
            N[i, j] = (float(row['alpha']) - 1.0) / float(prior_lambda)
    except Exception as exc:
        console.print(
            f'[yellow]Found Phase B CSV but could not parse it ({exc}); '
            f'will rerun Phase B.[/yellow]'
        )
        return None

    console.print(f'[green]Loaded Phase B counts from cache:[/green] {mc_csv}')
    return N


# ---------------------------------------------------------------------------
# Phase C MCMC entry point
# ---------------------------------------------------------------------------

def run_phase_c_mcmc(
    start_date,
    end_date,
    seconds_interval,
    labels_csv,
    N_phase_b=None,           # optional; if None, loaded from Phase B cache or computed
    em_result=None,           # optional; if None, loaded from cache or run via run_phase_c
    lambda_prior=0.3,
    eta=0.3,
    n_burn=500,
    n_sample=1000,
    thin=5,
    k_steps=(1, 5, 20, 60),
    kernel_half_width=5,
    trim_quantile=0.01,
    detrend=0,
    seed=42,
    output_dir='regime_results',
    use_chain_cache=True,
    console=None,
):
    """
    Gibbs sampler for the Markov-switching SDE with FFBS state updates.

    Caching behaviour
    -----------------
    * Aggregated returns (Phase C input): cached by data_collection (handled
      transparently in load_series).
    * Phase B count matrix: loaded from
      regime_results/mc_<labels_stem>_<int>s_results.csv if present;
      otherwise run_markov_chain is invoked.
    * EM warm start: loaded from
      regime_results/phase_c_<from>_to_<to>_<int>s_{theta,probs}.csv if
      present; otherwise run_phase_c is invoked.
    * Final MCMC chain: loaded from
      regime_results/phase_c_mcmc_<from>_to_<to>_<int>s_chain.npz when
      use_chain_cache=True and the file is present.
    """
    os.makedirs(output_dir, exist_ok=True)
    console = console or Console()
    rng = np.random.default_rng(seed)

    stem_em = (
        f"phase_c_{pd.to_datetime(start_date).strftime('%Y-%m-%d')}_to_"
        f"{pd.to_datetime(end_date).strftime('%Y-%m-%d')}_{seconds_interval}s"
    )
    stem = (
        f"phase_c_mcmc_{pd.to_datetime(start_date).strftime('%Y-%m-%d')}_to_"
        f"{pd.to_datetime(end_date).strftime('%Y-%m-%d')}_{seconds_interval}s"
    )

    chain_path    = os.path.join(output_dir, f'{stem}_chain.npz')
    summary_path  = os.path.join(output_dir, f'{stem}_summary.csv')
    forecast_path = os.path.join(output_dir, f'{stem}_forecast.csv')

    # ------------------------------------------------------------------ #
    # Short-circuit when a complete chain is already on disk
    # ------------------------------------------------------------------ #
    if (use_chain_cache
            and os.path.exists(chain_path)
            and os.path.exists(summary_path)
            and os.path.exists(forecast_path)):
        console.print(
            f'[green]MCMC outputs already present at[/green] {chain_path} — '
            'loading instead of resampling.'
        )
        npz = np.load(chain_path, allow_pickle=False)
        chain_P = npz['chain_P']
        chain_theta = [
            {k: float(npz[f'theta_{k}'][i])
             for k in ('kappa', 'm', 'alpha', 'beta', 'c')}
            for i in range(chain_P.shape[0])
        ]
        return {
            'chain_theta':  chain_theta,
            'chain_P':      chain_P,
            'chain_S':      npz['chain_S'] if 'chain_S' in npz.files else None,
            'chain_loglik': npz['chain_loglik'],
            'p_filt_mean':  npz['p_filt_mean'],
            'summary':      pd.read_csv(summary_path),
            'df_forecast':  pd.read_csv(forecast_path),
            'acceptance_rate_cubic': float(
                npz['acceptance_rate_cubic']
            ) if 'acceptance_rate_cubic' in npz.files else float('nan'),
        }

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    console.print(
        f'[cyan]MCMC Phase C — loading series at {seconds_interval}s[/cyan]'
    )
    x_prev, dx, dt, dt_t = load_series(
        start_date, end_date, seconds_interval,
        kernel_half_width=kernel_half_width,
        trim_quantile=trim_quantile,
        detrend=detrend,
    )
    console.print(f'  {len(dx)} increments loaded.')

    # ------------------------------------------------------------------ #
    # Phase B count matrix (cached or recomputed)
    # ------------------------------------------------------------------ #
    if N_phase_b is None:
        N_phase_b = _try_load_phase_b_counts(
            labels_csv, seconds_interval, output_dir, lambda_prior, console,
        )
        if N_phase_b is None:
            console.print(
                '[yellow]Phase B cache miss — running run_markov_chain.[/yellow]'
            )
            from markov_chain import run_markov_chain
            mc = run_markov_chain(
                labels_csv, seconds_interval=seconds_interval,
                output_dir=output_dir, console=console,
                prior_lambda=lambda_prior,
            )
            if mc is None:
                raise RuntimeError(
                    'Phase B returned None — too few valid windows for chain '
                    'estimation. Extend the date range.'
                )
            N_phase_b = mc['N']

    # ------------------------------------------------------------------ #
    # EM warm start (cached or recomputed)
    # ------------------------------------------------------------------ #
    if em_result is None:
        em_result = _try_load_em_result(stem_em, output_dir, console)
    if em_result is None:
        console.print(
            '[yellow]No EM result supplied or cached — running '
            'phase_c.run_phase_c to generate the warm start.[/yellow]'
        )
        from phase_c import run_phase_c
        em_result = run_phase_c(
            start_date, end_date, seconds_interval,
            labels_csv=labels_csv,
            output_dir=output_dir,
            console=console,
        )

    theta_em = em_result['theta']
    P_em     = em_result['P']
    gamma_em = em_result['gamma']
    sigma2   = float(em_result.get('sigma2', np.var(dx) / dt))

    if len(gamma_em) != len(dx):
        raise RuntimeError(
            f'EM gamma length ({len(gamma_em)}) does not match the current '
            f'series length ({len(dx)}). The cached EM run was on a different '
            f'window — delete {stem_em}_probs.csv to force recomputation.'
        )

    # ------------------------------------------------------------------ #
    # Initial state sequence: ancestral sample from gamma
    # ------------------------------------------------------------------ #
    gamma_sums = gamma_em.sum(axis=1, keepdims=True)
    gamma_norm = np.where(gamma_sums > 0, gamma_em / gamma_sums, 0.5)
    cum = np.cumsum(gamma_norm, axis=1)
    u = rng.uniform(size=len(gamma_norm))
    S_init = (u[:, None] >= cum[:, :-1]).sum(axis=1).astype(np.int8)
    console.print(
        f'  Initial S: {(S_init == 0).mean():.1%} single-well  '
        f'{(S_init == 1).mean():.1%} multi-well'
    )

    # ------------------------------------------------------------------ #
    # Priors
    # ------------------------------------------------------------------ #
    alpha_prior = np.ones((2, 2)) + lambda_prior * np.asarray(
        N_phase_b, dtype=float
    )

    prior_ou = {
        'kappa':  theta_em['kappa'],
        'm':      theta_em['m'],
        'a_mean': theta_em['kappa'] * theta_em['m'],
        'a_var':  1e4,
        'b_mean': -theta_em['kappa'],
        'b_var':  1e4,
    }

    proposal_scale = {
        'alpha': max(abs(theta_em['alpha']) * 0.05, 1e-4),
        'beta':  max(abs(theta_em['beta'])  * 0.05, 1e-4),
        'c':     max(abs(theta_em['c'])     * 0.05, 1e-4),
    }

    # ------------------------------------------------------------------ #
    # Window-to-observation assignment for label emission
    # ------------------------------------------------------------------ #
    labels_df = pd.read_csv(
        labels_csv, parse_dates=['window_start', 'window_end'],
    )
    labels_df = labels_df[
        labels_df['seconds_interval'] == seconds_interval
    ]
    labels_df = labels_df.drop_duplicates('window_start').reset_index(drop=True)
    if 'p_multiwell' not in labels_df.columns:
        labels_df['p_multiwell'] = 0.5

    window_assignments, p_label_pairs = build_window_assignments(
        dt_t, labels_df,
    )

    # ------------------------------------------------------------------ #
    # Run Gibbs
    # ------------------------------------------------------------------ #
    chain = run_gibbs(
        x_prev, dx, dt,
        theta_init=theta_em, P_init=P_em, S_init=S_init,
        sigma2=sigma2, alpha_prior=alpha_prior,
        window_assignments=window_assignments,
        p_label_pairs=p_label_pairs,
        prior_ou=prior_ou,
        proposal_scale=proposal_scale,
        n_burn=n_burn, n_sample=n_sample, thin=thin,
        eta=eta, rng=rng, console=console,
    )

    # ------------------------------------------------------------------ #
    # Summaries + forecast
    # ------------------------------------------------------------------ #
    summary_df = summarise_chain(
        chain['chain_theta'], chain['chain_P'],
        chain['chain_loglik'], n_burn,
        acceptance_rate_cubic=chain['acceptance_rate_cubic'],
        console=console,
    )

    df_forecast = posterior_predictive_forecast(
        chain['p_filt_mean'], chain['chain_P'], k_steps=k_steps,
    )

    # ------------------------------------------------------------------ #
    # Persist
    # ------------------------------------------------------------------ #
    np.savez_compressed(
        chain_path,
        chain_P                = chain['chain_P'],
        chain_S                = chain['chain_S'],
        chain_loglik           = chain['chain_loglik'],
        p_filt_mean            = chain['p_filt_mean'],
        acceptance_rate_cubic  = np.array(chain['acceptance_rate_cubic']),
        **{
            f'theta_{k}': np.array([t[k] for t in chain['chain_theta']])
            for k in ('kappa', 'm', 'alpha', 'beta', 'c')
        },
    )
    console.print(f'[green]Wrote[/green] {chain_path}')

    summary_df.to_csv(summary_path, index=False)
    console.print(f'[green]Wrote[/green] {summary_path}')

    df_forecast.to_csv(forecast_path, index=False)
    console.print(f'[green]Wrote[/green] {forecast_path}')

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    plots.plot_phase_c_gamma_timeline(
        chain['p_filt_mean'],
        os.path.join(output_dir, f'{stem}_gamma_post.png'),
        datetimes=pd.to_datetime(dt_t),
    )
    plots.plot_mcmc_kappa_posterior(
        np.array([t['kappa'] for t in chain['chain_theta']]),
        theta_em['kappa'],
        os.path.join(output_dir, f'{stem}_kappa_post.png'),
    )
    plots.plot_mcmc_loglik_trace(
        chain['chain_loglik'], n_burn,
        os.path.join(output_dir, f'{stem}_loglik_trace.png'),
    )

    console.print(
        f"[green]Acceptance rate (cubic MH): "
        f"{chain['acceptance_rate_cubic']:.2f}[/green]"
    )

    return {**chain, 'summary': summary_df, 'df_forecast': df_forecast}


# ---------------------------------------------------------------------------
# Standalone entry
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from regime_estimation import normalize_window_boundaries

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    seconds_interval = 30
    window_type = 'weekly'

    start_date, end_date = normalize_window_boundaries(
        start_date, end_date, window_type,
    )

    fname = (
        f"regime_labels_{start_date.strftime('%Y-%m-%d')}_to_"
        f"{end_date.strftime('%Y-%m-%d')}_{window_type}.csv"
    )
    labels_csv = os.path.join('regime_results', fname)

    if not os.path.exists(labels_csv):
        # Fall back to the most recent labels CSV produced for this window_type
        candidates = sorted(glob.glob(os.path.join(
            'regime_results', f'regime_labels_*_{window_type}.csv',
        )))
        if not candidates:
            raise FileNotFoundError(
                f'No regime_labels CSV found in regime_results/. '
                f'Run regime_estimation.py first.'
            )
        labels_csv = candidates[-1]

    _console = Console()
    _console.print(f'[cyan]Labels : {labels_csv}[/cyan]')

    run_phase_c_mcmc(
        start_date, end_date, seconds_interval,
        labels_csv=labels_csv,
        lambda_prior=0.3,
        eta=0.3,
        n_burn=500,
        n_sample=2000,
        thin=5,
        k_steps=(1, 5, 20, 60),
        console=_console,
    )
