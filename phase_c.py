"""
Phase C — parametric Markov-switching SDE.

Two-state hidden chain S_t in {0, 1} with parametric drifts:
    state 0  ('single-well')   mu_1(x) = -kappa (x - m)
    state 1  ('double-well')   mu_2(x) = -alpha (x - c)^3 + beta (x - c)

Shared diffusion sigma. Euler-Maruyama emission per step:
    Delta x_t  |  x_{t-1}, S_t = s   ~   N(  mu_s(x_{t-1}) * dt,  sigma^2 * dt  )

Pipeline (run_phase_c):
  1. Load aggregated log-prices over [start_date, end_date] at one interval.
     Cross-gap increments (where dt > 1.5 * nominal) are dropped.
  2. Initialise sigma^2 = var(Delta x) / dt and hold it fixed throughout.
  3. Initialise drift parameters by fitting the parametric forms to per-window
     km_df drift curves saved by Phase A (regime_results/km/), pooled by the
     Phase A regime label (single-well vs double-well).
  4. Initial transition matrix is near-identity P = [[1-eps, eps], [eps, 1-eps]].
     Initial state prior pi_0 is supplied by the caller (e.g. the empirical
     stationary distribution from markov_chain.run_markov_chain). Default
     uniform [0.5, 0.5].
  5. EM loop: Hamilton forward filter + Kim backward smoother (E-step),
     analytical weighted least squares for OU + 1D-profiled WLS for the
     cubic + closed-form transition update (M-step), until log-likelihood
     converges (relative tol) or max_iter reached.

Outputs (under output_dir):
  phase_c_<from>_to_<to>_<int>s_probs.csv     filtered + smoothed series
  phase_c_<from>_to_<to>_<int>s_theta.csv     final parameters and P
  phase_c_<from>_to_<to>_<int>s_loglik.csv    log-lik per EM iteration
  phase_c_<from>_to_<to>_<int>s_gamma.png     smoothed gamma_t timeline
  phase_c_<from>_to_<to>_<int>s_drifts.png    fitted mu_1, mu_2 curves
"""

import glob
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy.optimize import least_squares, minimize_scalar
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import data_collection as dc
import plots


STATES = ['single-well', 'double-well']
EPS = 1e-300


# ---------------------------------------------------------------------------
# Parametric drifts
# ---------------------------------------------------------------------------

def mu_single(x, kappa, m):
    return -kappa * (x - m)


def mu_double(x, alpha, beta, c):
    u = x - c
    return -alpha * u ** 3 + beta * u


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _aggregated_returns_path(
    start_date, end_date, seconds_interval,
    kernel_half_width, trim_quantile, detrend,
    data_dir='data',
):
    """Path of the cached aggregated-returns CSV per data_collection naming."""
    def _to_dt(d):
        if isinstance(d, str):
            return datetime.strptime(d, '%Y-%m-%d')
        return d

    start_dt = _to_dt(start_date)
    end_dt = _to_dt(end_date)
    kernel_tag = f'_k{kernel_half_width}' if kernel_half_width > 0 else ''
    trim_tag = f'_trim{trim_quantile}' if trim_quantile > 0 else ''
    detrend_tag = '_detrended' if detrend else ''
    return os.path.join(
        data_dir,
        f'ETHUSDT-aggReturns-{start_dt.strftime("%Y-%m-%d")}_to_'
        f'{end_dt.strftime("%Y-%m-%d")}-{seconds_interval}sec'
        f'{kernel_tag}{trim_tag}{detrend_tag}.csv',
    )


def load_series(
    start_date,
    end_date,
    seconds_interval,
    kernel_half_width=5,
    trim_quantile=0.01,
    detrend=0,
):
    """
    Load aggregated log-prices over [start_date, end_date] at one sampling
    interval and emit (x_prev, dx, dt, datetimes) with cross-gap increments
    dropped.

    If a cached CSV matching the data_collection naming convention exists in
    ./data, it is read directly; otherwise the aggregation is run.

    Returns
    -------
    x_prev    : (N,) log-price at t-1
    dx        : (N,) increment x_t - x_{t-1}
    dt        : float, nominal step in seconds
    dt_t      : (N,) datetime of x_t (for plotting), aligned with dx
    """
    cached_path = _aggregated_returns_path(
        start_date, end_date, seconds_interval,
        kernel_half_width, trim_quantile, detrend,
    )
    if os.path.exists(cached_path):
        df = pd.read_csv(cached_path, parse_dates=['datetime'])
        if df.empty:
            df = dc.aggregate_log_returns_range(
                start_date, end_date, seconds_interval,
                kernel_half_width=kernel_half_width,
                trim_quantile=trim_quantile,
                detrend=detrend,
            )
    else:
        df = dc.aggregate_log_returns_range(
            start_date, end_date, seconds_interval,
            kernel_half_width=kernel_half_width,
            trim_quantile=trim_quantile,
            detrend=detrend,
        )
    if df.empty:
        raise ValueError('Empty aggregated returns dataframe.')

    df = df.sort_values('datetime').reset_index(drop=True)
    x = df['log_first_price'].values.astype(float)
    ts = pd.to_datetime(df['datetime']).values

    dt_arr = (np.diff(ts) / np.timedelta64(1, 's')).astype(float)
    finite = np.isfinite(np.diff(x)) & np.isfinite(x[:-1])
    valid = (dt_arr > 0) & (dt_arr <= 1.5 * seconds_interval) & finite

    x_prev = x[:-1][valid]
    dx = np.diff(x)[valid]
    dt_t = ts[1:][valid]

    return x_prev, dx, float(seconds_interval), dt_t


# ---------------------------------------------------------------------------
# KM-curve based drift initialisation
# ---------------------------------------------------------------------------

def _load_km_for_window(window_start_str, window_end_str, seconds_interval, output_dir):
    km_path = os.path.join(
        output_dir, 'km',
        f'km_{window_start_str}_to_{window_end_str}_{seconds_interval}s.csv',
    )
    if os.path.exists(km_path):
        return pd.read_csv(km_path)
    return None


def pool_km_drift_curves(
    labels_csv, regime_label, seconds_interval, output_dir, annualize=True,
):
    """
    Pool per-window km_df drift curves for windows whose Phase A label matches
    `regime_label` at the requested seconds_interval. Bin-aggregate (mean) onto
    a common bin_center grid and return (x, mu_hat).
    """
    df = pd.read_csv(labels_csv)
    sub = df[(df['regime'] == regime_label)
             & (df['seconds_interval'] == seconds_interval)]

    pooled = []
    for _, row in sub.iterrows():
        km = _load_km_for_window(
            str(row['window_start']), str(row['window_end']),
            seconds_interval, output_dir,
        )
        if km is None:
            continue
        clean = km.dropna(subset=['drift'])[['bin_center', 'drift']]
        if not clean.empty:
            pooled.append(clean)

    if not pooled:
        return np.array([]), np.array([])

    pooled_df = pd.concat(pooled, ignore_index=True)
    pooled_df['bin'] = np.round(pooled_df['bin_center'], 4)
    agg = pooled_df.groupby('bin')['drift'].mean().reset_index()
    x = agg['bin'].values.astype(float)
    mu = agg['drift'].values.astype(float)
    if annualize:
        sec_per_year = 365.25 * 24 * 3600
        mu = mu * sec_per_year
    return x, mu


def fit_initial_theta_from_km(labels_csv, seconds_interval, output_dir, console=None):
    """
    Fit OU and cubic drifts to pooled km drift curves. Falls back to heuristic
    defaults when no km files are available for one of the regimes.

    Retained for comparison; the active initialiser is
    ``fit_initial_theta_moments`` which fits to the Phase C observation pairs
    directly and avoids the circular dependency on Phase A labels.
    """
    console = console or Console()

    # State 0: OU drift  -kappa*(x-m) = (kappa*m) + (-kappa)*x
    x1, mu1 = pool_km_drift_curves(
        labels_csv, 'single-well', seconds_interval, output_dir,
    )
    if len(x1) >= 5:
        slope, intercept = np.polyfit(x1, mu1, 1)
        kappa = max(-slope, 1e-9)
        m = intercept / kappa
    else:
        warnings.warn('No single-well km_df available; using OU defaults.')
        kappa, m = 1.0, 0.0

    # State 1: cubic drift  -alpha*(x-c)^3 + beta*(x-c).
    x2, mu2 = pool_km_drift_curves(
        labels_csv, 'double-well', seconds_interval, output_dir,
    )
    if len(x2) >= 5:
        c0 = float(np.mean(x2))
        def _resid(p):
            a, b, c = p
            return mu2 - mu_double(x2, a, b, c)
        res = least_squares(_resid, x0=[1.0, 1.0, c0])
        alpha, beta, c = res.x
        alpha = max(float(alpha), 1e-9)
        beta = float(beta)
        c = float(c)
    else:
        warnings.warn('No double-well km_df available; using cubic defaults.')
        alpha, beta, c = 1.0, 1.0, 0.0

    return {'kappa': float(kappa), 'm': float(m),
            'alpha': float(alpha), 'beta': float(beta), 'c': float(c)}


def fit_initial_theta_moments(x_prev, dx, dt, console=None):
    """
    Initialise OU and cubic drift parameters directly from the Phase C
    observation pairs (x_prev, dx/dt) using OLS + 2-means clustering.
    No dependence on Phase A labels or KM curves.
    """
    console = console or Console()

    r = dx / dt
    N = len(r)
    if N < 10:
        warnings.warn(
            f'Only {N} observation pairs available; falling back to defaults.'
        )
        return {'kappa': 1.0, 'm': float(np.mean(x_prev) if N else 0.0),
                'alpha': 1.0, 'beta': 1.0, 'c': 0.0}

    # Global OU fit -> state 0 init
    slope, intercept = np.polyfit(x_prev, r, 1)
    kappa_init = max(-slope, 1e-4)
    if kappa_init > 0:
        m_init = intercept / kappa_init
    else:
        m_init = float(np.mean(x_prev))

    # K-means split on standardised (x_prev, r) -> state 1 candidate
    Z = StandardScaler().fit_transform(np.column_stack([x_prev, r]))
    labels = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(Z)

    r_hat = intercept + slope * x_prev
    resid = np.abs(r - r_hat)
    mean_resid = [resid[labels == k].mean() if (labels == k).any() else -np.inf
                  for k in range(2)]
    dw_cluster = int(np.argmax(mean_resid))
    mask_dw = labels == dw_cluster

    if mask_dw.sum() < 5:
        warnings.warn(
            'Double-well candidate cluster has <5 points; using cubic defaults.'
        )
        alpha_init, beta_init, c_init = 1.0, 1.0, float(np.mean(x_prev))
    else:
        c0 = float(np.mean(x_prev[mask_dw]))
        alpha_init, beta_init, c_init = _wls_cubic(
            x_prev[mask_dw], r[mask_dw],
            w=np.ones(int(mask_dw.sum())),
            theta_prev={'alpha': 1.0, 'beta': 1.0, 'c': c0},
        )

    return {
        'kappa': float(kappa_init),
        'm':     float(m_init),
        'alpha': float(max(alpha_init, 1e-4)),
        'beta':  float(beta_init),
        'c':     float(c_init),
    }


# ---------------------------------------------------------------------------
# Emission likelihoods (log space for stability)
# ---------------------------------------------------------------------------

def emission_log_b(x_prev, dx, dt, theta, sigma2):
    """log b[T,2] = log p(dx_t | x_{t-1}, S_t=s) for s in {0,1}."""
    var = sigma2 * dt
    inv_2var = 0.5 / var
    norm_const = -0.5 * np.log(2.0 * np.pi * var)
    mu1 = mu_single(x_prev, theta['kappa'], theta['m']) * dt
    mu2 = mu_double(x_prev, theta['alpha'], theta['beta'], theta['c']) * dt
    log_b = np.empty((len(dx), 2))
    log_b[:, 0] = norm_const - inv_2var * (dx - mu1) ** 2
    log_b[:, 1] = norm_const - inv_2var * (dx - mu2) ** 2
    return log_b


# ---------------------------------------------------------------------------
# Hamilton forward filter (scaled, log-space normaliser)
# ---------------------------------------------------------------------------

def hamilton_filter(log_b, P, pi0):
    """
    Forward recursion with one-step normalisers c_t.
    Returns (p_filt[T,2], log_lik = sum_t log c_t, log_c[T]).
    """
    T = log_b.shape[0]
    p_filt = np.empty((T, 2))
    log_c = np.empty(T)

    # t = 0
    log_pi = np.log(np.clip(pi0, EPS, None))
    z = log_pi + log_b[0]
    mz = float(np.max(z))
    log_c[0] = mz + np.log(np.sum(np.exp(z - mz)))
    p_filt[0] = np.exp(z - log_c[0])

    for t in range(1, T):
        p_pred = p_filt[t - 1] @ P              # (2,)
        log_p = np.log(np.clip(p_pred, EPS, None))
        z = log_p + log_b[t]
        mz = float(np.max(z))
        log_c[t] = mz + np.log(np.sum(np.exp(z - mz)))
        p_filt[t] = np.exp(z - log_c[t])

    return p_filt, float(np.sum(log_c)), log_c


# ---------------------------------------------------------------------------
# Kim backward smoother (scaled)
# ---------------------------------------------------------------------------

def kim_smoother(p_filt, log_b, log_c, P):
    """
    Returns (gamma[T,2], xi[T-1,2,2]).

    Scaled backward variable beta_t with recursion
        beta_t(i) = sum_j P[i,j] * b_{t+1}(j)/c_{t+1} * beta_{t+1}(j)
    so that gamma_t = p_filt_t * beta_t (already sums to 1).
    xi_t(i,j) = Pr(S_t=i, S_{t+1}=j | x_{1:T})
              = p_filt_t(i) * P[i,j] * b_{t+1}(j)/c_{t+1} * beta_{t+1}(j).
    """
    T, K = p_filt.shape
    gamma = np.empty((T, K))
    xi = np.empty((T - 1, K, K))

    beta = np.ones(K)
    gamma[T - 1] = p_filt[T - 1]

    for t in range(T - 2, -1, -1):
        b_next_scaled = np.exp(log_b[t + 1] - log_c[t + 1])  # (K,)
        weighted = b_next_scaled * beta                       # (K,)
        beta_new = P @ weighted                               # (K,)

        xi_t = p_filt[t][:, None] * P * weighted[None, :]
        s = xi_t.sum()
        if s > 0:
            xi_t /= s
        xi[t] = xi_t

        gamma_t = p_filt[t] * beta_new
        s = gamma_t.sum()
        if s > 0:
            gamma_t /= s
        gamma[t] = gamma_t

        beta = beta_new

    return gamma, xi


# ---------------------------------------------------------------------------
# M-step
# ---------------------------------------------------------------------------

def _wls_linear(x, y, w):
    """Weighted linear regression y ~ a + b*x; returns (a, b)."""
    W = np.maximum(w, 0.0)
    sw = W.sum()
    if sw <= 1e-12:
        return 0.0, 0.0
    sx = (W * x).sum() / sw
    sy = (W * y).sum() / sw
    sxx = (W * x * x).sum() / sw
    sxy = (W * x * y).sum() / sw
    var_x = sxx - sx * sx
    if var_x <= 0:
        return float(sy), 0.0
    b = (sxy - sx * sy) / var_x
    a = sy - b * sx
    return float(a), float(b)


def _wls_cubic(x, y, w, theta_prev):
    """
    Weighted nonlinear LS for y = -alpha*(x-c)^3 + beta*(x-c).
    Profiled over c (1D Brent); for each c the (alpha, beta) sub-problem
    is linear and solved in closed form.
    """
    W = np.maximum(w, 0.0)
    if W.sum() <= 1e-9:
        return theta_prev['alpha'], theta_prev['beta'], theta_prev['c']

    def _fit_at_c(c):
        u = x - c
        A11 = (W * u ** 6).sum()
        A12 = -(W * u ** 4).sum()
        A22 = (W * u ** 2).sum()
        b1 = -(W * y * u ** 3).sum()
        b2 = (W * y * u).sum()
        det = A11 * A22 - A12 * A12
        if abs(det) < 1e-30:
            return None
        alpha = (A22 * b1 - A12 * b2) / det
        beta = (A11 * b2 - A12 * b1) / det
        mu_pred = -alpha * u ** 3 + beta * u
        sse = (W * (y - mu_pred) ** 2).sum()
        return alpha, beta, sse

    def _obj(c):
        out = _fit_at_c(c)
        return 1e30 if out is None else out[2]

    c0 = theta_prev['c']
    half = max(float(np.std(x)), 1.0)
    try:
        res = minimize_scalar(
            _obj, bracket=(c0 - half, c0, c0 + half), method='brent',
        )
        c_opt = float(res.x)
    except Exception:
        c_opt = c0

    fit = _fit_at_c(c_opt)
    if fit is None:
        return theta_prev['alpha'], theta_prev['beta'], theta_prev['c']
    alpha, beta, _ = fit
    return float(max(alpha, 1e-9)), float(beta), c_opt


def m_step(x_prev, dx, dt, gamma, xi, theta, sigma2):
    """Update kappa, m, alpha, beta, c, P. sigma2 fixed."""
    rt = dx / dt
    w0 = gamma[:, 0]
    w1 = gamma[:, 1]

    a0, b0 = _wls_linear(x_prev, rt, w0)
    new_kappa = max(-b0, 1e-9)
    new_m = a0 / new_kappa

    new_alpha, new_beta, new_c = _wls_cubic(x_prev, rt, w1, theta)

    xi_sum = xi.sum(axis=0)
    row_sum = xi_sum.sum(axis=1, keepdims=True)
    safe = np.where(row_sum == 0, 1.0, row_sum)
    P_new = xi_sum / safe
    zero_rows = (row_sum.flatten() == 0)
    if zero_rows.any():
        P_new[zero_rows] = 0.5

    return (
        {'kappa': new_kappa, 'm': new_m,
         'alpha': new_alpha, 'beta': new_beta, 'c': new_c},
        P_new,
    )


# ---------------------------------------------------------------------------
# EM driver
# ---------------------------------------------------------------------------

def run_em(
    x_prev, dx, dt, theta, P, pi0, sigma2,
    max_iter=50, tol=1e-4, console=None,
):
    console = console or Console()
    log_lik_trace = []
    prev_ll = -np.inf

    for it in range(max_iter):
        log_b = emission_log_b(x_prev, dx, dt, theta, sigma2)
        p_filt, log_lik, log_c = hamilton_filter(log_b, P, pi0)
        gamma, xi = kim_smoother(p_filt, log_b, log_c, P)
        log_lik_trace.append(log_lik)

        if it > 0:
            rel = abs(log_lik - prev_ll) / max(abs(prev_ll), 1.0)
            console.print(
                f'[cyan]EM iter {it:3d}[/cyan]  log-lik={log_lik:.6e}  '
                f'rel.delta={rel:.3e}'
            )
            if rel < tol:
                console.print('[green]Converged.[/green]')
                break
        else:
            console.print(f'[cyan]EM iter {it:3d}[/cyan]  log-lik={log_lik:.6e}')

        theta, P = m_step(x_prev, dx, dt, gamma, xi, theta, sigma2)
        prev_ll = log_lik

    # Final E-step with the most recent theta
    log_b = emission_log_b(x_prev, dx, dt, theta, sigma2)
    p_filt, log_lik, log_c = hamilton_filter(log_b, P, pi0)
    gamma, xi = kim_smoother(p_filt, log_b, log_c, P)
    log_lik_trace.append(log_lik)

    return {
        'theta': theta, 'P': P,
        'gamma': gamma, 'p_filt': p_filt, 'xi': xi,
        'log_lik': log_lik,
        'log_lik_trace': np.array(log_lik_trace),
    }


# ---------------------------------------------------------------------------
# Forward-looking diagnostics
# ---------------------------------------------------------------------------

def predict_k_steps(
    p_filt: np.ndarray,
    A: np.ndarray,
    k_steps: list[int],
) -> pd.DataFrame:
    """
    Compute k-step ahead regime probabilities for every time step t.

        P(S_{t+k} = s | x_{1:t}) = [p_filt[t] @ A^k]_s

    Returns a long-format DataFrame with columns
    ['t', 'k', 'p_single_ahead', 'p_double_ahead'].
    """
    from numpy.linalg import matrix_power
    rows = []
    T = len(p_filt)
    for k in k_steps:
        Ak = matrix_power(A, int(k))
        p_ahead = p_filt @ Ak
        for t in range(T):
            rows.append({
                't': int(t),
                'k': int(k),
                'p_single_ahead': float(p_ahead[t, 0]),
                'p_double_ahead': float(p_ahead[t, 1]),
            })
    return pd.DataFrame(rows)


def estimate_kappa_series(
    x_prev: np.ndarray,
    dx: np.ndarray,
    dt: float,
    dt_t: np.ndarray,
    window_size: int = 2000,
    step: int = 200,
) -> pd.DataFrame:
    """
    Estimate the OU mean-reversion rate kappa in a rolling window over the
    Phase C observation pairs (x_prev, dx). A declining kappa(t) is the
    critical-slowing-down signature that precedes a single->multi-well
    transition.

    Uses OLS of (dx/dt) on x_prev per window: intercept gives kappa*m,
    slope gives -kappa.
    """
    r = dx / dt
    T = len(r)
    if T <= window_size:
        return pd.DataFrame(columns=['datetime', 'kappa', 'kappa_m', 'n_obs'])

    rows = []
    idx = np.arange(0, T - window_size, step)
    for i in idx:
        sl = slice(i, i + window_size)
        slope, intercept = np.polyfit(x_prev[sl], r[sl], 1)
        kappa = max(-slope, 0.0)
        if kappa > 1e-9:
            m = intercept / kappa
        else:
            m = float(np.mean(x_prev[sl]))
        rows.append({
            'datetime': pd.Timestamp(dt_t[i + window_size // 2]),
            'kappa': float(kappa),
            'kappa_m': float(m),
            'n_obs': int(window_size),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase C entry point
# ---------------------------------------------------------------------------

def run_phase_c(
    start_date,
    end_date,
    seconds_interval,
    labels_csv,
    pi_init=None,
    epsilon_P=1e-3,
    sigma2=None,
    kernel_half_width=5,
    trim_quantile=0.01,
    detrend=0,
    max_iter=50,
    tol=1e-4,
    output_dir='regime_results',
    console=None,
):
    """
    Run Phase C end-to-end.

    Parameters
    ----------
    labels_csv : path to a Phase A regime labels CSV (used only to find
                 representative single-/double-well windows whose km_df
                 informs the parametric drift initialisation).
    pi_init    : (2,) initial-state prior. Default uniform [0.5, 0.5].
                 Pass markov_chain.run_markov_chain(...)['pi'] to use the
                 empirical stationary distribution.
    epsilon_P  : near-identity off-diagonal for the initial transition matrix.
    sigma2     : if None, estimated as var(dx)/dt and held fixed.
    """
    os.makedirs(output_dir, exist_ok=True)
    console = console or Console()

    console.print(f'[cyan]Phase C — loading log-price series at {seconds_interval}s ...[/cyan]')
    x_prev, dx, dt, dt_t = load_series(
        start_date, end_date, seconds_interval,
        kernel_half_width=kernel_half_width,
        trim_quantile=trim_quantile,
        detrend=detrend,
    )
    console.print(f'  {len(dx)} increments after gap filtering.')

    if sigma2 is None:
        sigma2 = float(np.var(dx) / dt)
    console.print(f'  sigma^2 (held fixed) = {sigma2:.4e}')

    theta = fit_initial_theta_moments(x_prev, dx, dt, console=console)
    console.print(
        f'  initial theta: kappa={theta["kappa"]:.4g}  m={theta["m"]:.4g}  '
        f'alpha={theta["alpha"]:.4g}  beta={theta["beta"]:.4g}  c={theta["c"]:.4g}'
    )

    P0 = np.array([[1 - epsilon_P, epsilon_P],
                   [epsilon_P, 1 - epsilon_P]])

    if pi_init is None:
        pi_init = np.array([0.5, 0.5])
    pi_init = np.asarray(pi_init, dtype=float)
    pi_init = pi_init / pi_init.sum()

    out = run_em(
        x_prev, dx, dt, theta, P0, pi_init, sigma2,
        max_iter=max_iter, tol=tol, console=console,
    )

    stem = (
        f"phase_c_{pd.to_datetime(start_date).strftime('%Y-%m-%d')}_to_"
        f"{pd.to_datetime(end_date).strftime('%Y-%m-%d')}_{seconds_interval}s"
    )

    df_probs = pd.DataFrame({
        'datetime': pd.to_datetime(dt_t),
        'x_prev': x_prev,
        'dx': dx,
        'p_filt_single': out['p_filt'][:, 0],
        'p_filt_double': out['p_filt'][:, 1],
        'gamma_single': out['gamma'][:, 0],
        'gamma_double': out['gamma'][:, 1],
    })
    probs_path = os.path.join(output_dir, f'{stem}_probs.csv')
    df_probs.to_csv(probs_path, index=False)
    console.print(f'[green]Wrote[/green] {probs_path}')

    theta_rows = [{'param': k, 'value': v} for k, v in out['theta'].items()]
    theta_rows.append({'param': 'sigma2', 'value': sigma2})
    theta_rows.append({'param': 'P_00', 'value': float(out['P'][0, 0])})
    theta_rows.append({'param': 'P_01', 'value': float(out['P'][0, 1])})
    theta_rows.append({'param': 'P_10', 'value': float(out['P'][1, 0])})
    theta_rows.append({'param': 'P_11', 'value': float(out['P'][1, 1])})
    theta_path = os.path.join(output_dir, f'{stem}_theta.csv')
    pd.DataFrame(theta_rows).to_csv(theta_path, index=False)
    console.print(f'[green]Wrote[/green] {theta_path}')

    lltrace_path = os.path.join(output_dir, f'{stem}_loglik.csv')
    pd.DataFrame({
        'iter': np.arange(len(out['log_lik_trace'])),
        'log_lik': out['log_lik_trace'],
    }).to_csv(lltrace_path, index=False)
    console.print(f'[green]Wrote[/green] {lltrace_path}')

    plots.plot_phase_c_gamma_timeline(
        out['gamma'],
        os.path.join(output_dir, f'{stem}_gamma.png'),
        datetimes=pd.to_datetime(dt_t),
    )
    plots.plot_phase_c_drifts(
        x_prev, out['theta'],
        os.path.join(output_dir, f'{stem}_drifts.png'),
    )

    # k-step ahead regime forecast
    k_steps = [1, 5, 20, 60]
    df_forecast = predict_k_steps(out['p_filt'], out['P'], k_steps)
    forecast_path = os.path.join(output_dir, f'{stem}_forecast.csv')
    df_forecast.to_csv(forecast_path, index=False)
    console.print(f'[green]Wrote[/green] {forecast_path}')

    # rolling kappa(t) — critical-slowing-down signal
    df_kappa = estimate_kappa_series(x_prev, dx, float(dt), dt_t)
    kappa_path = os.path.join(output_dir, f'{stem}_kappa.csv')
    df_kappa.to_csv(kappa_path, index=False)
    console.print(f'[green]Wrote[/green] {kappa_path}')

    if not df_kappa.empty:
        plots.plot_kappa_series(
            df_kappa, os.path.join(output_dir, f'{stem}_kappa.png'),
        )
    plots.plot_forecast(
        df_forecast,
        os.path.join(output_dir, f'{stem}_forecast_k5.png'),
        k=5,
    )

    summary = Table(title='Phase C — Final parameters')
    summary.add_column('Parameter', style='cyan')
    summary.add_column('Value', justify='right', style='green')
    for k, v in out['theta'].items():
        summary.add_row(k, f'{v:.6g}')
    summary.add_row('sigma^2', f'{sigma2:.6g}')
    summary.add_row('P[0,0]', f"{out['P'][0, 0]:.6g}")
    summary.add_row('P[0,1]', f"{out['P'][0, 1]:.6g}")
    summary.add_row('P[1,0]', f"{out['P'][1, 0]:.6g}")
    summary.add_row('P[1,1]', f"{out['P'][1, 1]:.6g}")
    summary.add_row('log-lik', f"{out['log_lik']:.6g}")
    console.print(summary)

    xi_sum = out['xi'].sum(axis=0)
    counts = Table(title='Phase C — Expected transition counts (sum_t xi_t)')
    counts.add_column('From \\ To', style='cyan')
    for s in STATES:
        counts.add_column(s, justify='right')
    for i, s_from in enumerate(STATES):
        counts.add_row(s_from, *[f'{xi_sum[i, j]:.2f}' for j in range(2)])
    console.print(counts)

    return {
        **out,
        'df_forecast': df_forecast,
        'df_kappa': df_kappa,
    }


# ---------------------------------------------------------------------------
# Standalone entry
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from datetime import datetime

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    seconds_interval = 30

    csvs = sorted(glob.glob(os.path.join('regime_results', 'regime_labels_*.csv')))
    if not csvs:
        raise FileNotFoundError(
            'No regime_labels CSV in regime_results/. Run regime_estimation.py first.'
        )
    labels_csv = csvs[-1]

    _console = Console()
    _console.print(f'[cyan]Labels file : {labels_csv}[/cyan]')

    # To use the empirical stationary distribution from markov_chain.py:
    # from markov_chain import run_markov_chain
    # mc = run_markov_chain(labels_csv, seconds_interval=seconds_interval)
    # pi_init = mc['pi'] if mc else None
    pi_init = None
  
    run_phase_c(
        start_date, end_date, seconds_interval,
        labels_csv=labels_csv,
        pi_init=pi_init,   
        max_iter=30,
        tol=1e-4,
        console=_console,
    )
