"""
Phase C — parametric Markov-switching SDE.

Two-state hidden chain S_t in {0, 1} with parametric drifts:
    state 0  ('single-well')   mu_1(x) = -kappa (x - m)
    state 1  ('muti-well')   mu_2(x) = -alpha (x - c)^3 + beta (x - c)

Shared diffusion sigma. Euler-Maruyama emission per step:
    Delta x_t  |  x_{t-1}, S_t = s   ~   N(  mu_s(x_{t-1}) * dt,  sigma^2 * dt  )

Pipeline (run_phase_c):
  1. Load aggregated log-prices over [start_date, end_date] at one interval.
     Cross-gap increments (where dt > 1.5 * nominal) are dropped.
  2. Initialise sigma^2 = var(Delta x) / dt and hold it fixed throughout.
  3. Initialise drift parameters by fitting the parametric forms to per-window
     km_df drift curves saved by Phase A (regime_results/km/), pooled by the
     Phase A regime label (single-well vs multi-well).
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


STATES = ['single-well', 'multi-well']
EPS = 1e-300
PHASE_C_OUTPUT_DIR = 'phase_c_results'
SEC_PER_YEAR = 365.25 * 24 * 3600


def annualize_theta(theta: dict) -> dict:
    """Return a copy of theta with rate parameters scaled to per-year."""
    return {
        'kappa_ann': float(theta['kappa']) * SEC_PER_YEAR,
        'alpha_ann': float(theta['alpha']) * SEC_PER_YEAR,
        'beta_ann':  float(theta['beta'])  * SEC_PER_YEAR,
        'm':         float(theta['m']),    # log-price, no time unit
        'c':         float(theta['c']),
    }


def sigma_annualized(sigma2) -> np.ndarray:
    """sqrt(sigma2 * SEC_PER_YEAR) — broadcasts scalar to length-2 vector."""
    s2 = np.atleast_1d(np.asarray(sigma2, dtype=float))
    if s2.size == 1:
        s2 = np.array([float(s2[0]), float(s2[0])])
    return np.sqrt(s2[:2] * SEC_PER_YEAR)


# ---------------------------------------------------------------------------
# Parametric drifts
# ---------------------------------------------------------------------------

def mu_single(x, kappa, m):
    return -kappa * (x - m)


def mu_multi(x, alpha, beta, c):
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

    # State 0: OU drift  -kappa*(x-m) = (kappa*m) + (-kappa)*x.
    # We need kappa in per-second units (the EM convention); request the
    # pooled mu in the same units rather than annualised.
    x1, mu1 = pool_km_drift_curves(
        labels_csv, 'single-well', seconds_interval, output_dir,
        annualize=False,
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
        labels_csv, 'multi-well', seconds_interval, output_dir,
        annualize=False,
    )
    if len(x2) >= 5:
        c0 = float(np.mean(x2))
        def _resid(p):
            a, b, c = p
            return mu2 - mu_multi(x2, a, b, c)
        res = least_squares(_resid, x0=[1.0, 1.0, c0])
        alpha, beta, c = res.x
        alpha = max(float(alpha), 1e-9)
        beta = float(beta)
        c = float(c)
    else:
        warnings.warn('No multi-well km_df available; using cubic defaults.')
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
            'Multi-well candidate cluster has <5 points; using cubic defaults.'
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
    """
    log b[T,2] = log p(dx_t | x_{t-1}, S_t=s) for s in {0,1}.

    sigma2 may be a scalar (shared noise) or a length-2 array
    [sigma2_state0, sigma2_state1] for regime-specific noise. The scalar form
    is broadcast to both states, preserving backwards compatibility with
    callers that pre-date the regime-specific switch (e.g. phase_c_mcmc).
    """
    s2 = np.atleast_1d(np.asarray(sigma2, dtype=float))
    if s2.size == 1:
        s2 = np.array([float(s2[0]), float(s2[0])])
    else:
        s2 = s2[:2].astype(float)
    var = s2 * dt                                  # (2,)
    inv_2var = 0.5 / var
    norm_const = -0.5 * np.log(2.0 * np.pi * var)
    mu1 = mu_single(x_prev, theta['kappa'], theta['m']) * dt
    mu2 = mu_multi(x_prev, theta['alpha'], theta['beta'], theta['c']) * dt
    log_b = np.empty((len(dx), 2))
    log_b[:, 0] = norm_const[0] - inv_2var[0] * (dx - mu1) ** 2
    log_b[:, 1] = norm_const[1] - inv_2var[1] * (dx - mu2) ** 2
    return log_b


# ---------------------------------------------------------------------------
# GP label prior — soft emission tilt
# ---------------------------------------------------------------------------

def build_window_assignments(dt_t, labels_df):
    """
    Map each observation timestamp to the Phase A window index it falls in.
    Observations outside all windows get -1 (uniform p_label).

    Returns
    -------
    window_assignments : (T,) int array of window indices (or -1)
    p_label_pairs      : (W+1, 2) of [1-p_mw, p_mw] per window;
                         the last row is the uniform default [0.5, 0.5].
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


def augmented_log_b(log_b, window_assignments, p_label_pairs, eta=0.3):
    """
    Tilt emission log-likelihoods by the Phase A GP label likelihood:

        log b_tilde_s(t) = log b_s(t) + eta * log p_label(s | window_t)

    Window -1 (no Phase A coverage) maps to the uniform default row.
    """
    if eta == 0.0 or p_label_pairs is None:
        return log_b
    idx = window_assignments.copy()
    n_w = p_label_pairs.shape[0] - 1   # last row is uniform default
    idx = np.where(idx < 0, n_w, idx)
    return log_b + eta * np.log(np.clip(p_label_pairs[idx], 1e-6, 1.0 - 1e-6))


# ---------------------------------------------------------------------------
# Dwell-time Dirichlet prior on the transition matrix
# ---------------------------------------------------------------------------

def build_dwell_alpha_prior(
    n_observations: int,
    dt_seconds: float,
    target_dwell_seconds: float,
    prior_strength: float,
) -> np.ndarray:
    """
    Symmetric 2x2 Dirichlet pseudo-count matrix that pulls each row of P
    toward (1 - dt/D, dt/D) where D = target_dwell_seconds.

    Total prior mass per row is prior_strength * n_observations, so larger
    prior_strength enforces the target dwell more strictly.

    Returns alpha[i,j] with diag = N0*(1 - dt/D), off-diag = N0*(dt/D).
    """
    D_steps = max(target_dwell_seconds / dt_seconds, 1.0 + 1e-9)
    p_off = 1.0 / D_steps
    p_on = 1.0 - p_off
    N0 = max(prior_strength * float(n_observations), 1.0)
    alpha = np.array([[N0 * p_on, N0 * p_off],
                      [N0 * p_off, N0 * p_on]], dtype=float)
    return alpha


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


def m_step(x_prev, dx, dt, gamma, xi, theta, sigma2,
           alpha_prior_P=None, regime_specific_sigma2=False):
    """
    Update kappa, m, alpha, beta, c, P, and (optionally) sigma2.

    alpha_prior_P : optional (2,2) Dirichlet pseudo-counts added to xi_sum
                    before row normalisation. Lets us pull P toward a target
                    mean dwell time without disturbing the SDE M-steps.
    regime_specific_sigma2 : if True, sigma2 is returned as a 2-vector
                    [sigma2_0, sigma2_1] derived from gamma-weighted residual
                    SSR per state under the newly updated drift parameters.
                    If False, sigma2 is returned unchanged (held fixed at the
                    pre-EM data estimate).
    """
    rt = dx / dt
    w0 = gamma[:, 0]
    w1 = gamma[:, 1]

    a0, b0 = _wls_linear(x_prev, rt, w0)
    new_kappa = max(-b0, 1e-9)
    new_m = a0 / new_kappa

    new_alpha, new_beta, new_c = _wls_cubic(x_prev, rt, w1, theta)

    new_theta = {'kappa': new_kappa, 'm': new_m,
                 'alpha': new_alpha, 'beta': new_beta, 'c': new_c}

    if regime_specific_sigma2:
        # Per-state Gaussian MLE for the variance rate using the *updated*
        # drifts: sigma2_s = sum_t gamma_t[s] * (dx_t - mu_s*dt)^2 / (N_s * dt)
        mu1 = mu_single(x_prev, new_kappa, new_m) * dt
        mu2 = mu_multi(x_prev, new_alpha, new_beta, new_c) * dt
        resid0_sq = (dx - mu1) ** 2
        resid1_sq = (dx - mu2) ** 2
        N0 = max(float(w0.sum()), 1e-9)
        N1 = max(float(w1.sum()), 1e-9)
        SSR0 = float((w0 * resid0_sq).sum())
        SSR1 = float((w1 * resid1_sq).sum())
        new_sigma2 = np.array([SSR0 / (N0 * dt), SSR1 / (N1 * dt)], dtype=float)
        new_sigma2 = np.maximum(new_sigma2, 1e-30)
    else:
        new_sigma2 = sigma2

    xi_sum = xi.sum(axis=0)
    if alpha_prior_P is not None:
        xi_sum = xi_sum + np.asarray(alpha_prior_P, dtype=float)
    row_sum = xi_sum.sum(axis=1, keepdims=True)
    safe = np.where(row_sum == 0, 1.0, row_sum)
    P_new = xi_sum / safe
    zero_rows = (row_sum.flatten() == 0)
    if zero_rows.any():
        P_new[zero_rows] = 0.5

    return new_theta, P_new, new_sigma2


# ---------------------------------------------------------------------------
# EM driver
# ---------------------------------------------------------------------------

def run_em(
    x_prev, dx, dt, theta, P, pi0, sigma2,
    max_iter=50, tol=1e-4, console=None,
    alpha_prior_P=None,
    window_assignments=None, p_label_pairs=None, eta=0.0,
    regime_specific_sigma2=False,
):
    """
    EM with optional Dirichlet prior on P (mean-dwell enforcement) and a
    Phase A GP label tilt on the emission log-likelihood.

    alpha_prior_P : (2,2) Dirichlet pseudo-counts added to xi_sum in M-step.
    window_assignments, p_label_pairs, eta : if eta > 0, the emission
        log-likelihoods are tilted by eta * log p_multiwell at each step,
        pulling the smoothed gamma toward the GP prior within each window.
    regime_specific_sigma2 : if True, the M-step also re-estimates sigma2
        separately for each state and threads the (2,) vector through the
        next E-step.
    """
    console = console or Console()
    log_lik_trace = []
    prev_ll = -np.inf

    def _emit(theta_, sigma2_):
        lb = emission_log_b(x_prev, dx, dt, theta_, sigma2_)
        if eta > 0.0 and window_assignments is not None and p_label_pairs is not None:
            lb = augmented_log_b(lb, window_assignments, p_label_pairs, eta=eta)
        return lb

    for it in range(max_iter):
        log_b = _emit(theta, sigma2)
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

        theta, P, sigma2 = m_step(
            x_prev, dx, dt, gamma, xi, theta, sigma2,
            alpha_prior_P=alpha_prior_P,
            regime_specific_sigma2=regime_specific_sigma2,
        )
        prev_ll = log_lik

    # Final E-step with the most recent theta
    log_b = _emit(theta, sigma2)
    p_filt, log_lik, log_c = hamilton_filter(log_b, P, pi0)
    gamma, xi = kim_smoother(p_filt, log_b, log_c, P)
    log_lik_trace.append(log_lik)

    return {
        'theta': theta, 'P': P,
        'sigma2': sigma2,
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
    ['t', 'k', 'p_single_ahead', 'p_multi_ahead'].
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
                'p_multi_ahead': float(p_ahead[t, 1]),
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
    Rolling-window OU mean-reversion rate kappa over the Phase C observation
    pairs (x_prev, dx). A declining kappa(t) is the critical-slowing-down
    signature that precedes a single -> multi-well transition.

    Uses OLS of (dx/dt) on x_prev per window: intercept gives kappa*m,
    slope gives -kappa. The returned column ``kappa_ann`` is annualised
    (multiplied by SEC_PER_YEAR); per-second values are easy to derive but
    too small to read in plots.
    """
    r = dx / dt
    T = len(r)
    if T <= window_size:
        return pd.DataFrame(columns=['datetime', 'kappa_ann', 'kappa_m', 'n_obs'])

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
            'kappa_ann': float(kappa) * SEC_PER_YEAR,
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
    regime_specific_sigma2=False,
    kernel_half_width=5,
    trim_quantile=0.01,
    detrend=0,
    max_iter=50,
    tol=1e-4,
    output_dir=PHASE_C_OUTPUT_DIR,
    target_dwell_seconds=7 * 86400.0,
    dwell_prior_strength=10.0,
    eta_label=0.3,
    console=None,
):
    """
    Run Phase C end-to-end.

    Parameters
    ----------
    labels_csv : path to a Phase A regime labels CSV. Used for parametric
                 drift initialisation AND for the GP-derived p_multiwell
                 prior tilt on the emission likelihoods.
    pi_init    : (2,) initial-state prior. Default uniform [0.5, 0.5].
                 Pass markov_chain.run_markov_chain(...)['pi'] to use the
                 empirical stationary distribution.
    epsilon_P  : near-identity off-diagonal for the initial transition matrix.
    sigma2     : if None, estimated from the (already trimmed & detrended)
                 dx series as var(dx)/dt. Held fixed when
                 ``regime_specific_sigma2`` is False; otherwise used as the
                 initialisation and re-estimated per state each M-step.
    regime_specific_sigma2 : if True, the EM re-estimates one sigma2 per
                 state from gamma-weighted residual SSR. The final pair is
                 reported (annualised) and saved to the theta CSV.
    target_dwell_seconds : enforced mean dwell time (default 1 week). The
                 M-step adds a Dirichlet prior on each row of P with
                 P[i,i] ~ 1 - dt / target_dwell_seconds.
    dwell_prior_strength : prior mass = strength * len(dx). Larger values
                 pull P closer to the target dwell. Pass 0 to disable.
    eta_label  : strength of the Phase A GP p_multiwell tilt on the
                 emission log-likelihoods. Pass 0 to disable.
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

    # dx already reflects the trim_quantile + detrend pipeline (load_series
    # reads the cached aggregated returns CSV that bakes those in), so the
    # data-driven sigma2 below is computed *after* trim/detrend by construction.
    if sigma2 is None:
        s2_init = float(np.var(dx) / dt)
        if regime_specific_sigma2:
            sigma2 = np.array([s2_init, s2_init], dtype=float)
        else:
            sigma2 = s2_init
    else:
        if regime_specific_sigma2 and np.ndim(sigma2) == 0:
            sigma2 = np.array([float(sigma2), float(sigma2)], dtype=float)

    s_ann_init = sigma_annualized(sigma2)
    if regime_specific_sigma2:
        console.print(
            f'  sigma^2 (regime-specific, post trim/detrend) initial: '
            f'[{float(np.asarray(sigma2)[0]):.4e}, {float(np.asarray(sigma2)[1]):.4e}]'
        )
        console.print(
            f'  -> annualised sigma_init: '
            f'[{s_ann_init[0]:.4f}, {s_ann_init[1]:.4f}]'
        )
    else:
        console.print(
            f'  sigma^2 (held fixed, post trim/detrend) = {float(sigma2):.4e}  '
            f'-> annualised sigma = {s_ann_init[0]:.4f}'
        )

    theta = fit_initial_theta_moments(x_prev, dx, dt, console=console)
    theta_ann = annualize_theta(theta)
    console.print(
        f'  initial theta (annualised): '
        f'kappa_ann={theta_ann["kappa_ann"]:.4g}  m={theta_ann["m"]:.4g}  '
        f'alpha_ann={theta_ann["alpha_ann"]:.4g}  '
        f'beta_ann={theta_ann["beta_ann"]:.4g}  c={theta_ann["c"]:.4g}'
    )

    P0 = np.array([[1 - epsilon_P, epsilon_P],
                   [epsilon_P, 1 - epsilon_P]])

    if pi_init is None:
        pi_init = np.array([0.5, 0.5])
    pi_init = np.asarray(pi_init, dtype=float)
    pi_init = pi_init / pi_init.sum()

    # Dwell-time enforcement: Dirichlet prior on each row of P pulled toward
    # 1 - dt/D, with D = target_dwell_seconds. Strength scales with series
    # length so the prior holds its weight in long runs.
    alpha_prior_P = None
    if dwell_prior_strength > 0.0:
        alpha_prior_P = build_dwell_alpha_prior(
            n_observations=len(dx),
            dt_seconds=float(dt),
            target_dwell_seconds=float(target_dwell_seconds),
            prior_strength=float(dwell_prior_strength),
        )
        target_p_off = dt / target_dwell_seconds
        console.print(
            f'  dwell prior: D_target={target_dwell_seconds:.0f}s '
            f'(~{target_dwell_seconds / 86400:.1f}d), '
            f'P[i,i]_target={1 - target_p_off:.6f}, '
            f'mass={alpha_prior_P.sum():.3g}'
        )

    # GP label tilt: load Phase A windows (at this interval) and build
    # per-observation p_multiwell vectors used to tilt the emission log-b.
    window_assignments = None
    p_label_pairs = None
    if eta_label > 0.0:
        if not os.path.exists(labels_csv):
            raise FileNotFoundError(
                f'Phase A labels CSV not found at {labels_csv}. '
                'Run regime_estimation first, or pass eta_label=0 to skip the GP tilt.'
            )
        labels_df = pd.read_csv(
            labels_csv, parse_dates=['window_start', 'window_end'],
        )
        # Coerce seconds_interval to int so filtering survives CSVs where the
        # column was written as string/float by older pandas versions.
        labels_df['seconds_interval'] = pd.to_numeric(
            labels_df['seconds_interval'], errors='coerce'
        ).astype('Int64')
        if 'p_multiwell' not in labels_df.columns:
            raise ValueError(
                f'Phase A labels CSV {labels_csv} lacks the p_multiwell column. '
                'Re-run regime_estimation, or pass eta_label=0 to skip the GP tilt.'
            )
        sub = labels_df[labels_df['seconds_interval'] == int(seconds_interval)]
        sub = sub.drop_duplicates('window_start').reset_index(drop=True)
        if sub.empty:
            available = sorted(labels_df['seconds_interval'].dropna().unique().tolist())
            raise ValueError(
                f'No Phase A rows for seconds_interval={seconds_interval} in '
                f'{labels_csv} (available intervals: {available}). Re-run '
                'regime_estimation with this interval, or pass eta_label=0.'
            )
        if sub['p_multiwell'].isna().all():
            raise ValueError(
                f'Phase A p_multiwell values are all NaN for seconds_interval='
                f'{seconds_interval} in {labels_csv}. Re-run regime_estimation, '
                'or pass eta_label=0.'
            )
        window_assignments, p_label_pairs = build_window_assignments(dt_t, sub)
        covered = (window_assignments >= 0).mean()
        console.print(
            f'  GP label tilt: eta={eta_label}, '
            f'{covered:.1%} of observations covered by a Phase A window.'
        )

    out = run_em(
        x_prev, dx, dt, theta, P0, pi_init, sigma2,
        max_iter=max_iter, tol=tol, console=console,
        alpha_prior_P=alpha_prior_P,
        window_assignments=window_assignments,
        p_label_pairs=p_label_pairs,
        eta=eta_label,
        regime_specific_sigma2=regime_specific_sigma2,
    )
    sigma2 = out['sigma2']
    sigma2_arr = np.atleast_1d(np.asarray(sigma2, dtype=float))
    if sigma2_arr.size == 1:
        sigma2_arr = np.array([float(sigma2_arr[0]), float(sigma2_arr[0])])
    sigma_ann_arr = np.sqrt(sigma2_arr * SEC_PER_YEAR)

    stem = (
        f"phase_c_{pd.to_datetime(start_date).strftime('%Y-%m-%d')}_to_"
        f"{pd.to_datetime(end_date).strftime('%Y-%m-%d')}_{seconds_interval}s"
    )

    df_probs = pd.DataFrame({
        'datetime': pd.to_datetime(dt_t),
        'x_prev': x_prev,
        'dx': dx,
        'p_filt_single': out['p_filt'][:, 0],
        'p_filt_multi': out['p_filt'][:, 1],
        'gamma_single': out['gamma'][:, 0],
        'gamma_multi': out['gamma'][:, 1],
    })
    probs_path = os.path.join(output_dir, f'{stem}_probs.csv')
    df_probs.to_csv(probs_path, index=False)
    console.print(f'[green]Wrote[/green] {probs_path}')

    theta_ann_final = annualize_theta(out['theta'])
    theta_rows = [{'param': k, 'value': v} for k, v in out['theta'].items()]
    # Annualised echoes for human reading. Round-trip code (e.g. MCMC) should
    # read the raw rate parameters above; the *_ann rows are diagnostic.
    theta_rows += [
        {'param': 'kappa_ann', 'value': theta_ann_final['kappa_ann']},
        {'param': 'alpha_ann', 'value': theta_ann_final['alpha_ann']},
        {'param': 'beta_ann',  'value': theta_ann_final['beta_ann']},
    ]
    # sigma2: always emit a scalar 'sigma2' (mean across states when
    # regime-specific) for backwards-compat with phase_c_mcmc. Per-state
    # columns 'sigma2_0' / 'sigma2_1' are added regardless so the user can
    # see what came out.
    sigma2_scalar = float(sigma2_arr.mean())
    theta_rows += [
        {'param': 'sigma2',   'value': sigma2_scalar},
        {'param': 'sigma2_0', 'value': float(sigma2_arr[0])},
        {'param': 'sigma2_1', 'value': float(sigma2_arr[1])},
        {'param': 'sigma_ann_0', 'value': float(sigma_ann_arr[0])},
        {'param': 'sigma_ann_1', 'value': float(sigma_ann_arr[1])},
        {'param': 'regime_specific_sigma2', 'value': float(bool(regime_specific_sigma2))},
    ]
    theta_rows += [
        {'param': 'P_00', 'value': float(out['P'][0, 0])},
        {'param': 'P_01', 'value': float(out['P'][0, 1])},
        {'param': 'P_10', 'value': float(out['P'][1, 0])},
        {'param': 'P_11', 'value': float(out['P'][1, 1])},
    ]
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

    summary = Table(title='Phase C — Final parameters (annualised rates)')
    summary.add_column('Parameter', style='cyan')
    summary.add_column('Value', justify='right', style='green')
    summary.add_row('kappa_ann (yr⁻¹)',          f"{theta_ann_final['kappa_ann']:.6g}")
    summary.add_row('m (log-price)',             f"{theta_ann_final['m']:.6g}")
    summary.add_row('alpha_ann (yr⁻¹·logp⁻²)',   f"{theta_ann_final['alpha_ann']:.6g}")
    summary.add_row('beta_ann (yr⁻¹)',           f"{theta_ann_final['beta_ann']:.6g}")
    summary.add_row('c (log-price)',             f"{theta_ann_final['c']:.6g}")
    summary.add_row('P[0,0]', f"{out['P'][0, 0]:.6g}")
    summary.add_row('P[0,1]', f"{out['P'][0, 1]:.6g}")
    summary.add_row('P[1,0]', f"{out['P'][1, 0]:.6g}")
    summary.add_row('P[1,1]', f"{out['P'][1, 1]:.6g}")
    summary.add_row('log-lik', f"{out['log_lik']:.6g}")
    console.print(summary)

    sigma_table = Table(
        title=(
            'Phase C — Volatility (annualised)' +
            ('  [regime-specific]' if regime_specific_sigma2 else '  [shared]')
        )
    )
    sigma_table.add_column('State', style='cyan')
    sigma_table.add_column('sigma2 (per-second)', justify='right')
    sigma_table.add_column('sigma_annualised', justify='right', style='green')
    for i, s in enumerate(STATES):
        sigma_table.add_row(s, f'{sigma2_arr[i]:.6e}', f'{sigma_ann_arr[i]:.4f}')
    console.print(sigma_table)

    xi_sum = out['xi'].sum(axis=0)
    counts = Table(title='Phase C — Expected transition counts (sum_t xi_t)')
    counts.add_column('From \\ To', style='cyan')
    for s in STATES:
        counts.add_column(s, justify='right')
    for i, s_from in enumerate(STATES):
        counts.add_row(s_from, *[f'{xi_sum[i, j]:.2f}' for j in range(2)])
    console.print(counts)

    # Time-share per state: mean of smoothed gamma_t (also report filtered).
    gamma_share = out['gamma'].mean(axis=0)
    pfilt_share = out['p_filt'].mean(axis=0)
    share = Table(title='Phase C — Share of time in each state')
    share.add_column('State', style='cyan')
    share.add_column('Smoothed (gamma)', justify='right', style='green')
    share.add_column('Filtered (p_filt)', justify='right', style='yellow')
    for i, s in enumerate(STATES):
        share.add_row(s, f'{gamma_share[i]:.4f}', f'{pfilt_share[i]:.4f}')
    console.print(share)

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
    from markov_chain import run_markov_chain
    mc = run_markov_chain(labels_csv, seconds_interval=seconds_interval)
    pi_init = mc['pi'] if mc else None
    #pi_init = None
  
    run_phase_c(
        start_date, end_date, seconds_interval,
        labels_csv=labels_csv,
        pi_init=pi_init,   
        max_iter=30,
        tol=1e-4,
        console=_console,
    )
