## Sequential GP Potential Field: Implementation Instructions

## Context and goal
"""
The current pipeline implements a per-window Gaussian process in Phase A to estimate the drift field mu(x) 
and potential landscape. This is a static snapshot approach: the GP is fit independently in each window
and does not leverage temporal continuity. 


This instruction implements Level 3: replace the per-window GP in Phase A with
a **sequential Gaussian process** that maintains a continuously updated
posterior over the drift field mu(x, t). The GP evolves over time via a
**state space (Kalman filter) representation**, which makes sequential updating
O(N) rather than O(N^3).

**What you get at the end:**

- `p_multiwell(t)` — a continuous signal updated at every evaluation point,
  not just weekly
- `barrier_height(t)` — posterior mean and variance of the potential barrier
- `kramers_rate(t)` — expected escape rate with uncertainty
- A forecast of each quantity at horizon tau that degrades gracefully with
  the GP temporal length scale rather than collapsing to a base rate

**Implementation:** Pure numpy/scipy. No JAX, no TensorFlow, no BayesNewton,
no objax. Works on any Python >= 3.10 with standard scientific stack.
The Matern 3/2 temporal kernel has an exact state space (SDE) representation
with a 2-dimensional state vector. The spatial RBF kernel is evaluated as a
standard kernel matrix at a fixed inducing point grid. The full model is a
Kalman filter whose observation matrix is the spatial kernel evaluated at the
current price location against the inducing grid.

-----

## Dependencies

No new dependencies beyond what is already in the project:


```python
"""
import numpy as np
import scipy.linalg
import scipy.optimize
print('Dependencies OK')

## Architecture overview

"""
phase_gp.py
    matern32_sde()          — closed-form Matern 3/2 SDE matrices
    rbf_kernel()            — RBF spatial kernel matrix
    KalmanGPDriftModel      — sequential GP via Kalman filter, pure numpy
        initialise()        — set inducing points and initial state
        update()            — one Kalman predict-update step per batch
        predict()           — posterior mean and full covariance at query points
        sample_drift()      — joint posterior samples (spatially coherent)
        optimise_hp()       — marginal likelihood maximisation via scipy
    ema_demean_drift()      — causal EMA drift demeaning
    topology_from_gp()      — p_multiwell, barrier, Kramers from GP posterior
    forecast_topology()     — topology forecast at future horizons
    run_phase_gp()          — main entry point

plots.py
    plot_gp_topology_series()
    plot_gp_potential_snapshots()
```

Phase A (`regime_estimation.py`) is NOT modified.
Phase B (`markov_chain.py`) is NOT modified.
"""

"""
Phase GP — Sequential Gaussian Process over the drift field mu(x, t).

Models the EMA-demeaned price drift as a Gaussian process in both
log-price space x and time t:

    r_hat_t = r_t - EMA(r_t, halflife=tau)     EMA drift demeaning
    r_hat_t ~ mu(x_{t-1}, t) + eps_t
    eps_t   ~ N(0, sigma2 / dt)                 Euler-Maruyama noise
    mu(x,t) ~ GP(0, k_rbf(x,x') * k_matern32(t,t'))

Implementation: pure numpy/scipy Kalman filter. No JAX or TensorFlow.

The temporal Matern 3/2 kernel has an exact stochastic differential
equation (SDE) representation with a 2-dimensional state vector. This
gives O(N) sequential inference via the Kalman filter — one predict-
update step per observation, with constant-size state regardless of
how much history has been seen.

The spatial RBF kernel is handled via a fixed grid of M inducing points
{z_1, ..., z_M} in log-price space. At each time step, the observation
matrix H maps the 2M-dimensional Kalman state (M inducing points, each
with a 2-d Matern state) to the observed drift at x_{t-1}.

The temporal kernel allows the potential shape to deform over time,
enabling genuine single-well <-> double-well transitions to be detected
and forecast continuously rather than only at weekly window boundaries.

Posterior samples are drawn from the JOINT GP posterior (full covariance
matrix over the spatial grid) so that spatially coherent drift curves
are produced. This suppresses spurious zero crossings from independent
marginal sampling at correlated adjacent grid points.

Outputs (under output_dir):
    phase_gp_<stem>_topology.csv     p_multiwell(t), barrier(t), kramers(t)
    phase_gp_<stem>_forecast.csv     topology forecasts at multiple horizons
    phase_gp_<stem>_params.csv       learned kernel hyperparameters
    phase_gp_<stem>_topology.png     topology signal time series
    phase_gp_<stem>_potential.png    U(x) snapshots at selected times
"""

### Imports

import os
import warnings
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import expm, solve, cholesky, cho_solve, cho_factor
from scipy.optimize import minimize
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from data_collection import load_series
from regime_estimation import _greedy_well_filter, normalize_window_boundaries
import plots


## Change 1 — Matern 3/2 SDE matrices

The Matern 3/2 kernel k(t, t’) = sigma2 * (1 + sqrt(3)|tau|/ell) *
exp(-sqrt(3)|tau|/ell) has an exact SDE representation:

```
df(t) = F f(t) dt + L dW(t)
```

with 2-dimensional state [f(t), f’(t)]. This gives exact O(N) inference
via the Kalman filter — no approximation needed.

```python
def matern32_sde(lengthscale, sigma2=1.0):
    """
    Return the SDE matrices for a Matern 3/2 GP in one dimension.

    The continuous-time SDE is:
        d[f, f'] = F [f, f']^T dt + L dW
        y        = H [f, f']^T + noise

    Parameters
    ----------
    lengthscale : float, temporal length scale in the same units as t.
                  If t is in days, lengthscale is in days.
    sigma2      : float, marginal variance of the GP. Default 1.0.

    Returns
    -------
    F     : (2, 2) drift matrix of the SDE
    L     : (2, 1) noise coupling vector
    Qc    : (1, 1) spectral density (scalar, wrapped in array)
    H     : (1, 2) observation matrix — selects the function value
    P_inf : (2, 2) stationary covariance (initial state covariance)

    Notes
    -----
    These matrices are derived in Solin & Sarkka (2014) "Explicit Link
    Between Periodic Covariance Functions and State Space Models."
    The Matern 3/2 SDE is exact — not an approximation.
    """
    lam = np.sqrt(3.0) / lengthscale

    F = np.array([
        [0.0,      1.0],
        [-lam**2, -2.0 * lam],
    ])

    L = np.array([[0.0], [1.0]])

    # Spectral density: Qc = 4 * sigma2 * lam^3
    Qc = np.array([[4.0 * sigma2 * lam**3]])

    # Observation matrix: observe the function value, not its derivative
    H = np.array([[1.0, 0.0]])

    # Stationary covariance P_inf (solution to the Lyapunov equation)
    P_inf = sigma2 * np.array([
        [1.0,   -lam],
        [-lam,   lam**2],
    ])

    return F, L, Qc, H, P_inf


def matern32_discrete(F, L, Qc, dt):
    """
    Discretise the Matern 3/2 SDE for a fixed time step dt.

    Returns A (transition matrix) and Q (process noise covariance)
    for the discrete recursion:
        f_{t+1} = A f_t + q_t,   q_t ~ N(0, Q)

    Uses the matrix exponential for A and the closed-form solution
    for Q = P_inf - A P_inf A^T.

    Parameters
    ----------
    F  : (2, 2) SDE drift matrix from matern32_sde()
    L  : (2, 1) SDE noise coupling from matern32_sde()
    Qc : (1, 1) spectral density from matern32_sde()
    dt : float, time step in the same units as the lengthscale

    Returns
    -------
    A : (2, 2) discrete transition matrix
    Q : (2, 2) discrete process noise covariance
    """
    A = expm(F * dt)
    # Process noise: Q = integral_0^dt A(dt-s) L Qc L^T A(dt-s)^T ds
    # For Matern 3/2, closed form: Q = P_inf - A P_inf A^T
    # where P_inf is the stationary covariance.
    # Re-derive P_inf from F, L, Qc via the discrete Lyapunov equation
    # to avoid passing it separately.
    # Numerically stable: use the augmented matrix approach.
    n = F.shape[0]
    Z = np.zeros((n, n))
    M = np.block([
        [-F,            L @ Qc @ L.T],
        [Z,             F.T         ],
    ])
    expM = expm(M * dt)
    Q = expM[n:, n:].T @ expM[:n, n:]
    return A, Q


## Change 2 — Spatial RBF kernel

The spatial kernel is evaluated as a matrix at inducing points and at
query points. No special library needed — it is two lines of numpy.

def rbf_kernel(x1, x2, lengthscale, variance=1.0):
    """
    RBF (squared exponential) kernel matrix.

    k(x, x') = variance * exp(-0.5 * (x - x')^2 / lengthscale^2)

    Parameters
    ----------
    x1          : (N,) or (N, 1) first set of points
    x2          : (M,) or (M, 1) second set of points
    lengthscale : float, spatial length scale in log-price units
    variance    : float, marginal variance

    Returns
    -------
    K : (N, M) kernel matrix
    """
    x1 = np.asarray(x1).reshape(-1, 1)
    x2 = np.asarray(x2).reshape(-1, 1)
    sq_dist = np.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
    return variance * np.exp(-0.5 * sq_dist / lengthscale ** 2)

## Change 3 — Kalman GP drift model

This is the core of the implementation. The state vector has shape
(2 * M,) where M is the number of spatial inducing points and 2 is the
Matern state dimension. Each inducing point carries its own independent
Matern temporal state [f_m(t), f_m’(t)].

The observation at (x_t, r_hat_t) connects to the state via:
r_hat_t = sum_m k_rbf(x_t, z_m) * f_m(t) + eps_t

where z_m are the inducing points and f_m(t) is the Matern state
evaluated at inducing point m.

```python
class KalmanGPDriftModel:
    """
    Sequential GP over the EMA-demeaned drift field mu(x, t).

    Implementation: Kalman filter in pure numpy. No JAX, no TensorFlow.

    State space structure
    ---------------------
    M spatial inducing points z_1, ..., z_M, each with a 2-d Matern
    temporal state [f_m(t), f_m'(t)]. Total state dimension: 2M.

    State layout (column-major Matern pairs):
        state = [f_1, f'_1, f_2, f'_2, ..., f_M, f'_M]   shape (2M,)

    Transition (block-diagonal, one Matern block per inducing point):
        state_{t+1} = A_block @ state_t + noise_t
        noise_t ~ N(0, Q_block)

    Observation at (x_t, r_hat_t):
        r_hat_t = H_t @ state_t + eps_t
        eps_t   ~ N(0, obs_noise)

    where H_t = [k(x_t, z_1), 0, k(x_t, z_2), 0, ..., k(x_t, z_M), 0]
    (spatial kernel weights interleaved with zeros for the derivative states).

    Prediction at query point (x_q, t_q):
        mu(x_q) = K(x_q, Z) @ K(Z, Z)^{-1} @ f_state
    where f_state = [f_1(t_q), f_2(t_q), ..., f_M(t_q)] extracted from
    the Kalman state.

    Usage
    -----
    model = KalmanGPDriftModel(
        spatial_lengthscale=0.05,
        temporal_lengthscale_days=7.0,
        spatial_variance=1.0,
        sigma2=1e-4,
        dt=30.0,
    )
    model.initialise(x_range=(7.0, 8.5), n_inducing=20)
    model.update(x_prev_batch, r_hat_batch, t_seconds_batch)
    mu_mean, K_post = model.predict(x_grid, full_cov=True)
    samples = model.sample_drift(x_grid, n_samples=200)
    """

    def __init__(
        self,
        spatial_lengthscale=0.05,
        temporal_lengthscale_days=7.0,
        spatial_variance=1.0,
        sigma2=1e-4,
        dt=30.0,
    ):
        """
        Parameters
        ----------
        spatial_lengthscale      : RBF length scale in log-price units.
                                   0.05 means ~5% log-price correlation width.
        temporal_lengthscale_days: Matern 3/2 length scale in days.
                                   Controls how fast the potential deforms.
                                   Initialise from Phase B mean dwell times.
        spatial_variance         : RBF marginal variance. Will be optimised.
        sigma2                   : Euler-Maruyama diffusion variance from
                                   Phase C (var(dx)/dt). Observation noise
                                   is set to sigma2/dt — the correct EM scale
                                   for the scaled increment r = dx/dt.
        dt                       : nominal sampling interval in seconds.
                                   Used to scale sigma2 to obs_noise and to
                                   discretise the Matern SDE.
        """
        self.spatial_ls  = spatial_lengthscale
        self.temporal_ls = temporal_lengthscale_days   # days
        self.spatial_var = spatial_variance
        self.sigma2      = sigma2
        self.dt          = dt
        self.obs_noise   = sigma2 / dt   # correct EM noise: sigma2/dt

        # Inducing points — set by initialise()
        self.inducing_x  = None   # (M,)
        self.M           = None   # number of inducing points

        # Kalman state — set by initialise(), updated by update()
        self.state_mean  = None   # (2M,)  current state mean
        self.state_cov   = None   # (2M, 2M) current state covariance

        # Discrete transition matrices — set by initialise()
        self.A_block     = None   # (2M, 2M) block-diagonal transition
        self.Q_block     = None   # (2M, 2M) block-diagonal process noise

        # Log of marginal likelihood accumulated for HP optimisation
        self._log_lik    = 0.0
        self._n_obs      = 0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialise(self, x_range, n_inducing=20):
        """
        Set up inducing points and initial Kalman state.

        Parameters
        ----------
        x_range    : (2,) tuple, (x_min, x_max) of observed log-prices.
        n_inducing : number of spatial inducing points. 20 is sufficient
                     for a log-price range of ~1.5 units. Increase to 40
                     for a wider range or longer spatial length scale.
        """
        self.inducing_x = np.linspace(x_range[0], x_range[1], n_inducing)
        self.M = n_inducing

        # Build discrete Matern matrices for nominal dt
        dt_days = self.dt / 86400.0
        F, L, Qc, H_1d, P_inf = matern32_sde(
            self.temporal_ls, sigma2=self.spatial_var,
        )
        A_1, Q_1 = matern32_discrete(F, L, Qc, dt_days)

        # Block-diagonal: M independent Matern processes
        # State layout: [f_1, f'_1, f_2, f'_2, ..., f_M, f'_M]
        self.A_block = np.kron(np.eye(self.M), A_1)   # (2M, 2M)
        self.Q_block = np.kron(np.eye(self.M), Q_1)   # (2M, 2M)

        # Initial state: zero mean, stationary covariance
        self.state_mean = np.zeros(2 * self.M)
        P_inf_block = np.kron(np.eye(self.M), P_inf)
        self.state_cov  = P_inf_block.copy()

        # Cache the 1d SDE matrices for reuse in variable-dt updates
        self._F    = F
        self._L    = L
        self._Qc   = Qc
        self._P_inf_1d = P_inf

        self._log_lik = 0.0
        self._n_obs   = 0

    # ------------------------------------------------------------------
    # Observation matrix
    # ------------------------------------------------------------------

def _obs_matrix(self, x_obs):
        """
        Build the observation matrix H for a single observation at x_obs.

        H shape: (1, 2M)
        H[0, 2m]   = k_rbf(x_obs, z_m)   (function value weight)
        H[0, 2m+1] = 0                    (derivative not observed)

        The observation equation is:
            r_hat = H @ state + eps,  eps ~ N(0, obs_noise)
        """
        k_vec = rbf_kernel(
            np.array([x_obs]),
            self.inducing_x,
            self.spatial_ls,
            self.spatial_var,
        ).flatten()   # (M,)

        H = np.zeros((1, 2 * self.M))
        H[0, 0::2] = k_vec   # every other element (function values)
        return H

    # ------------------------------------------------------------------
    # Sequential update
    # ------------------------------------------------------------------

def update(self, x_prev, r_hat, t_seconds):
        """
        Sequential Kalman update: incorporate a batch of EMA-demeaned
        drift observations. Processes observations in temporal order.

        Parameters
        ----------
        x_prev    : (N,) log-price at t-1 (absolute, not demeaned)
        r_hat     : (N,) EMA-demeaned scaled increment (dx/dt - EMA(dx/dt))
        t_seconds : (N,) observation times in seconds since epoch

        The state is updated in-place. Call this method once per window
        (batch update within a window is equivalent to sequential update
        because the observations are processed in order).
        """
        if self.state_mean is None:
            raise RuntimeError('Call initialise() before update().')

        dt_days = self.dt / 86400.0
        prev_t  = None

        for i in range(len(r_hat)):
            # --- Predict step ---
            # If time gap differs significantly from nominal dt (e.g. gap
            # between windows), recompute A and Q for the actual gap.
            if prev_t is not None:
                actual_dt_days = (t_seconds[i] - prev_t) / 86400.0
                if abs(actual_dt_days - dt_days) > 0.1 * dt_days:
                    A_i, Q_i = matern32_discrete(
                        self._F, self._L, self._Qc, actual_dt_days,
                    )
                    A_block_i = np.kron(np.eye(self.M), A_i)
                    Q_block_i = np.kron(np.eye(self.M), Q_i)
                else:
                    A_block_i = self.A_block
                    Q_block_i = self.Q_block
            else:
                A_block_i = self.A_block
                Q_block_i = self.Q_block

            m_pred = A_block_i @ self.state_mean
            P_pred = A_block_i @ self.state_cov @ A_block_i.T + Q_block_i

            # --- Update step ---
            H = self._obs_matrix(x_prev[i])   # (1, 2M)
            S = H @ P_pred @ H.T + self.obs_noise   # (1, 1) scalar
            K = P_pred @ H.T / S[0, 0]              # (2M, 1) Kalman gain
            innov = r_hat[i] - (H @ m_pred)[0]

            self.state_mean = m_pred + K[:, 0] * innov
            # Joseph form for numerical stability: P = (I - KH) P (I - KH)^T + K R K^T
            IKH = np.eye(2 * self.M) - K @ H
            self.state_cov = (IKH @ P_pred @ IKH.T
                              + self.obs_noise * K @ K.T)

            # Accumulate log marginal likelihood for HP optimisation
            # log N(innov; 0, S)
            self._log_lik += (
                -0.5 * np.log(2 * np.pi * S[0, 0])
                - 0.5 * innov ** 2 / S[0, 0]
            )
            self._n_obs  += 1
            prev_t = t_seconds[i]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, x_grid, full_cov=False):
        """
        Posterior mean and covariance of mu(x, t_now) at spatial grid.

        Extracts the function-value states [f_1, f_2, ..., f_M] from the
        current Kalman state and projects to the query grid via the RBF
        kernel.

        Parameters
        ----------
        x_grid   : (N_q,) query points in log-price space
        full_cov : if True, return full (N_q, N_q) posterior covariance.
                   If False, return marginal variances (N_q,).
                   full_cov=True is required for joint posterior sampling.

        Returns
        -------
        mu_mean : (N_q,) posterior mean drift at x_grid
        mu_cov  : (N_q,) marginal variances  [full_cov=False]
                  (N_q, N_q) full covariance [full_cov=True]
        """
        # Extract function-value states (every other element starting at 0)
        f_mean = self.state_mean[0::2]   # (M,)
        # Extract (M, M) sub-block of state_cov for function values
        idx = np.arange(0, 2 * self.M, 2)
        P_ff = self.state_cov[np.ix_(idx, idx)]   # (M, M)

        # Kernel matrices
        K_qz = rbf_kernel(x_grid, self.inducing_x,
                           self.spatial_ls, self.spatial_var)   # (N_q, M)
        K_zz = rbf_kernel(self.inducing_x, self.inducing_x,
                           self.spatial_ls, self.spatial_var)   # (M, M)

        # Add jitter to K_zz for numerical stability
        K_zz_jit = K_zz + 1e-6 * np.eye(self.M)

        # Posterior mean: K_qz @ K_zz^{-1} @ f_mean
        # Solve K_zz alpha = f_mean rather than explicit inverse
        alpha = solve(K_zz_jit, f_mean, assume_a='pos')   # (M,)
        mu_mean = K_qz @ alpha   # (N_q,)

        # Posterior covariance
        # V = K_zz^{-1} @ K_qz^T  (solution)
        V = solve(K_zz_jit, K_qz.T, assume_a='pos')   # (M, N_q)

        if full_cov:
            # Full posterior covariance
            # K_post = K_qq - K_qz K_zz^{-1} K_zq + K_qz K_zz^{-1} P_ff K_zz^{-1} K_zq
            K_qq = rbf_kernel(x_grid, x_grid,
                              self.spatial_ls, self.spatial_var)   # (N_q, N_q)
            mu_cov = (K_qq
                      - K_qz @ V
                      + K_qz @ solve(K_zz_jit, P_ff @ V, assume_a='pos'))
            # Add jitter for Cholesky stability downstream
            mu_cov += 1e-8 * np.eye(len(x_grid))
        else:
            # Marginal variances only (diagonal of full covariance)
            K_qq_diag = self.spatial_var * np.ones(len(x_grid))
            mu_cov = (K_qq_diag
                      - np.sum(K_qz * V.T, axis=1)
                      + np.sum((K_qz @ solve(K_zz_jit, P_ff,
                                             assume_a='pos')) * K_qz, axis=1))
            mu_cov = np.maximum(mu_cov, 0.0)

        return mu_mean, mu_cov

    # ------------------------------------------------------------------
    # Joint posterior sampling
    # ------------------------------------------------------------------

    def sample_drift(self, x_grid, n_samples=200, rng=None):
        """
        Draw spatially coherent samples from the joint GP posterior.

        Samples are drawn from N(mu_mean, K_post) where K_post is the
        full (N_q x N_q) posterior covariance. This ensures adjacent
        grid points move together as the spatial kernel dictates,
        suppressing spurious zero crossings from independent marginal
        sampling at correlated nearby points.

        Parameters
        ----------
        x_grid    : (N_q,) spatial query grid in log-price
        n_samples : number of joint posterior draws

        Returns
        -------
        samples : (N_q, n_samples) spatially coherent drift curves.
                  Each column is one draw from N(mu_mean, K_post).
        """
        rng = rng or np.random.default_rng(42)
        mu_mean, K_post = self.predict(x_grid, full_cov=True)

        # Cholesky decomposition for efficient sampling
        try:
            L_chol = cholesky(K_post, lower=True)
        except np.linalg.LinAlgError:
            # Fallback: add more jitter if initial jitter insufficient
            K_post += 1e-6 * np.eye(len(x_grid))
            L_chol = cholesky(K_post, lower=True)

        # Draw standard normal samples and transform
        # z shape: (N_q, n_samples)
        z = rng.standard_normal((len(x_grid), n_samples))
        samples = mu_mean[:, None] + L_chol @ z   # (N_q, n_samples)
        return samples        


## Change 5 — Topology extraction from GP posterior

```python
def topology_from_gp(
    model,
    x_range,
    n_grid=200,
    n_samples=200,
    min_crossing_sep=10,
    min_barrier_fraction=0.1,
    annualize=True,
    rng=None,
):
    """
    Compute topology statistics from the current GP posterior.

    The GP posterior is queried at the current Kalman state (t = now).
    No time argument is needed — the Kalman state already encodes the
    posterior conditioned on all observations seen so far.

    Parameters
    ----------
    model              : KalmanGPDriftModel with data up to t_now
    x_range            : (2,) tuple, (x_min, x_max) for spatial grid
    n_grid             : number of spatial grid points for integration
    n_samples          : number of joint posterior samples for p_multiwell
    min_crossing_sep   : minimum grid-point separation between zero
                         crossings (suppresses spurious wiggles). With
                         n_grid=200 spanning 1.5 log-price units,
                         min_crossing_sep=10 requires crossings to be
                         at least 0.075 log-price apart.
    min_barrier_fraction: well is kept only if its barrier height is at
                         least this fraction of total U range.
    annualize          : multiply drift by seconds-per-year before
                         integrating. Topology is unchanged; barrier
                         heights are in annualised units.

    Returns
    -------
    dict with keys:
        p_multiwell    P(n_wells >= 2 | data so far)
        mean_n_wells   posterior mean well count
        barrier_mean   posterior mean barrier height
        barrier_std    posterior std of barrier height
        kramers_mean   unscaled Kramers rate exp(-barrier/D), D=sigma2/2
        kramers_std    std of Kramers rate across samples
        well_locations from posterior mean potential
        u_range        total range of posterior mean potential
    """
    rng = rng or np.random.default_rng(42)
    x_grid = np.linspace(x_range[0], x_range[1], n_grid)

    # Posterior mean drift — used for well_locations and barriers
    mu_mean, _ = model.predict(x_grid, full_cov=False)

    # Joint posterior samples — spatially coherent
    f_samples = model.sample_drift(x_grid, n_samples=n_samples, rng=rng)
    # f_samples: (n_grid, n_samples)

    sec_per_year = 365.25 * 24 * 3600
    if annualize:
        mu_mean   = mu_mean   * sec_per_year
        f_samples = f_samples * sec_per_year

    # --- Zero crossing count per sample ---
    n_stable = np.empty(n_samples, dtype=int)
    for j in range(n_samples):
        cross_idx = np.where(np.diff(np.sign(f_samples[:, j])) < 0)[0]
        if len(cross_idx) > 1 and min_crossing_sep > 0:
            gaps = np.diff(cross_idx)
            keep = np.concatenate([[True], gaps >= min_crossing_sep])
            cross_idx = cross_idx[keep]
        n_stable[j] = len(cross_idx)

    p_multiwell  = float(np.mean(n_stable >= 2))
    mean_n_wells = float(np.mean(n_stable))

    # --- Posterior mean potential (one integration of the mean drift) ---
    U_mean = -cumulative_trapezoid(mu_mean, x_grid, initial=0.0)
    U_mean -= U_mean.min()
    u_range = float(U_mean.max())

    if u_range < 1e-12:
        return {
            'p_multiwell':  round(p_multiwell, 4),
            'mean_n_wells': round(mean_n_wells, 2),
            'barrier_mean': 0.0, 'barrier_std': 0.0,
            'kramers_mean': 0.0, 'kramers_std': 0.0,
            'well_locations': [], 'u_range': 0.0,
        }

    kept_idx, barriers = _greedy_well_filter(
        x_grid, U_mean,
        threshold=min_barrier_fraction * u_range,
        min_well_separation=0.0,
    )
    well_locations = [float(x_grid[i]) for i in kept_idx]

    # --- Barrier height distribution across samples ---
    barrier_samples = np.empty(n_samples)
    for j in range(n_samples):
        U_j = -cumulative_trapezoid(f_samples[:, j], x_grid, initial=0.0)
        U_j -= U_j.min()
        u_range_j = float(U_j.max())
        if u_range_j < 1e-9:
            barrier_samples[j] = 0.0
            continue
        _, bars_j = _greedy_well_filter(
            x_grid, U_j,
            threshold=min_barrier_fraction * u_range_j,
            min_well_separation=0.0,
        )
        barrier_samples[j] = max(bars_j) if bars_j else 0.0

    barrier_mean = float(np.mean(barrier_samples))
    barrier_std  = float(np.std(barrier_samples))

    # --- Kramers rate: exp(-barrier / D), D = sigma2/2 ---
    D = model.sigma2 / 2.0
    if D > 0:
        kramers_samples = np.exp(-barrier_samples / D)
    else:
        kramers_samples = np.zeros(n_samples)
    kramers_mean = float(np.mean(kramers_samples))
    kramers_std  = float(np.std(kramers_samples))

    return {
        'p_multiwell':   round(p_multiwell, 4),
        'mean_n_wells':  round(mean_n_wells, 2),
        'barrier_mean':  round(barrier_mean, 6),
        'barrier_std':   round(barrier_std, 6),
        'kramers_mean':  round(kramers_mean, 8),
        'kramers_std':   round(kramers_std, 8),
        'well_locations': well_locations,
        'u_range':        round(u_range, 4),
    }            


