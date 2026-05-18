# Level 3 — Sequential GP Potential Field: Implementation Instructions

## Context and goal

The current pipeline estimates U(x) independently per weekly window (Phase A)
and fits a parametric switching SDE (Phase C) with parameters fixed across the
full estimation period. The topology signal p_multiwell_w is a weekly scalar
that is computed but never used in Phase C.

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

```bash
pip install numpy scipy pandas rich scikit-learn matplotlib
```

All of these are already present. Confirm:

```python
import numpy as np
import scipy.linalg
import scipy.optimize
print('Dependencies OK')
```

-----

## Architecture overview

```
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
Phase C (`phase_c.py`) is NOT modified.

-----

## File: `phase_gp.py`

Create this file from scratch.

### Module docstring

```python
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
```

### Imports

```python
import os
import warnings
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import expm, solve, cholesky, cho_solve, cho_factor
from scipy.optimize import minimize
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from phase_c import load_series
from regime_estimation import _greedy_well_filter, normalize_window_boundaries
import plots
```

-----

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
```

-----

## Change 2 — Spatial RBF kernel

The spatial kernel is evaluated as a matrix at inducing points and at
query points. No special library needed — it is two lines of numpy.

```python
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
```

-----

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

    # ------------------------------------------------------------------
    # Hyperparameter optimisation
    # ------------------------------------------------------------------

    def optimise_hp(
        self,
        x_prev_subset,
        r_hat_subset,
        t_seconds_subset,
        n_restarts=3,
        console=None,
    ):
        """
        Optimise kernel hyperparameters by maximising the accumulated
        log marginal likelihood using scipy L-BFGS-B.

        Runs a fresh Kalman filter on a subset of the data for each
        candidate hyperparameter set. Tries n_restarts random initialisations
        and keeps the best result.

        Parameters
        ----------
        x_prev_subset    : (N,) observations to optimise on (e.g. first
                           month of data, or a downsampled subset)
        r_hat_subset     : (N,) corresponding EMA-demeaned drift
        t_seconds_subset : (N,) corresponding times
        n_restarts       : number of random restarts for the optimiser

        Updates self.spatial_ls, self.temporal_ls, self.spatial_var in-place.
        Re-runs initialise() with the optimised hyperparameters.
        Call this after the first few windows of data, before the main loop.
        """
        console = console or Console()
        x_range = (float(x_prev_subset.min()), float(x_prev_subset.max()))
        n_ind   = self.M if self.M is not None else 20

        def _neg_log_lik(log_params):
            sp_ls, tmp_ls, sp_var = np.exp(log_params)
            # Clamp to sensible range
            sp_ls  = np.clip(sp_ls,  1e-3, 2.0)
            tmp_ls = np.clip(tmp_ls, 0.5,  180.0)
            sp_var = np.clip(sp_var, 1e-6, 1e2)

            # Fresh model with candidate params
            m = KalmanGPDriftModel(
                spatial_lengthscale=sp_ls,
                temporal_lengthscale_days=tmp_ls,
                spatial_variance=sp_var,
                sigma2=self.sigma2,
                dt=self.dt,
            )
            m.initialise(x_range, n_inducing=n_ind)
            m.update(x_prev_subset, r_hat_subset, t_seconds_subset)
            return -m._log_lik   # minimise negative log likelihood

        best_nll  = np.inf
        best_pars = np.log([self.spatial_ls, self.temporal_ls, self.spatial_var])

        rng_hp = np.random.default_rng(0)
        for restart in range(n_restarts):
            if restart == 0:
                p0 = np.log([self.spatial_ls, self.temporal_ls, self.spatial_var])
            else:
                # Random initialisation
                p0 = np.log([
                    rng_hp.uniform(0.01, 0.5),    # spatial_ls
                    rng_hp.uniform(2.0, 60.0),    # temporal_ls
                    rng_hp.uniform(0.01, 10.0),   # spatial_var
                ])

            try:
                res = minimize(
                    _neg_log_lik, p0,
                    method='L-BFGS-B',
                    bounds=[
                        (np.log(1e-3), np.log(2.0)),     # spatial_ls
                        (np.log(0.5),  np.log(180.0)),   # temporal_ls
                        (np.log(1e-6), np.log(1e2)),     # spatial_var
                    ],
                    options={'maxiter': 100, 'ftol': 1e-6},
                )
                if res.fun < best_nll:
                    best_nll  = res.fun
                    best_pars = res.x
                console.print(
                    f'  [cyan]HP opt restart {restart+1}/{n_restarts}[/cyan]  '
                    f'nll={res.fun:.4e}  '
                    f'sp_ls={np.exp(res.x[0]):.4f}  '
                    f'tmp_ls={np.exp(res.x[1]):.2f}d  '
                    f'sp_var={np.exp(res.x[2]):.4f}'
                )
            except Exception as exc:
                console.print(f'  [yellow]Restart {restart+1} failed: {exc}[/yellow]')

        # Update hyperparameters and re-initialise state
        self.spatial_ls  = float(np.exp(best_pars[0]))
        self.temporal_ls = float(np.exp(best_pars[1]))
        self.spatial_var = float(np.exp(best_pars[2]))
        self.initialise(x_range, n_inducing=n_ind)

        console.print(
            f'[green]HP optimised:[/green] '
            f'spatial_ls={self.spatial_ls:.4f}  '
            f'temporal_ls={self.temporal_ls:.2f}d  '
            f'spatial_var={self.spatial_var:.4f}'
        )

    def get_params(self):
        """Return current hyperparameters as a dict for logging."""
        return {
            'spatial_lengthscale':       self.spatial_ls,
            'temporal_lengthscale_days': self.temporal_ls,
            'spatial_variance':          self.spatial_var,
            'obs_noise':                 self.obs_noise,
        }
```

-----

## Change 4 — EMA drift demeaning

```python
def ema_demean_drift(r, dt_t, dt, halflife_days=14.0):
    """
    Remove the slow-moving drift trend from scaled increments r = dx/dt
    using a causal exponential moving average (EMA).

    The EMA captures the price trend component of the drift. Subtracting
    it leaves only the restoring force structure (wells, barriers) that
    the GP should model.

    The halflife should be several times the OU mean-reversion timescale
    (1/kappa) so the EMA tracks genuine trend without absorbing within-
    regime mean-reversion. Two to four weeks is appropriate for ETH at
    30-second intervals.

    Parameters
    ----------
    r             : (N,) scaled increment array, r_t = dx_t / dt
    dt_t          : (N,) observation datetimes (numpy datetime64 or
                    anything pd.to_datetime accepts)
    dt            : float, nominal sampling interval in seconds.
    halflife_days : float, EMA halflife in days. Default 14.

    Returns
    -------
    r_hat : (N,) EMA-demeaned drift — what the GP observes as mu(x, t)
    r_bar : (N,) EMA trend estimate — the removed component

    Notes
    -----
    Uses pd.Series.ewm with times= and halflife= in time-unit strings
    so the halflife is wall-clock time, not observation count. This is
    critical at 30-second intervals where a '14-day halflife' should
    mean 14 days of real time, not 14 observations.
    """
    halflife_str = f'{halflife_days * 24 * 3600:.0f}s'
    r_series = pd.Series(r, index=pd.to_datetime(dt_t))
    r_bar_series = r_series.ewm(
        halflife=halflife_str,
        times=r_series.index,
        adjust=False,   # causal: no future data used
    ).mean()

    r_bar = r_bar_series.values
    r_hat = r - r_bar
    return r_hat, r_bar
```

-----

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
```

-----

## Change 6 — Topology forecast

With the Kalman-GP, forecasting the potential at a future time t+tau
means propagating the Kalman state forward by tau using the discrete
transition matrix A^k (where k = tau / dt_days steps). This gives
a predictive state with wider covariance — the posterior uncertainty
grows as the temporal kernel allows the potential to drift.

```python
def forecast_topology(
    model,
    forecast_horizons_days,
    x_range,
    n_grid=200,
    n_samples=200,
    min_crossing_sep=10,
    min_barrier_fraction=0.1,
    annualize=True,
    rng=None,
):
    """
    Forecast topology at future times by propagating the Kalman state.

    For each horizon tau, the Kalman state is propagated forward tau/dt
    steps using the transition matrix A_block. This widens the posterior
    covariance — uncertainty grows with horizon at a rate determined by
    the Matern temporal length scale. The longer the length scale, the
    slower the uncertainty grows, meaning longer-range forecasts are
    informative.

    Parameters
    ----------
    model                  : KalmanGPDriftModel at t_now
    forecast_horizons_days : list of float, e.g. [1.0, 3.0, 7.0, 14.0]
    x_range                : (2,) tuple, spatial evaluation range

    Returns
    -------
    DataFrame with columns:
        horizon_days, p_multiwell, barrier_mean, barrier_std,
        kramers_mean, kramers_std
    """
    rng   = rng or np.random.default_rng(42)
    rows  = []
    dt_days = model.dt / 86400.0

    for h in forecast_horizons_days:
        # Number of Kalman prediction steps for this horizon
        n_steps = max(1, int(round(h / dt_days)))

        # Propagate state forward n_steps steps (no observations)
        m_fwd = model.A_block  # start with one-step A
        # A^n_steps via repeated squaring
        A_n = np.linalg.matrix_power(model.A_block, n_steps)
        m_pred = A_n @ model.state_mean
        P_pred = model.state_cov.copy()
        for _ in range(n_steps):
            P_pred = model.A_block @ P_pred @ model.A_block.T + model.Q_block

        # Build a temporary model with the forecasted state
        fwd_model = KalmanGPDriftModel(
            spatial_lengthscale=model.spatial_ls,
            temporal_lengthscale_days=model.temporal_ls,
            spatial_variance=model.spatial_var,
            sigma2=model.sigma2,
            dt=model.dt,
        )
        fwd_model.inducing_x = model.inducing_x
        fwd_model.M          = model.M
        fwd_model.A_block    = model.A_block
        fwd_model.Q_block    = model.Q_block
        fwd_model._F         = model._F
        fwd_model._L         = model._L
        fwd_model._Qc        = model._Qc
        fwd_model._P_inf_1d  = model._P_inf_1d
        fwd_model.state_mean = m_pred
        fwd_model.state_cov  = P_pred

        topo = topology_from_gp(
            fwd_model, x_range,
            n_grid=n_grid, n_samples=n_samples,
            min_crossing_sep=min_crossing_sep,
            min_barrier_fraction=min_barrier_fraction,
            annualize=annualize,
            rng=rng,
        )

        rows.append({
            'horizon_days':  h,
            'p_multiwell':   topo['p_multiwell'],
            'barrier_mean':  topo['barrier_mean'],
            'barrier_std':   topo['barrier_std'],
            'kramers_mean':  topo['kramers_mean'],
            'kramers_std':   topo['kramers_std'],
        })

    return pd.DataFrame(rows)
```

-----

## Change 7 — Main entry point `run_phase_gp`

```python
def run_phase_gp(
    start_date,
    end_date,
    seconds_interval,
    labels_csv,
    sigma2=None,
    spatial_lengthscale=0.05,
    temporal_lengthscale_days=7.0,
    spatial_variance=1.0,
    ema_halflife_days=14.0,
    n_inducing=20,
    n_grid=200,
    n_samples=200,
    hp_opt_after_n_windows=2,
    hp_opt_n_restarts=3,
    forecast_horizons_days=(1.0, 3.0, 7.0, 14.0),
    topology_every_n_obs=500,
    kernel_half_width=5,
    trim_quantile=0.01,
    min_crossing_sep=10,
    min_barrier_fraction=0.1,
    output_dir='regime_results',
    seed=42,
    console=None,
):
    """
    Run the sequential Kalman-GP drift field model.

    Processes Phase A windows sequentially. Within each window, the full
    batch of observations is fed to the Kalman filter. Topology statistics
    are computed every topology_every_n_obs observations.

    Parameters
    ----------
    labels_csv               : Phase A labels CSV — used to read
                               p_multiwell_w for comparison only.
    sigma2                   : if None, estimated as var(dx)/dt.
    temporal_lengthscale_days: Matern length scale in days. Should match
                               Phase B mean dwell time. Will be refined
                               by HP optimisation.
    ema_halflife_days        : EMA halflife for drift demeaning. Default
                               14 days. Set detrend=0 in load_series —
                               EMA handles trend removal.
    hp_opt_after_n_windows   : run HP optimisation after this many windows
                               have been processed. 2 = after two weeks.
                               Uses the accumulated data as the optimisation
                               dataset.
    topology_every_n_obs     : compute topology every this many observations.
                               500 at 30s = ~4 hours.
    """
    os.makedirs(output_dir, exist_ok=True)
    console = console or Console()
    rng = np.random.default_rng(seed)

    # --- Load full series (no preprocessing detrend — EMA handles it) ---
    console.print(
        f'[cyan]Phase GP — loading series at {seconds_interval}s[/cyan]'
    )
    x_prev, dx, dt, dt_t = load_series(
        start_date, end_date, seconds_interval,
        kernel_half_width=kernel_half_width,
        trim_quantile=trim_quantile,
        detrend=0,   # EMA demeaning replaces preprocessing detrend
    )
    console.print(f'  {len(dx)} increments loaded.')

    r = dx / dt   # raw scaled increment (dx/dt)

    if sigma2 is None:
        sigma2 = float(np.var(dx) / dt)
    console.print(f'  sigma2          = {sigma2:.4e}')
    console.print(f'  obs_noise       = {sigma2/dt:.4e}  (sigma2/dt)')

    # --- EMA drift demeaning ---
    console.print(
        f'  EMA halflife    = {ema_halflife_days:.1f} days'
    )
    r_hat, r_bar = ema_demean_drift(r, dt_t, dt,
                                    halflife_days=ema_halflife_days)
    console.print(
        f'  drift mean before: {r.mean():.4e}  after: {r_hat.mean():.4e}'
    )

    t_seconds = (
        pd.to_datetime(dt_t).astype(np.int64) / 1e9
    ).values.astype(float)

    # --- x range for spatial grid ---
    x_min = float(np.percentile(x_prev, 1))
    x_max = float(np.percentile(x_prev, 99))
    x_range = (x_min, x_max)
    console.print(f'  x range         = [{x_min:.4f}, {x_max:.4f}]')

    # --- Build model ---
    model = KalmanGPDriftModel(
        spatial_lengthscale=spatial_lengthscale,
        temporal_lengthscale_days=temporal_lengthscale_days,
        spatial_variance=spatial_variance,
        sigma2=sigma2,
        dt=float(dt),
    )
    model.initialise(x_range=x_range, n_inducing=n_inducing)
    console.print(
        f'  Kalman state dim = {2 * n_inducing}  '
        f'(2 x {n_inducing} inducing points)'
    )

    # --- Load Phase A labels for comparison ---
    labels_df = pd.read_csv(
        labels_csv, parse_dates=['window_start', 'window_end']
    )
    labels_df = labels_df[
        labels_df['seconds_interval'] == seconds_interval
    ].copy()

    # Assign each observation to its Phase A window
    dt_series  = pd.Series(pd.to_datetime(dt_t))
    window_idx = np.full(len(dt_t), -1, dtype=int)
    for i, row in labels_df.iterrows():
        mask = (
            (dt_series >= row['window_start'])
            & (dt_series < row['window_end'])
        )
        window_idx[mask] = i

    # --- Sequential processing ---
    topology_rows = []
    forecast_rows = []
    n_windows     = labels_df.index.nunique()
    window_count  = 0

    # Accumulate observations for HP optimisation
    hp_x_accum   = []
    hp_r_accum   = []
    hp_t_accum   = []
    hp_done       = False

    with Progress(
        SpinnerColumn(),
        '[progress.description]{task.description}',
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task('Sequential Kalman-GP', total=len(dx))

        for w_idx in sorted(labels_df.index):
            obs_mask = (window_idx == w_idx)
            if not obs_mask.any():
                continue

            x_w     = x_prev[obs_mask]
            r_hat_w = r_hat[obs_mask]   # EMA-demeaned — NOT raw r
            t_w     = t_seconds[obs_mask]
            dt_w    = dt_t[obs_mask]

            # Accumulate for HP optimisation
            hp_x_accum.append(x_w)
            hp_r_accum.append(r_hat_w)
            hp_t_accum.append(t_w)
            window_count += 1

            # HP optimisation once after hp_opt_after_n_windows windows
            if (not hp_done
                    and window_count >= hp_opt_after_n_windows):
                console.print(
                    f'[yellow]Optimising hyperparameters after '
                    f'{window_count} windows...[/yellow]'
                )
                hp_x = np.concatenate(hp_x_accum)
                hp_r = np.concatenate(hp_r_accum)
                hp_t = np.concatenate(hp_t_accum)
                # Downsample to at most 5000 points for speed
                if len(hp_x) > 5000:
                    idx_sub = np.linspace(0, len(hp_x)-1, 5000, dtype=int)
                    hp_x, hp_r, hp_t = hp_x[idx_sub], hp_r[idx_sub], hp_t[idx_sub]
                model.optimise_hp(
                    hp_x, hp_r, hp_t,
                    n_restarts=hp_opt_n_restarts,
                    console=console,
                )
                hp_done = True

            # Feed this window's observations to the Kalman filter
            model.update(x_w, r_hat_w, t_w)

            # Topology at regular intervals within the window
            n_w      = int(obs_mask.sum())
            eval_idx = np.arange(topology_every_n_obs - 1, n_w,
                                 topology_every_n_obs)
            if len(eval_idx) == 0:
                eval_idx = np.array([n_w - 1])

            row = labels_df.loc[w_idx]
            p_multi_phase_a = float(row.get('p_multiwell', np.nan))

            for local_i in eval_idx:
                dt_query = pd.Timestamp(dt_w[local_i])
                topo = topology_from_gp(
                    model, x_range,
                    n_grid=n_grid, n_samples=n_samples,
                    min_crossing_sep=min_crossing_sep,
                    min_barrier_fraction=min_barrier_fraction,
                    rng=rng,
                )
                topology_rows.append({
                    'datetime':       dt_query,
                    'p_multiwell_gp': topo['p_multiwell'],
                    'p_multiwell_a':  p_multi_phase_a,
                    'mean_n_wells':   topo['mean_n_wells'],
                    'barrier_mean':   topo['barrier_mean'],
                    'barrier_std':    topo['barrier_std'],
                    'kramers':        topo['kramers_mean'],
                    'kramers_std':    topo['kramers_std'],
                    'u_range':        topo['u_range'],
                    'window_start':   str(row['window_start'].date()),
                })

            # Forecast from end of this window
            df_fc = forecast_topology(
                model,
                forecast_horizons_days=list(forecast_horizons_days),
                x_range=x_range,
                n_grid=n_grid, n_samples=n_samples,
                min_crossing_sep=min_crossing_sep,
                min_barrier_fraction=min_barrier_fraction,
                rng=rng,
            )
            df_fc['window_end'] = str(row['window_end'].date())
            forecast_rows.append(df_fc)

            prog.advance(task, advance=int(obs_mask.sum()))

    # --- Assemble and save outputs ---
    df_topology = pd.DataFrame(topology_rows)
    df_forecast = pd.concat(forecast_rows, ignore_index=True)
    df_params   = pd.DataFrame([model.get_params()])

    stem = (
        f"phase_gp_{pd.Timestamp(start_date).strftime('%Y-%m-%d')}_to_"
        f"{pd.Timestamp(end_date).strftime('%Y-%m-%d')}_{seconds_interval}s"
    )

    topo_path = os.path.join(output_dir, f'{stem}_topology.csv')
    df_topology.to_csv(topo_path, index=False)
    console.print(f'[green]Wrote[/green] {topo_path}')

    fc_path = os.path.join(output_dir, f'{stem}_forecast.csv')
    df_forecast.to_csv(fc_path, index=False)
    console.print(f'[green]Wrote[/green] {fc_path}')

    params_path = os.path.join(output_dir, f'{stem}_params.csv')
    df_params.to_csv(params_path, index=False)
    console.print(f'[green]Wrote[/green] {params_path}')

    plots.plot_gp_topology_series(
        df_topology,
        os.path.join(output_dir, f'{stem}_topology.png'),
    )
    plots.plot_gp_potential_snapshots(
        model, x_range,
        df_topology['datetime'].values,
        os.path.join(output_dir, f'{stem}_potential.png'),
        n_snapshots=6,
    )

    return {
        'model':       model,
        'df_topology': df_topology,
        'df_forecast': df_forecast,
        'df_params':   df_params,
        'r_hat':       r_hat,
        'r_bar':       r_bar,
    }
```

-----

## Change 8 — Standalone entry point

```python
if __name__ == '__main__':
    from datetime import datetime
    import glob
    from markov_chain import run_markov_chain
    from regime_estimation import normalize_window_boundaries

    start_date       = datetime(2024, 1, 1)
    end_date         = datetime(2024, 12, 31)
    seconds_interval = 30
    window_type      = 'weekly'

    start_date, end_date = normalize_window_boundaries(
        start_date, end_date, window_type
    )

    csvs = sorted(glob.glob(
        os.path.join('regime_results', f'regime_labels_*_{window_type}.csv')
    ))
    if not csvs:
        raise FileNotFoundError(
            'No regime_labels CSV found. Run regime_estimation.py first.'
        )
    labels_csv = csvs[-1]

    _console = Console()

    # Use Phase B dwell times to initialise temporal length scale
    mc = run_markov_chain(
        labels_csv,
        seconds_interval=seconds_interval,
        console=_console,
    )
    if mc is not None:
        mean_dwell_days = float(np.mean(mc['dwells'])) * 7.0
        mean_dwell_days = float(np.clip(mean_dwell_days, 3.0, 60.0))
    else:
        mean_dwell_days = 7.0
    _console.print(
        f'[cyan]Temporal length scale init: {mean_dwell_days:.1f} days[/cyan]'
    )

    run_phase_gp(
        start_date, end_date, seconds_interval,
        labels_csv=labels_csv,
        temporal_lengthscale_days=mean_dwell_days,
        ema_halflife_days=14.0,
        n_inducing=20,
        n_grid=200,
        n_samples=200,
        hp_opt_after_n_windows=2,
        hp_opt_n_restarts=3,
        topology_every_n_obs=500,
        forecast_horizons_days=(1.0, 3.0, 7.0, 14.0),
        console=_console,
    )
```

-----

## Change 9 — Plot stubs to add to `plots.py`

```python
def plot_gp_topology_series(df_topology, out_path):
    """
    Four-panel time series:
      1. p_multiwell_gp (sequential GP) vs p_multiwell_a (Phase A weekly)
      2. barrier_mean +/- 1 std shading
      3. kramers on log scale
      4. mean_n_wells
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    dt  = pd.to_datetime(df_topology['datetime'])
    fmt = mdates.AutoDateFormatter(mdates.AutoDateLocator())

    axes[0].plot(dt, df_topology['p_multiwell_gp'],
                 color='steelblue', linewidth=0.8, label='GP (sequential)')
    if 'p_multiwell_a' in df_topology.columns:
        axes[0].step(dt, df_topology['p_multiwell_a'],
                     color='tomato', linewidth=0.8,
                     alpha=0.6, label='Phase A (weekly)')
    axes[0].axhline(0.5, color='black', linewidth=0.5, linestyle=':')
    axes[0].set_ylabel('P(multi-well)')
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=7)

    axes[1].plot(dt, df_topology['barrier_mean'],
                 color='darkorange', linewidth=0.8)
    axes[1].fill_between(
        dt,
        df_topology['barrier_mean'] - df_topology['barrier_std'],
        df_topology['barrier_mean'] + df_topology['barrier_std'],
        alpha=0.3, color='darkorange',
    )
    axes[1].set_ylabel('Barrier height ΔU')

    axes[2].semilogy(dt, np.maximum(df_topology['kramers'], 1e-30),
                     color='purple', linewidth=0.8)
    axes[2].set_ylabel('Kramers rate Γ')

    axes[3].plot(dt, df_topology['mean_n_wells'],
                 color='green', linewidth=0.8)
    axes[3].axhline(1.5, color='black', linewidth=0.5, linestyle=':')
    axes[3].set_ylabel('Mean well count')
    axes[3].xaxis.set_major_formatter(fmt)

    fig.suptitle('Sequential Kalman-GP topology signal', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved {out_path}')


def plot_gp_potential_snapshots(model, x_range, datetimes, out_path,
                                n_snapshots=6):
    """
    Grid of U(x) snapshots. Because the Kalman-GP state encodes the
    posterior at the most recently seen observation, this function plots
    snapshots by replaying the model state at evenly spaced times
    stored in df_topology. Pass df_topology['datetime'].values as
    datetimes and note that each snapshot reflects the GP state *after*
    all observations up to that datetime.
    """
    import matplotlib.pyplot as plt

    idx = np.linspace(0, len(datetimes) - 1, n_snapshots, dtype=int)
    fig, axes = plt.subplots(2, n_snapshots // 2, figsize=(14, 6))
    axes = axes.flatten()

    x_grid = np.linspace(x_range[0], x_range[1], 200)
    # Note: model reflects the state at end of the full series.
    # To show snapshots at intermediate times, topology_rows must store
    # the Kalman state at each evaluation point, or snapshots must be
    # extracted during the main loop. For now, all panels show the
    # same final posterior mean — clearly label this in the title.
    mu_mean, _ = model.predict(x_grid, full_cov=False)
    U_mean = -cumulative_trapezoid(mu_mean, x_grid, initial=0.0)
    U_mean -= U_mean.min()

    for k, (ax, dt_q) in enumerate(
        zip(axes, pd.to_datetime(datetimes[idx]))
    ):
        ax.plot(x_grid, U_mean, color='steelblue', linewidth=1.2)
        ax.set_title(str(dt_q.date()), fontsize=8)
        ax.set_xlabel('log-price', fontsize=7)
        ax.set_ylabel('U(x)', fontsize=7)
        ax.tick_params(labelsize=6)

    fig.suptitle(
        'GP posterior mean potential U(x) — final state '
        '(extend run_phase_gp to store intermediate states for true snapshots)',
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved {out_path}')
```

**Note on snapshots:** Because the Kalman filter is sequential and the
state is updated in-place, true intermediate-time snapshots require
storing the state (state_mean, state_cov) at each topology evaluation
point. To implement this properly, add a `snapshots` list to the main
loop and append `(dt_query, model.state_mean.copy(), model.state_cov.copy())`
at each topology evaluation. Then `plot_gp_potential_snapshots` can
restore each saved state and call `predict` to get the correct U(x)
at that time. This is optional for initial validation.

-----

## Acceptance checks

**After Change 1 (SDE matrices):**

```python
F, L, Qc, H, P_inf = matern32_sde(lengthscale=7.0, sigma2=1.0)

# F eigenvalues must have negative real part (stable SDE)
eigs = np.linalg.eigvals(F)
assert np.all(eigs.real < 0), f'SDE not stable: {eigs}'

# P_inf must be positive definite (valid covariance)
assert np.all(np.linalg.eigvalsh(P_inf) > 0), 'P_inf not PD'

# Discrete transition must be contracting
A, Q = matern32_discrete(F, L, Qc, dt=1.0)
assert np.all(np.abs(np.linalg.eigvals(A)) < 1), 'A not contracting'
assert np.all(np.linalg.eigvalsh(Q) > 0), 'Q not PD'
print('Matern 3/2 SDE OK')
```

**After Change 2 (RBF kernel):**

```python
z = np.linspace(7.0, 8.5, 20)
K = rbf_kernel(z, z, lengthscale=0.05, variance=1.0)
assert K.shape == (20, 20)
assert np.all(np.linalg.eigvalsh(K) > -1e-9), 'RBF kernel not PSD'
# Diagonal must equal variance
np.testing.assert_allclose(np.diag(K), 1.0, rtol=1e-6)
print('RBF kernel OK')
```

**After Change 3 (KalmanGPDriftModel):**

```python
dt_val = 30.0
sigma2 = 1e-4
model  = KalmanGPDriftModel(
    spatial_lengthscale=0.05,
    temporal_lengthscale_days=7.0,
    spatial_variance=1.0,
    sigma2=sigma2,
    dt=dt_val,
)
model.initialise(x_range=(7.0, 8.5), n_inducing=10)

# Verify obs_noise = sigma2/dt
assert abs(model.obs_noise - sigma2 / dt_val) < 1e-14, \
    f'obs_noise wrong: {model.obs_noise} != {sigma2/dt_val}'

# State shapes
assert model.state_mean.shape == (20,)   # 2 * 10 inducing
assert model.state_cov.shape  == (20, 20)

# Feed synthetic EMA-demeaned observations
rng    = np.random.default_rng(0)
x_test = rng.uniform(7.0, 8.5, 200)
r_test = rng.normal(0, np.sqrt(sigma2 / dt_val), 200)
t_test = np.linspace(0, 86400.0 * 7, 200)
model.update(x_test, r_test, t_test)

# Predict
x_grid  = np.linspace(7.0, 8.5, 50)
mu_mean, mu_var = model.predict(x_grid, full_cov=False)
assert mu_mean.shape == (50,)
assert np.all(mu_var >= 0), 'Negative marginal variance'

# Full covariance
mu_mean2, K_post = model.predict(x_grid, full_cov=True)
assert K_post.shape == (50, 50)
assert np.all(np.linalg.eigvalsh(K_post) > -1e-6), 'K_post not PSD'
np.testing.assert_allclose(mu_mean, mu_mean2, rtol=1e-10)
print('KalmanGPDriftModel OK')
```

**After joint sampling:**

```python
samples = model.sample_drift(x_grid, n_samples=100, rng=rng)
assert samples.shape == (50, 100)

# Adjacent grid points must be more correlated than distant ones
corr_adj = np.corrcoef(samples[24], samples[25])[0, 1]
corr_far = np.corrcoef(samples[0],  samples[49])[0, 1]
assert corr_adj > corr_far, \
    f'Samples not spatially coherent: adj={corr_adj:.3f} far={corr_far:.3f}'
print(f'Joint sampling OK  adj_corr={corr_adj:.3f}  far_corr={corr_far:.3f}')
```

**After Change 4 (EMA demeaning):**

```python
N     = 10000
dt    = 30.0
times = pd.date_range('2024-01-01', periods=N, freq='30s')
r_ou  = rng.normal(0, 0.01, N)
trend = np.linspace(0, 0.001, N)
r_raw = r_ou + trend

r_hat, r_bar = ema_demean_drift(r_raw, times.values, dt, halflife_days=14.0)
assert abs(r_hat.mean()) < abs(r_raw.mean())

from scipy.stats import pearsonr
rho, _ = pearsonr(r_bar[300:], trend[300:])
assert rho > 0.8, f'EMA did not track trend: rho={rho:.3f}'
print('EMA demeaning OK')
```

**After Change 5 (topology_from_gp):**

```python
topo = topology_from_gp(model, x_range=(7.0, 8.5))
assert 0.0 <= topo['p_multiwell'] <= 1.0
assert topo['barrier_mean'] >= 0.0
assert topo['barrier_std']  >= 0.0
print('Topology extraction OK')
```

**After Change 6 (forecast_topology):**

```python
df_fc = forecast_topology(
    model,
    forecast_horizons_days=[1.0, 7.0, 14.0],
    x_range=(7.0, 8.5),
)
assert len(df_fc) == 3
# Uncertainty must not decrease with horizon
assert df_fc['barrier_std'].iloc[-1] >= df_fc['barrier_std'].iloc[0], \
    'Forecast uncertainty should grow with horizon'
print('Forecast OK')
```

**Full integration check after run_phase_gp:**

```python
out = run_phase_gp(...)
df  = out['df_topology']

assert df['p_multiwell_gp'].between(0, 1).all()
assert (df['barrier_mean'] >= 0).all()

# EMA check
assert abs(out['r_hat'].mean()) < abs(out['r_bar'].mean() +
    out['r_hat'].mean()), 'EMA did not reduce drift mean'

# Correlation with Phase A
from scipy.stats import spearmanr
rho, pval = spearmanr(
    df.groupby('window_start')['p_multiwell_gp'].mean(),
    df.groupby('window_start')['p_multiwell_a'].mean(),
)
print(f'Phase A vs GP rho = {rho:.3f}  p = {pval:.3f}')
assert rho > 0.2, 'GP and Phase A uncorrelated — check spatial_lengthscale'

print('Full integration check passed.')
```

-----

## Tuning notes

**`temporal_lengthscale_days`:** The most important parameter. Initialise
from Phase B mean dwell times. After HP optimisation, inspect the value:
below 2 days means the GP is chasing short-term noise; above 60 days means
it is too rigid to detect weekly transitions. HP optimisation with L-BFGS-B
and 3 restarts reliably finds a sensible value if the data is informative.

**`spatial_lengthscale`:** Controls how smooth the drift is across log-price.
0.05 (~5% log-price) is appropriate for ETH. If p_multiwell is always near
0 or 1 (no uncertainty), the spatial kernel is too smooth and the GP is
interpolating rather than inferring — reduce spatial_lengthscale. If
p_multiwell is always near 0.5 (maximum uncertainty), the kernel may be
too rough — increase spatial_lengthscale.

**`n_inducing`:** 20 is sufficient for a log-price range of ~1.5 units.
Increasing to 40 improves spatial resolution but doubles the Kalman state
dimension (cost scales as (2M)^2 per observation). Profile before increasing
beyond 30.

**`topology_every_n_obs`:** 500 at 30 seconds = ~4 hours. Each topology
evaluation draws n_samples joint samples (Cholesky of n_grid x n_grid
matrix) and integrates each to get a potential. At n_grid=200, n_samples=200,
this is ~0.1 seconds per evaluation on a modern CPU. Reducing to 100
observations gives ~hourly resolution at 4x the compute cost.

**`hp_opt_after_n_windows`:** HP optimisation runs once after this many
windows. 2 windows = 2 weeks of data, enough for a reasonable first
estimate. The optimisation uses L-BFGS-B on the accumulated log marginal
likelihood with n_restarts random initialisations. Runtime: a few seconds
for a 5000-point subset. Do not optimise at every window — the model is
sequential and early data should not be revisited repeatedly.

**`ema_halflife_days`:** Should be several multiples of 1/kappa (the OU
mean-reversion timescale from Phase C). For ETH with kappa ~1e-5/s,
1/kappa ~1.2 days, so a 14-day halflife is ~12x the mean-reversion
timescale. If the demeaned r_hat still shows a visible trend in a rolling
mean plot, shorten the halflife. If p_multiwell is uniformly near 0.5
everywhere, the halflife is too short and is absorbing mean-reversion — lengthen it.

**Memory:** The Kalman state is (2M, 2M) — with M=20 this is a 40x40
matrix, a fixed 12.8 KB regardless of how many observations have been
processed. Memory does not grow with T. The only memory cost that grows is
the topology_rows and forecast_rows lists accumulated during the loop —
these are O(T / topology_every_n_obs) rows, trivially small.