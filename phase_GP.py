"""
Phase GP — Sequential Gaussian Process over the drift field mu(x, t).

This is the new Phase C of the pipeline. It replaces the per-window static
GP/EM/MCMC approach with a single sequential Kalman-filter GP whose state
is continuously updated as new observations arrive. The GP factors as a
spatial RBF kernel over log-price times an exact Matern 3/2 state-space
representation in time, giving O(N) inference and a constant-size state.

    r_t      = dx_t / dt                                 scaled increment
    r_hat_t  = r_t - EMA(r_t, halflife=tau_ema)          causal EMA demean
    r_hat_t ~ mu(x_{t-1}, t) + eps_t
    eps_t   ~ N(0, sigma2 / dt)                          Euler-Maruyama noise
    mu(x,t) ~ GP(0, k_rbf(x, x') * k_matern32(t, t'))

Outputs (under output_dir):
    phase_gp_<stem>_topology.csv     p_multiwell(t), barrier(t), kramers(t)
    phase_gp_<stem>_forecast.csv     topology forecasts at multiple horizons
    phase_gp_<stem>_params.csv       learned kernel hyperparameters
    phase_gp_<stem>_topology.png     topology signal time series
    phase_gp_<stem>_potential.png    U(x) snapshots at selected times
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import expm, solve, cholesky
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

import data_collection as dc
from data_collection import load_series, ema_demean_drift
from regime_estimation import (
    _greedy_well_filter,
    normalize_window_boundaries,
    run_phase_a,
)
from markov_chain import run_markov_chain
import plots


# ---------------------------------------------------------------------------
# Matern 3/2 state-space matrices
# ---------------------------------------------------------------------------

def matern32_sde(lengthscale, sigma2=1.0):
    """
    Exact state-space form of a Matern 3/2 GP in 1D.

    Returns F (2,2), L (2,1), Qc (1,1), H (1,2), P_inf (2,2) — the SDE
    drift, noise coupling, spectral density, observation matrix, and
    stationary covariance.
    """
    lam = np.sqrt(3.0) / lengthscale

    F = np.array([
        [0.0,      1.0],
        [-lam**2, -2.0 * lam],
    ])
    L = np.array([[0.0], [1.0]])
    Qc = np.array([[4.0 * sigma2 * lam**3]])
    H = np.array([[1.0, 0.0]])
    # Stationary covariance: solution of F P + P Fᵀ + L Qc Lᵀ = 0.
    P_inf = sigma2 * np.array([
        [1.0,   0.0],
        [0.0,   lam**2],
    ])
    return F, L, Qc, H, P_inf


def matern32_discrete(F, L, Qc, dt):
    """
    Discretise the Matern 3/2 SDE for a fixed step dt.

    Returns A (2,2) transition and Q (2,2) process noise covariance.
    """
    A = expm(F * dt)
    n = F.shape[0]
    Z = np.zeros((n, n))
    M = np.block([
        [-F,            L @ Qc @ L.T],
        [Z,             F.T         ],
    ])
    expM = expm(M * dt)
    Q = expM[n:, n:].T @ expM[:n, n:]
    return A, Q


# ---------------------------------------------------------------------------
# Spatial RBF kernel
# ---------------------------------------------------------------------------

def rbf_kernel(x1, x2, lengthscale, variance=1.0):
    """RBF (squared exponential) kernel matrix k(x, x') = variance * exp(-d^2 / 2 ell^2)."""
    x1 = np.asarray(x1).reshape(-1, 1)
    x2 = np.asarray(x2).reshape(-1, 1)
    sq_dist = np.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
    return variance * np.exp(-0.5 * sq_dist / lengthscale ** 2)


# ---------------------------------------------------------------------------
# Kalman GP drift model
# ---------------------------------------------------------------------------

class KalmanGPDriftModel:
    """
    Sequential GP over the EMA-demeaned drift field mu(x, t).

    State layout (one Matern block per inducing point):
        state = [f_1, f'_1, ..., f_M, f'_M]   shape (2M,)
    """

    def __init__(
        self,
        spatial_lengthscale=0.05,
        temporal_lengthscale_days=7.0,
        spatial_variance=1.0,
        sigma2=1e-4,
        dt=30.0,
    ):
        self.spatial_ls = spatial_lengthscale
        self.temporal_ls = temporal_lengthscale_days
        self.spatial_var = spatial_variance
        self.sigma2 = sigma2
        self.dt = dt
        self.obs_noise = sigma2 / dt

        self.inducing_x = None
        self.M = None

        self.state_mean = None
        self.state_cov = None

        self.A_block = None
        self.Q_block = None
        self._K_zz_inv = None
        self._K_zz_jit = None
        self._I_2M = None

        self._log_lik = 0.0
        self._n_obs = 0

    def _recompute_hp_dependent(self):
        """Refresh HP-dependent caches (K_zz_inv, A_block, Q_block, SDE matrices)
        without touching state_mean / state_cov. Used by HP optimisation to swap
        in new kernel parameters while preserving the accumulated Kalman state.
        """
        K_zz = rbf_kernel(self.inducing_x, self.inducing_x,
                          self.spatial_ls, self.spatial_var)
        self._K_zz_jit = K_zz + 1e-6 * np.eye(self.M)
        self._K_zz_inv = solve(self._K_zz_jit, np.eye(self.M), assume_a='pos')

        dt_days = self.dt / 86400.0
        F, L, Qc, _H_1d, P_inf = matern32_sde(
            self.temporal_ls, sigma2=self.spatial_var,
        )
        A_1, Q_1 = matern32_discrete(F, L, Qc, dt_days)
        eye_M = np.eye(self.M)
        self.A_block = np.kron(eye_M, A_1)
        self.Q_block = np.kron(eye_M, Q_1)
        self._F = F
        self._L = L
        self._Qc = Qc
        self._P_inf_1d = P_inf

    def initialise(self, x_range, n_inducing=20, data_x=None):
        """
        Place inducing points across x_range. If `data_x` is supplied, drop
        inducing points outside the observed [min, max] of `data_x` so empty
        regions of input space do not waste filter capacity.
        """
        inducing = np.linspace(x_range[0], x_range[1], n_inducing)
        if data_x is not None and len(data_x) > 0:
            d_lo, d_hi = float(np.min(data_x)), float(np.max(data_x))
            keep = (inducing >= d_lo) & (inducing <= d_hi)
            inducing = inducing[keep]
            if len(inducing) < 2:
                inducing = np.linspace(d_lo, d_hi, max(2, n_inducing))
        self.inducing_x = inducing
        self.M = len(self.inducing_x)

        self._recompute_hp_dependent()
        self._I_2M = np.eye(2 * self.M)

        # P_inf: f block = K_zz (GP spatial prior), f' block = lam² × K_zz.
        # predict() uses K_qq - K_qz @ K_ZZ^{-1} @ K_qz.T + posterior term,
        # which recovers K(x,x) only when P_ff starts at K_ZZ.
        lam_sq = self._P_inf_1d[1, 1] / self._P_inf_1d[0, 0]
        idx_f  = np.arange(0, 2 * self.M, 2)
        idx_fp = np.arange(1, 2 * self.M, 2)
        P_inf_block = np.zeros((2 * self.M, 2 * self.M))
        P_inf_block[np.ix_(idx_f,  idx_f )] = self._K_zz_jit
        P_inf_block[np.ix_(idx_fp, idx_fp)] = lam_sq * self._K_zz_jit

        self.state_mean = np.zeros(2 * self.M)
        self.state_cov = P_inf_block.copy()

        self._log_lik = 0.0
        self._n_obs = 0

    def _obs_matrix(self, x_obs):
        k_vec = rbf_kernel(
            np.array([x_obs]),
            self.inducing_x,
            self.spatial_ls,
            self.spatial_var,
        ).flatten()
        # H = K_{xZ} @ K_{ZZ}^{-1}: consistent with predict(), which computes
        # mu(x) = K_{xZ} @ K_{ZZ}^{-1} @ f.  Using raw K_{xZ} here inflates
        # the effective observation magnitude relative to what predict() reads
        # back, causing over-aggressive Kalman gain and eventual FP instability.
        H = np.zeros((1, 2 * self.M))
        H[0, 0::2] = self._K_zz_inv @ k_vec
        return H

    def update(self, x_prev, r_hat, t_seconds):
        """Sequential Kalman update over a batch of observations in temporal order."""
        if self.state_mean is None:
            raise RuntimeError('Call initialise() before update().')

        dt_days = self.dt / 86400.0
        prev_t = None

        for i in range(len(r_hat)):
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

            if not (np.isfinite(m_pred).all() and np.isfinite(P_pred).all()):
                warnings.warn(
                    f'KalmanGP: state overflow at observation {i} '
                    f'(max |m_pred|={np.max(np.abs(m_pred)):.3e}, '
                    f'max |P_pred|={np.max(np.abs(P_pred)):.3e}). '
                    'Aborting update, log_lik set to -inf.',
                    RuntimeWarning, stacklevel=2,
                )
                self._log_lik = -np.inf
                break

            H = self._obs_matrix(x_prev[i])
            S = H @ P_pred @ H.T + self.obs_noise
            raw_s = float(S[0, 0])
            if raw_s < 1e-15:
                warnings.warn(
                    f'KalmanGP: innovation variance S={raw_s:.3e} at observation {i} '
                    f'(obs_noise={self.obs_noise:.3e}). Flooring to 1e-15. '
                    'P_pred may have lost positive-definiteness.',
                    RuntimeWarning, stacklevel=2,
                )
            s_val = max(raw_s, 1e-15)
            K = P_pred @ H.T / s_val
            innov = r_hat[i] - (H @ m_pred)[0]

            self.state_mean = m_pred + K[:, 0] * innov
            IKH = self._I_2M - K @ H
            P_post = IKH @ P_pred @ IKH.T + self.obs_noise * K @ K.T
            P_sym = (P_post + P_post.T) * 0.5
            asym = np.max(np.abs(P_post - P_sym))
            if asym > 1e-8 * np.max(np.abs(P_post)):
                warnings.warn(
                    f'KalmanGP: P asymmetry {asym:.3e} at observation {i}. Symmetrising.',
                    RuntimeWarning, stacklevel=2,
                )
            self.state_cov = P_sym

            self._log_lik += (
                -0.5 * np.log(2 * np.pi * s_val)
                - 0.5 * innov ** 2 / s_val
            )
            self._n_obs += 1
            prev_t = t_seconds[i]

    def predict(self, x_grid, full_cov=False):
        """Posterior mean and covariance of mu(x, t_now) at x_grid."""
        f_mean = self.state_mean[0::2]
        idx = np.arange(0, 2 * self.M, 2)
        P_ff = self.state_cov[np.ix_(idx, idx)]

        K_qz = rbf_kernel(x_grid, self.inducing_x,
                          self.spatial_ls, self.spatial_var)

        alpha = self._K_zz_inv @ f_mean
        mu_mean = K_qz @ alpha

        V = self._K_zz_inv @ K_qz.T

        if full_cov:
            K_qq = rbf_kernel(x_grid, x_grid,
                              self.spatial_ls, self.spatial_var)
            mu_cov = (K_qq
                      - K_qz @ V
                      + K_qz @ (self._K_zz_inv @ P_ff @ V))
            # Jitter proportional to prior variance: robust when K_qq is nearly
            # singular (dense grid relative to small spatial_ls).
            mu_cov += max(1e-6 * self.spatial_var, 1e-10) * np.eye(len(x_grid))
        else:
            K_qq_diag = self.spatial_var * np.ones(len(x_grid))
            mu_cov = (K_qq_diag
                      - np.sum(K_qz * V.T, axis=1)
                      + np.sum((K_qz @ self._K_zz_inv @ P_ff) * K_qz, axis=1))
            mu_cov = np.maximum(mu_cov, 0.0)

        return mu_mean, mu_cov

    def sample_drift(self, x_grid, n_samples=200, rng=None):
        """Joint posterior samples from N(mu_mean, K_post), shape (N_q, n_samples)."""
        rng = rng or np.random.default_rng(42)
        mu_mean, K_post = self.predict(x_grid, full_cov=True)

        # Geometric jitter ladder: K_post can become ill-conditioned when the
        # spatial_var/spatial_ls ratio is large or before HP optimisation runs.
        jitter = max(1e-8 * self.spatial_var, 1e-12)
        L_chol = None
        for _ in range(12):
            try:
                L_chol = cholesky(K_post + jitter * np.eye(len(x_grid)), lower=True)
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        if L_chol is None:
            raise np.linalg.LinAlgError(
                'sample_drift Cholesky failed — K_post not PD even with large jitter.'
            )

        z = rng.standard_normal((len(x_grid), n_samples))
        return mu_mean[:, None] + L_chol @ z

    def optimise_hp_gpflow(
        self,
        x_prev_subset,
        r_hat_subset,
        t_seconds_subset,
        console=None,
    ):
        """
        Alternative HP optimisation using GPflow SGPR with a separable
        SquaredExponential(x) × Matern32(t) kernel.

        The temporal input is in days; spatial input is log-price.
        Likelihood variance is fixed to sigma2/dt (known from Euler-Maruyama).
        """
        try:
            import gpflow
        except ImportError:
            raise ImportError('gpflow is required for this method: pip install gpflow')

        console = console or Console()

        t_days = t_seconds_subset / 86400.0
        X = np.column_stack([x_prev_subset, t_days]).astype(np.float64)
        Y_raw = r_hat_subset.reshape(-1, 1).astype(np.float64)

        if len(X) > 5000:
            idx = np.linspace(0, len(X) - 1, 5000, dtype=int)
            X, Y_raw = X[idx], Y_raw[idx]

        # Normalise targets to O(1) so kernel/noise variances are numerically
        # well-conditioned in GPflow's Softplus transform.
        y_scale = float(np.std(Y_raw)) or 1.0
        Y = Y_raw / y_scale

        x_ind = np.linspace(X[:, 0].min(), X[:, 0].max(), 10)
        t_ind = np.linspace(X[:, 1].min(), X[:, 1].max(), 5)
        xx, tt = np.meshgrid(x_ind, t_ind)
        Z = np.column_stack([xx.ravel(), tt.ravel()]).astype(np.float64)

        k_spatial = gpflow.kernels.SquaredExponential(
            variance=1.0,  # in normalised-Y space; rescaled back after opt
            lengthscales=float(self.spatial_ls),
            active_dims=[0],
        )
        k_temporal = gpflow.kernels.Matern32(
            variance=1.0,
            lengthscales=float(self.temporal_ls),
            active_dims=[1],
        )
        gpflow.set_trainable(k_temporal.variance, False)

        gp = gpflow.models.SGPR(
            data=(X, Y),
            kernel=k_spatial * k_temporal,
            inducing_variable=Z,
        )

        gpflow.optimizers.Scipy().minimize(
            gp.training_loss,
            gp.trainable_variables,
            options={'maxiter': 200},
        )

        new_spatial_ls  = float(k_spatial.lengthscales.numpy())
        new_temporal_ls = float(k_temporal.lengthscales.numpy())
        # Unscale kernel variance back to original units
        new_spatial_var = float(k_spatial.variance.numpy()) * y_scale**2

        self.spatial_ls  = float(np.clip(new_spatial_ls,  1e-3, 0.1))
        self.temporal_ls = float(np.clip(new_temporal_ls, 0.5,  180.0))
        # spatial_var lower clip relaxed to 1e-12: per-second drift is O(1e-8),
        # so a meaningful prior on var(mu) lives near 1e-16 in per-second units.
        # The old 1e-6 floor left HP opt with a ~10^10-too-wide prior, collapsing
        # posterior mean toward 0 and inflating sigma/mu ratio.
        self.spatial_var = float(np.clip(new_spatial_var, 1e-12, 1e2))

        # Refresh HP-dependent caches in-place; keep accumulated state_mean /
        # state_cov from the pre-HP-opt Kalman filter rather than resetting.
        self._recompute_hp_dependent()

        console.print(
            f'[green]GPflow HP optimised:[/green] '
            f'spatial_ls={self.spatial_ls:.4f}  '
            f'temporal_ls={self.temporal_ls:.2f}d  '
            f'spatial_var={self.spatial_var:.4f}'
        )

    def optimise_hp(
        self,
        x_prev_subset,
        r_hat_subset,
        t_seconds_subset,
        n_restarts=3,
        method='scipy',
        x_range=None,
        console=None,
    ):
        """Maximise the marginal log-likelihood over kernel HPs.

        method='scipy'  — multi-restart L-BFGS-B (default, no extra deps)
        method='gpflow' — GPflow SGPR with separable kernel (requires gpflow)
        x_range — inducing-point bracket (typically the full series percentile
                  range); falls back to the subset's min/max when omitted.
        """
        if method == 'gpflow':
            return self.optimise_hp_gpflow(
                x_prev_subset, r_hat_subset, t_seconds_subset, console=console,
            )
        console = console or Console()
        if x_range is None:
            x_range = (float(x_prev_subset.min()), float(x_prev_subset.max()))
        n_ind = self.M if self.M is not None else 20

        def _neg_log_lik(log_params):
            sp_ls, tmp_ls, sp_var = np.exp(log_params)
            sp_ls = np.clip(sp_ls, 1e-3, 0.1)
            tmp_ls = np.clip(tmp_ls, 0.5, 180.0)
            sp_var = np.clip(sp_var, 1e-12, 1e2)

            m = KalmanGPDriftModel(
                spatial_lengthscale=sp_ls,
                temporal_lengthscale_days=tmp_ls,
                spatial_variance=sp_var,
                sigma2=self.sigma2,
                dt=self.dt,
            )
            m.initialise(x_range, n_inducing=n_ind, data_x=x_prev_subset)
            m.update(x_prev_subset, r_hat_subset, t_seconds_subset)
            ll = m._log_lik
            if not np.isfinite(ll):
                warnings.warn(
                    f'HP opt: non-finite log_lik={ll} at '
                    f'sp_ls={np.exp(log_params[0]):.4f} '
                    f'tmp_ls={np.exp(log_params[1]):.2f}d '
                    f'sp_var={np.exp(log_params[2]):.4f}. Returning penalty.',
                    RuntimeWarning, stacklevel=2,
                )
            return 1e10 if not np.isfinite(ll) else -ll

        best_nll = np.inf
        best_pars = np.log([self.spatial_ls, self.temporal_ls, self.spatial_var])

        rng_hp = np.random.default_rng(0)
        for restart in range(n_restarts):
            if restart == 0:
                p0 = np.log([self.spatial_ls, self.temporal_ls, self.spatial_var])
            else:
                p0 = np.log([
                    rng_hp.uniform(0.005, 0.1),
                    rng_hp.uniform(2.0, 60.0),
                    rng_hp.uniform(0.01, 10.0),
                ])

            try:
                res = minimize(
                    _neg_log_lik, p0,
                    method='L-BFGS-B',
                    bounds=[
                        (np.log(1e-3),  np.log(0.1)),
                        (np.log(0.5),   np.log(180.0)),
                        (np.log(1e-12), np.log(1e2)),
                    ],
                    options={'maxiter': 100, 'ftol': 1e-6},
                )
                if res.fun < best_nll:
                    best_nll = res.fun
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

        self.spatial_ls = float(np.exp(best_pars[0]))
        self.temporal_ls = float(np.exp(best_pars[1]))
        self.spatial_var = float(np.exp(best_pars[2]))
        # Swap in new HPs while keeping the accumulated Kalman state.
        self._recompute_hp_dependent()

        console.print(
            f'[green]HP optimised:[/green] '
            f'spatial_ls={self.spatial_ls:.4f}  '
            f'temporal_ls={self.temporal_ls:.2f}d  '
            f'spatial_var={self.spatial_var:.4f}'
        )

    def get_params(self):
        return {
            'spatial_lengthscale':       self.spatial_ls,
            'temporal_lengthscale_days': self.temporal_ls,
            'spatial_variance':          self.spatial_var,
            'obs_noise':                 self.obs_noise,
        }


# ---------------------------------------------------------------------------
# Topology extraction
# ---------------------------------------------------------------------------

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
    """Topology statistics from the current GP posterior — p_multiwell, barrier, Kramers."""
    rng = rng or np.random.default_rng(42)
    x_grid = np.linspace(x_range[0], x_range[1], n_grid)

    mu_mean, mu_var = model.predict(x_grid, full_cov=False)
    f_samples = model.sample_drift(x_grid, n_samples=n_samples, rng=rng)

    sec_per_year = 365.25 * 24 * 3600
    if annualize:
        mu_mean = mu_mean * sec_per_year
        f_samples = f_samples * sec_per_year
        mu_var = mu_var * (sec_per_year ** 2)

    # Identifiability diagnostic: mean posterior std vs mean |drift|. A value
    # ≳1 means p_multiwell is mostly tracking posterior uncertainty rather
    # than mean-field structure.
    mu_std_mean = float(np.mean(np.sqrt(np.maximum(mu_var, 0.0))))
    mu_mag_mean = float(np.mean(np.abs(mu_mean)))
    mu_std_to_mean = (
        float(mu_std_mean / mu_mag_mean) if mu_mag_mean > 0 else float('inf')
    )

    U_mean = -cumulative_trapezoid(mu_mean, x_grid, initial=0.0)
    U_mean -= U_mean.min()
    u_range = float(U_mean.max())

    # Per-sample well count and barrier — both use the same greedy filter so
    # p_multiwell and barrier_mean speak about the same topological features.
    n_stable = np.empty(n_samples, dtype=int)
    barrier_samples = np.empty(n_samples)
    for j in range(n_samples):
        U_j = -cumulative_trapezoid(f_samples[:, j], x_grid, initial=0.0)
        U_j -= U_j.min()
        u_range_j = float(U_j.max())
        if u_range_j < 1e-9:
            n_stable[j] = 0
            barrier_samples[j] = 0.0
            continue
        kept_idx_j, bars_j = _greedy_well_filter(
            x_grid, U_j,
            threshold=min_barrier_fraction * u_range_j,
            min_well_separation=0.0,
        )
        n_stable[j] = len(kept_idx_j)
        barrier_samples[j] = max(bars_j) if bars_j else 0.0

    p_multiwell = float(np.mean(n_stable >= 2))
    mean_n_wells = float(np.mean(n_stable))

    if u_range < 1e-12:
        return {
            'p_multiwell':  round(p_multiwell, 4),
            'mean_n_wells': round(mean_n_wells, 2),
            'barrier_mean': 0.0, 'barrier_std': 0.0,
            'kramers_mean': 0.0, 'kramers_std': 0.0,
            'well_locations': [], 'u_range': 0.0,
            'mu_std_to_mean': round(mu_std_to_mean, 4),
        }

    kept_idx, _ = _greedy_well_filter(
        x_grid, U_mean,
        threshold=min_barrier_fraction * u_range,
        min_well_separation=0.0,
    )
    well_locations = [float(x_grid[i]) for i in kept_idx]

    barrier_mean = float(np.mean(barrier_samples))
    barrier_std = float(np.std(barrier_samples))

    # D shares units with the barrier: annualise both or neither, so the
    # Arrhenius exponent stays dimensionless.
    D = model.sigma2 / 2.0
    if annualize:
        D = D * sec_per_year
    if D > 0:
        kramers_samples = np.exp(-barrier_samples / D)
    else:
        kramers_samples = np.zeros(n_samples)
    kramers_mean = float(np.mean(kramers_samples))
    kramers_std = float(np.std(kramers_samples))

    return {
        'p_multiwell':   round(p_multiwell, 4),
        'mean_n_wells':  round(mean_n_wells, 2),
        'barrier_mean':  round(barrier_mean, 6),
        'barrier_std':   round(barrier_std, 6),
        'kramers_mean':  round(kramers_mean, 8),
        'kramers_std':   round(kramers_std, 8),
        'well_locations': well_locations,
        'u_range':        round(u_range, 4),
        'mu_std_to_mean': round(mu_std_to_mean, 4),
    }


# ---------------------------------------------------------------------------
# Topology forecast
# ---------------------------------------------------------------------------

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
    """Forecast topology at each horizon by propagating the Kalman state forward."""
    rng = rng or np.random.default_rng(42)
    rows = []
    dt_days = model.dt / 86400.0

    for h in forecast_horizons_days:
        n_steps = max(1, int(round(h / dt_days)))

        A_n = np.linalg.matrix_power(model.A_block, n_steps)
        m_pred = A_n @ model.state_mean
        P_pred = model.state_cov.copy()
        for _ in range(n_steps):
            P_pred = model.A_block @ P_pred @ model.A_block.T + model.Q_block

        fwd_model = KalmanGPDriftModel(
            spatial_lengthscale=model.spatial_ls,
            temporal_lengthscale_days=model.temporal_ls,
            spatial_variance=model.spatial_var,
            sigma2=model.sigma2,
            dt=model.dt,
        )
        fwd_model.inducing_x = model.inducing_x
        fwd_model.M = model.M
        fwd_model._K_zz_inv = model._K_zz_inv
        fwd_model.A_block = model.A_block
        fwd_model.Q_block = model.Q_block
        fwd_model._F = model._F
        fwd_model._L = model._L
        fwd_model._Qc = model._Qc
        fwd_model._P_inf_1d = model._P_inf_1d
        fwd_model.state_mean = m_pred
        fwd_model.state_cov = P_pred

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


# ---------------------------------------------------------------------------
# Pipeline glue — prepare Phase A and Phase B if missing
# ---------------------------------------------------------------------------

def _ensure_phase_a_labels(
    start_date,
    end_date,
    seconds_interval,
    window_type,
    kernel_half_width,
    trim_quantile,
    output_dir,
    console,
):
    """
    Return the path to the Phase A labels CSV for this (range, interval,
    window_type). Run Phase A if the CSV does not exist.
    """
    fname = (
        f'regime_labels_{start_date.strftime("%Y-%m-%d")}_to_'
        f'{end_date.strftime("%Y-%m-%d")}_{seconds_interval}s_{window_type}.csv'
    )
    out_path = os.path.join(output_dir, fname)
    if os.path.exists(out_path):
        console.print(f'[green]Phase A labels found:[/green] {fname}')
        return out_path

    console.print('[yellow]Phase A labels missing — running Phase A.[/yellow]')
    run_phase_a(
        start_date=start_date,
        end_date=end_date,
        seconds_interval=seconds_interval,
        kernel_half_width=kernel_half_width,
        trim_quantile=trim_quantile,
        output_dir=output_dir,
        window_type=window_type,
        console=console,
    )
    if not os.path.exists(out_path):
        raise FileNotFoundError(
            f'Phase A did not produce expected labels CSV: {out_path}'
        )
    return out_path


def _phase_b_dwell_days(labels_csv, window_type, output_dir, console):
    """
    Run Phase B (markov_chain) and return the mean dwell time in days.

    Falls back to 7.0 days if Phase B cannot be fitted (eg. only one regime
    observed in the supplied window range).
    """
    sec_per_window = dc.window_seconds(window_type)
    try:
        mc = run_markov_chain(
            labels_csv,
            window_type=window_type,
            output_dir=output_dir,
            console=console,
        )
    except Exception as exc:
        console.print(
            f'[yellow]Phase B failed ({exc}); '
            'falling back to 7-day dwell heuristic.[/yellow]'
        )
        return 7.0

    if mc is None:
        return 7.0

    dwells = np.asarray(mc['dwells'], dtype=float)
    finite = dwells[np.isfinite(dwells)]
    if finite.size == 0:
        return 7.0

    mean_dwell_windows = float(np.mean(finite))
    mean_dwell_days = mean_dwell_windows * sec_per_window / 86400.0
    return float(np.clip(mean_dwell_days, 3.0, 60.0))


# ---------------------------------------------------------------------------
# Phase A / Phase GP correlation check
# ---------------------------------------------------------------------------

def check_topology_correlation(df_topology, console=None):
    """
    Print Pearson and Spearman correlations between p_multiwell_gp (sequential
    Kalman-GP) and p_multiwell_a (Phase A batch classifier).

    Only rows where Phase A produced a label (p_multiwell_a not NaN) are used.
    Returns a dict with keys 'n', 'pearson_r', 'pearson_p', 'spearman_r',
    'spearman_p', or an empty dict if there is insufficient paired data.
    """
    console = console or Console()

    if 'p_multiwell_a' not in df_topology.columns or 'p_multiwell_gp' not in df_topology.columns:
        console.print('[yellow]Correlation check skipped: topology DataFrame missing required columns.[/yellow]')
        return {}

    paired = df_topology[['p_multiwell_gp', 'p_multiwell_a']].dropna()
    n = len(paired)

    if n < 3:
        console.print(f'[yellow]Correlation check skipped: only {n} paired observations.[/yellow]')
        return {}

    gp_vals = paired['p_multiwell_gp'].values
    a_vals  = paired['p_multiwell_a'].values

    pr, pp = pearsonr(gp_vals, a_vals)
    sr, sp = spearmanr(gp_vals, a_vals)

    console.print(
        f'\n[bold]Phase A vs Phase GP topology correlation[/bold]  (n={n} paired snapshots)\n'
        f'  Pearson  r = {pr:+.3f}  (p={pp:.3g})\n'
        f'  Spearman r = {sr:+.3f}  (p={sp:.3g})'
    )

    return {
        'n': n,
        'pearson_r': round(pr, 4), 'pearson_p': round(pp, 6),
        'spearman_r': round(sr, 4), 'spearman_p': round(sp, 6),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_phase_gp(
    start_date,
    end_date,
    seconds_interval,
    phase_a_seconds_interval=None,
    window_type='weekly',
    labels_csv=None,
    sigma2=None,
    spatial_lengthscale=0.03,
    temporal_lengthscale_days=None,
    spatial_variance=1.0,
    ema_halflife_days=None,
    n_inducing=20,
    n_grid=200,
    n_samples=200,
    hp_opt_after_n_windows=2,
    hp_opt_n_restarts=3,
    hp_opt_method='scipy',
    forecast_horizons_days=(1.0, 3.0, 7.0, 14.0),
    topology_every_n_obs=500,
    kernel_half_width=5,
    trim_quantile=0.01,
    min_crossing_sep=10,
    min_barrier_fraction=0.1,
    output_dir='regime_results',
    gp_output_dir=None,
    seed=42,
    console=None,
):
    """
    Run the sequential Kalman-GP drift field model end-to-end.

    Self-contained: if Phase A labels are missing they are produced by calling
    run_phase_a; if no temporal lengthscale / EMA halflife is supplied they are
    drawn from Phase B (markov_chain) mean dwell times. Raw data is downloaded
    on demand via data_collection.ensure_data through Phase A.

    output_dir    — where Phase A labels and Phase B chains live (read/written).
    gp_output_dir — where Phase GP artefacts (topology / forecast / params /
                    plots) are written. Defaults to output_dir for backwards
                    compatibility.
    """
    os.makedirs(output_dir, exist_ok=True)
    if gp_output_dir is None:
        gp_output_dir = output_dir
    os.makedirs(gp_output_dir, exist_ok=True)
    console = console or Console()
    rng = np.random.default_rng(seed)
    seconds_interval = int(seconds_interval)
    _phase_a_si = int(phase_a_seconds_interval) if phase_a_seconds_interval else seconds_interval

    # --- Snap window boundaries (Phase A uses the same rule) ---
    start_date, end_date = normalize_window_boundaries(
        start_date, end_date, window_type, console=console,
    )

    # --- Phase A: produce regime labels if missing ---
    if labels_csv is None:
        labels_csv = _ensure_phase_a_labels(
            start_date, end_date, _phase_a_si, window_type,
            kernel_half_width, trim_quantile, output_dir, console,
        )

    # --- Phase B: dwell times drive both Matern temporal_ls and EMA halflife ---
    mean_dwell_days = _phase_b_dwell_days(
        labels_csv, window_type, output_dir, console,
    )
    if temporal_lengthscale_days is None:
        # Half of Phase B mean dwell — Phase A regime transitions resolve on
        # roughly half the dwell horizon, so a tighter temporal kernel tracks
        # them better than the full-dwell value.
        temporal_lengthscale_days = mean_dwell_days / 2.0
    if ema_halflife_days is None:
        ema_halflife_days = mean_dwell_days * 4.0
    ema_str = (
        f'{ema_halflife_days:.2f}d'
        if ema_halflife_days and np.isfinite(ema_halflife_days)
        else 'DISABLED'
    )
    console.print(
        f'[cyan]Phase B mean dwell:[/cyan] {ema_halflife_days:.2f} days  '
        f'-> temporal_ls={temporal_lengthscale_days:.2f}d, '
        f'ema_halflife={ema_str}'
    )

    # --- Load full per-window series ---
    console.print(
        f'[cyan]Phase GP — loading series at {seconds_interval}s '
        f'({window_type})[/cyan]'
    )
    x_prev, dx, dt, dt_t = load_series(
        start_date, end_date, seconds_interval,
        kernel_half_width=kernel_half_width,
        trim_quantile=trim_quantile,
        window_type=window_type,
    )
    console.print(f'  {len(dx)} increments loaded.')

    r = dx / dt

    # --- EMA drift demeaning (Phase B-derived halflife) ---
    r_hat, r_bar = ema_demean_drift(r, dt_t, halflife_days=ema_halflife_days)
    console.print(
        f'  drift mean before: {r.mean():.4e}  after: {r_hat.mean():.4e}'
    )

    if sigma2 is None:
        # Use the post-EMA residual (in dx-units: r_hat * dt) so the drift
        # component is removed before estimating the diffusion variance.
        sigma2 = float(np.var(r_hat * dt) / dt)
    console.print(f'  sigma2          = {sigma2:.4e}')
    console.print(f'  obs_noise       = {sigma2/dt:.4e}  (sigma2/dt)')

    t_seconds = (
        pd.to_datetime(dt_t).astype(np.int64) / 1e9
    ).values.astype(float)

    x_min = float(np.percentile(x_prev, 1))
    x_max = float(np.percentile(x_prev, 99))
    x_range = (x_min, x_max)
    console.print(f'  x range         = [{x_min:.4f}, {x_max:.4f}]')

    if spatial_lengthscale is None:
        # Half of Phase A's 0.3 × std(x) heuristic — keeps the spatial GP
        # tight enough to resolve features at the weekly excursion scale.
        spatial_lengthscale = 0.15 * float(np.std(x_prev))
        console.print(
            f'  spatial_ls      = {spatial_lengthscale:.4f}  '
            f'(0.15 × std(x_prev); derived from data)'
        )

    model = KalmanGPDriftModel(
        spatial_lengthscale=spatial_lengthscale,
        temporal_lengthscale_days=temporal_lengthscale_days,
        spatial_variance=spatial_variance,
        sigma2=sigma2,
        dt=float(dt),
    )
    model.initialise(x_range=x_range, n_inducing=n_inducing, data_x=x_prev)
    console.print(
        f'  Kalman state dim = {2 * model.M}  '
        f'({model.M} inducing points; requested {n_inducing})'
    )

    # --- Phase A labels for downstream comparison ---
    labels_df = pd.read_csv(
        labels_csv, parse_dates=['window_start', 'window_end']
    )
    labels_df = labels_df[
        labels_df['seconds_interval'] == _phase_a_si
    ].copy().reset_index(drop=True)

    dt_series = pd.Series(pd.to_datetime(dt_t))
    window_idx = np.full(len(dt_t), -1, dtype=int)
    for i, row in labels_df.iterrows():
        mask = (
            (dt_series >= row['window_start'])
            & (dt_series < row['window_end'] + pd.Timedelta(days=1))
        )
        window_idx[mask] = i

    topology_rows = []
    forecast_rows = []
    snapshots = []

    hp_x_accum = []
    hp_r_accum = []
    hp_t_accum = []
    hp_done = False
    window_count = 0

    with Progress(
        SpinnerColumn(),
        '[progress.description]{task.description}',
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task('Sequential Kalman-GP', total=len(dx))

        for w_idx in labels_df.index:
            obs_mask = (window_idx == w_idx)
            if not obs_mask.any():
                continue

            x_w = x_prev[obs_mask]
            r_hat_w = r_hat[obs_mask]
            t_w = t_seconds[obs_mask]
            dt_w = dt_t[obs_mask]

            hp_x_accum.append(x_w)
            hp_r_accum.append(r_hat_w)
            hp_t_accum.append(t_w)
            window_count += 1

            if (not hp_done) and (window_count >= hp_opt_after_n_windows):
                console.print(
                    f'[yellow]Optimising hyperparameters after '
                    f'{window_count} windows...[/yellow]'
                )
                hp_x = np.concatenate(hp_x_accum)
                hp_r = np.concatenate(hp_r_accum)
                hp_t = np.concatenate(hp_t_accum)
                if len(hp_x) > 5000:
                    idx_sub = np.linspace(0, len(hp_x) - 1, 5000, dtype=int)
                    hp_x, hp_r, hp_t = hp_x[idx_sub], hp_r[idx_sub], hp_t[idx_sub]
                model.optimise_hp(
                    hp_x, hp_r, hp_t,
                    n_restarts=hp_opt_n_restarts,
                    method=hp_opt_method,
                    x_range=x_range,
                    console=console,
                )
                hp_done = True

            model.update(x_w, r_hat_w, t_w)

            n_w = int(obs_mask.sum())
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
                    'mu_std_to_mean': topo['mu_std_to_mean'],
                    'window_start':   str(pd.Timestamp(row['window_start']).date()),
                })
                snapshots.append(
                    (dt_query, model.state_mean.copy(), model.state_cov.copy())
                )

            df_fc = forecast_topology(
                model,
                forecast_horizons_days=list(forecast_horizons_days),
                x_range=x_range,
                n_grid=n_grid, n_samples=n_samples,
                min_crossing_sep=min_crossing_sep,
                min_barrier_fraction=min_barrier_fraction,
                rng=rng,
            )
            df_fc['window_end'] = str(pd.Timestamp(row['window_end']).date())
            forecast_rows.append(df_fc)

            prog.advance(task, advance=int(obs_mask.sum()))

    df_topology = pd.DataFrame(topology_rows)
    df_forecast = (
        pd.concat(forecast_rows, ignore_index=True)
        if forecast_rows else pd.DataFrame()
    )
    df_params = pd.DataFrame([model.get_params()])

    stem = (
        f"phase_gp_{pd.Timestamp(start_date).strftime('%Y-%m-%d')}_to_"
        f"{pd.Timestamp(end_date).strftime('%Y-%m-%d')}_{seconds_interval}s"
    )

    topo_path = os.path.join(gp_output_dir, f'{stem}_topology.csv')
    df_topology.to_csv(topo_path, index=False)
    console.print(f'[green]Wrote[/green] {topo_path}')

    fc_path = os.path.join(gp_output_dir, f'{stem}_forecast.csv')
    df_forecast.to_csv(fc_path, index=False)
    console.print(f'[green]Wrote[/green] {fc_path}')

    params_path = os.path.join(gp_output_dir, f'{stem}_params.csv')
    df_params.to_csv(params_path, index=False)
    console.print(f'[green]Wrote[/green] {params_path}')

    if not df_topology.empty:
        plots.plot_gp_topology_series(
            df_topology,
            os.path.join(gp_output_dir, f'{stem}_topology.png'),
        )
        plots.plot_gp_potential_snapshots(
            model, x_range,
            snapshots,
            os.path.join(gp_output_dir, f'{stem}_potential.png'),
            n_snapshots=6,
        )
        plots.plot_km_vs_gp_overlay(
            model, x_range,
            snapshots,
            labels_df=labels_df,
            km_dir=os.path.join(output_dir, 'km'),
            phase_a_seconds_interval=_phase_a_si,
            out_path=os.path.join(gp_output_dir, f'{stem}_km_vs_gp.png'),
            n_snapshots=6,
        )

    if not df_topology.empty and 'mu_std_to_mean' in df_topology.columns:
        finite_ratio = df_topology['mu_std_to_mean'].replace(
            [np.inf, -np.inf], np.nan,
        ).dropna()
        if len(finite_ratio) > 0:
            console.print(
                f'[bold]Posterior σ(μ) / |μ| ratio[/bold]  '
                f'mean={finite_ratio.mean():.2f}  '
                f'median={finite_ratio.median():.2f}  '
                f'(≳1 means p_multiwell is uncertainty-driven)'
            )

    corr = check_topology_correlation(df_topology, console=console)

    return {
        'model':        model,
        'df_topology':  df_topology,
        'df_forecast':  df_forecast,
        'df_params':    df_params,
        'r_hat':        r_hat,
        'r_bar':        r_bar,
        'snapshots':    snapshots,
        'correlation':  corr,
    }


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from datetime import datetime

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    seconds_interval = 30
    window_type = 'weekly'

    _console = Console()
    run_phase_gp(
        start_date, end_date, seconds_interval,
        window_type=window_type,
        n_inducing=20,
        n_grid=200,
        n_samples=200,
        hp_opt_after_n_windows=2,
        hp_opt_n_restarts=3,
        topology_every_n_obs=500,
        forecast_horizons_days=(1.0, 3.0, 7.0, 14.0),
        console=_console,
    )
