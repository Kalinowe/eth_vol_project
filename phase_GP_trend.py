"""
Non-stationary Kalman-GP drift model with an explicit x-independent trend.

Motivation
----------
The stationary KalmanGPDriftModel (phase_GP.py) treats every observation as
purely a sample of mu(x_t).  When the price trends within a window the GP
absorbs the trend into a sloped mu(x), so U(x) = -integral(mu) becomes
monotonic instead of U-shaped even when the underlying x-dependent structure
is a single well.

This module adds a scalar trend state beta(t) to the Kalman state.  The
observation model becomes

    r_hat_t = beta(t) + mu(x_{t-1}, t) + eps_t,   eps_t ~ N(0, obs_noise)

with two priors:

    beta(t)  ~ Matern 3/2(temporal_ls=trend_ls,    variance=trend_var)
    mu(x, t) ~ GP(0, RBF(x; spatial_ls, spatial_var) * Matern 3/2(t; temporal_ls))

To keep beta and mu identifiable:
  * trend_ls >> temporal_ls   (beta is slower than mu)
  * trend_var > spatial_var    (beta has a broader prior so it absorbs the
                                constant-in-x signal preferentially)

Topology is computed from mu only via predict() so a trending market does
not inject a fake linear -beta*x term into U(x).  Total drift mu + beta
is exposed via predict_total() for KM-overlay plots, since KM estimates
the total drift the price actually experiences.

State layout (size 2 + 2M):
    state = [beta, beta', f_1, f'_1, ..., f_M, f'_M]

Public API matches KalmanGPDriftModel so topology_from_gp() and the plotting
helpers work without changes:
    initialise(x_range, n_inducing, data_x=None)
    update(x_prev, r_hat, t_seconds)
    predict(x_grid, full_cov=False)              -> GP component mu(x)
    sample_drift(x_grid, n_samples, rng)         -> GP component samples
    predict_total(x_grid, full_cov=False)        -> mu(x) + beta
    beta_mean, beta_std                           -> current trend posterior
    get_params()
"""

from __future__ import annotations

import warnings
import numpy as np
from scipy.linalg import solve, cholesky

from phase_GP import (
    matern32_sde,
    matern32_discrete,
    rbf_kernel,
    _SEC_PER_YEAR,  # re-exported for callers if needed
)


__all__ = ['KalmanGPDriftWithTrendModel', '_SEC_PER_YEAR']


class KalmanGPDriftWithTrendModel:
    """Non-stationary Kalman-GP drift model with an explicit scalar trend."""

    def __init__(
        self,
        spatial_lengthscale=0.05,
        temporal_lengthscale_days=7.0,
        spatial_variance=1.0,
        trend_lengthscale_days=30.0,
        trend_variance=None,
        sigma2=1e-4,
        dt=30.0,
    ):
        self.spatial_ls = float(spatial_lengthscale)
        self.temporal_ls = float(temporal_lengthscale_days)
        self.spatial_var = float(spatial_variance)
        self.trend_ls = float(trend_lengthscale_days)
        # Default trend variance: 10x spatial_variance.  Strongly biases the
        # Kalman split so that constant-in-x signal flows to beta first.
        self.trend_var = float(
            trend_variance if trend_variance is not None else 10.0 * spatial_variance
        )
        self.sigma2 = float(sigma2)
        self.dt = float(dt)
        self.obs_noise = self.sigma2 / self.dt

        self.inducing_x = None
        self.M = None

        self.state_mean = None
        self.state_cov = None

        self.A_block = None
        self.Q_block = None
        self._K_zz_inv = None
        self._K_zz_jit = None
        self._I_aug = None

        # Per-component SDE caches (needed when an observation arrives off the
        # nominal dt grid; we rebuild A/Q locally for that step).
        self._F_f = self._L_f = self._Qc_f = None
        self._P_inf_f_1d = None
        self._F_b = self._L_b = self._Qc_b = None
        self._P_inf_b_1d = None

        self._log_lik = 0.0
        self._n_obs = 0

    # ------------------------------------------------------------------
    # HP-dependent caches
    # ------------------------------------------------------------------
    def _recompute_hp_dependent(self):
        K_zz = rbf_kernel(
            self.inducing_x, self.inducing_x,
            self.spatial_ls, self.spatial_var,
        )
        self._K_zz_jit = K_zz + 1e-6 * np.eye(self.M)
        self._K_zz_inv = solve(self._K_zz_jit, np.eye(self.M), assume_a='pos')

        dt_days = self.dt / 86400.0

        # GP spatial-temporal block: Matern 3/2 in time, RBF in x.
        F_f, L_f, Qc_f, _H1d_f, P_inf_f = matern32_sde(
            self.temporal_ls, sigma2=self.spatial_var,
        )
        A_1_f, Q_1_f = matern32_discrete(F_f, L_f, Qc_f, dt_days)
        K_zz_norm = self._K_zz_jit / self.spatial_var
        A_block_f = np.kron(np.eye(self.M), A_1_f)
        Q_block_f = np.kron(K_zz_norm, Q_1_f)

        # Scalar trend block: Matern 3/2 in time.
        F_b, L_b, Qc_b, _H1d_b, P_inf_b = matern32_sde(
            self.trend_ls, sigma2=self.trend_var,
        )
        A_1_b, Q_1_b = matern32_discrete(F_b, L_b, Qc_b, dt_days)

        # Augmented block-diagonal A, Q (beta block first).
        d_aug = 2 + 2 * self.M
        A_aug = np.zeros((d_aug, d_aug))
        Q_aug = np.zeros((d_aug, d_aug))
        A_aug[:2, :2] = A_1_b
        Q_aug[:2, :2] = Q_1_b
        A_aug[2:, 2:] = A_block_f
        Q_aug[2:, 2:] = Q_block_f

        self.A_block = A_aug
        self.Q_block = Q_aug
        self._F_f, self._L_f, self._Qc_f = F_f, L_f, Qc_f
        self._P_inf_f_1d = P_inf_f
        self._F_b, self._L_b, self._Qc_b = F_b, L_b, Qc_b
        self._P_inf_b_1d = P_inf_b

    def _build_step_matrices(self, dt_days):
        """Build A, Q for a non-nominal time step (gap in observations)."""
        A_1_b, Q_1_b = matern32_discrete(self._F_b, self._L_b, self._Qc_b, dt_days)
        A_1_f, Q_1_f = matern32_discrete(self._F_f, self._L_f, self._Qc_f, dt_days)
        K_zz_norm = self._K_zz_jit / self.spatial_var

        d_aug = 2 + 2 * self.M
        A = np.zeros((d_aug, d_aug))
        Q = np.zeros((d_aug, d_aug))
        A[:2, :2] = A_1_b
        Q[:2, :2] = Q_1_b
        A[2:, 2:] = np.kron(np.eye(self.M), A_1_f)
        Q[2:, 2:] = np.kron(K_zz_norm, Q_1_f)
        return A, Q

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def initialise(self, x_range, n_inducing=20, data_x=None):
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
        d_aug = 2 + 2 * self.M
        self._I_aug = np.eye(d_aug)

        # Stationary prior covariance.
        lam_sq_f = self._P_inf_f_1d[1, 1] / self._P_inf_f_1d[0, 0]
        idx_f = 2 + np.arange(0, 2 * self.M, 2)
        idx_fp = 2 + np.arange(1, 2 * self.M, 2)

        P_inf = np.zeros((d_aug, d_aug))
        P_inf[:2, :2] = self._P_inf_b_1d
        P_inf[np.ix_(idx_f, idx_f)] = self._K_zz_jit
        P_inf[np.ix_(idx_fp, idx_fp)] = lam_sq_f * self._K_zz_jit

        self.state_mean = np.zeros(d_aug)
        self.state_cov = P_inf.copy()

        self._log_lik = 0.0
        self._n_obs = 0

    # ------------------------------------------------------------------
    # Observation matrix
    # ------------------------------------------------------------------
    def _obs_matrix(self, x_obs):
        k_vec = rbf_kernel(
            np.array([x_obs]),
            self.inducing_x,
            self.spatial_ls,
            self.spatial_var,
        ).flatten()
        d_aug = 2 + 2 * self.M
        H = np.zeros((1, d_aug))
        H[0, 0] = 1.0                              # beta picks up directly
        H[0, 2::2] = self._K_zz_inv @ k_vec        # GP f component
        return H

    # ------------------------------------------------------------------
    # Sequential Kalman update
    # ------------------------------------------------------------------
    def update(self, x_prev, r_hat, t_seconds):
        if self.state_mean is None:
            raise RuntimeError('Call initialise() before update().')

        dt_days_nominal = self.dt / 86400.0
        prev_t = None

        for i in range(len(r_hat)):
            if prev_t is not None:
                actual_dt_days = (t_seconds[i] - prev_t) / 86400.0
                if abs(actual_dt_days - dt_days_nominal) > 0.1 * dt_days_nominal:
                    A_step, Q_step = self._build_step_matrices(actual_dt_days)
                else:
                    A_step, Q_step = self.A_block, self.Q_block
            else:
                A_step, Q_step = self.A_block, self.Q_block

            m_pred = A_step @ self.state_mean
            P_pred = A_step @ self.state_cov @ A_step.T + Q_step

            if not (np.isfinite(m_pred).all() and np.isfinite(P_pred).all()):
                warnings.warn(
                    f'KalmanGP-trend: state overflow at observation {i} '
                    f'(max |m_pred|={np.max(np.abs(m_pred)):.3e}). '
                    'Aborting update.',
                    RuntimeWarning, stacklevel=2,
                )
                self._log_lik = -np.inf
                break

            H = self._obs_matrix(x_prev[i])
            S = H @ P_pred @ H.T + self.obs_noise
            raw_s = float(S[0, 0])
            if raw_s < 1e-15:
                warnings.warn(
                    f'KalmanGP-trend: innovation variance S={raw_s:.3e} at obs {i}. '
                    'Flooring to 1e-15.',
                    RuntimeWarning, stacklevel=2,
                )
            s_val = max(raw_s, 1e-15)

            K_gain = P_pred @ H.T / s_val
            innov = r_hat[i] - (H @ m_pred)[0]

            self.state_mean = m_pred + K_gain[:, 0] * innov
            IKH = self._I_aug - K_gain @ H
            P_post = IKH @ P_pred @ IKH.T + self.obs_noise * K_gain @ K_gain.T
            P_sym = (P_post + P_post.T) * 0.5
            self.state_cov = P_sym

            self._log_lik += -0.5 * np.log(2 * np.pi * s_val) - 0.5 * innov ** 2 / s_val
            self._n_obs += 1
            prev_t = t_seconds[i]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, x_grid, full_cov=False):
        """Posterior mean/cov of the GP component mu(x, t_now). Excludes beta."""
        idx_f = 2 + np.arange(0, 2 * self.M, 2)
        f_mean = self.state_mean[idx_f]
        P_ff = self.state_cov[np.ix_(idx_f, idx_f)]

        K_qz = rbf_kernel(
            x_grid, self.inducing_x, self.spatial_ls, self.spatial_var,
        )
        alpha = self._K_zz_inv @ f_mean
        mu_mean = K_qz @ alpha
        V = self._K_zz_inv @ K_qz.T

        if full_cov:
            K_qq = rbf_kernel(
                x_grid, x_grid, self.spatial_ls, self.spatial_var,
            )
            mu_cov = (
                K_qq - K_qz @ V + K_qz @ (self._K_zz_inv @ P_ff @ V)
            )
            mu_cov += max(1e-6 * self.spatial_var, 1e-10) * np.eye(len(x_grid))
        else:
            K_qq_diag = self.spatial_var * np.ones(len(x_grid))
            mu_cov = (
                K_qq_diag
                - np.sum(K_qz * V.T, axis=1)
                + np.sum(V * (P_ff @ V), axis=0)
            )
            mu_cov = np.maximum(mu_cov, 0.0)

        return mu_mean, mu_cov

    def predict_total(self, x_grid, full_cov=False):
        """Total drift mu(x) + beta at t_now (with combined posterior variance)."""
        mu_mean, mu_cov = self.predict(x_grid, full_cov=full_cov)
        beta_mean = float(self.state_mean[0])
        beta_var = float(max(self.state_cov[0, 0], 0.0))
        # Cov(beta, f) is non-zero after updates; pull it in for honest variance.
        # The mu-component at x_q has weights w(x_q) = K_qz @ K_zz^-1 on f.
        idx_f = 2 + np.arange(0, 2 * self.M, 2)
        cov_beta_f = self.state_cov[0:1, idx_f]    # (1, M); Cov(beta, f_j)
        K_qz = rbf_kernel(
            x_grid, self.inducing_x, self.spatial_ls, self.spatial_var,
        )
        W = K_qz @ self._K_zz_inv                  # (N_q, M)
        # cov_mu_beta[i] = Cov(mu(x_i), beta) = W[i,:] @ Cov(f, beta)
        cov_mu_beta = (W @ cov_beta_f.T).ravel()

        total_mean = mu_mean + beta_mean
        if full_cov:
            # Cov(mu(x_i)+beta, mu(x_j)+beta)
            #   = Cov(mu(x_i), mu(x_j))
            #     + Cov(beta, beta)
            #     + Cov(mu(x_i), beta) + Cov(mu(x_j), beta)
            total_cov = (
                mu_cov + beta_var
                + cov_mu_beta[:, None] + cov_mu_beta[None, :]
            )
        else:
            total_cov = np.maximum(
                mu_cov + beta_var + 2.0 * cov_mu_beta, 0.0,
            )
        return total_mean, total_cov

    def sample_drift(self, x_grid, n_samples=200, rng=None):
        """Joint samples of the GP component mu(x). Used by topology_from_gp."""
        rng = rng or np.random.default_rng(42)
        mu_mean, K_post = self.predict(x_grid, full_cov=True)

        jitter = max(1e-8 * self.spatial_var, 1e-12)
        L_chol = None
        for _ in range(12):
            try:
                L_chol = cholesky(
                    K_post + jitter * np.eye(len(x_grid)), lower=True,
                )
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        if L_chol is None:
            raise np.linalg.LinAlgError(
                'sample_drift Cholesky failed even with large jitter.'
            )
        z = rng.standard_normal((len(x_grid), n_samples))
        return mu_mean[:, None] + L_chol @ z

    # ------------------------------------------------------------------
    # Posterior summaries
    # ------------------------------------------------------------------
    @property
    def beta_mean(self):
        if self.state_mean is None:
            return 0.0
        return float(self.state_mean[0])

    @property
    def beta_std(self):
        if self.state_cov is None:
            return 0.0
        return float(np.sqrt(max(self.state_cov[0, 0], 0.0)))

    def get_params(self):
        return {
            'spatial_lengthscale':       self.spatial_ls,
            'temporal_lengthscale_days': self.temporal_ls,
            'spatial_variance':          self.spatial_var,
            'trend_lengthscale_days':    self.trend_ls,
            'trend_variance':            self.trend_var,
            'obs_noise':                 self.obs_noise,
            'beta_final':                self.beta_mean,
            'beta_std_final':            self.beta_std,
        }
