"""
Phase GP — Sequential Gaussian Process over the drift field mu(x, t).

Phase C of the pipeline: a single sequential Kalman-filter GP whose state
is continuously updated as new observations arrive. The GP factors as a
spatial RBF kernel over log-price times an exact Matern 3/2 state-space
representation in time, giving O(N) inference and a constant-size state.

    r_t      = (dx_t / dt) * sec_per_year                annualised drift [/year]
    r_hat_t  = r_t - EMA(r_t, halflife=tau_ema)          causal EMA demean
    r_hat_t ~ mu(x_{t-1}, t) + eps_t
    eps_t   ~ N(0, sigma2 / dt)                          Euler-Maruyama noise
    mu(x,t) ~ GP(0, k_rbf(x, x') * k_matern32(t, t'))

All Kalman state, sigma2, and topology outputs are in annualised units
(/year for drift, /year² for variance). spatial_var=1.0 is calibrated to
this convention.

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
from rich.console import Console

from regime_estimation import _greedy_well_filter


_SEC_PER_YEAR = 365.25 * 24 * 3600


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

    F = np.array(
        [
            [0.0, 1.0],
            [-(lam**2), -2.0 * lam],
        ]
    )
    L = np.array([[0.0], [1.0]])
    Qc = np.array([[4.0 * sigma2 * lam**3]])
    H = np.array([[1.0, 0.0]])
    # Stationary covariance: solution of F P + P Fᵀ + L Qc Lᵀ = 0.
    P_inf = sigma2 * np.array(
        [
            [1.0, 0.0],
            [0.0, lam**2],
        ]
    )
    return F, L, Qc, H, P_inf


def matern32_discrete(F, L, Qc, dt):
    """
    Discretise the Matern 3/2 SDE for a fixed step dt.

    Returns A (2,2) transition and Q (2,2) process noise covariance.
    """
    A = expm(F * dt)
    n = F.shape[0]
    Z = np.zeros((n, n))
    M = np.block(
        [
            [-F, L @ Qc @ L.T],
            [Z, F.T],
        ]
    )
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
    return variance * np.exp(-0.5 * sq_dist / lengthscale**2)


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
        K_zz = rbf_kernel(
            self.inducing_x, self.inducing_x, self.spatial_ls, self.spatial_var
        )
        self._K_zz_jit = K_zz + 1e-6 * np.eye(self.M)
        self._K_zz_inv = solve(self._K_zz_jit, np.eye(self.M), assume_a="pos")

        dt_days = self.dt / 86400.0
        F, L, Qc, _H_1d, P_inf = matern32_sde(
            self.temporal_ls,
            sigma2=self.spatial_var,
        )
        A_1, Q_1 = matern32_discrete(F, L, Qc, dt_days)
        eye_M = np.eye(self.M)
        K_zz_norm = self._K_zz_jit / self.spatial_var  # correlation matrix, diag ≈ 1
        self.A_block = np.kron(eye_M, A_1)
        self.Q_block = np.kron(K_zz_norm, Q_1)
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
        idx_f = np.arange(0, 2 * self.M, 2)
        idx_fp = np.arange(1, 2 * self.M, 2)
        P_inf_block = np.zeros((2 * self.M, 2 * self.M))
        P_inf_block[np.ix_(idx_f, idx_f)] = self._K_zz_jit
        P_inf_block[np.ix_(idx_fp, idx_fp)] = lam_sq * self._K_zz_jit

        self.state_mean = np.zeros(2 * self.M)
        self.state_cov = P_inf_block.copy()

        self._log_lik = 0.0
        self._n_obs = 0

    def reproject_to_range(self, x_lo, x_hi, n_inducing=None, margin=0.1):
        """
        Move all inducing points into [x_lo - margin·w, x_hi + margin·w]
        (w = max(x_hi - x_lo, 1e-4)) while preserving the accumulated Kalman
        posterior via GP interpolation.

        Returns the expanded (lo, hi) range so callers can pass it directly to
        topology_from_gp without recomputing.
        """
        x_width = max(x_hi - x_lo, 1e-4)
        x_lo = x_lo - margin * x_width
        x_hi = x_hi + margin * x_width

        if n_inducing is None:
            n_inducing = self.M

        Z_old = self.inducing_x.copy()
        M_old = self.M
        K_zz_old_inv = self._K_zz_inv.copy()

        idx_f = np.arange(0, 2 * M_old, 2)
        idx_fp = np.arange(1, 2 * M_old, 2)
        f_mean = self.state_mean[idx_f]
        fp_mean = self.state_mean[idx_fp]
        P_ff = self.state_cov[np.ix_(idx_f, idx_f)]
        P_fpfp = self.state_cov[np.ix_(idx_fp, idx_fp)]
        P_ffp = self.state_cov[np.ix_(idx_f, idx_fp)]

        # New inducing layout and HP-dependent caches
        self.inducing_x = np.linspace(x_lo, x_hi, n_inducing)
        self.M = n_inducing
        self._recompute_hp_dependent()
        self._I_2M = np.eye(2 * self.M)

        # GP interpolation weights: W = K(Z_new, Z_old) @ K(Z_old)^{-1}
        K_new_old = rbf_kernel(
            self.inducing_x, Z_old, self.spatial_ls, self.spatial_var
        )
        W = K_new_old @ K_zz_old_inv  # (M_new, M_old)

        # Residual prior covariance not explained by the old inducing points
        prior_resid = self._K_zz_jit - K_new_old @ K_zz_old_inv @ K_new_old.T
        lam_sq = self._P_inf_1d[1, 1] / self._P_inf_1d[0, 0]

        # Projected state mean
        idx_f_new = np.arange(0, 2 * self.M, 2)
        idx_fp_new = np.arange(1, 2 * self.M, 2)
        new_mean = np.zeros(2 * self.M)
        new_mean[idx_f_new] = W @ f_mean
        new_mean[idx_fp_new] = W @ fp_mean

        # Projected state covariance
        P_ff_new = prior_resid + W @ P_ff @ W.T
        P_fpfp_new = lam_sq * prior_resid + W @ P_fpfp @ W.T
        P_ffp_new = W @ P_ffp @ W.T

        new_cov = np.zeros((2 * self.M, 2 * self.M))
        new_cov[np.ix_(idx_f_new, idx_f_new)] = P_ff_new
        new_cov[np.ix_(idx_fp_new, idx_fp_new)] = P_fpfp_new
        new_cov[np.ix_(idx_f_new, idx_fp_new)] = P_ffp_new
        new_cov[np.ix_(idx_fp_new, idx_f_new)] = P_ffp_new.T

        self.state_mean = new_mean
        self.state_cov = (new_cov + new_cov.T) * 0.5
        return x_lo, x_hi

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
            raise RuntimeError("Call initialise() before update().")

        dt_days = self.dt / 86400.0
        prev_t = None

        for i in range(len(r_hat)):
            if prev_t is not None:
                actual_dt_days = (t_seconds[i] - prev_t) / 86400.0
                if abs(actual_dt_days - dt_days) > 0.1 * dt_days:
                    A_i, Q_i = matern32_discrete(
                        self._F,
                        self._L,
                        self._Qc,
                        actual_dt_days,
                    )
                    K_zz_norm_i = self._K_zz_jit / self.spatial_var
                    A_block_i = np.kron(np.eye(self.M), A_i)
                    Q_block_i = np.kron(K_zz_norm_i, Q_i)
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
                    f"KalmanGP: state overflow at observation {i} "
                    f"(max |m_pred|={np.max(np.abs(m_pred)):.3e}, "
                    f"max |P_pred|={np.max(np.abs(P_pred)):.3e}). "
                    "Aborting update, log_lik set to -inf.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._log_lik = -np.inf
                break

            H = self._obs_matrix(x_prev[i])
            S = H @ P_pred @ H.T + self.obs_noise
            raw_s = float(S[0, 0])
            if raw_s < 1e-15:
                warnings.warn(
                    f"KalmanGP: innovation variance S={raw_s:.3e} at observation {i} "
                    f"(obs_noise={self.obs_noise:.3e}). Flooring to 1e-15. "
                    "P_pred may have lost positive-definiteness.",
                    RuntimeWarning,
                    stacklevel=2,
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
                    f"KalmanGP: P asymmetry {asym:.3e} at observation {i}. Symmetrising.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.state_cov = P_sym

            self._log_lik += -0.5 * np.log(2 * np.pi * s_val) - 0.5 * innov**2 / s_val
            self._n_obs += 1
            prev_t = t_seconds[i]

    def predict(self, x_grid, full_cov=False):
        """Posterior mean and covariance of mu(x, t_now) at x_grid."""
        f_mean = self.state_mean[0::2]
        idx = np.arange(0, 2 * self.M, 2)
        P_ff = self.state_cov[np.ix_(idx, idx)]

        K_qz = rbf_kernel(x_grid, self.inducing_x, self.spatial_ls, self.spatial_var)

        alpha = self._K_zz_inv @ f_mean
        mu_mean = K_qz @ alpha

        V = self._K_zz_inv @ K_qz.T

        if full_cov:
            K_qq = rbf_kernel(x_grid, x_grid, self.spatial_ls, self.spatial_var)
            mu_cov = K_qq - K_qz @ V + K_qz @ (self._K_zz_inv @ P_ff @ V)
            # Jitter proportional to prior variance: robust when K_qq is nearly
            # singular (dense grid relative to small spatial_ls).
            mu_cov += max(1e-6 * self.spatial_var, 1e-10) * np.eye(len(x_grid))
        else:
            # diag(K_qz K_zz^-1 P_ff K_zz^-1 K_qz^T) = diag(V.T P_ff V)
            # since V = K_zz^-1 K_qz.T (and K_zz^-1 is symmetric).
            # The previous form omitted the right-hand K_zz^-1, inflating
            # the posterior variance by ~spatial_var.
            K_qq_diag = self.spatial_var * np.ones(len(x_grid))
            mu_cov = (
                K_qq_diag - np.sum(K_qz * V.T, axis=1) + np.sum(V * (P_ff @ V), axis=0)
            )
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
                "sample_drift Cholesky failed — K_post not PD even with large jitter."
            )

        z = rng.standard_normal((len(x_grid), n_samples))
        return mu_mean[:, None] + L_chol @ z

    def optimise_hp(
        self,
        x_prev_subset,
        r_hat_subset,
        t_seconds_subset,
        n_restarts=3,
        x_range=None,
        bounds_range=None,
        fix_spatial_var=False,
        console=None,
    ):
        """Maximise the marginal log-likelihood over kernel HPs via multi-restart L-BFGS-B.

        x_range       — inducing-point bracket for MLL evaluation; falls back
                        to the subset's min/max.
        bounds_range  — x-range used ONLY for computing ls_lo/ls_hi bounds.
                        Defaults to x_range. Pass the typical per-window range
                        here when x_range covers the full multi-year history so
                        that the bounds are calibrated to the scale at which
                        topology is actually evaluated (per window after reproject).
        fix_spatial_var — when True, optimise only spatial_ls and temporal_ls
            (2-parameter MLE). self.spatial_var is held fixed throughout.
            Use this when spatial_var has been set externally from KM data,
            since the MLL is nearly flat in spatial_var at typical crypto SNR.
            When False (default), all three HPs are jointly optimised.
        """
        console = console or Console()
        if x_range is None:
            x_range = (float(x_prev_subset.min()), float(x_prev_subset.max()))
        n_ind = self.M if self.M is not None else 20

        # Lower bound on spatial_ls: 1/3 of the inducing spacing.  Allowing ls
        # below the inducing gap risks free wiggle between inducing points, but
        # with a dense grid (N_INDUCING≥40) this rarely causes fake multi-well
        # artefacts while letting the optimizer capture KM-like fine structure.
        # Upper bound = half the x-range, capped at 0.1: a lengthscale equal
        # to the full window range makes the GP nearly constant (p_multi → 0).
        # bounds_range should be the WINDOW-scale range, not the global range,
        # so that the bounds are calibrated to where topology is evaluated.
        _brange = bounds_range if bounds_range is not None else x_range
        x_width = _brange[1] - _brange[0]
        ls_lo = max(x_width / (3 * n_ind), 1e-4)
        ls_hi = max(min(x_width / 2.0, 0.1), ls_lo * 1.5)

        # --- shared inner model factory ---
        def _run_model(sp_ls, tmp_ls, sp_var):
            sp_ls = np.clip(sp_ls, ls_lo, ls_hi)
            tmp_ls = np.clip(tmp_ls, 0.5, 180.0)
            sp_var = np.clip(sp_var, 1e-12, 1e6)
            m = KalmanGPDriftModel(
                spatial_lengthscale=sp_ls,
                temporal_lengthscale_days=tmp_ls,
                spatial_variance=sp_var,
                sigma2=self.sigma2,
                dt=self.dt,
            )
            m.initialise(x_range, n_inducing=n_ind, data_x=x_prev_subset)
            m.update(x_prev_subset, r_hat_subset, t_seconds_subset)
            return m._log_lik

        if fix_spatial_var:
            # --- 2-parameter optimisation: spatial_ls, temporal_ls only ---
            # spatial_var is fixed at self.spatial_var and not updated.
            fixed_sp_var = self.spatial_var

            def _neg_log_lik_2(log_params):
                sp_ls, tmp_ls = np.exp(log_params)
                ll = _run_model(sp_ls, tmp_ls, fixed_sp_var)
                if not np.isfinite(ll):
                    warnings.warn(
                        f"HP opt (ls-only): non-finite log_lik={ll} at "
                        f"sp_ls={sp_ls:.4f} tmp_ls={tmp_ls:.2f}d. "
                        "Returning penalty.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                return 1e10 if not np.isfinite(ll) else -ll

            best_nll = np.inf
            best_pars = np.log([self.spatial_ls, self.temporal_ls])
            rng_hp = np.random.default_rng(0)

            for restart in range(n_restarts):
                p0 = (
                    np.log([self.spatial_ls, self.temporal_ls])
                    if restart == 0
                    else np.log(
                        [
                            rng_hp.uniform(ls_lo, ls_hi),
                            rng_hp.uniform(2.0, 60.0),
                        ]
                    )
                )
                try:
                    res = minimize(
                        _neg_log_lik_2,
                        p0,
                        method="L-BFGS-B",
                        bounds=[
                            (np.log(ls_lo), np.log(ls_hi)),
                            (np.log(0.5), np.log(180.0)),
                        ],
                        options={"maxiter": 100, "ftol": 1e-6},
                    )
                    if res.fun < best_nll:
                        best_nll = res.fun
                        best_pars = res.x
                    console.print(
                        f"  [cyan]HP opt (ls-only) restart {restart + 1}/{n_restarts}[/cyan]  "
                        f"nll={res.fun:.4e}  "
                        f"sp_ls={np.exp(res.x[0]):.4f}  "
                        f"tmp_ls={np.exp(res.x[1]):.2f}d  "
                        f"sp_var={fixed_sp_var:.4g} [fixed]"
                    )
                except Exception as exc:
                    console.print(
                        f"  [yellow]Restart {restart + 1} failed: {exc}[/yellow]"
                    )

            self.spatial_ls = float(np.exp(best_pars[0]))
            self.temporal_ls = float(np.exp(best_pars[1]))
            # spatial_var intentionally NOT updated here.
            self._recompute_hp_dependent()
            console.print(
                f"[green]HP optimised (ls-only):[/green] "
                f"spatial_ls={self.spatial_ls:.4f}  "
                f"temporal_ls={self.temporal_ls:.2f}d  "
                f"spatial_var={self.spatial_var:.4g} [unchanged]"
            )

        else:
            # --- 3-parameter optimisation: spatial_ls, temporal_ls, spatial_var ---
            def _neg_log_lik_3(log_params):
                sp_ls, tmp_ls, sp_var = np.exp(log_params)
                ll = _run_model(sp_ls, tmp_ls, sp_var)
                if not np.isfinite(ll):
                    warnings.warn(
                        f"HP opt: non-finite log_lik={ll} at "
                        f"sp_ls={sp_ls:.4f} "
                        f"tmp_ls={tmp_ls:.2f}d "
                        f"sp_var={sp_var:.4f}. Returning penalty.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                return 1e10 if not np.isfinite(ll) else -ll

            best_nll = np.inf
            best_pars = np.log([self.spatial_ls, self.temporal_ls, self.spatial_var])
            rng_hp = np.random.default_rng(0)

            for restart in range(n_restarts):
                p0 = (
                    np.log([self.spatial_ls, self.temporal_ls, self.spatial_var])
                    if restart == 0
                    else np.log(
                        [
                            rng_hp.uniform(ls_lo, ls_hi),
                            rng_hp.uniform(2.0, 60.0),
                            rng_hp.uniform(0.01, 10.0),
                        ]
                    )
                )
                try:
                    res = minimize(
                        _neg_log_lik_3,
                        p0,
                        method="L-BFGS-B",
                        bounds=[
                            (np.log(ls_lo), np.log(ls_hi)),
                            (np.log(0.5), np.log(180.0)),
                            (np.log(1e-12), np.log(1e6)),
                        ],
                        options={"maxiter": 100, "ftol": 1e-6},
                    )
                    if res.fun < best_nll:
                        best_nll = res.fun
                        best_pars = res.x
                    console.print(
                        f"  [cyan]HP opt restart {restart + 1}/{n_restarts}[/cyan]  "
                        f"nll={res.fun:.4e}  "
                        f"sp_ls={np.exp(res.x[0]):.4f}  "
                        f"tmp_ls={np.exp(res.x[1]):.2f}d  "
                        f"sp_var={np.exp(res.x[2]):.4f}"
                    )
                except Exception as exc:
                    console.print(
                        f"  [yellow]Restart {restart + 1} failed: {exc}[/yellow]"
                    )

            self.spatial_ls = float(np.exp(best_pars[0]))
            self.temporal_ls = float(np.exp(best_pars[1]))
            self.spatial_var = float(np.exp(best_pars[2]))
            self._recompute_hp_dependent()
            console.print(
                f"[green]HP optimised:[/green] "
                f"spatial_ls={self.spatial_ls:.4f}  "
                f"temporal_ls={self.temporal_ls:.2f}d  "
                f"spatial_var={self.spatial_var:.4f}"
            )

    def get_params(self):
        return {
            "spatial_lengthscale": self.spatial_ls,
            "temporal_lengthscale_days": self.temporal_ls,
            "spatial_variance": self.spatial_var,
            "obs_noise": self.obs_noise,
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
    rng=None,
):
    """Topology statistics from the current GP posterior — p_multiwell, barrier, Kramers.

    Assumes the GP state is already in annualised units (mu in /year, mu_var in
    /year²); see run_phase_gp where r is multiplied by _SEC_PER_YEAR.
    """
    rng = rng or np.random.default_rng(42)
    x_grid = np.linspace(x_range[0], x_range[1], n_grid)

    mu_mean, mu_var = model.predict(x_grid, full_cov=False)
    f_samples = model.sample_drift(x_grid, n_samples=n_samples, rng=rng)

    # Identifiability diagnostic: mean posterior std vs mean |drift|. A value
    # ≳1 means p_multiwell is mostly tracking posterior uncertainty rather
    # than mean-field structure.
    mu_std_mean = float(np.mean(np.sqrt(np.maximum(mu_var, 0.0))))
    mu_mag_mean = float(np.mean(np.abs(mu_mean)))
    mu_std_to_mean = (
        float(mu_std_mean / mu_mag_mean) if mu_mag_mean > 0 else float("inf")
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
            x_grid,
            U_j,
            threshold=min_barrier_fraction * u_range_j,
            min_well_separation=0.0,
        )
        n_stable[j] = len(kept_idx_j)
        barrier_samples[j] = max(bars_j) if bars_j else 0.0

    p_multiwell = float(np.mean(n_stable >= 2))
    mean_n_wells = float(np.mean(n_stable))

    if u_range < 1e-12:
        return {
            "p_multiwell": round(p_multiwell, 4),
            "mean_n_wells": round(mean_n_wells, 2),
            "barrier_mean": 0.0,
            "barrier_std": 0.0,
            "kramers_mean": 0.0,
            "kramers_std": 0.0,
            "well_locations": [],
            "u_range": 0.0,
            "mu_std_to_mean": round(mu_std_to_mean, 4),
        }

    kept_idx, _ = _greedy_well_filter(
        x_grid,
        U_mean,
        threshold=min_barrier_fraction * u_range,
        min_well_separation=0.0,
    )
    well_locations = [float(x_grid[i]) for i in kept_idx]

    barrier_mean = float(np.mean(barrier_samples))
    barrier_std = float(np.std(barrier_samples))

    # Kramers exponent barrier/D must be dimensionless.
    # sigma2 is stored as var(r_hat_ann) * dt_sec with units [/year]² · sec, so
    # D_year = sigma2 / (2 · sec_per_year) brings D into [/year], matching the
    # barrier units (U = -∫mu dx with mu in /year).
    D = model.sigma2 / (2.0 * _SEC_PER_YEAR)
    if D > 0:
        kramers_samples = np.exp(-barrier_samples / D)
    else:
        kramers_samples = np.zeros(n_samples)
    kramers_mean = float(np.mean(kramers_samples))
    kramers_std = float(np.std(kramers_samples))

    return {
        "p_multiwell": round(p_multiwell, 4),
        "mean_n_wells": round(mean_n_wells, 2),
        "barrier_mean": round(barrier_mean, 6),
        "barrier_std": round(barrier_std, 6),
        "kramers_mean": round(kramers_mean, 8),
        "kramers_std": round(kramers_std, 8),
        "well_locations": well_locations,
        "u_range": round(u_range, 4),
        "mu_std_to_mean": round(mu_std_to_mean, 4),
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
            fwd_model,
            x_range,
            n_grid=n_grid,
            n_samples=n_samples,
            min_crossing_sep=min_crossing_sep,
            min_barrier_fraction=min_barrier_fraction,
            rng=rng,
        )

        rows.append(
            {
                "horizon_days": h,
                "p_multiwell": topo["p_multiwell"],
                "barrier_mean": topo["barrier_mean"],
                "barrier_std": topo["barrier_std"],
                "kramers_mean": topo["kramers_mean"],
                "kramers_std": topo["kramers_std"],
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------------------------


def step_and_describe(
    model,
    x_prev,
    r_hat,
    t_seconds,
    *,
    x_range,
    n_inducing,
    n_grid=200,
    n_samples=200,
    min_crossing_sep=10,
    min_barrier_fraction=0.1,
    rng=None,
):
    """Kalman update then topology extraction.

    Calls ``model.update`` then ``topology_from_gp`` and returns the topology
    dict.  Use alongside ``model.reproject_to_range`` when a reproject is
    needed before the update.
    """
    model.update(x_prev, r_hat, t_seconds)
    return topology_from_gp(
        model,
        x_range,
        n_grid=n_grid,
        n_samples=n_samples,
        min_crossing_sep=min_crossing_sep,
        min_barrier_fraction=min_barrier_fraction,
        rng=rng,
    )


def daily_replay(
    model,
    x_all,
    dx_all,
    dt_t_pd,
    dt_step,
    *,
    record_from,
    reproject_window_days=30,
    n_inducing,
    n_grid=200,
    n_samples=200,
    min_crossing_sep=10,
    min_barrier_fraction=0.1,
    rng=None,
):
    """Generator: replay the Kalman filter day by day over ``dt_t_pd``.

    Yields ``(date, topo_dict, x_d)`` for every day at or after
    ``record_from``.  Days before ``record_from`` advance the model state but
    are not yielded (warm-up).

    The inducing grid is reprojected once per ISO week using the trailing
    ``reproject_window_days`` x-range — identical to the training loop in
    RunGP.py Step 8.  The reproject fires at the start of each ISO week,
    before any of that week's daily updates, matching training exactly.
    """
    import pandas as _pd  # already imported at module level; re-alias for clarity

    record_ts = _pd.Timestamp(record_from).normalize()
    dt_norm = dt_t_pd.normalize()
    all_days = sorted(dt_norm.unique())
    if not all_days:
        return

    # Group days by ISO week, preserving chronological order.
    day_series = _pd.Series(all_days)
    week_labels = day_series.dt.to_period("W").values

    # Initial reproject: use only data strictly before the first day to be
    # processed — no future information enters the inducing grid placement.
    first_day = all_days[0]
    x_pre = x_all[dt_norm < first_day]
    if len(x_pre) >= 2:
        topo_range = model.reproject_to_range(
            float(x_pre.min()), float(x_pre.max()), n_inducing=n_inducing
        )
    else:
        # No prior data available (GP starts cold); use only the very first
        # bar's price as a single-point fallback — model.reproject_to_range
        # will add a small margin automatically.
        x_first = x_all[dt_norm == first_day]
        x0 = float(x_first.mean()) if len(x_first) else 0.0
        topo_range = model.reproject_to_range(x0, x0, n_inducing=n_inducing)

    for wk in sorted(set(week_labels)):
        wk_days = sorted(d for d, w in zip(all_days, week_labels) if w == wk)
        # Reproject fires *before* this week's updates, so only data up to
        # (but not including) the first day of this week is permissible.
        wk_cutoff = wk_days[0] - _pd.Timedelta(days=1)
        roll_lo = wk_cutoff - _pd.Timedelta(days=reproject_window_days - 1)
        roll_mask = (dt_norm >= roll_lo) & (dt_norm <= wk_cutoff)
        x_roll = x_all[roll_mask]
        if len(x_roll) < 2:
            x_roll = x_all[dt_norm <= wk_cutoff]
        if len(x_roll) >= 2:
            topo_range = model.reproject_to_range(
                float(x_roll.min()), float(x_roll.max()), n_inducing=n_inducing
            )

        for d in wk_days:
            mask = dt_norm == d
            x_d = x_all[mask]
            dx_d = dx_all[mask]
            t_d = (np.asarray(dt_t_pd[mask].astype(np.int64)) / 1e9).astype(float)
            r_hat_d = (dx_d / dt_step) * _SEC_PER_YEAR

            model.update(x_d, r_hat_d, t_d)

            if d < record_ts:
                continue  # warm-up: advance state only

            topo_d = topology_from_gp(
                model,
                topo_range,
                n_grid=n_grid,
                n_samples=n_samples,
                min_crossing_sep=min_crossing_sep,
                min_barrier_fraction=min_barrier_fraction,
                rng=rng,
            )
            yield d, topo_d, x_d
