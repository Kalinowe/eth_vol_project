import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from phase_GP import topology_from_gp, _SEC_PER_YEAR


# ---------------------------------------------------------------------------
# Geometric well-jump detection  (price-only; no topology used)
# ---------------------------------------------------------------------------


def detect_price_jumps(
    log_price: pd.Series,
    stable_days: int = 4,
    stable_thr: float = 0.05,
    jump_thr: float = 0.1,
    settle_days: int = 4,
) -> pd.DataFrame:
    """Detect well-jump events from a daily log-price series.

    Identical logic to ``_detect_jumps`` in backtest_jumps.py.  Returns a
    DataFrame with columns: event_id, pre_stable_start, pre_stable_end,
    jump_start, post_stable_start, pre_log_price, post_log_price,
    log_price_change, direction.  Returns an empty DataFrame if no events.
    """
    lp = log_price.sort_index()
    roll_range = lp.rolling(stable_days, min_periods=stable_days).apply(
        lambda w: w.max() - w.min(), raw=True
    )
    dates = lp.index.tolist()
    n = len(dates)
    events = []
    i = 0
    while i < n:
        if pd.isna(roll_range.iloc[i]) or roll_range.iloc[i] >= stable_thr:
            i += 1
            continue
        stable_end_idx = i
        anchor_lp = lp.iloc[i]

        j = i + 1
        while j < n and abs(lp.iloc[j] - anchor_lp) < jump_thr:
            j += 1
        if j >= n:
            break
        jump_start_date = dates[j]

        k = j + 1
        settled = False
        while k + settle_days <= n:
            window = lp.iloc[k : k + settle_days]
            if window.max() - window.min() < stable_thr:
                settled = True
                break
            k += 1
        if not settled:
            i = j + 1
            continue

        post_stable_start = dates[k]
        post_lp_median = float(lp.iloc[k : k + settle_days].median())

        pre_start_idx = stable_end_idx
        while pre_start_idx > 0:
            c = pre_start_idx - 1
            if not pd.isna(roll_range.iloc[c]) and roll_range.iloc[c] < stable_thr:
                pre_start_idx = c
            else:
                break
        pre_stable_start = dates[pre_start_idx]
        pre_lp_median = float(lp.iloc[pre_start_idx : stable_end_idx + 1].median())
        log_change = post_lp_median - pre_lp_median
        events.append(
            {
                "event_id": len(events) + 1,
                "pre_stable_start": pd.Timestamp(pre_stable_start),
                "pre_stable_end": pd.Timestamp(dates[stable_end_idx]),
                "jump_start": pd.Timestamp(jump_start_date),
                "post_stable_start": pd.Timestamp(post_stable_start),
                "pre_log_price": pre_lp_median,
                "post_log_price": post_lp_median,
                "log_price_change": log_change,
                "direction": "up" if log_change > 0 else "down",
            }
        )
        i = k + settle_days
    return pd.DataFrame(events)


# ---------------------------------------------------------------------------
# Sequential Kalman-GP snapshots and overlays
# ---------------------------------------------------------------------------


def plot_topology_snapshots(
    model,
    x_range_global,
    snapshots,
    snapped_start,
    snapped_end,
    phase_gp_si,
    out_path,
    n_grid=200,
    n_samples=200,
    rng=None,
):
    """
    Grid of panels (one per snapshot) showing the potential U(x) = -∫μ(x)dx,
    the integral of the GP posterior drift.  Each panel title shows the date,
    p_multiwell, σ/μ, and β (when available).  A ±2σ posterior envelope is
    drawn from drift samples integrated in the same way.
    """
    rng = rng or np.random.default_rng(0)
    if not snapshots:
        print(f"[plot_topology_snapshots] no snapshots; skipping {out_path}")
        return

    n = len(snapshots)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.5 * cols, 3.5 * rows), squeeze=False
    )
    axes_flat = axes.flatten()

    saved_mean = model.state_mean.copy()
    saved_cov = model.state_cov.copy()
    saved_inducing = model.inducing_x.copy()

    for k, ax in enumerate(axes_flat):
        if k >= n:
            ax.axis("off")
            continue

        snap = snapshots[k]
        dt_query, sm, sc, snap_x_range, snap_inducing_x = snap[:5]
        model.state_mean = sm
        model.state_cov = sc
        if not np.array_equal(model.inducing_x, snap_inducing_x):
            model.inducing_x = snap_inducing_x
            model._recompute_hp_dependent()

        x_grid = np.linspace(snap_x_range[0], snap_x_range[1], n_grid)

        # Posterior mean drift → integrate to potential
        mu_mean, mu_var = model.predict(x_grid, full_cov=False)
        U_mean = -cumulative_trapezoid(mu_mean, x_grid, initial=0.0)
        U_mean -= U_mean.min()

        # Drift samples → integrate → baseline-shift → std band
        f_samples = model.sample_drift(x_grid, n_samples=n_samples, rng=rng)
        U_samples = -cumulative_trapezoid(f_samples, x_grid, axis=0, initial=0.0)
        U_samples -= U_samples.min(axis=0, keepdims=True)
        U_std = U_samples.std(axis=1)

        ax.fill_between(
            x_grid,
            U_mean - 2 * U_std,
            U_mean + 2 * U_std,
            color="steelblue",
            alpha=0.2,
            linewidth=0,
            label="±2σ",
        )
        ax.plot(x_grid, U_mean, color="steelblue", linewidth=1.4, label="mean")
        # Mark inducing points on x-axis
        ax.scatter(
            snap_inducing_x,
            np.zeros(len(snap_inducing_x)),
            marker="|",
            color="darkgreen",
            s=60,
            zorder=5,
            label="inducing",
        )

        topo = topology_from_gp(model, snap_x_range, n_grid=50, n_samples=50, rng=rng)
        beta_val = getattr(model, "beta_mean", None)
        beta_str = f"  β={beta_val:+.1f}" if beta_val is not None else ""
        ax.set_title(
            f"{pd.Timestamp(dt_query).date()}  "
            f"p={topo['p_multiwell']:.2f}  "
            f"σ/μ={topo['mu_std_to_mean']:.1f}{beta_str}",
            fontsize=8,
        )
        ax.set_xlabel("log-price", fontsize=7)
        ax.set_ylabel("U(x) = −∫μ dx", fontsize=7)
        ax.set_xlim(snap_x_range[0], snap_x_range[1])
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.legend(fontsize=6, loc="best")

    model.state_mean = saved_mean
    model.state_cov = saved_cov
    if not np.array_equal(model.inducing_x, saved_inducing):
        model.inducing_x = saved_inducing
        model._recompute_hp_dependent()

    fig.suptitle(
        f"GP potential U(x) snapshots  |  "
        f"{snapped_start.date()} – {snapped_end.date()}, {phase_gp_si}s",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_drift_with_km(
    model,
    snapshots,
    snapped_start,
    snapped_end,
    phase_a_si,
    phase_gp_si,
    output_dir,
    out_path,
    spatial_var_source,
    use_reproject,
    n_grid=200,
    rng=None,
    kernel_hw_phase_a=0,
    trim_quantile=0.0,
):
    """
    Grid of panels (one per snapshot) showing GP posterior drift overlaid with
    Phase A KM bin estimates.

    Y-axis uses a weighted percentile of KM drift values so sparse boundary
    bins with extreme drift do not inflate the scale.
    """
    rng = rng or np.random.default_rng(0)
    if not snapshots:
        print(f"[plot_drift_with_km] no snapshots; skipping {out_path}")
        return

    km_dir = os.path.join(output_dir, "km")
    _kernel_tag = f"_k{kernel_hw_phase_a}" if kernel_hw_phase_a > 0 else ""
    _trim_tag = f"_trim{trim_quantile}" if trim_quantile > 0 else ""
    _km_suffix = f"_{phase_a_si}s{_kernel_tag}{_trim_tag}.csv"

    # ---- Y-axis bounds: weighted percentile of KM drift across the full period ----
    all_drifts, all_weights = [], []
    for fname in sorted(os.listdir(km_dir)):
        if not fname.endswith(_km_suffix):
            continue
        parts = fname.replace(".csv", "").split("_")
        try:
            km_start = pd.Timestamp(parts[1])
            km_end = pd.Timestamp(parts[3])
        except Exception:
            continue
        if km_start < pd.Timestamp(snapped_start) or km_end > pd.Timestamp(
            snapped_end
        ) + pd.Timedelta(days=2):
            continue
        df = pd.read_csv(os.path.join(km_dir, fname)).dropna(subset=["drift"])
        if df.empty:
            continue
        all_drifts.extend((df["drift"].values * _SEC_PER_YEAR).tolist())
        all_weights.extend(df["weight"].values.tolist())

    if all_drifts:
        arr = np.asarray(all_drifts)
        wts = np.asarray(all_weights, dtype=float)
        wts = wts / wts.sum()
        order = np.argsort(arr)
        cdf = np.cumsum(wts[order])
        q2 = arr[order[np.searchsorted(cdf, 0.02)]]
        q98 = arr[order[np.searchsorted(cdf, 0.98)]]
        pad = 0.25 * max(abs(q2), abs(q98), 1.0)
        y_lo, y_hi = q2 - pad, q98 + pad
    else:
        y_lo, y_hi = -200.0, 200.0

    n = len(snapshots)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.5 * cols, 3.5 * rows), squeeze=False
    )
    axes_flat = axes.flatten()

    # Preserve model state across all panels.
    saved_mean = model.state_mean.copy()
    saved_cov = model.state_cov.copy()
    saved_inducing = model.inducing_x.copy()

    for k, ax in enumerate(axes_flat):
        if k >= n:
            ax.axis("off")
            continue

        snap = snapshots[k]
        dt_query, sm, sc, snap_x_range, snap_inducing_x = snap[:5]
        win_start = snap[5] if len(snap) > 5 else pd.Timestamp(snapped_start)

        model.state_mean = sm
        model.state_cov = sc
        if not np.array_equal(model.inducing_x, snap_inducing_x):
            model.inducing_x = snap_inducing_x
            model._recompute_hp_dependent()

        x_grid = np.linspace(snap_x_range[0], snap_x_range[1], n_grid)
        # Use total drift (mu + beta) for comparison with KM, which estimates
        # total drift.  Falls back to mu-only for the stationary model.
        _predict_total = getattr(model, "predict_total", model.predict)
        mu_mean, mu_var = _predict_total(x_grid, full_cov=False)
        mu_std = np.sqrt(np.maximum(mu_var, 0.0))
        beta_val = getattr(model, "beta_mean", None)

        ax.axhline(0, color="grey", linewidth=0.5)
        if beta_val is not None:
            ax.axhline(
                beta_val,
                color="orange",
                linewidth=0.9,
                linestyle="--",
                alpha=0.8,
                label=f"β={beta_val:+.1f}",
            )
        ax.fill_between(
            x_grid,
            mu_mean - 2 * mu_std,
            mu_mean + 2 * mu_std,
            color="steelblue",
            alpha=0.2,
            label="GP ±2σ",
        )
        ax.plot(x_grid, mu_mean, color="steelblue", linewidth=1.4, label="GP total")

        # Find the KM CSV whose window contains dt_query.
        km_df = None
        for fname in sorted(os.listdir(km_dir)):
            if not fname.endswith(_km_suffix):
                continue
            parts = fname.replace(".csv", "").split("_")
            try:
                km_start = pd.Timestamp(parts[1])
                km_end = pd.Timestamp(parts[3])
            except Exception:
                continue
            if km_start <= pd.Timestamp(dt_query) <= km_end + pd.Timedelta(days=1):
                cand = pd.read_csv(os.path.join(km_dir, fname)).dropna(subset=["drift"])
                if not cand.empty:
                    km_df = cand
                    break

        if km_df is not None:
            km_sorted = km_df.sort_values("bin_center")
            drift_ann = km_sorted["drift"].values * _SEC_PER_YEAR
            w = km_sorted["weight"].values.astype(float)
            sz = 5 + 25 * w / max(w.max(), 1.0)
            ax.scatter(
                km_sorted["bin_center"].values,
                drift_ann,
                s=sz,
                c="crimson",
                alpha=0.45,
                edgecolors="none",
                zorder=4,
                label="KM bins",
            )

        ax.scatter(
            model.inducing_x,
            np.zeros(model.M),
            marker="|",
            color="darkgreen",
            s=60,
            zorder=5,
            label="inducing",
        )
        ax.set_xlim(snap_x_range[0], snap_x_range[1])
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(
            f"{pd.Timestamp(win_start).date()} – {pd.Timestamp(dt_query).date()}",
            fontsize=8,
        )
        ax.set_xlabel("log-price", fontsize=7)
        ax.set_ylabel("drift [/yr]", fontsize=7)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.legend(fontsize=6, loc="best")

    model.state_mean = saved_mean
    model.state_cov = saved_cov
    if not np.array_equal(model.inducing_x, saved_inducing):
        model.inducing_x = saved_inducing
        model._recompute_hp_dependent()

    fig.suptitle(
        f"GP drift + KM estimates  |  {snapped_start.date()} – {snapped_end.date()}, "
        f"{phase_gp_si}s  |  sp_var={spatial_var_source}  "
        f"reproject={use_reproject}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_logprice_topology(
    model,
    snapshots,
    x_prev_all,
    dt_t_all,
    snapped_start,
    snapped_end,
    phase_gp_si,
    out_path,
    spatial_var_source,
    use_reproject,
    n_grid=200,
    events_df=None,
):
    """
    Grid of panels (one per snapshot) showing:
      - log-price time series for that window (blue line)
      - GP posterior mean zero-crossings as horizontal lines:
          green dashed  = stable well  (drift: + → −, price attracted here)
          red dotted    = unstable fix (drift: − → +, price repelled)
      - vertical orange lines at jump_start dates (from events_df, if provided)
      - p_multiwell and well count in the panel title.
    """
    if not snapshots:
        print(f"[plot_logprice_topology] no snapshots; skipping {out_path}")
        return

    dt_arr = pd.to_datetime(dt_t_all)

    n = len(snapshots)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.5 * cols, 3.2 * rows), squeeze=False
    )
    axes_flat = axes.flatten()

    # Preserve model state.
    saved_mean = model.state_mean.copy()
    saved_cov = model.state_cov.copy()
    saved_inducing = model.inducing_x.copy()

    for k, ax in enumerate(axes_flat):
        if k >= n:
            ax.axis("off")
            continue

        snap = snapshots[k]
        dt_query, sm, sc, snap_x_range, snap_inducing_x = snap[:5]
        win_start = (
            snap[5]
            if len(snap) > 5
            else (
                pd.Timestamp(snapped_start)
                if k == 0
                else pd.Timestamp(snapshots[k - 1][0])
            )
        )
        win_end = pd.Timestamp(dt_query)

        # Slice log-price for this window.
        mask = (dt_arr >= win_start) & (dt_arr <= win_end)
        times_w = dt_arr[mask]
        x_w = np.asarray(x_prev_all)[mask]

        ax.plot(
            times_w, x_w, color="steelblue", linewidth=0.7, alpha=0.8, label="log-price"
        )

        # Restore GP state for this snapshot.
        model.state_mean = sm
        model.state_cov = sc
        if not np.array_equal(model.inducing_x, snap_inducing_x):
            model.inducing_x = snap_inducing_x
            model._recompute_hp_dependent()

        # Find zero-crossings of the GP posterior mean drift.
        x_grid = np.linspace(snap_x_range[0], snap_x_range[1], n_grid)
        mu_mean, _ = model.predict(x_grid, full_cov=False)
        sign_arr = np.sign(mu_mean)
        cross_idx = np.where(np.diff(sign_arr) != 0)[0]

        for ci in cross_idx:
            # Linear interpolation to the exact zero.
            dx = x_grid[ci + 1] - x_grid[ci]
            frac = -mu_mean[ci] / (mu_mean[ci + 1] - mu_mean[ci] + 1e-30)
            x_cross = x_grid[ci] + frac * dx
            # Stable well: drift slope negative at crossing (+ → −).
            is_stable = mu_mean[ci + 1] < mu_mean[ci]
            ax.axhline(
                x_cross,
                color="green" if is_stable else "crimson",
                linestyle="--" if is_stable else ":",
                linewidth=1.1 if is_stable else 0.8,
                alpha=0.85,
                label=("well" if is_stable else "saddle")
                if ci == cross_idx[0]
                else "_",
            )

        # Draw vertical lines at jump_start dates that fall in this window.
        if events_df is not None and not events_df.empty and len(times_w):
            t_lo, t_hi = times_w.min(), times_w.max()
            jump_col = "jump_start" if "jump_start" in events_df.columns else None
            if jump_col:
                for _, ev in events_df.iterrows():
                    jt = pd.Timestamp(ev[jump_col])
                    if t_lo <= jt <= t_hi:
                        ax.axvline(
                            jt,
                            color="darkorange",
                            linewidth=1.2,
                            linestyle="-",
                            alpha=0.85,
                            label="jump"
                            if "jump" not in [h.get_label() for h in ax.get_lines()]
                            else "_",
                        )

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_ylabel("log-price", fontsize=7)

        # p_multiwell from the GP state (reuse stored topo info via quick predict).
        n_stable = sum(1 for ci in cross_idx if mu_mean[ci + 1] < mu_mean[ci])
        ax.set_title(
            f"{win_start.date()} – {win_end.date()}\nwells≈{n_stable}",
            fontsize=7,
        )
        if k == 0:
            ax.legend(fontsize=6, loc="best")

    model.state_mean = saved_mean
    model.state_cov = saved_cov
    if not np.array_equal(model.inducing_x, saved_inducing):
        model.inducing_x = saved_inducing
        model._recompute_hp_dependent()

    fig.suptitle(
        f"Log-price & GP topology  |  {pd.Timestamp(snapped_start).date()} – {pd.Timestamp(snapped_end).date()}, "
        f"{phase_gp_si}s  |  sp_var={spatial_var_source}  "
        f"reproject={use_reproject}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Backtest analysis plots
# ---------------------------------------------------------------------------


def plot_backtest_boxes(
    pre: pd.DataFrame,
    null: pd.DataFrame,
    out_path: str,
    *,
    signals: list,
    offsets: list,
    null_buffer_days: int,
    null_sample_size: int,
    trend_window: int,
    burn_in_start,
    backtest_start,
    backtest_end,
    phase_gp_si: int,
    kernel_hw: int,
    phase_a_si: int,
    kernel_hw_phase_a: int,
) -> None:
    n_events = pre["event_id"].nunique() if "event_id" in pre.columns else len(pre)
    offsets_str = ", ".join(f"{o:+d}d" for o in offsets)
    fig, axes = plt.subplots(len(signals), 1, figsize=(10, 2.4 * len(signals)))
    if len(signals) == 1:
        axes = [axes]
    for ax, sig in zip(axes, signals):
        data, labels, colors = [], [], []
        for off in offsets:
            pv = pre.loc[pre["offset_days"] == off, sig].dropna().values
            nv = null.loc[null["offset_days"] == off, sig].dropna().values
            data.extend([pv, nv])
            labels.extend(
                [f"pre {off:+d}d  (n={len(pv)})", f"null {off:+d}d  (n={len(nv)})"]
            )
            colors.extend(["#d6604d", "#4393c3"])
        positions = np.arange(len(data))
        bp = ax.boxplot(
            data, positions=positions, widths=0.7, patch_artist=True, showfliers=False
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(sig, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(
        f"Pre-jump (red) vs null calm (blue)\n"
        f"offsets: {offsets_str}   events: {n_events}   null buffer: {null_buffer_days}d   "
        f"null draws/offset: {null_sample_size}   trend window: {trend_window}d\n"
        f"burn-in: {burn_in_start.date()} -> {backtest_start.date()}   "
        f"backtest: {backtest_start.date()} -> {backtest_end.date()}   "
        f"GP si: {phase_gp_si}s k={kernel_hw}   KM si: {phase_a_si}s k={kernel_hw_phase_a}",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_backtest_overview(
    daily: pd.DataFrame,
    events: pd.DataFrame,
    out_path: str,
    *,
    slope_z_fn,
    trend_window: int,
    offsets: list,
    backtest_start,
    backtest_end,
    burn_in_start,
    phase_gp_si: int,
    kernel_hw: int,
    phase_a_si: int,
    kernel_hw_phase_a: int,
) -> None:
    """Price + signal panels with jump markers."""
    daily = daily.copy()
    slope_z_vals = np.full(len(daily), np.nan)
    for i in range(len(daily)):
        lo = daily.index[i] - pd.Timedelta(days=trend_window - 1)
        tail_pm = np.asarray(
            daily.loc[
                (daily.index >= lo) & (daily.index <= daily.index[i]), "p_multiwell"
            ],
            dtype=float,
        )
        _, z = slope_z_fn(tail_pm)
        slope_z_vals[i] = z
    daily["slope_z_p_multiwell"] = slope_z_vals
    panels = [
        ("price_usd", "ETH/USDT [$]", "log"),
        ("p_multiwell", "p_multiwell", None),
        ("slope_z_p_multiwell", "slope_z_p_multiwell", None),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 8), sharex=True)
    for ax, (col, ylab, yscale) in zip(axes, panels):
        if col in daily.columns:
            ax.plot(daily.index, daily[col], color="#1f4e79", lw=0.9)
        if yscale:
            ax.set_yscale(yscale)
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(alpha=0.3)
        for _, ev in events.iterrows():
            ax.axvline(
                pd.Timestamp(ev["jump_start"]), color="crimson", lw=0.6, alpha=0.7
            )
    axes[-1].set_xlabel("date")
    n_events = len(events)
    offsets_str = ", ".join(f"{o:+d}d" for o in offsets)
    fig.suptitle(
        f"Kalman-GP topology   backtest: {backtest_start.date()} -> {backtest_end.date()}   "
        f"({n_events} jumps, red lines = jump_start)\n"
        f"burn-in: {burn_in_start.date()} -> {backtest_start.date()}   "
        f"GP si: {phase_gp_si}s k={kernel_hw}   KM si: {phase_a_si}s k={kernel_hw_phase_a}   "
        f"offsets: {offsets_str}   trend window: {trend_window}d",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    print(
        "This module exposes plotting helpers; import its functions from the pipeline scripts."
    )
