"""
Backtest.py — self-contained backtest of Kalman-GP topology signals as
predictors of ETH price well-jumps.

STRUCTURE
---------
There is a single continuous GP run from BURN_IN_START to BACKTEST_END:

  [BURN_IN_START, BACKTEST_START)  — burn-in: model warms up, topology
                                     recorded for look-back buffer.
  [BACKTEST_START, BACKTEST_END]   — backtest: topology recorded every day,
                                     well-jumps detected, signals evaluated.

GP INITIALISATION
-----------------
  • Uses init_gp_pipeline() from Initialise_GP.py for KM pre-period, spatial_var,
    sigma2, and model initialisation.
  • Burn-in uses daily_replay() from gaussian_process.py.
  • Backtest uses a direct daily update loop.

DATA-LEAKAGE GUARANTEE
-----------------------
1. The GP runs strictly forward in time; topology at day d uses data <= d only.
2. Jump labels use a price-only geometric criterion (rolling range + displacement
   thresholds) with no reference to any topology signal.
3. The backtest analysis reads only from the two CSV files written by the GP run.

SIGNALS TESTED
--------------
  p_multiwell           — level: fraction of posterior samples showing >=2 wells
  barrier_snr           — level: barrier_mean / barrier_std (uncertainty-adj.)
  slope_p_multiwell     — trend: OLS slope of p_multiwell over TREND_WINDOW days
  slope_z_p_multiwell   — trend: OLS z-stat of p_multiwell over TREND_WINDOW days

OUTPUTS  (all under BACKTEST_DIR / {backtest_tag}/)
-------
  daily_topology.csv    — day-by-day topology across full period
  events.csv            — detected well-jump events
  per_event.csv         — signal values at each (event, offset) pair
  null_samples.csv      — same schema for null (calm) dates
  summary.csv           — per-signal Mann-Whitney stats + rank-biserial IC vs null
  per_signal_boxes.png  — boxplots: pre-jump vs null at each look-back offset
  events_overview.png   — price + signal panels with jump markers
"""

from __future__ import annotations

import gc
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy.stats import mannwhitneyu

import data_collection as dc
from data_collection import load_series
from gaussian_process import daily_replay, topology_from_gp, _SEC_PER_YEAR
from kramers_moyal import iter_windows, estimate_km
from Initialise_GP import init_gp_pipeline
import plots


# =============================================================================
# KM MONTHLY RECALIBRATION (backtest)
# =============================================================================


def _recalibrate_km_backtest(model, month_start, month_end, console, regime_output_dir):
    """Run KM on one completed backtest month and update model.spatial_var.

    The KM CSV is cached: if it already exists from a previous run it is read
    directly without re-aggregating 30 s data or re-running estimate_km.
    """
    km_dir = regime_output_dir
    os.makedirs(km_dir, exist_ok=True)
    kernel_tag = f"_k{KM_KERNEL_HW}" if KM_KERNEL_HW > 0 else ""
    km_csv = os.path.join(
        km_dir,
        f"km_{month_start.strftime('%Y-%m-%d')}_to_"
        f"{month_end.strftime('%Y-%m-%d')}_{KM_SI}s{kernel_tag}.csv",
    )

    if os.path.exists(km_csv):
        km_df = pd.read_csv(km_csv)
        console.print(
            f"  KM recalib {month_start.strftime('%Y-%m')}: loaded cached CSV"
        )
    else:
        x_a, dx_a, _, _ = load_series(
            month_start,
            month_end,
            KM_SI,
            kernel_half_width=KM_KERNEL_HW,
            trim_quantile=TRIM_QUANTILE,
            ema_halflife_days=0.0,
            window_type="monthly",
        )
        if len(x_a) == 0:
            console.print(
                f"[yellow]  KM recalib {month_start.strftime('%Y-%m')}: no data; skipping[/yellow]"
            )
            return
        log_prices_a = np.concatenate([x_a, x_a[-1:] + dx_a[-1:]]) if len(dx_a) else x_a
        km_df = estimate_km(
            log_prices_a,
            KM_SI,
            n_bins=N_BINS,
            weight_threshold=WEIGHT_THRESHOLD,
        )
        km_df.to_csv(km_csv, index=False)

    km_valid = km_df.dropna(subset=["drift"])
    if km_valid.empty:
        console.print(
            f"[yellow]  KM recalib {month_start.strftime('%Y-%m')}: "
            "all bins below threshold; skipping[/yellow]"
        )
        return

    d_ann = km_valid["drift"].values * _SEC_PER_YEAR
    if len(d_ann) >= 10:
        lo = float(np.quantile(d_ann, 0.02))
        hi = float(np.quantile(d_ann, 0.98))
        d_trim = d_ann[(d_ann >= lo) & (d_ann <= hi)]
        if len(d_trim) >= 3:
            d_ann = d_trim
    new_sp_var = float(np.var(d_ann))

    if new_sp_var < 1.0:
        console.print(
            f"[yellow]  KM recalib {month_start.strftime('%Y-%m')}: "
            f"spatial_var={new_sp_var:.4g} too small; skipping[/yellow]"
        )
        return

    old_sp_var = float(model.spatial_var)
    model.spatial_var = new_sp_var
    model._recompute_hp_dependent()
    console.print(
        f"  [green]KM recalib[/green] {month_start.strftime('%Y-%m')}: "
        f"spatial_var {old_sp_var:.4g} \u2192 {new_sp_var:.4g} /yr\u00b2"
    )


# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Backtest period ---------------------------------------------------------
BURN_IN_START = datetime(2024, 1, 1)  # first day the GP runs
BACKTEST_START = datetime(2024, 7, 1)  # first day topology is recorded
BACKTEST_END = datetime(2025, 12, 31)  # last day

# --- KM initialisation data --------------------------------------------------
KM_INIT_MONTHS = 0  # 0 = use burn-in period itself; >0 = causal pre-period
# EMA half-life in monthly windows for KM spatial-var weighting.
# Newer windows get higher weight; None = uniform (same as RunGP default).
KM_SPATIAL_VAR_EMA_HALFLIFE = None  # e.g. 6 → 6-month half-life

# --- KM configuration -------------------------------------------------------
KM_SI = 30  # aggregation interval for KM estimation (seconds)
KM_KERNEL_HW = 3  # kernel half-width for KM aggregation
N_BINS = 200  # KM histogram bins
WEIGHT_THRESHOLD = 5  # min weight for KM bin to be included
MIN_BARRIER_FRACTION = 0.1  # min barrier / well-depth fraction

# --- GP configuration --------------------------------------------------------
GP_SI = 900  # aggregation interval for the GP (seconds)
KERNEL_HW = 100  # kernel half-width for 900 s aggregation
TRIM_QUANTILE = 0.01  # trim extreme micro-returns
N_INDUCING = 10  # inducing points
TEMPORAL_LS_DAYS = 10.0  # Matern temporal lengthscale (days)
SPATIAL_LENGTHSCALE_INIT = None  # None -> (x_hi-x_lo)/N_INDUCING
SPATIAL_VAR_MODE = (
    "km"  # "fixed" or "km" (recalibrate from KM each month; more responsive)
)
SPATIAL_VAR_FIXED = 1500.0  # used only when SPATIAL_VAR_MODE == "fixed"
SIGMA2 = None  # None -> var(r_hat*dt)/dt from calm periods data
USE_REPROJECT = True
REPROJECT_WINDOW_DAYS = 30  # trailing days for rolling x-range reproject

# --- EMA demeaning -----------------------------------------------------------
EMA_HALFLIFE_DAYS = None  # None -> no demeaning; set e.g. 14.0 to enable

# --- Topology ----------------------------------------------------------------
N_GRID = 200
N_SAMPLES = 120
MIN_CROSSING_SEP = 10
SEED = 42

# --- Well-jump detection (price-geometric; no topology used here) -------------
STABLE_DAYS = 4  # rolling-window size to assess price stability
STABLE_THR = 0.05  # log-price range < this -> "stable"
JUMP_THR = 0.1  # displacement from anchor > this -> "jumping"
SETTLE_DAYS = 4  # consecutive stable days needed to declare new well

# --- Backtest analysis -------------------------------------------------------
OFFSETS = [-2, -1]  # days before jump_start to sample signals
TREND_WINDOW = 5  # look-back for slope-z statistics (must be >= 4)
NULL_BUFFER_DAYS = 10  # calm date must be >= this far from any jump
NULL_SAMPLE_SIZE = 200  # null draws per offset

# --- Output ------------------------------------------------------------------
BACKTEST_DIR = "backtests"


# =============================================================================
# SIGNALS
# =============================================================================

SIGNALS = [
    "p_multiwell",
    "barrier_snr",
    "slope_p_multiwell",
    "slope_z_p_multiwell",
]

ALT_HYPOTHESIS = {
    "p_multiwell": "higher",
    "barrier_snr": "lower",
    "slope_p_multiwell": "higher",
    "slope_z_p_multiwell": "higher",
}


# =============================================================================
# GP RUN + BACKTEST REPLAY
# =============================================================================


def _run_gp_and_record(console: Console, rng, out_dir: str) -> tuple[str, str]:
    """Run the Kalman-GP over [BURN_IN_START, BACKTEST_END] as a single loop.

    Uses init_gp_pipeline from Initialise_GP.py for model initialisation (Steps 1-6),
    then daily_replay for burn-in, and a direct daily update loop for the
    backtest period (no blob serialisation overhead).

    Writes:
      {out_dir}/daily_topology.csv
      {out_dir}/events.csv

    Returns (daily_path, events_path).
    """
    regime_output_dir = os.path.join(out_dir, "km")

    console.print(
        f"[bold]Backtest run[/bold]  "
        f"burn-in=[{BURN_IN_START.date()}, {BACKTEST_START.date()})  "
        f"backtest=[{BACKTEST_START.date()}, {BACKTEST_END.date()}]  "
        f"-> {out_dir}/"
    )

    # ---- Pre-fetch: ensure all raw data + 900s aggregation is on disk -------
    # Do this before burn-in so there is no pause between burn-in and backtest.
    burnin_end = (pd.Timestamp(BACKTEST_START) - pd.Timedelta(days=1)).to_pydatetime()
    console.rule(f"[bold cyan]Pre-fetch — Ensure raw data + aggregate {GP_SI} s series")
    dc.ensure_data(BURN_IN_START, BACKTEST_END)
    for _w_start, _w_end in iter_windows(BURN_IN_START, BACKTEST_END, "monthly"):
        dc.aggregate_log_returns_range(
            _w_start,
            _w_end,
            GP_SI,
            kernel_half_width=KERNEL_HW,
            trim_quantile=TRIM_QUANTILE,
            ema_halflife_days=EMA_HALFLIFE_DAYS,
        )
    # Pre-aggregate 30s KM data for the full period so that
    # _recalibrate_km_backtest can read each backtest month on demand.
    if SPATIAL_VAR_MODE == "km":
        for _w_start, _w_end in iter_windows(BURN_IN_START, BACKTEST_END, "monthly"):
            dc.aggregate_log_returns_range(
                _w_start,
                _w_end,
                KM_SI,
                kernel_half_width=KM_KERNEL_HW,
                trim_quantile=TRIM_QUANTILE,
                ema_halflife_days=0.0,
            )

    # ---- Steps 1-6: initialise model via shared pipeline --------------------
    result = init_gp_pipeline(
        BURN_IN_START,
        burnin_end,
        km_init_months=KM_INIT_MONTHS,
        km_ema_halflife_windows=KM_SPATIAL_VAR_EMA_HALFLIFE,
        km_si=KM_SI,
        km_kernel_hw=KM_KERNEL_HW,
        gp_si=GP_SI,
        kernel_hw=KERNEL_HW,
        trim_quantile=TRIM_QUANTILE,
        n_bins=N_BINS,
        weight_threshold=WEIGHT_THRESHOLD,
        n_inducing=N_INDUCING,
        temporal_ls_days=TEMPORAL_LS_DAYS,
        spatial_lengthscale_init=SPATIAL_LENGTHSCALE_INIT,
        spatial_var_mode=SPATIAL_VAR_MODE,
        spatial_var_fixed=SPATIAL_VAR_FIXED,
        sigma2_override=SIGMA2,
        sigma2_stable_days=STABLE_DAYS,
        sigma2_stable_thr=STABLE_THR,
        ema_halflife_days=EMA_HALFLIFE_DAYS,
        km_output_dir=regime_output_dir,
        console=console,
    )
    model = result["model"]
    x_burnin = result["x_all"]
    dx_burnin = result["dx_all"]
    r_hat_burnin = result["r_hat_all"]
    dt_step = result["dt_step"]
    dt_t_pd = result["dt_t_pd"]

    # ---- Step 7: daily_replay over burn-in ----------------------------------
    # Records topology for every burn-in day so the look-back window for slope
    # signals is fully populated before the first backtest day.
    console.rule(
        f"[bold cyan]Step 7 — GP burn-in replay  [{BURN_IN_START.date()}, {burnin_end}]"
    )
    burn_in_rows: list[dict] = []
    for d, topo_d, x_d in daily_replay(
        model,
        x_burnin,
        dx_burnin,
        dt_t_pd,
        dt_step,
        record_from=pd.Timestamp(BURN_IN_START),
        r_hat_all=r_hat_burnin,
        reproject_window_days=REPROJECT_WINDOW_DAYS,
        n_inducing=N_INDUCING,
        n_grid=N_GRID,
        n_samples=N_SAMPLES,
        min_crossing_sep=MIN_CROSSING_SEP,
        min_barrier_fraction=MIN_BARRIER_FRACTION,
        rng=rng,
    ):
        burn_in_rows.append(
            {
                "date": d,
                "log_price": float(np.median(x_d)),
                "price_usd": float(np.exp(np.median(x_d))),
                "p_multiwell": topo_d["p_multiwell"],
                "barrier_mean": topo_d["barrier_mean"],
                "barrier_std": topo_d["barrier_std"],
                "barrier_snr": topo_d["barrier_mean"] / topo_d["barrier_std"]
                if topo_d["barrier_std"] > 0
                else 0.0,
            }
        )
        console.print(
            f"  {d.date()}  $={burn_in_rows[-1]['price_usd']:.0f}"
            f"  p_multi={topo_d['p_multiwell']:.2f}"
            f"  barrier={topo_d['barrier_mean']:.3f}\u00b1{topo_d['barrier_std']:.3f}"
        )
    del dx_burnin, r_hat_burnin
    # Keep x_burnin and dt_t_pd alive — the Step 8 reprojection look-back
    # window may extend into the burn-in period for the first few weeks.
    gc.collect()

    console.rule(
        "[bold cyan]Step 8 — Backtest day-by-day  "
        f"[{BACKTEST_START.date()}, {BACKTEST_END.date()}]"
    )

    # Load full backtest series
    x_bt, dx_bt, dt_bt, dt_t_bt = load_series(
        BACKTEST_START,
        BACKTEST_END,
        GP_SI,
        kernel_half_width=KERNEL_HW,
        ema_halflife_days=EMA_HALFLIFE_DAYS,
        window_type="monthly",
    )
    dt_t_bt_pd = pd.to_datetime(dt_t_bt)
    r_hat_bt = (dx_bt / dt_bt) * _SEC_PER_YEAR

    dt_norm_bt = dt_t_bt_pd.normalize()
    all_bt_days = sorted(dt_norm_bt.unique())

    # Group by ISO week for reprojection
    day_series_bt = pd.Series(all_bt_days)
    week_labels_bt = day_series_bt.dt.to_period("W").values

    # Month-end lookup for KM recalibration
    # Build recalibration trigger from the actual last data day of each
    # calendar month in the backtest series.  Avoids mismatch between
    # calendar month-ends and the real last 900 s bar of the month.
    # Keys stored as datetime.date to avoid pd.Timestamp hash/type issues.
    _month_to_last: dict[tuple, pd.Timestamp] = {}
    for _d in all_bt_days:
        _ym = (_d.year, _d.month)
        if _ym not in _month_to_last or _d > _month_to_last[_ym]:
            _month_to_last[_ym] = _d
    recalib_day_map: dict[object, tuple] = {}
    for (_yr, _mo), _last_d in _month_to_last.items():
        _m_start = datetime(_yr, _mo, 1)
        _m_end = (
            datetime(_yr + 1, 1, 1) - timedelta(days=1)
            if _mo == 12
            else datetime(_yr, _mo + 1, 1) - timedelta(days=1)
        )
        recalib_day_map[_last_d.date()] = (_m_start, _m_end)  # .date() for safe lookup
    console.print(
        f"  KM recalibration scheduled on: "
        + ", ".join(str(k) for k in sorted(recalib_day_map))
    )

    # Current topo_range from model's inducing grid
    topo_range = (float(model.inducing_x.min()), float(model.inducing_x.max()))
    last_reproject_wk = None

    backtest_rows: list[dict] = []
    backtest_monthly_snaps: list[tuple] = []
    for wk in sorted(set(week_labels_bt)):
        wk_days = sorted(d for d, w in zip(all_bt_days, week_labels_bt) if w == wk)

        # Weekly reproject
        if last_reproject_wk is None or wk != last_reproject_wk:
            wk_cutoff = wk_days[0] - pd.Timedelta(days=1)
            roll_lo = wk_cutoff - pd.Timedelta(days=REPROJECT_WINDOW_DAYS - 1)
            roll_mask = (dt_t_bt_pd.normalize() >= roll_lo) & (
                dt_t_bt_pd.normalize() <= wk_cutoff
            )
            x_roll = x_bt[roll_mask]
            # If the look-back window extends into the burn-in period, prepend
            # burn-in observations so early backtest weeks get a full window.
            if roll_lo < pd.Timestamp(BACKTEST_START):
                bi_mask = (dt_t_pd.normalize() >= roll_lo) & (
                    dt_t_pd.normalize() <= wk_cutoff
                )
                x_roll = np.concatenate([x_burnin[bi_mask], x_roll])
            if len(x_roll) >= 2:
                topo_range = model.reproject_to_range(
                    float(x_roll.min()), float(x_roll.max()), n_inducing=N_INDUCING
                )
            last_reproject_wk = wk

        for d in wk_days:
            mask = dt_norm_bt == d
            x_d = x_bt[mask]
            r_hat_d = r_hat_bt[mask]
            t_d = (np.asarray(dt_t_bt_pd[mask].astype(np.int64)) / 1e9).astype(float)

            if len(x_d) == 0:
                continue

            model.update(x_d, r_hat_d, t_d)

            # Month-end KM recalibration (uses 30s data, not 900s) and snapshot.
            # Use .date() for lookup to avoid pd.Timestamp hash/type issues.
            d_ts = pd.Timestamp(d).normalize()
            if d_ts.date() in recalib_day_map:
                m_start_r, m_end_r = recalib_day_map[d_ts.date()]
                if SPATIAL_VAR_MODE == "km":
                    _recalibrate_km_backtest(
                        model, m_start_r, m_end_r, console, regime_output_dir
                    )
                backtest_monthly_snaps.append(
                    (
                        d_ts,
                        model.state_mean.copy(),
                        model.state_cov.copy(),
                        topo_range,
                        model.inducing_x.copy(),
                        pd.Timestamp(m_start_r),
                        pd.Timestamp(m_end_r),
                    )
                )

            topo_d = topology_from_gp(
                model,
                topo_range,
                n_grid=N_GRID,
                n_samples=N_SAMPLES,
                min_crossing_sep=MIN_CROSSING_SEP,
                min_barrier_fraction=MIN_BARRIER_FRACTION,
                rng=rng,
            )
            backtest_rows.append(
                {
                    "date": pd.Timestamp(d).normalize(),
                    "log_price": float(np.median(x_d)),
                    "price_usd": float(np.exp(np.median(x_d))),
                    "p_multiwell": topo_d["p_multiwell"],
                    "barrier_mean": topo_d["barrier_mean"],
                    "barrier_std": topo_d["barrier_std"],
                    "barrier_snr": topo_d["barrier_mean"] / topo_d["barrier_std"]
                    if topo_d["barrier_std"] > 0
                    else 0.0,
                }
            )
            console.print(
                f"  {d.date()}  $={backtest_rows[-1]['price_usd']:.0f}"
                f"  p_multi={topo_d['p_multiwell']:.2f}"
                f"  barrier={topo_d['barrier_mean']:.3f}\u00b1{topo_d['barrier_std']:.3f}"
            )
    del x_bt, dx_bt, r_hat_bt, x_burnin, dt_t_pd
    gc.collect()

    # ---- Detect well-jumps on backtest period only --------------------------
    console.rule("[bold cyan]Well-jump detection (price-geometric, backtest period)")
    all_rows = burn_in_rows + backtest_rows
    daily_df = pd.DataFrame(all_rows).set_index("date")
    daily_df.index = pd.to_datetime(daily_df.index)
    events_df = _detect_jumps(
        daily_df[daily_df.index >= pd.Timestamp(BACKTEST_START)], console
    )

    # ---- Write CSV files ----------------------------------------------------
    daily_path = os.path.join(out_dir, "daily_topology.csv")
    events_path = os.path.join(out_dir, "events.csv")
    daily_df.reset_index().to_csv(daily_path, index=False)
    console.print(f"  Wrote {daily_path}")
    events_df.to_csv(events_path, index=False)
    if not events_df.empty:
        console.print(f"  Wrote {events_path} ({len(events_df)} events)")
    else:
        console.print("[yellow]  No events detected; wrote empty events.csv[/yellow]")

    del all_rows, daily_df, events_df
    gc.collect()

    return daily_path, events_path, model, backtest_monthly_snaps, regime_output_dir


# =============================================================================
# JUMP DETECTION  (pure price; topology not used)
# =============================================================================


def _detect_jumps(daily_df: pd.DataFrame, console: Console) -> pd.DataFrame:
    """Price-geometric well-jump detection.

    Only log_price is used.  No topology signal influences the labels.
    """
    lp = daily_df["log_price"]
    roll_range = lp.rolling(STABLE_DAYS, min_periods=STABLE_DAYS).apply(
        lambda w: w.max() - w.min(), raw=True
    )
    dates = daily_df.index.tolist()
    n = len(dates)
    events = []
    i = 0
    while i < n:
        if pd.isna(roll_range.iloc[i]) or roll_range.iloc[i] >= STABLE_THR:
            i += 1
            continue
        stable_end_idx = i
        anchor_lp = lp.iloc[i]

        j = i + 1
        while j < n and abs(lp.iloc[j] - anchor_lp) < JUMP_THR:
            j += 1
        if j >= n:
            break
        jump_start_date = dates[j]

        k = j + 1
        settled = False
        while k + SETTLE_DAYS <= n:
            window = lp.iloc[k : k + SETTLE_DAYS]
            if window.max() - window.min() < STABLE_THR:
                settled = True
                break
            k += 1
        if not settled:
            i = j + 1
            continue

        post_stable_start = dates[k]
        post_lp_median = float(lp.iloc[k : k + SETTLE_DAYS].median())

        pre_start_idx = stable_end_idx
        while pre_start_idx > 0:
            c = pre_start_idx - 1
            if not pd.isna(roll_range.iloc[c]) and roll_range.iloc[c] < STABLE_THR:
                pre_start_idx = c
            else:
                break
        pre_stable_start = dates[pre_start_idx]
        pre_lp_median = float(lp.iloc[pre_start_idx : stable_end_idx + 1].median())

        log_change = post_lp_median - pre_lp_median
        events.append(
            {
                "event_id": len(events) + 1,
                "pre_stable_start": pre_stable_start.date(),
                "pre_stable_end": dates[stable_end_idx].date(),
                "jump_start": jump_start_date.date(),
                "post_stable_start": post_stable_start.date(),
                "pre_log_price": round(pre_lp_median, 5),
                "post_log_price": round(post_lp_median, 5),
                "pre_price_usd": round(float(np.exp(pre_lp_median)), 1),
                "post_price_usd": round(float(np.exp(post_lp_median)), 1),
                "log_price_change": round(log_change, 5),
                "direction": "up" if log_change > 0 else "down",
            }
        )
        console.print(
            f"  [bold]Event {len(events)}[/bold]  "
            f"{'UP' if log_change > 0 else 'DOWN'}  "
            f"{pre_stable_start.date()} -> {jump_start_date.date()} -> {post_stable_start.date()}  "
            f"${np.exp(pre_lp_median):.0f} -> ${np.exp(post_lp_median):.0f}  "
            f"\u0394log={log_change:+.3f}"
        )
        i = k + SETTLE_DAYS

    if not events:
        console.print("[yellow]  No well-jumps detected.[/yellow]")
    return pd.DataFrame(events)


# =============================================================================
# BACKTEST ANALYSIS (reads CSV only; no model state in scope)
# =============================================================================


def _slope_z(y: np.ndarray) -> tuple[float, float]:
    """OLS slope vs index and two-sided z-stat. (nan, nan) if degenerate."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 3:
        return float("nan"), float("nan")
    x = np.arange(n, dtype=float)
    x_c = x - x.mean()
    sxx = float(np.sum(x_c * x_c))
    if sxx <= 0:
        return float("nan"), float("nan")
    y_c = y - y.mean()
    slope = float(np.sum(x_c * y_c) / sxx)
    resid = y_c - slope * x_c
    s2 = float(np.sum(resid * resid) / (n - 2))
    se = float(np.sqrt(s2 / sxx)) if s2 > 0 else 0.0
    z = slope / se if se > 0 else float("nan")
    return slope, z


def _sample_signals(daily: pd.DataFrame, date: pd.Timestamp) -> dict | None:
    """Compute all signals at *date* using only data <= date."""
    if date not in daily.index:
        return None
    row = daily.loc[date]
    bm = float(row["barrier_mean"])
    bst = float(row["barrier_std"])
    snr = bm / bst if bst > 0 else np.nan

    lo = date - pd.Timedelta(days=TREND_WINDOW - 1)
    tail = daily.loc[(daily.index >= lo) & (daily.index <= date)]
    sl_pm, z_pm = _slope_z(tail["p_multiwell"].values)
    return {
        "date": date,
        "p_multiwell": float(row["p_multiwell"]),
        "barrier_snr": snr,
        "slope_p_multiwell": sl_pm,
        "slope_z_p_multiwell": z_pm,
    }


def _bh_correct(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction over a 1-D array of p-values."""
    out = np.full_like(pvals, np.nan, dtype=float)
    finite = np.isfinite(pvals)
    idx = np.where(finite)[0]
    if len(idx) == 0:
        return out
    m = len(idx)
    order = idx[np.argsort(pvals[idx])]
    ranks = np.arange(1, m + 1)
    p_adj = np.minimum(1.0, pvals[order] * m / ranks)
    for i in range(m - 2, -1, -1):
        p_adj[i] = min(p_adj[i], p_adj[i + 1])
    out[order] = p_adj
    return out


def _run_backtest(
    console: Console,
    rng,
    out_dir: str,
    daily_path: str,
    events_path: str,
) -> None:
    """Statistical backtest: are signals elevated before detected jumps?"""
    console.rule("[bold cyan]Load topology + events from CSV")
    daily = pd.read_csv(daily_path, parse_dates=["date"]).set_index("date").sort_index()
    daily.index = daily.index.normalize()  # ensure midnight for offset lookups
    events = pd.read_csv(events_path, parse_dates=["jump_start"])
    console.print(
        f"  {len(daily)} daily rows  "
        f"({daily.index.min().date()} -> {daily.index.max().date()})"
    )
    console.print(f"  {len(events)} jump events")

    if events.empty:
        console.print("[yellow]No events — backtest analysis skipped.[/yellow]")
        return

    # ---- Pre-jump samples ---------------------------------------------------
    console.rule("[bold cyan]Pre-jump signal samples")
    pre_rows = []
    for _, ev in events.iterrows():
        jstart = pd.Timestamp(ev["jump_start"]).normalize()
        for off in OFFSETS:
            d = jstart + pd.Timedelta(days=off)
            s = _sample_signals(daily, d)
            if s is None:
                continue
            s.update(
                {
                    "event_id": int(ev["event_id"]),
                    "offset_days": off,
                    "jump_start": jstart,
                    "direction": ev["direction"],
                }
            )
            pre_rows.append(s)
    per_event = pd.DataFrame(pre_rows)
    per_event.to_csv(os.path.join(out_dir, "per_event.csv"), index=False)
    console.print(f"  {len(per_event)} pre-jump sample rows")

    # ---- Null samples --------------------------------------------------------
    console.rule("[bold cyan]Null (calm) samples")
    jump_dates = pd.to_datetime(events["jump_start"]).dt.normalize().values
    min_abs = np.min(
        np.abs(np.array([(daily.index - jd).days.astype(int) for jd in jump_dates])),
        axis=0,
    )
    calm_mask = min_abs >= NULL_BUFFER_DAYS
    calm_dates = daily.index[calm_mask]
    console.print(
        f"  {len(calm_dates)} calm dates (>= {NULL_BUFFER_DAYS}d from any jump)"
    )
    if len(calm_dates) < 10:
        console.print("[red]  Too few calm dates — null distribution unreliable.[/red]")

    null_rows = []
    for off in OFFSETS:
        if len(calm_dates) == 0:
            break
        picks = rng.choice(
            calm_dates,
            size=NULL_SAMPLE_SIZE,
            replace=len(calm_dates) < NULL_SAMPLE_SIZE,
        )
        for d in picks:
            s = _sample_signals(daily, pd.Timestamp(d))
            if s is None:
                continue
            s["offset_days"] = off
            null_rows.append(s)
    null_df = pd.DataFrame(null_rows)
    null_df.to_csv(os.path.join(out_dir, "null_samples.csv"), index=False)

    # ---- Summary statistics -------------------------------------------------
    console.rule("[bold cyan]Mann-Whitney tests")
    summary_rows = []
    for sig in SIGNALS:
        alt = ALT_HYPOTHESIS[sig]
        mw_alt = "greater" if alt == "higher" else "less"
        for off in OFFSETS:
            pre_vals = (
                per_event.loc[per_event["offset_days"] == off, sig].dropna().values
            )
            null_vals = null_df.loc[null_df["offset_days"] == off, sig].dropna().values
            if len(pre_vals) == 0 or len(null_vals) < 5:
                p_val, u_stat = np.nan, np.nan
            else:
                try:
                    u_stat, p_val = mannwhitneyu(
                        pre_vals, null_vals, alternative=mw_alt
                    )
                except ValueError:
                    p_val, u_stat = np.nan, np.nan
            _n1, _n2 = len(pre_vals), len(null_vals)
            ic = (
                float(2 * u_stat / (_n1 * _n2) - 1)
                if np.isfinite(u_stat) and _n1 > 0 and _n2 > 0
                else np.nan
            )
            summary_rows.append(
                {
                    "signal": sig,
                    "offset_days": off,
                    "alt": alt,
                    "n_pre": len(pre_vals),
                    "pre_mean": float(np.mean(pre_vals)) if len(pre_vals) else np.nan,
                    "pre_median": float(np.median(pre_vals))
                    if len(pre_vals)
                    else np.nan,
                    "null_mean": float(np.mean(null_vals))
                    if len(null_vals)
                    else np.nan,
                    "null_median": float(np.median(null_vals))
                    if len(null_vals)
                    else np.nan,
                    "mw_u": float(u_stat) if np.isfinite(u_stat) else np.nan,
                    "mw_p_raw": float(p_val) if np.isfinite(p_val) else np.nan,
                    "ic": ic,
                }
            )
    summary = pd.DataFrame(summary_rows)

    # BH correction over the actual number of finite tests.
    p_raw = summary["mw_p_raw"].to_numpy(dtype=float)
    summary["mw_p_bh"] = _bh_correct(p_raw)
    console.print(
        f"  BH correction over {int(np.isfinite(p_raw).sum())} finite tests "
        f"(out of {len(p_raw)} total = {len(SIGNALS)} signals x {len(OFFSETS)} offsets)"
    )
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    # ---- Console table -------------------------------------------------------
    table = Table(title="Pre-jump vs null  (one-sided Mann-Whitney)")
    table.add_column("signal", style="bold")
    table.add_column("alt")
    for off in OFFSETS:
        table.add_column(f"d={off}  p_raw / p_bh / IC", justify="right")
    for sig in SIGNALS:
        row = [sig, ALT_HYPOTHESIS[sig]]
        for off in OFFSETS:
            s = summary[(summary["signal"] == sig) & (summary["offset_days"] == off)]
            if s.empty:
                row.append("-")
                continue
            p_r = float(s["mw_p_raw"].values[0])
            p_b = float(s["mw_p_bh"].values[0])
            ic_v = float(s["ic"].values[0])
            if not np.isfinite(p_r):
                row.append("-")
                continue
            raw_mark = "**" if p_r < 0.01 else ("*" if p_r < 0.05 else "")
            bh_mark = (
                "\u2021"
                if np.isfinite(p_b) and p_b < 0.05
                else ("\u2020" if np.isfinite(p_b) and p_b < 0.10 else "")
            )
            p_b_str = f"{p_b:.3f}{bh_mark}" if np.isfinite(p_b) else "nan"
            ic_str = f"{ic_v:+.3f}" if np.isfinite(ic_v) else "nan"
            row.append(f"{p_r:.3f}{raw_mark} / {p_b_str} / {ic_str}")
        table.add_row(*row)
    console.print(table)
    console.print("  * p_raw<0.05  ** p_raw<0.01  \u2020 p_bh<0.10  \u2021 p_bh<0.05")
    console.print(
        "  p_bh: p value adjusted with Benjamini-Hochberg FDR correction.  IC: information coefficient (rank-biserial correlation)."
    )

    # ---- Plots ---------------------------------------------------------------
    console.rule("[bold cyan]Plots")
    plots.plot_backtest_boxes(
        per_event,
        null_df,
        os.path.join(out_dir, "per_signal_boxes.png"),
        signals=SIGNALS,
        offsets=OFFSETS,
        null_buffer_days=NULL_BUFFER_DAYS,
        null_sample_size=NULL_SAMPLE_SIZE,
        trend_window=TREND_WINDOW,
        burn_in_start=BURN_IN_START,
        backtest_start=BACKTEST_START,
        backtest_end=BACKTEST_END,
        gp_si=GP_SI,
        kernel_hw=KERNEL_HW,
        km_si=KM_SI,
        km_kernel_hw=KM_KERNEL_HW,
    )
    plots.plot_backtest_overview(
        daily,
        events,
        os.path.join(out_dir, "events_overview.png"),
        slope_z_fn=_slope_z,
        trend_window=TREND_WINDOW,
        offsets=OFFSETS,
        backtest_start=BACKTEST_START,
        backtest_end=BACKTEST_END,
        burn_in_start=BURN_IN_START,
        gp_si=GP_SI,
        kernel_hw=KERNEL_HW,
        km_si=KM_SI,
        km_kernel_hw=KM_KERNEL_HW,
    )
    console.print(f"[green]All outputs written to {out_dir}/[/green]")


# =============================================================================
# MONTHLY PLOT GENERATION
# =============================================================================


def _generate_monthly_plots(
    console: Console,
    model,
    backtest_monthly_snaps: list,
    regime_output_dir: str,
    daily_path: str,
    events_path: str,
    out_dir: str,
    rng,
) -> None:
    """Drift grid, potential grid (one file each), plus per-month overview plots."""
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not backtest_monthly_snaps:
        console.print(
            "[yellow]No monthly snapshots available; skipping monthly plots.[/yellow]"
        )
        return

    daily = pd.read_csv(daily_path, parse_dates=["date"]).set_index("date").sort_index()
    daily.index = daily.index.normalize()
    events = pd.read_csv(events_path, parse_dates=["jump_start"])

    km_dir = regime_output_dir
    y_lo_drift, y_hi_drift = plots.km_drift_ylim(km_dir, KM_SI, KM_KERNEL_HW)

    console.rule("[bold cyan]Monthly plots")

    # --- Single grid: all months drift ---
    plots.plot_all_months_drift(
        model,
        backtest_monthly_snaps,
        os.path.join(plots_dir, "all_months_drift.png"),
        km_dir=km_dir,
        km_si=KM_SI,
        km_kernel_hw=KM_KERNEL_HW,
        n_grid=N_GRID,
        y_lo=y_lo_drift,
        y_hi=y_hi_drift,
        backtest_start=BACKTEST_START,
        backtest_end=BACKTEST_END,
    )

    # --- Single grid: all months potential ---
    plots.plot_all_months_potential(
        model,
        backtest_monthly_snaps,
        os.path.join(plots_dir, "all_months_potential.png"),
        n_grid=N_GRID,
        n_samples=N_SAMPLES,
        rng=rng,
        backtest_start=BACKTEST_START,
        backtest_end=BACKTEST_END,
    )

    # --- Per-month overview: price + p_multiwell + barrier_snr ---
    for snap in backtest_monthly_snaps:
        m_start = (
            pd.Timestamp(snap[5])
            if len(snap) > 5
            else pd.Timestamp(snap[0]).replace(day=1)
        )
        m_end = pd.Timestamp(snap[6]) if len(snap) > 6 else pd.Timestamp(snap[0])
        month_label = m_start.strftime("%Y-%m")

        ov_path = os.path.join(plots_dir, f"{month_label}_overview.png")
        plots.plot_monthly_overview(
            daily,
            events,
            m_start,
            m_end,
            ov_path,
            stable_days=STABLE_DAYS,
            stable_thr=STABLE_THR,
        )

    console.print(f"[green]Plots written to {plots_dir}/[/green]")


# =============================================================================
# ENTRY POINT
# =============================================================================


def main() -> None:
    console = Console()
    ss = np.random.SeedSequence(SEED)
    gp_rng, analysis_rng, plot_rng = [np.random.default_rng(s) for s in ss.spawn(3)]

    backtest_tag = (
        f"{BACKTEST_START.strftime('%Y-%m-%d')}_{BACKTEST_END.strftime('%Y-%m-%d')}"
    )
    out_dir = os.path.join(BACKTEST_DIR, backtest_tag)
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: run the GP (burn-in + backtest) and detect jumps.
    daily_path, events_path, model, backtest_monthly_snaps, regime_output_dir = (
        _run_gp_and_record(console, gp_rng, out_dir)
    )

    # Step 2: pure CSV-based signal analysis.
    _run_backtest(console, analysis_rng, out_dir, daily_path, events_path)

    # Step 3: per-month visualisations.
    _generate_monthly_plots(
        console,
        model,
        backtest_monthly_snaps,
        regime_output_dir,
        daily_path,
        events_path,
        out_dir,
        plot_rng,
    )
    del model


if __name__ == "__main__":
    main()
