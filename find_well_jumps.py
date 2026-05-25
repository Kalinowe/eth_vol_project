"""
find_well_jumps.py — replay Kalman-GP day-by-day through 2025, detect
well-jump events (stable well → large price transition → new stable well),
and write results to well_jumps_2025.csv.

A well-jump is detected by a simple state machine on rolling log-price:
  STABLE   : trailing STABLE_DAYS log-price range < STABLE_THR
  JUMPING  : current log-price has moved > JUMP_THR from the stable anchor
  SETTLED  : rolling range < STABLE_THR again in the new price zone

Output columns
--------------
  event_id           : sequential integer
  pre_stable_start   : first date of the pre-jump stable period
  pre_stable_end     : last date before price left old well
  jump_start         : first date price exceeded JUMP_THR from anchor
  post_stable_start  : first date of post-jump stable period
  pre_log_price      : median log-price in pre-stable window
  post_log_price     : median log-price in post-stable window
  pre_price_usd      : exp(pre_log_price)
  post_price_usd     : exp(post_log_price)
  log_price_change   : post - pre log-price
  direction          : 'up' or 'down'
  max_p_multiwell    : max p_multiwell in [pre_stable_end, post_stable_start]
  max_barrier_mean   : max barrier_mean over same window
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from rich.console import Console

from data_collection import load_series
from phase_GP import KalmanGPDriftModel, topology_from_gp, _SEC_PER_YEAR
from update.state_io import _restore_model_from_snap

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
STATE_PKL = os.path.join(
    "gp_results",
    "900s",
    "no_hp",
    "gp_2024-01-01_to_2025-12-31_900s_kmvar_nonehp_reproject_state.pkl",
)
START_SNAP_DATE = "2024-11-30"  # warm-start snapshot; replay from day after
RECORD_FROM = "2025-01-01"  # first day to record topology / price
REPLAY_END = "2025-12-31"  # last day to record

# Well-jump detection thresholds
STABLE_DAYS = 7  # rolling window to assess stability
STABLE_THR = 0.07  # log-price range within window to call it "stable"
JUMP_THR = 0.12  # log-price move from anchor to call it "jumping"
SETTLE_DAYS = 5  # days of stability needed to call new well "settled"

REPROJECT_MARGIN = 0.10
REPROJECT_CADENCE = 7  # days between reprojections
REPROJECT_WINDOW = 30  # trailing days used to compute new x-range
N_GRID = 200
N_SAMPLES = 120
MIN_CROSSING_SEP = 10
MIN_BARRIER_FRAC = 0.1
SEED = 42

OUT_CSV = "well_jumps_2025.csv"
# ---------------------------------------------------------------------------


def main() -> None:
    console = Console()
    rng = np.random.default_rng(SEED)

    # ---- Load blob -----------------------------------------------------------
    with open(STATE_PKL, "rb") as fh:
        blob = pickle.load(fh)
    cfg = blob["config"]
    si = int(cfg["phase_gp_seconds_interval"])
    khw = int(cfg["kernel_half_width"])
    trim = float(cfg["trim_quantile"])
    n_ind = int(cfg["n_inducing"])

    # ---- Find warm-start snapshot -------------------------------------------
    target_ts = pd.Timestamp(START_SNAP_DATE)
    best_snap, best_snap_dt = None, None
    for snap in blob.get("snapshots", []):
        snap_dt = pd.Timestamp(snap[0])
        if snap_dt <= target_ts:
            if best_snap_dt is None or snap_dt > best_snap_dt:
                best_snap, best_snap_dt = snap, snap_dt
    if best_snap is None:
        raise RuntimeError(f"No snapshot at or before {START_SNAP_DATE}")
    console.print(f"Warm-start snapshot: {best_snap_dt.date()}")

    model = _restore_model_from_snap(blob, best_snap)
    replay_start = (best_snap_dt + pd.Timedelta(days=1)).normalize().to_pydatetime()
    replay_end = datetime.strptime(REPLAY_END, "%Y-%m-%d")
    record_from = pd.Timestamp(RECORD_FROM).normalize()

    # ---- Load series ---------------------------------------------------------
    console.rule("Load series")
    x_all, dx_all, dt_step, dt_t_all = load_series(
        replay_start,
        replay_end,
        si,
        kernel_half_width=khw,
        trim_quantile=trim,
        window_type="monthly",
    )
    console.print(f"  {len(x_all)} obs  x=[{x_all.min():.4f}, {x_all.max():.4f}]")

    dt_t_pd = pd.to_datetime(dt_t_all)
    topo_range = model.reproject_to_range(
        float(x_all.min()), float(x_all.max()), n_inducing=n_ind
    )
    last_reproject_d: pd.Timestamp | None = None

    # ---- Day-by-day replay ---------------------------------------------------
    console.rule("Replay")
    rows = []
    for d in sorted(dt_t_pd.normalize().unique()):
        # Rolling-window reproject
        if last_reproject_d is None or (d - last_reproject_d).days >= REPROJECT_CADENCE:
            roll_lo = d - pd.Timedelta(days=REPROJECT_WINDOW - 1)
            roll_mask = (dt_t_pd.normalize() >= roll_lo) & (dt_t_pd.normalize() <= d)
            x_roll = x_all[roll_mask]
            if len(x_roll) < 2:
                x_roll = x_all[dt_t_pd.normalize() <= d]
            if len(x_roll) >= 2:
                topo_range = model.reproject_to_range(
                    float(x_roll.min()), float(x_roll.max()), n_inducing=n_ind
                )
                last_reproject_d = d

        mask = dt_t_pd.normalize() == d
        x_d = x_all[mask]
        dx_d = dx_all[mask]
        t_d = (np.asarray(dt_t_pd[mask].astype(np.int64)) / 1e9).astype(float)
        r_hat_d = (dx_d / dt_step) * _SEC_PER_YEAR
        model.update(x_d, r_hat_d, t_d)

        if d < record_from:
            continue

        topo_d = topology_from_gp(
            model,
            topo_range,
            n_grid=N_GRID,
            n_samples=N_SAMPLES,
            min_crossing_sep=MIN_CROSSING_SEP,
            min_barrier_fraction=MIN_BARRIER_FRAC,
            rng=rng,
        )
        rows.append(
            {
                "date": d,
                "log_price": float(np.median(x_d)),
                "price_usd": float(np.exp(np.median(x_d))),
                "p_multiwell": topo_d["p_multiwell"],
                "barrier_mean": topo_d["barrier_mean"],
                "barrier_std": topo_d["barrier_std"],
            }
        )
        console.print(
            f"  {d.date()}  log_p={rows[-1]['log_price']:.4f} "
            f"(${rows[-1]['price_usd']:.0f})  "
            f"p_multi={topo_d['p_multiwell']:.2f}  "
            f"barrier={topo_d['barrier_mean']:.3f}"
        )

    if not rows:
        console.print("[red]No rows recorded.[/red]")
        return

    df = pd.DataFrame(rows).set_index("date")
    df.index = pd.to_datetime(df.index)

    # ---- Well-jump detection -------------------------------------------------
    console.rule("Detecting well jumps")

    lp = df["log_price"]
    events = []

    # State machine: scan for stable → jump → stable transitions
    # Precompute rolling range (high - low over trailing STABLE_DAYS)
    roll_range = lp.rolling(STABLE_DAYS, min_periods=STABLE_DAYS).apply(
        lambda w: w.max() - w.min(), raw=True
    )

    dates = df.index.tolist()
    n = len(dates)
    i = 0
    while i < n:
        # Look for a stable window ending at i
        if pd.isna(roll_range.iloc[i]) or roll_range.iloc[i] >= STABLE_THR:
            i += 1
            continue

        # Found stable period at i.  Record anchor.
        stable_end_idx = i
        anchor_lp = lp.iloc[i]

        # Walk forward until price departs by JUMP_THR
        j = i + 1
        while j < n and abs(lp.iloc[j] - anchor_lp) < JUMP_THR:
            j += 1
        if j >= n:
            break  # never departed

        jump_start_idx = j
        jump_start_date = dates[j]
        jump_lp = lp.iloc[j]

        # Walk forward until new stable window
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

        # Find the start of the pre-stable period (walk back from stable_end_idx)
        pre_start_idx = stable_end_idx
        while pre_start_idx > 0:
            candidate = pre_start_idx - 1
            if (
                not pd.isna(roll_range.iloc[candidate])
                and roll_range.iloc[candidate] < STABLE_THR
            ):
                pre_start_idx = candidate
            else:
                break
        pre_stable_start = dates[pre_start_idx]
        pre_lp_median = float(lp.iloc[pre_start_idx : stable_end_idx + 1].median())

        log_change = post_lp_median - pre_lp_median
        direction = "up" if log_change > 0 else "down"

        # Topology during transition window
        trans = df.loc[dates[stable_end_idx] : post_stable_start]
        max_pmulti = float(trans["p_multiwell"].max())
        max_barrier = float(trans["barrier_mean"].max())

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
                "direction": direction,
                "max_p_multiwell": round(max_pmulti, 4),
                "max_barrier_mean": round(max_barrier, 4),
            }
        )
        console.print(
            f"  [bold]Event {len(events)}[/bold]  {direction.upper()}  "
            f"{pre_stable_start.date()} → {jump_start_date.date()} → {post_stable_start.date()}  "
            f"${np.exp(pre_lp_median):.0f} → ${np.exp(post_lp_median):.0f}  "
            f"Δlog={log_change:+.3f}"
        )

        # Advance past the new stable period
        i = k + SETTLE_DAYS

    if not events:
        console.print(
            "[yellow]No well jumps detected with current thresholds.[/yellow]"
        )
        return

    out = pd.DataFrame(events)
    out.to_csv(OUT_CSV, index=False)
    console.print(f"\n[green]Wrote {len(events)} events → {OUT_CSV}[/green]")
    console.print(out.to_string(index=False))

    # Also save the daily replay for reference
    df.reset_index().rename(columns={"date": "date"}).to_csv(
        "well_jumps_2025_daily.csv", index=False
    )
    console.print("[green]Daily replay → well_jumps_2025_daily.csv[/green]")


if __name__ == "__main__":
    main()
