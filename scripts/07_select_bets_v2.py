#!/usr/bin/env python3
"""
07_select_bets.py

Selects "best bets" from calibrated props.

Logic:
- Load props_current_calibrated.csv
- Restrict to latest available week
- Compute edge = |prob_over_cal - 0.5|
- Filter rows with edge >= threshold (default 0.05)
- Assign pick direction and stake
- Save to output/best_bets.csv
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# Config
INPUT = Path("data/props/props_current_calibrated.csv")
OUTDIR = Path("output")

# --- Selection Rules (Injected) ---
EDGE_THRESHOLD_STD = 0.08
EDGE_THRESHOLD_HIGH = 0.12
MAX_PICKS_PER_ENTITY = 5  # per game/player cap
WIND_EXCLUDE_MPH = 15.0
PRECIP_EXCLUDE_PCT = 0.60
EXTREME_WIND_MPH = 20.0
EXTREME_COLD_F = 10.0
VARIANCE_RATIO_MAX = 0.20  # RMSE/line cap
# ------------------------------------
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTPUT = OUTDIR / "best_bets.csv"
EDGE_THRESHOLD = EDGE_THRESHOLD_STD  # 5% edge

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT}")

    df = pd.read_csv(INPUT)
    if "prob_over_cal" not in df.columns or "prob_under_cal" not in df.columns:
        raise ValueError("Calibrated probabilities missing from input.")

    if "week" not in df.columns:
        raise ValueError("No 'week' column in input; cannot filter to current week.")

    # Restrict to latest week
    current_week = df["week"].max()
    df = df[df["week"] == current_week].copy()
    print(f"✓ Filtering to latest week = {current_week}, {len(df)} rows remain")

    # Compute edge: distance from 0.5
    df["edge"] = (df["prob_over_cal"] - 0.5).abs()

    # Filter by threshold
    best = df[df["edge"] >= EDGE_THRESHOLD].copy()

    # Decide pick direction (over/under)
    best["pick"] = best.apply(
        lambda r: "over" if r["prob_over_cal"] > 0.5 else "under", axis=1
    )

    # Flat staking plan
    best["stake_units"] = 1.0

    # Add run timestamp
    best["run_ts_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Select columns for output
    keep_cols = [
        "player_id", "player_name", "recent_team", "opponent_team",
        "season", "week", "market", "line",
        "prob_over_cal", "prob_under_cal",
        "edge", "pick", "stake_units", "run_ts_utc"
    ]
    existing = [c for c in keep_cols if c in best.columns]
    best_out = best[existing].reset_index(drop=True)

    best_out.to_csv(OUTPUT, index=False)
    print(f"✓ Wrote {OUTPUT} with {len(best_out)} rows (edge ≥ {EDGE_THRESHOLD}, week={current_week})")

if __name__ == "__main__":
    main()


def _stake_units_from_edge(edge, questionable=False):
    if edge is None:
        return 0.0
    if edge >= EDGE_THRESHOLD_HIGH:
        base = 1.5
    elif edge >= EDGE_THRESHOLD_STD:
        base = 1.0
    else:
        base = 0.0
    if questionable:
        base = min(base, 0.5)
    return round(base, 2)


def _is_off_market(line, consensus, step=1.0):
    try:
        if consensus is None or line is None:
            return False
        return abs(float(line) - float(consensus)) > step
    except Exception:
        return False

def _cap_per_entity(records, key_fields=('player_id','game_id'), cap=MAX_PICKS_PER_ENTITY):
    grouped = {}
    out = []
    for r in records:
        k = tuple(r.get(kf) for kf in key_fields)
        grouped.setdefault(k, [])
        if len(grouped[k]) < cap:
            grouped[k].append(r)
            out.append(r)
    return out
