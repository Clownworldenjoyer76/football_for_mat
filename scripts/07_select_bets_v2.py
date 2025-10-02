#!/usr/bin/env python3
"""
07_select_bets_v2.py

Selects "best bets" from calibrated props.

Key changes in v2:
- Accepts optional --season (int). If provided and not present in the input, fail fast.
- If --season is NOT provided, auto-selects the LATEST season in the file, then the LATEST week in that season.
- Prints the chosen (season, week) and row counts so CI logs are unambiguous.
- Adds a cap per (player_id, game_id) to avoid over-concentration, keeping previous defaults.

Inputs
------
- data/props/props_current_calibrated.csv (must include: season, week, prob_over_cal, prob_under_cal)

Outputs
-------
- output/best_bets.csv (single combined output)
- output/best_bets_top_<season>_wk<week>.csv (season/week-scoped)

Notes
-----
- EDGE_THRESHOLD default is 0.05 (5 percentage points from 50/50).
- Stake sizing is a simple monotonic mapping from edge with an optional 'questionable' dampener.
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np

INPUT = Path("data/props/props_current_calibrated.csv")
OUTDIR = Path("output")
OUTDIR.mkdir(parents=True, exist_ok=True)

EDGE_THRESHOLD = float(os.getenv("EDGE_THRESHOLD", "0.05"))
MAX_PICKS_PER_ENTITY = int(os.getenv("MAX_PICKS_PER_ENTITY", "2"))  # per (player_id, game_id)

def _choose_scope(df: pd.DataFrame, want_season: int | None):
    if "season" not in df.columns or "week" not in df.columns:
        raise ValueError("Input is missing 'season' or 'week' columns.")
    seasons = sorted(set(int(s) for s in df["season"].dropna().unique()))
    if not seasons:
        raise ValueError("No season values found in input.")
    if want_season is not None:
        if want_season not in seasons:
            raise SystemExit(f"ERROR: requested season={want_season} not found in input seasons {seasons}.")
        season = want_season
    else:
        season = max(seasons)

    df_season = df[df["season"].astype(int) == int(season)].copy()
    if df_season.empty:
        raise SystemExit(f"ERROR: no rows for season={season}.")
    week = int(df_season["week"].max())
    df_week = df_season[df_season["week"].astype(int) == week].copy()
    if df_week.empty:
        raise SystemExit(f"ERROR: no rows for (season,week)=({season},{week}).")
    return season, week, df_week

def _stake_units_from_edge(edge: float, questionable: bool = False) -> float:
    # Basic piecewise Kelly-like sizing (bounded 0–2.0 units)
    # You can tune these breakpoints if desired.
    e = float(edge)
    if e < 0.05:
        u = 0.0
    elif e < 0.07:
        u = 0.5
    elif e < 0.10:
        u = 1.0
    elif e < 0.15:
        u = 1.5
    else:
        u = 2.0
    if questionable:
        u *= 0.75
    return round(u, 2)

def _has_questionable_tag(row: pd.Series) -> bool:
    # Optional hook: mark 'questionable' if any known tag/column suggests risk
    for col in ("status", "questionable", "injury_status"):
        if col in row.index:
            val = str(row[col]).strip().lower()
            if val in ("q", "questionable", "doubtful", "out"):
                return True
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None, help="Target season; if absent, auto=latest season in file.")
    ap.add_argument("--edge_threshold", type=float, default=EDGE_THRESHOLD, help="Minimum calibrated edge to include (default 0.05).")
    ap.add_argument("--input", default=str(INPUT), help="Path to calibrated props CSV.")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"ERROR: {args.input} not found.")

    df = pd.read_csv(args.input)

    # Ensure calibrated probabilities exist
    if "prob_over_cal" not in df.columns or "prob_under_cal" not in df.columns:
        raise SystemExit("ERROR: Calibrated probabilities missing (need 'prob_over_cal' & 'prob_under_cal').")

    # Choose scope: latest season → latest week (or explicit --season)
    season, week, cur = _choose_scope(df, args.season)
    print(f"Selecting best bets for season={season}, week={week}")

    # Compute edge and pick direction
    cur["edge"] = (cur["prob_over_cal"] - 0.5).abs()
    picks = cur[cur["edge"] >= float(args.edge_threshold)].copy()

    # Direction
    picks["pick"] = np.where(picks["prob_over_cal"] >= 0.5, "over", "under")

    # Stake sizing
    picks["stake_units"] = [
        _stake_units_from_edge(e, _has_questionable_tag(row)) for e, row in zip(picks["edge"], picks.itertuples(index=False))
    ]

    # Optional: cap per (player_id, game_id)
    records = picks.to_dict(orient="records")
    records = _cap_per_entity(records, key_fields=("player_id", "game_id"), cap=MAX_PICKS_PER_ENTITY)

    # Output frame
    best_out = pd.DataFrame.from_records(records)
    if best_out.empty:
        print(f"WARNING: No picks met edge ≥ {args.edge_threshold} for (season,week)=({season},{week}).")
    keep_cols = [c for c in [
        "player_id","player_name","recent_team","opponent_team","game_id",
        "season","week","market","line",
        "prob_over_cal","prob_under_cal",
        "edge","pick","stake_units"
    ] if c in best_out.columns]
    best_out = best_out[keep_cols].copy()

    # Write scoped AND generic outputs
    OUTDIR.mkdir(parents=True, exist_ok=True)
    scoped = OUTDIR / f"best_bets_top_{season}_wk{week}.csv"
    generic = OUTDIR / "best_bets.csv"
    best_out.to_csv(scoped, index=False)
    best_out.to_csv(generic, index=False)

    print(f"✓ Wrote {scoped} and {generic} with {len(best_out)} rows (edge ≥ {args.edge_threshold})")

if __name__ == "__main__":
    main()
