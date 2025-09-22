#!/usr/bin/env python3
"""
Stamp outcomes for previously-logged props and build/refresh
data/props/history_props_with_outcomes.csv.

Usage (defaults shown):
  python scripts/08_finalize_props_outcomes.py \
    --pending data/props/history_pending.csv \
    --weekly  data/weekly/latest.csv \
    --out     data/props/history_props_with_outcomes.csv \
    --join    player_week   # or player_game if you have game_id in both

The pending file should include: season, week, market, player_id (or gsis_id),
line, prob_over (plus anything else you have like game_id, run_ts).
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

# Map prop market -> column name in weekly stats
MARKET_COL_MAP = {
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
    "receptions":     "receptions",
}

def load_pending(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Pending file not found: {path}")
    df = pd.read_csv(path)
    if "player_id" not in df.columns and "gsis_id" in df.columns:
        df["player_id"] = df["gsis_id"]
    need = {"season","week","market","player_id","line","prob_over"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"pending file missing: {sorted(missing)}")
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    return df

def load_weekly(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Weekly stats file not found: {path}")
    df = pd.read_csv(path)
    # unify id
    if "player_id" not in df.columns:
        for alt in ["gsis_id","player_gsis_id","pfr_id"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "player_id"})
                break
    if "season" in df.columns: df["season"] = df["season"].astype(int)
    if "week"   in df.columns: df["week"]   = df["week"].astype(int)
    return df

def attach_actuals(pending: pd.DataFrame, weekly: pd.DataFrame, join: str) -> pd.DataFrame:
    out = []
    for mk, grp in pending.groupby("market"):
        stat_col = MARKET_COL_MAP.get(str(mk))
        if not stat_col or stat_col not in weekly.columns:
            tmp = grp.copy()
            tmp["actual_stat"] = pd.NA
            tmp["over_actual"] = pd.NA
            out.append(tmp); continue
        if join == "player_game" and "game_id" in grp.columns and "game_id" in weekly.columns:
            merged = grp.merge(weekly[["player_id","game_id", stat_col]], on=["player_id","game_id"], how="left")
        else:
            merged = grp.merge(weekly[["player_id","season","week", stat_col]], on=["player_id","season","week"], how="left")
        merged["actual_stat"] = merged[stat_col]
        merged["over_actual"] = (merged["actual_stat"].astype(float) > merged["line"].astype(float)).astype("Int64")
        out.append(merged)
    return pd.concat(out, ignore_index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pending", default="data/props/history_pending.csv")
    ap.add_argument("--weekly",  default="data/weekly/latest.csv")
    ap.add_argument("--out",     default="data/props/history_props_with_outcomes.csv")
    ap.add_argument("--join",    default="player_week", choices=["player_week","player_game"])
    args = ap.parse_args()

    p_pending, p_weekly, p_out = Path(args.pending), Path(args.weekly), Path(args.out)
    p_out.parent.mkdir(parents=True, exist_ok=True)

    pend = load_pending(p_pending)
    wk   = load_weekly(p_weekly)
    stamped = attach_actuals(pend, wk, args.join)

    keep_cols = [c for c in ["season","week","market","player_id","game_id","line","prob_over","run_ts","actual_stat","over_actual"] if c in stamped.columns]
    stamped = stamped[keep_cols]

    # Append & dedupe
    if p_out.exists():
        prev = pd.read_csv(p_out)
        all_rows = pd.concat([prev, stamped], ignore_index=True)
    else:
        all_rows = stamped
    subset_cols = [c for c in ["season","week","market","player_id","line"] if c in all_rows.columns]
    if subset_cols:
        all_rows = all_rows.drop_duplicates(subset=subset_cols, keep="last")
    all_rows.to_csv(p_out, index=False)
    print("✓ Wrote", p_out.as_posix())

    # Clear pending now that we stamped outcomes
    try:
        p_pending.unlink(missing_ok=True)
        print("Cleared", p_pending.as_posix())
    except Exception:
        pass

if __name__ == "__main__":
    main()
