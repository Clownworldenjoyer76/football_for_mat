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
# Includes your current markets and generic names.
MARKET_COL_MAP = {
    # your current markets
    "qb_passing_yards": "passing_yards",
    "rb_rushing_yards": "rushing_yards",
    "wr_rec_yards": "receiving_yards",
    "wrte_receptions": "receptions",

    # generic fallbacks (if you ever log markets with these exact names)
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
    "receptions":     "receptions",
}

def load_pending(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Pending file not found: {path}")
    df = pd.read_csv(path)

    # Normalize IDs
    if "player_id" not in df.columns and "gsis_id" in df.columns:
        df["player_id"] = df["gsis_id"]

    need = {"season","week","market","line","prob_over"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"pending file missing: {sorted(missing)}")

    # Coerce numeric
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    if "week" in df.columns:
        df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df["prob_over"] = pd.to_numeric(df["prob_over"], errors="coerce")

    # Trim market strings
    df["market"] = df["market"].astype(str).str.strip()

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

    # Normalize join keys
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    if "week" in df.columns:
        df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    if "player_name" in df.columns:
        df["player_name"] = df["player_name"].astype(str).str.strip()
    if "player_display_name" in df.columns and "player_name" not in df.columns:
        df = df.rename(columns={"player_display_name": "player_name"})
        df["player_name"] = df["player_name"].astype(str).str.strip()

    return df

def attach_actuals(pending: pd.DataFrame, weekly: pd.DataFrame, join: str) -> pd.DataFrame:
    out = []

    # Determine available join keys
    have_id_join = {"player_id","season","week"}.issubset(weekly.columns) and "player_id" in pending.columns
    have_name_join = {"player_name","season","week"}.issubset(weekly.columns) and "player_name" in pending.columns

    for mk, grp in pending.groupby("market"):
        stat_col = MARKET_COL_MAP.get(str(mk))
        if not stat_col or stat_col not in weekly.columns:
            tmp = grp.copy()
            tmp["actual_stat"] = pd.NA
            tmp["over_actual"] = pd.NA
            out.append(tmp)
            continue

        # Prefer game join if requested and available
        if join == "player_game" and "game_id" in grp.columns and "game_id" in weekly.columns:
            merged = grp.merge(
                weekly[["player_id","game_id", stat_col]],
                on=["player_id","game_id"],
                how="left"
            )
        else:
            if have_id_join:
                merged = grp.merge(
                    weekly[["player_id","season","week", stat_col]],
                    on=["player_id","season","week"],
                    how="left"
                )
            elif have_name_join:
                # fallback to name-based join (whitespace-trimmed)
                g = grp.copy()
                g["player_name"] = g["player_name"].astype(str).str.strip()
                merged = g.merge(
                    weekly[["player_name","season","week", stat_col]],
                    on=["player_name","season","week"],
                    how="left"
                )
            else:
                # cannot join
                merged = grp.copy()
                merged[stat_col] = pd.NA

        merged["actual_stat"] = pd.to_numeric(merged.get(stat_col), errors="coerce")
        merged["over_actual"] = (merged["actual_stat"] > pd.to_numeric(merged["line"], errors="coerce")).astype("Int64")
        out.append(merged)

    return pd.concat(out, ignore_index=True) if out else pending.copy()

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

    keep_cols = [c for c in [
        "season","week","market","player_id","player_name","recent_team","opponent_team",
        "game_id","line","prob_over","run_ts","actual_stat","over_actual"
    ] if c in stamped.columns]
    stamped = stamped[keep_cols]

    # Append & dedupe by stable identity
    if p_out.exists():
        prev = pd.read_csv(p_out)
        all_rows = pd.concat([prev, stamped], ignore_index=True)
    else:
        all_rows = stamped

    subset_cols = [c for c in ["season","week","market","player_id","line"] if c in all_rows.columns]
    if subset_cols:
        all_rows = all_rows.drop_duplicates(subset=subset_cols, keep="last")

    all_rows.to_csv(p_out, index=False)
    print("✓ Wrote", p_out.as_posix(), f"({len(stamped)} new rows, {len(all_rows)} total)")

    # Clear pending now that we stamped outcomes
    try:
        p_pending.unlink(missing_ok=True)
        print("Cleared", p_pending.as_posix())
    except Exception:
        pass

if __name__ == "__main__":
    main()
