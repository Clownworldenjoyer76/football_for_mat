#!/usr/bin/env python3

# === Season Guard (auto-added) ===
# Ensures outputs are for TARGET_SEASON (default 2025). Non-invasive: runs at process exit.
try:
    import os, atexit, sys
    import pandas as _pd
    TARGET_SEASON = int(os.getenv("TARGET_SEASON", "2025"))
    _SEASON_GUARD_PATHS = ["data/props/props_current.csv"]

    def _season_guard_check():
        for _p in _SEASON_GUARD_PATHS:
            try:
                if not _p or not isinstance(_p, str):
                    continue
                if not os.path.exists(_p):
                    continue
                _df = _pd.read_csv(_p)
                if "season" not in _df.columns:
                    continue
                _mx = _pd.to_numeric(_df["season"], errors="coerce").max()
                if int(_mx) != TARGET_SEASON:
                    print(f"ERROR: {_p} max(season)={int(_mx)} != {TARGET_SEASON}. Rebuild inputs for {TARGET_SEASON}.", file=sys.stderr)
                    sys.exit(1)
            except Exception as _e:
                print(f"WARNING: season guard issue for {_p}: {_e}", file=sys.stderr)

    atexit.register(_season_guard_check)
except Exception:
    pass
# === End Season Guard ===

"""
04_generate_props.py
Reads model predictions and produces market props with over/under probabilities.

Hard constraints in this version:
- Only generate props for TARGET_SEASON (default 2025)
- Restrict to the latest available WEEK for TARGET_SEASON (derived from data/weekly/latest.csv when available)
- Overwrite data/props/props_current.csv every run (no reuse/append of stale seasons)
- Per-market CSVs are also restricted to TARGET_SEASON + target week
- Fail fast if no rows exist for TARGET_SEASON
"""

from pathlib import Path
import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm

TARGET_SEASON = int(os.getenv("TARGET_SEASON", "2025"))
WEEKLY_LATEST = Path("data/weekly/latest.csv")
PRED_DIR = Path("data/predictions")
METRICS_DIR = Path("output")
OUT_DIR = Path("data/props")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_rmse(market: str):
    met_path = METRICS_DIR / f"{market}_metrics.json"
    if not met_path.exists():
        return None
    try:
        with open(met_path) as f:
            m = json.load(f)
        return m.get("metrics", {}).get("RMSE")
    except Exception:
        return None

def resolve_target_week() -> int | None:
    if not WEEKLY_LATEST.exists():
        return None
    try:
        usecols = ["season", "week"]
        dfw = pd.read_csv(WEEKLY_LATEST, usecols=lambda c: c in usecols)
        dfw = dfw[pd.to_numeric(dfw.get("season"), errors="coerce") == TARGET_SEASON]
        if dfw.empty or "week" not in dfw.columns:
            return None
        wk = int(pd.to_numeric(dfw["week"], errors="coerce").max())
        return wk
    except Exception:
        return None

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "recent_team" not in df.columns and "team" in df.columns:
        df["recent_team"] = df["team"]
    if "opponent_team" not in df.columns and "opponent" in df.columns:
        df["opponent_team"] = df["opponent"]
    return df

def process_market(pred_file: Path, target_week: int | None) -> pd.DataFrame:
    market = pred_file.stem
    df = pd.read_csv(pred_file)
    required = {"y_pred", "season", "week"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{pred_file} missing required columns: {missing}")
    df = df[pd.to_numeric(df["season"], errors="coerce") == TARGET_SEASON]
    if df.empty:
        raise ValueError(f"{pred_file}: no rows for season {TARGET_SEASON}")
    if target_week is None:
        target_week = int(pd.to_numeric(df["week"], errors="coerce").max())
    df = df[pd.to_numeric(df["week"], errors="coerce") == int(target_week)]
    if df.empty:
        raise ValueError(f"{pred_file}: no rows for (season,week)=({TARGET_SEASON},{target_week})")
    rmse = load_rmse(market)
    if not rmse or rmse <= 0:
        if {"y_true", "y_pred"}.issubset(df.columns) and len(df) > 1:
            rmse = max(1.0, float(np.std(df["y_true"] - df["y_pred"])))
        else:
            rmse = 10.0
    df["line"] = (df["y_pred"] * 2).round() / 2
    df["prob_over"] = 1 - norm.cdf(df["line"], loc=df["y_pred"], scale=rmse)
    df["prob_under"] = 1 - df["prob_over"]
    df["market"] = market
    df = ensure_columns(df)
    keep_cols_order = [
        "player_id", "player_name", "recent_team", "opponent_team", "game_id",
        "season", "week", "market", "line", "y_pred", "prob_over", "prob_under"
    ]
    keep_cols = [c for c in keep_cols_order if c in df.columns]
    df_out = df[keep_cols].copy()
    out_path = OUT_DIR / f"props_{market}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"✓ Wrote {out_path} ({len(df_out)} rows) — season={TARGET_SEASON} week={int(target_week)}")
    return df_out

def append_history_pending(df_all: pd.DataFrame):
    pending_path = OUT_DIR / "history_pending.csv"
    df = df_all.copy()
    if "run_ts" not in df.columns:
        df["run_ts"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    if "market" not in df.columns:
        df["market"] = "unknown"
    if pending_path.exists():
        prev = pd.read_csv(pending_path)
        combined = pd.concat([prev, df], ignore_index=True)
    else:
        combined = df
    subset_cols_pref = ["season", "week", "market", "player_id", "game_id", "line", "run_ts"]
    subset = [c for c in subset_cols_pref if c in combined.columns]
    combined = combined.drop_duplicates(subset=subset, keep="last") if subset else combined.drop_duplicates()
    combined.to_csv(pending_path, index=False)
    print(f"✓ Appended to {pending_path} ({len(df)} new rows)")

def main():
    if not PRED_DIR.exists():
        print(f"ERROR: predictions dir not found: {PRED_DIR}", file=sys.stderr)
        sys.exit(1)
    target_week = resolve_target_week()
    if target_week is not None:
        print(f"Using week={target_week} from {WEEKLY_LATEST} for TARGET_SEASON={TARGET_SEASON}")
    else:
        print("WARNING: Could not resolve target week from data/weekly/latest.csv; will use per-file max week")
    all_rows = []
    any_errors = 0
    for pred_file in sorted(PRED_DIR.glob("*.csv")):
        try:
            df_out = process_market(pred_file, target_week)
            all_rows.append(df_out)
        except Exception as e:
            any_errors += 1
            print(f"! Skipping {pred_file}: {e}", file=sys.stderr)
    if not all_rows:
        print("ERROR: No props generated for TARGET_SEASON. Ensure predictions exist for 2025.", file=sys.stderr)
        sys.exit(1)
    combined = pd.concat(all_rows, ignore_index=True)
    if "season" not in combined.columns or combined.empty:
        print("ERROR: Combined props missing 'season' or empty.", file=sys.stderr)
        sys.exit(1)
    smax = int(pd.to_numeric(combined["season"], errors="coerce").max())
    if smax != TARGET_SEASON:
        print(f"ERROR: Combined props season={smax} != {TARGET_SEASON}.", file=sys.stderr)
        sys.exit(1)
    combined_out = OUT_DIR / "props_current.csv"
    combined.to_csv(combined_out, index=False)
    print(f"✓ Wrote {combined_out} ({len(combined)} rows) — season={TARGET_SEASON}, week={int(combined['week'].max())}")
    try:
        append_history_pending(combined)
    except Exception as e:
        print(f"! Failed to append to history_pending.csv: {e}", file=sys.stderr)
    if any_errors:
        print(f"Completed with {any_errors} market(s) skipped.", file=sys.stderr)

if __name__ == "__main__":
    main()
