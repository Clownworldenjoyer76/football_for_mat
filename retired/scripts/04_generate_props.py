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

This version preserves legacy behavior (looking for per-market CSVs in data/predictions/)
and ADDS support for consolidated pregame predictions files in data/predictions/pregame/.
If legacy files for TARGET_SEASON are missing/empty, the script will use pregame files.

Hard constraints:
- Only generate props for TARGET_SEASON (default 2025)
- Restrict to the latest available WEEK for TARGET_SEASON (derived from data/weekly/latest.csv when available)
- Overwrite data/props/props_current.csv every run
- Per-market CSVs are restricted to TARGET_SEASON + target week
- Fail fast if no rows exist for TARGET_SEASON from any source
"""

from pathlib import Path
import os
import sys
import json
from datetime import datetime
import re

import numpy as np
import pandas as pd
from scipy.stats import norm

# -------- Config --------
TARGET_SEASON = int(os.getenv("TARGET_SEASON", "2025"))

WEEKLY_LATEST = Path("data/weekly/latest.csv")

# Legacy per-market predictions (what the old script consumed)
PRED_DIR = Path("data/predictions")

# New consolidated pregame predictions (what 04_predict_pregame.py writes)
PREGAME_DIR = Path("data/predictions/pregame")

METRICS_DIR = Path("output")
OUT_DIR = Path("data/props")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Mapping from target columns (in pregame predictions) -> legacy market names
PREGAME_TARGET_TO_MARKET = {
    "passing_yards"   : "qb_passing_yards",
    "rushing_yards"   : "rb_rushing_yards",
    "receiving_yards" : "wr_rec_yards",
    "receptions"      : "wrte_receptions",
}

# Preferred identifier columns if present
ID_COLS_PREF = [
    "player_id","player_name","recent_team","opponent_team","team","opponent","game_id","position",
    "season","week"
]

# -------- Utilities --------
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
        usecols = ["season","week"]
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

def compute_props_from_df(df: pd.DataFrame, market: str, target_week: int | None) -> pd.DataFrame:
    """
    df must have columns: season, week, y_pred (+ optional ids)
    Restricts to TARGET_SEASON and target_week and computes line/probabilities.
    """
    required = {"y_pred","season","week"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    df = df[pd.to_numeric(df["season"], errors="coerce") == TARGET_SEASON]
    if df.empty:
        raise ValueError(f"no rows for season {TARGET_SEASON}")

    if target_week is None:
        target_week = int(pd.to_numeric(df["week"], errors="coerce").max())
    df = df[pd.to_numeric(df["week"], errors="coerce") == int(target_week)]
    if df.empty:
        raise ValueError(f"no rows for (season,week)=({TARGET_SEASON},{target_week})")

    # RMSE (fallback conservative if unknown)
    rmse = load_rmse(market)
    if not rmse or rmse <= 0:
        rmse = 10.0

    df["line"] = (df["y_pred"] * 2).round() / 2
    df["prob_over"] = 1 - norm.cdf(df["line"], loc=df["y_pred"], scale=rmse)
    df["prob_under"] = 1 - df["prob_over"]
    df["market"] = market

    df = ensure_columns(df)

    keep_cols_order = [
        "player_id","player_name","recent_team","opponent_team","game_id",
        "season","week","market","line","y_pred","prob_over","prob_under"
    ]
    keep = [c for c in keep_cols_order if c in df.columns]
    return df[keep].copy()

# -------- Legacy source: per-market CSVs in data/predictions --------
def process_legacy_prediction_file(pred_file: Path, target_week: int | None) -> pd.DataFrame:
    market = pred_file.stem
    df = pd.read_csv(pred_file)
    # legacy expects y_pred + season/week in each file
    if "y_pred" not in df.columns:
        raise ValueError(f"{pred_file} missing y_pred column")
    return compute_props_from_df(df, market, target_week)

def collect_from_legacy(target_week: int | None) -> list[pd.DataFrame]:
    results = []
    for pred_file in sorted(PRED_DIR.glob("*.csv")):
        try:
            df_out = process_legacy_prediction_file(pred_file, target_week)
            results.append(df_out)
            print(f"✓ Legacy source: {pred_file} -> {len(df_out)} rows")
        except Exception as e:
            print(f"! Skipping legacy {pred_file}: {e}", file=sys.stderr)
    return results

# -------- Pregame source: consolidated files in data/predictions/pregame --------
def find_pregame_files_for_season() -> list[Path]:
    if not PREGAME_DIR.exists():
        return []
    files = list(PREGAME_DIR.glob("*.csv")) + list(PREGAME_DIR.glob("*.csv.gz"))
    # prefer files that include the season token in the filename if present
    season_token = str(TARGET_SEASON)
    prioritized = [p for p in files if season_token in p.name]
    return prioritized if prioritized else files

def split_pregame_to_markets(df_pre: pd.DataFrame, target_week: int | None) -> list[pd.DataFrame]:
    """
    For each known target column in the pregame predictions, produce a props dataframe.
    Requires columns: season, week and one or more target columns named per PREGAME_TARGET_TO_MARKET.
    """
    need_cols = {"season","week"}
    if not need_cols.issubset(df_pre.columns):
        raise ValueError("pregame predictions missing required 'season'/'week' columns")

    # restrict to TARGET_SEASON and target_week
    dfp = df_pre[pd.to_numeric(df_pre["season"], errors="coerce") == TARGET_SEASON].copy()
    if dfp.empty:
        raise ValueError(f"pregame file has no rows for season {TARGET_SEASON}")

    if target_week is None:
        target_week = int(pd.to_numeric(dfp["week"], errors="coerce").max())
    dfp = dfp[pd.to_numeric(dfp["week"], errors="coerce") == int(target_week)].copy()
    if dfp.empty:
        raise ValueError(f"pregame file has no rows for (season,week)=({TARGET_SEASON},{target_week})")

    # Pick identifier columns that are present
    id_cols = [c for c in ID_COLS_PREF if c in dfp.columns]
    if not id_cols:
        id_cols = [c for c in dfp.columns if c in ("season","week")]

    out_frames = []
    for target_col, market in PREGAME_TARGET_TO_MARKET.items():
        if target_col not in dfp.columns:
            continue
        tmp = dfp[id_cols].copy()
        tmp["y_pred"] = pd.to_numeric(dfp[target_col], errors="coerce")
        tmp = tmp[tmp["y_pred"].notna()]
        if tmp.empty:
            continue
        props_df = compute_props_from_df(tmp, market, target_week)
        out_frames.append(props_df)
        print(f"✓ Pregame source: {market} from column '{target_col}' -> {len(props_df)} rows")
    return out_frames

def collect_from_pregame(target_week: int | None) -> list[pd.DataFrame]:
    frames = []
    for p in find_pregame_files_for_season():
        try:
            df_pre = pd.read_csv(p, low_memory=False)
            frames.extend(split_pregame_to_markets(df_pre, target_week))
            # If we successfully produced frames from one pregame file, we can stop early.
            if frames:
                break
        except Exception as e:
            print(f"! Skipping pregame {p}: {e}", file=sys.stderr)
    return frames

# -------- Main --------
def main():
    if not PRED_DIR.exists() and not PREGAME_DIR.exists():
        print(f"ERROR: predictions directories not found: {PRED_DIR} or {PREGAME_DIR}", file=sys.stderr)
        sys.exit(1)

    target_week = resolve_target_week()
    if target_week is not None:
        print(f"Using week={target_week} from {WEEKLY_LATEST} for TARGET_SEASON={TARGET_SEASON}")
    else:
        print("WARNING: Could not resolve target week from data/weekly/latest.csv; will use per-file max week")

    all_rows = []

    # 1) Try legacy per-market predictions first (preserves existing behavior)
    legacy_rows = collect_from_legacy(target_week)
    all_rows.extend(legacy_rows)

    # 2) If legacy produced nothing for TARGET_SEASON, try pregame consolidated files
    if not all_rows:
        pregame_rows = collect_from_pregame(target_week)
        all_rows.extend(pregame_rows)

    if not all_rows:
        print("ERROR: No props generated for TARGET_SEASON from any source. "
              "Ensure predictions exist (legacy per-market or pregame consolidated) for 2025.",
              file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(all_rows, ignore_index=True)

    if "season" not in combined.columns or combined.empty:
        print("ERROR: Combined props missing 'season' or empty.", file=sys.stderr)
        sys.exit(1)

    smax = int(pd.to_numeric(combined["season"], errors="coerce").max())
    if smax != TARGET_SEASON:
        print(f"ERROR: Combined props season={smax} != {TARGET_SEASON}.", file=sys.stderr)
        sys.exit(1)

    # Write per-market props (already written in legacy path), but ensure outputs exist for all markets we created
    for market in sorted(combined["market"].unique()):
        dfm = combined[combined["market"] == market].copy()
        out_path = OUT_DIR / f"props_{market}.csv"
        dfm.to_csv(out_path, index=False)
        print(f"✓ Ensured {out_path} ({len(dfm)} rows)")

    # Combined output
    combined_out = OUT_DIR / "props_current.csv"
    combined.to_csv(combined_out, index=False)
    print(f"✓ Wrote {combined_out} ({len(combined)} rows) — season={TARGET_SEASON}, week={int(combined['week'].max())}")

    # Append to history_pending
    try:
        pending_path = OUT_DIR / "history_pending.csv"
        dfh = combined.copy()
        if "run_ts" not in dfh.columns:
            dfh["run_ts"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        if "market" not in dfh.columns:
            dfh["market"] = "unknown"
        if pending_path.exists():
            prev = pd.read_csv(pending_path)
            merged = pd.concat([prev, dfh], ignore_index=True)
        else:
            merged = dfh
        subset_cols_pref = ["season","week","market","player_id","game_id","line","run_ts"]
        subset = [c for c in subset_cols_pref if c in merged.columns]
        merged = merged.drop_duplicates(subset=subset, keep="last") if subset else merged.drop_duplicates()
        merged.to_csv(pending_path, index=False)
        print(f"✓ Appended to {pending_path} ({len(dfh)} new rows)")
    except Exception as e:
        print(f"! Failed to append to history_pending.csv: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
