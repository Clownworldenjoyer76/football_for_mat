#!/usr/bin/env python3
"""
Step 02 — Build features

What this does
--------------
Primary source:
- If present, reads data/weekly/latest.csv (preferred) and filters to TARGET_SEASON
Fallback sources:
- Loads weekly_all.* from data/features/ (parquet/csv/csv.gz)

Then it:
- Cleans & standardizes core columns (team/opponent, season/week, is_home)
- Merges injuries from CSV (no scraping) and adds an injury_flag
- Merges weather (season/week averages) if available
- Adds simple rolling averages over last 3 games for all numeric stats
- Writes:
    data/features/weekly_clean.csv.gz
    data/features/manifest.csv
    data/features/_build_features_summary.txt

A season guard at the end ensures max(season) == TARGET_SEASON (default 2025)
when using the preferred weekly source.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import os
import sys
import pandas as pd
import numpy as np

# ---------------- Config / Paths ----------------
TARGET_SEASON = int(os.getenv("TARGET_SEASON", "2025"))

PREFERRED_WEEKLY = Path("data/weekly/latest.csv")

SRC_CANDIDATES: List[Tuple[str, str]] = [
    ("parquet", "data/features/weekly_all.parquet"),
    ("csv",     "data/features/weekly_all.csv.gz"),
    ("csv",     "data/features/weekly_all.csv"),
]

OUT_DIR = Path("data/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "weekly_clean.csv.gz"
OUT_MAN = OUT_DIR / "manifest.csv"
OUT_SUM = OUT_DIR / "_build_features_summary.txt"

INJURY_FILE = Path("data/raw/injuries/injury_reports_latest.csv")
WEATHER_FILE = Path("data/raw/weather/game_weather_latest.csv")

TEAM_MAP = [("recent_team", "team")]
OPP_MAP  = [("opponent_team", "opponent")]

KEEP_BASE = [
    "player_id", "player_name", "player_display_name",
    "position", "position_group",
    "team", "opponent",
    "season", "week", "season_type",
    "game_id", "home_away"
]

# ---------------- Loaders ----------------
def load_source() -> Tuple[pd.DataFrame, Path, bool]:
    """
    Returns (df, src_path, from_preferred_weekly_flag)
    """
    if PREFERRED_WEEKLY.exists():
        try:
            df = pd.read_csv(PREFERRED_WEEKLY, low_memory=False)
            return df, PREFERRED_WEEKLY, True
        except Exception as e:
            print(f"WARNING: could not read {PREFERRED_WEEKLY}: {e}", file=sys.stderr)

    for kind, p in SRC_CANDIDATES:
        fp = Path(p)
        if fp.exists():
            if kind == "parquet":
                df = pd.read_parquet(fp)
            else:
                df = pd.read_csv(fp, low_memory=False)
            return df, fp, False

    raise SystemExit(
        "No weekly source found. Looked for:\n"
        f"  - {PREFERRED_WEEKLY}\n  - "
        + "\n  - ".join([p for _, p in SRC_CANDIDATES])
    )

def load_injuries() -> Optional[pd.DataFrame]:
    if INJURY_FILE.exists():
        try:
            return pd.read_csv(INJURY_FILE, low_memory=False)
        except Exception:
            return None
    return None

def load_weather() -> Optional[pd.DataFrame]:
    if WEATHER_FILE.exists():
        try:
            return pd.read_csv(WEATHER_FILE, low_memory=False)
        except Exception:
            return None
    return None

# ---------------- Cleaning ----------------
def standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for src, dst in TEAM_MAP:
        if src in df.columns and dst not in df.columns:
            df.rename(columns={src: dst}, inplace=True)
    for src, dst in OPP_MAP:
        if src in df.columns and dst not in df.columns:
            df.rename(columns={src: dst}, inplace=True)

    # season/week numeric
    for c in ["season", "week"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # is_home flag
    if "home_away" in df.columns and "is_home" not in df.columns:
        df["is_home"] = (df["home_away"].astype(str).str.upper() != "AWAY").astype(int)

    return df

def select_numeric(df: pd.DataFrame) -> List[str]:
    base_set = set(KEEP_BASE + ["is_home"])
    num_cols: List[str] = []
    for c, dt in df.dtypes.items():
        if c in base_set:
            continue
        if np.issubdtype(dt, np.number):
            num_cols.append(c)
    return num_cols

# ---------------- Feature Adds ----------------
def merge_injuries(df: pd.DataFrame, injuries: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Create injury_flag (0/1) from injuries CSV if available.
    Matches on player_display_name (fallback to player_name) against injuries['NAME'].
    """
    df = df.copy()
    df["injury_flag"] = 0

    if injuries is None or "NAME" not in injuries.columns:
        return df

    inj = injuries.copy()
    inj["NAME"] = inj["NAME"].astype(str).str.upper().str.strip()

    name_col = "player_display_name" if "player_display_name" in df.columns else (
               "player_name" if "player_name" in df.columns else None)
    if name_col is None:
        return df

    candidates = df[name_col].astype(str).str.upper().str.strip()
    df.loc[candidates.isin(inj["NAME"]), "injury_flag"] = 1
    return df

def merge_weather(df: pd.DataFrame, weather: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Attach season/week aggregated weather (mean) if available.
    """
    if weather is None:
        return df
    if not {"season", "week"}.issubset(df.columns):
        return df

    w = weather.copy()
    agg_cols = {}
    for col in ["temperature_2m", "windspeed_10m", "precipitation", "cloudcover"]:
        if col in w.columns:
            agg_cols[col] = "mean"
    if not agg_cols:
        return df

    if not {"season", "week"}.issubset(w.columns):
        return df

    w_avg = w.groupby(["season", "week"]).agg(agg_cols).reset_index()
    return df.merge(w_avg, on=["season", "week"], how="left")

def add_rolling(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """
    Add 3-game rolling means per player for numeric columns (if player_id present).
    """
    if not {"season", "week"}.issubset(df.columns):
        return df
    if "player_id" not in df.columns:
        return df

    df = df.sort_values(["player_id", "season", "week"]).copy()
    for c in numeric_cols:
        try:
            df[f"{c}_roll3"] = (
                df.groupby("player_id", group_keys=False)[c]
                  .transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(window=3, min_periods=1).mean())
            )
        except Exception:
            pass
    return df

# ---------------- Summary ----------------
def write_summary(df_out: pd.DataFrame,
                  src_path: Path,
                  numeric_cols: List[str],
                  injuries_present: bool,
                  weather_present: bool):
    lines = []
    lines.append(f"source: {src_path}")
    lines.append(f"rows: {len(df_out)}")
    lines.append(f"cols: {df_out.shape[1]}")
    lines.append("")
    lines.append(f"injuries_csv_present: {injuries_present}")
    lines.append(f"weather_csv_present: {weather_present}")
    if "injury_flag" in df_out.columns:
        lines.append(f"injury_flag_1_rows: {int((df_out['injury_flag'] == 1).sum())}")
    lines.append("")

    base_kept = [c for c in KEEP_BASE if c in df_out.columns]
    lines.append("base_kept: " + (", ".join(base_kept) if base_kept else "(none)"))

    lines.append(f"numeric_cols_count: {len(numeric_cols)}")
    if numeric_cols:
        sample = numeric_cols[:30]
        lines.append("numeric_cols_sample: " + ", ".join(sample))
    lines.append("")

    if "season" in df_out.columns:
        season_counts = df_out["season"].value_counts().sort_index().to_dict()
        lines.append(f"by_season_rows: {season_counts}")
    if "position" in df_out.columns:
        pos_counts = df_out["position"].astype(str).str.upper().value_counts().to_dict()
        lines.append(f"by_position_rows: {pos_counts}")
    if {"season", "week"}.issubset(df_out.columns):
        sw = (df_out.groupby(["season", "week"]).size()
              .reset_index(name="rows")
              .sort_values(["season", "week"])
              .tail(10))
        lines.append("last_10_season_week_rows:")
        for _, r in sw.iterrows():
            lines.append(f"  {int(r['season'])}-W{int(r['week'])}: {int(r['rows'])}")

    OUT_SUM.write_text("\n".join(lines) + "\n")

# ---------------- Main ----------------
def main():
    # Load source (prefer weekly/latest.csv)
    df, src_path, from_weekly = load_source()
    df = standardize_cols(df)

    # If we came from the preferred weekly file, hard-filter to TARGET_SEASON
    if from_weekly and "season" in df.columns:
        df = df[pd.to_numeric(df["season"], errors="coerce") == TARGET_SEASON].copy()

    # Determine bases / numeric
    base = [c for c in KEEP_BASE if c in df.columns]
    numeric_cols = select_numeric(df)

    # Coerce numerics
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # season/week to int if present
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").fillna(0).astype(int)
    if "week" in df.columns:
        df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int)

    # Injuries (CSV only; no scraping)
    inj_df = load_injuries()
    df = merge_injuries(df, inj_df)
    injuries_present = inj_df is not None

    # Weather (CSV if present)
    w_df = load_weather()
    df = merge_weather(df, w_df)
    weather_present = w_df is not None

    # Rolling stats
    df = add_rolling(df, numeric_cols)

    # Build output columns safely
    roll_cols = [f"{c}_roll3" for c in numeric_cols if f"{c}_roll3" in df.columns]
    extra_cols = []
    if "injury_flag" in df.columns:
        extra_cols.append("injury_flag")

    out_cols = base + numeric_cols + extra_cols + roll_cols
    out_cols = [c for c in out_cols if c in df.columns]  # guard

    df_out = df[out_cols].copy()

    # Season guard (only when we used preferred weekly)
    if from_weekly and "season" in df_out.columns and len(df_out):
        mx = int(pd.to_numeric(df_out["season"], errors="coerce").max())
        if mx != TARGET_SEASON:
            print(f"ERROR: weekly_clean max(season)={mx} != {TARGET_SEASON}. "
                  f"Check data/weekly/latest.csv.", file=sys.stderr)
            sys.exit(1)

    # Write data
    df_out.to_csv(OUT_CSV, index=False, compression="gzip")

    # Write manifest
    manifest = pd.DataFrame({
        "column": df_out.columns,
        "dtype": [str(df_out[c].dtype) for c in df_out.columns]
    })
    manifest.to_csv(OUT_MAN, index=False)

    # Write summary
    write_summary(df_out, src_path, numeric_cols, injuries_present, weather_present)

    print(f"wrote {OUT_CSV}  rows={len(df_out)} cols={len(out_cols)}")
    print(f"wrote {OUT_MAN}")
    print(f"wrote {OUT_SUM}")

if __name__ == "__main__":
    main()
