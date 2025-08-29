#!/usr/bin/env python3
"""
Step 02 — Build features

- Loads weekly_all.* from data/features/
- Cleans + standardizes
- Merges injuries and weather (if available)
- Adds rolling averages
- Outputs:
    data/features/weekly_clean.csv.gz
    data/features/manifest.csv
    data/features/_build_features_summary.txt
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ---------------- Paths ----------------
SRC_CANDIDATES = [
    ("parquet", "data/features/weekly_all.parquet"),
    ("csv",     "data/features/weekly_all.csv.gz"),
    ("csv",     "data/features/weekly_all.csv"),
]

OUT_DIR = Path("data/features"); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "weekly_clean.csv.gz"
OUT_MAN = OUT_DIR / "manifest.csv"
OUT_SUM = OUT_DIR / "_build_features_summary.txt"

INJURY_FILE = Path("data/raw/injuries/injury_reports_latest.csv")
WEATHER_FILE = Path("data/raw/weather/game_weather_latest.csv")

# ---------------- Config ----------------
TEAM_MAP = [("recent_team", "team")]
OPP_MAP  = [("opponent_team", "opponent")]

KEEP_BASE = [
    "player_id", "player_name", "player_display_name",
    "position", "position_group",
    "team", "opponent",
    "season", "week", "season_type"
]

# ---------------- Loaders ----------------
def load_source():
    for kind, p in SRC_CANDIDATES:
        fp = Path(p)
        if fp.exists():
            if kind == "parquet":
                df = pd.read_parquet(fp)
            else:
                df = pd.read_csv(fp, low_memory=False)
            return df, fp
    raise SystemExit("No weekly_all file found.")

def load_injuries():
    if INJURY_FILE.exists():
        return pd.read_csv(INJURY_FILE)
    return None

def load_weather():
    if WEATHER_FILE.exists():
        return pd.read_csv(WEATHER_FILE, parse_dates=["time"], low_memory=False)
    return None

# ---------------- Cleaning ----------------
def standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for src, dst in TEAM_MAP:
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})
    for src, dst in OPP_MAP:
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    for c in ["season", "week"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "home_away" in df.columns and "is_home" not in df.columns:
        df["is_home"] = (df["home_away"].astype(str).str.upper() != "AWAY").astype(int)

    return df

def select_numeric(df: pd.DataFrame):
    base_set = set(KEEP_BASE + ["home_away", "is_home"])
    num_cols = []
    for c, dt in df.dtypes.items():
        if c in base_set: continue
        if np.issubdtype(dt, np.number):
            num_cols.append(c)
    return num_cols

# ---------------- Feature Adds ----------------
def merge_injuries(df: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    # Injury file may not align; just mark if player shows up as OUT/DOUBTFUL
    if injuries is None or "NAME" not in injuries.columns:
        return df
    inj = injuries.copy()
    inj["NAME"] = inj["NAME"].str.upper().str.strip()
    df["injury_flag"] = df["player_display_name"].str.upper().str.strip().isin(inj["NAME"])
    return df

def merge_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    if weather is None:
        return df
    # Simplify: take noon UTC forecast per season/week/team
    if not {"season","week","team"}.issubset(df.columns):
        return df
    w = weather.copy()
    if "time" in w.columns:
        w["date"] = pd.to_datetime(w["time"]).dt.date
    # join keys are fuzzy; for now just attach season/week avg weather
    w_avg = w.groupby(["season","week"]).agg({
        "temperature_2m":"mean",
        "windspeed_10m":"mean",
        "precipitation":"mean",
        "cloudcover":"mean"
    }).reset_index()
    df = df.merge(w_avg, on=["season","week"], how="left")
    return df

def add_rolling(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    if not {"season","week","player_id"}.issubset(df.columns):
        return df
    df = df.sort_values(["player_id","season","week"])
    for c in numeric_cols:
        roll = (df.groupby("player_id")[c]
                  .transform(lambda x: x.rolling(window=3, min_periods=1).mean()))
        df[f"{c}_roll3"] = roll
    return df

# ---------------- Summary ----------------
def write_summary(df_out: pd.DataFrame, src_path: Path, numeric_cols: list):
    lines = []
    lines.append(f"source: {src_path}")
    lines.append(f"rows: {len(df_out)}")
    lines.append(f"cols: {df_out.shape[1]}")
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
    if {"season","week"}.issubset(df_out.columns):
        sw = (df_out.groupby(["season","week"]).size().reset_index(name="rows")
              .sort_values(["season","week"]).tail(10))
        lines.append("last_10_season_week_rows:")
        for _, r in sw.iterrows():
            lines.append(f"  {int(r['season'])}-W{int(r['week'])}: {int(r['rows'])}")

    OUT_SUM.write_text("\n".join(lines) + "\n")

# ---------------- Main ----------------
def main():
    df, src_path = load_source()
    df = standardize_cols(df)

    base = [c for c in KEEP_BASE if c in df.columns]
    numeric_cols = select_numeric(df)

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "season" in df: df["season"] = df["season"].fillna(0).astype(int)
    if "week" in df: df["week"] = df["week"].fillna(0).astype(int)

    # Merge injuries
    inj = load_injuries()
    df = merge_injuries(df, inj)

    # Merge weather
    w = load_weather()
    df = merge_weather(df, w)

    # Add rolling averages
    df = add_rolling(df, numeric_cols)

    out_cols = base + numeric_cols + ["injury_flag"] + [f"{c}_roll3" for c in numeric_cols if f"{c}_roll3" in df]
    df_out = df[out_cols].copy()

    df_out.to_csv(OUT_CSV, index=False)

    manifest = pd.DataFrame({
        "column": df_out.columns,
        "dtype": [str(df_out[c].dtype) for c in df_out.columns]
    })
    manifest.to_csv(OUT_MAN, index=False)

    write_summary(df_out, src_path, numeric_cols)

    print(f"wrote {OUT_CSV} rows={len(df_out)} cols={len(out_cols)}")
    print(f"wrote {OUT_MAN}")
    print(f"wrote {OUT_SUM}")

if __name__ == "__main__":
    main()
