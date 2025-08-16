#!/usr/bin/env python3
# Build features from data/features/weekly_all.* using actual columns
from pathlib import Path
import pandas as pd
import numpy as np

SRC_CANDIDATES = [
    ("parquet", "data/features/weekly_all.parquet"),
    ("csv",     "data/features/weekly_all.csv.gz"),
    ("csv",     "data/features/weekly_all.csv"),
]

OUT_DIR = Path("data/features"); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "weekly_clean.csv.gz"
OUT_MAN = OUT_DIR / "manifest.csv"
OUT_SUM = OUT_DIR / "_build_features_summary.txt"

# If your weekly_all has these, we will rename to canonical names.
TEAM_MAP = [("recent_team", "team")]
OPP_MAP  = [("opponent_team", "opponent")]

# Base ID/meta columns to keep if present.
KEEP_BASE = [
    "player_id", "player_name", "player_display_name",
    "position", "position_group",
    "team", "opponent",
    "season", "week", "season_type"
]

def load_source():
    for kind, p in SRC_CANDIDATES:
        fp = Path(p)
        if fp.exists():
            if kind == "parquet":
                df = pd.read_parquet(fp)
            else:
                df = pd.read_csv(fp, low_memory=False)
            return df, fp
    raise SystemExit("No weekly_all file found. Looked for: " + ", ".join([p for _, p in SRC_CANDIDATES]))

def standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for src, dst in TEAM_MAP:
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})
    for src, dst in OPP_MAP:
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    # season/week numeric
    for c in ["season", "week"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # optional is_home from home_away if present
    if "home_away" in df.columns and "is_home" not in df.columns:
        df["is_home"] = (df["home_away"].astype(str).str.upper() != "AWAY").astype(int)

    return df

def select_numeric(df: pd.DataFrame):
    base_set = set(KEEP_BASE + ["home_away", "is_home"])
    num_cols = []
    for c, dt in df.dtypes.items():
        if c in base_set:
            continue
        if np.issubdtype(dt, np.number):
            num_cols.append(c)
    return num_cols

def write_summary(df_out: pd.DataFrame, src_path: Path, numeric_cols: list):
    lines = []
    lines.append(f"source: {src_path}")
    lines.append(f"rows: {len(df_out)}")
    lines.append(f"cols: {df_out.shape[1]}")
    lines.append("")

    # show base columns kept
    base_kept = [c for c in KEEP_BASE if c in df_out.columns]
    lines.append("base_kept: " + (", ".join(base_kept) if base_kept else "(none)"))

    # numeric count + sample
    lines.append(f"numeric_cols_count: {len(numeric_cols)}")
    if numeric_cols:
        sample = numeric_cols[:30]
        lines.append("numeric_cols_sample: " + ", ".join(sample))
    lines.append("")

    # quick tallies (guard for missing cols)
    if "season" in df_out.columns:
        season_counts = (df_out["season"].value_counts().sort_index()).to_dict()
        lines.append(f"by_season_rows: {season_counts}")
    if "position" in df_out.columns:
        pos_counts = (df_out["position"].astype(str).str.upper().value_counts()).to_dict()
        lines.append(f"by_position_rows: {pos_counts}")
    if {"season", "week"}.issubset(df_out.columns):
        # last 10 season-week pairs by rows
        sw = (df_out.groupby(["season","week"]).size().reset_index(name="rows")
              .sort_values(["season","week"], ascending=[True, True]).tail(10))
        lines.append("last_10_season_week_rows:")
        for _, r in sw.iterrows():
            lines.append(f"  {int(r['season'])}-W{int(r['week'])}: {int(r['rows'])}")

    OUT_SUM.write_text("\n".join(lines) + "\n")

def main():
    df, src_path = load_source()
    df = standardize_cols(df)

    have = set(df.columns)
    base = [c for c in KEEP_BASE if c in have]
    numeric_cols = select_numeric(df)

    # coerce numeric cols
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # season/week to int if present
    if "season" in df: df["season"] = df["season"].fillna(0).astype(int)
    if "week"   in df: df["week"]   = df["week"].fillna(0).astype(int)

    out_cols = base + numeric_cols
    df_out = df[out_cols].copy()

    # write data
    df_out.to_csv(OUT_CSV, index=False)

    # write manifest
    manifest = pd.DataFrame({
        "column": out_cols,
        "dtype": [str(df_out[c].dtype) for c in out_cols]
    })
    manifest.to_csv(OUT_MAN, index=False)

    # write summary
    write_summary(df_out, src_path, numeric_cols)

    print(f"wrote {OUT_CSV}  rows={len(df_out)} cols={len(out_cols)}")
    print(f"wrote {OUT_MAN}")
    print(f"wrote {OUT_SUM}")

if __name__ == "__main__":
    main()
