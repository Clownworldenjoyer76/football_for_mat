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
OUT_SUM = OUT_DIR / "_build_features_summary.txt"

# column mappings present in your weekly_all
TEAM_MAP = [("recent_team", "team")]
OPP_MAP  = [("opponent_team", "opponent")]

# pass-through keys we want to preserve
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

    # rename team/opponent to canonical names if present
    for src, dst in TEAM_MAP:
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})
    for src, dst in OPP_MAP:
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    # ensure season/week numeric
    for c in ["season", "week"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # add is_home if you ever add home_away later (not present in weekly_all)
    if "home_away" in df.columns and "is_home" not in df.columns:
        df["is_home"] = (df["home_away"].astype(str).str.upper() != "AWAY").astype(int)

    return df

def select_numeric(df: pd.DataFrame):
    # keep *all* numeric/stat columns that actually exist
    # exclude ids/strings we already keep in KEEP_BASE
    base_set = set(KEEP_BASE + ["home_away", "is_home"])
    num_cols = [c for c, dt in df.dtypes.items()
                if (np.issubdtype(dt, np.number) and c not in base_set)]
    return num_cols

def main():
    df, src_path = load_source()
    df = standardize_cols(df)

    # build final column list using actual columns
    have = set(df.columns)
    base = [c for c in KEEP_BASE if c in have]
    numeric_cols = select_numeric(df)

    # coerce numeric cols
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # minimal NA handling
    if "season" in df: df["season"] = df["season"].fillna(0).astype(int)
    if "week" in df:   df["week"]   = df["week"].fillna(0).astype(int)

    out_cols = base + numeric_cols
    df_out = df[out_cols].copy()

    # write outputs
    df_out.to_csv(OUT_CSV, index=False)

    # quick manifest + summary
    manifest = pd.DataFrame({
        "column": out_cols,
        "dtype": [str(df_out[c].dtype) for c in out_cols]
    })
    manifest_path = OUT_DIR / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    lines = []
    lines.append(f"source: {src_path}")
    lines.append(f"rows: {len(df_out)}")
    lines.append(f"cols: {len(out_cols)}")
    lines.append(f"kept_base: {', '.join(base)}")
    lines.append(f"numeric_cols_count: {len(numeric_cols)}")
    # show first 30 numeric columns for sanity
    head_list = numeric_cols[:30]
    lines.append("numeric_cols_sample: " + (", ".join(head_list) if head_list else "(none)"))
    OUT_SUM.write_text("\n".join(lines) + "\n")

    print(f"wrote {OUT_CSV}  rows={len(df_out)} cols={len(out_cols)}")
    print(f"wrote {manifest_path}")
    print(f"wrote {OUT_SUM}")

if __name__ == "__main__":
    main()
