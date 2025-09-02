#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build ALL target labels from weekly data.

Inputs
------
data/features/weekly_all.csv.gz

Outputs
-------
data/features/targets/targets_all.csv
data/features/targets/targets_count.csv
data/features/targets/targets_share.csv

Notes
-----
- Fails fast with clear messages if input or columns are missing.
- Keeps minimal, analysis-friendly columns.
"""

from __future__ import annotations
import sys
import gzip
import io
from pathlib import Path
import pandas as pd


# ---------- Configuration ----------
REPO_ROOT = Path(__file__).resolve().parents[2]  # repo root from features/targets/*
INPUT_FILE = REPO_ROOT / "data" / "features" / "weekly_all.csv.gz"

OUTPUT_DIR = REPO_ROOT / "data" / "features" / "targets"
OUTPUT_ALL = OUTPUT_DIR / "targets_all.csv"
OUTPUT_COUNT = OUTPUT_DIR / "targets_count.csv"
OUTPUT_SHARE = OUTPUT_DIR / "targets_share.csv"

# Columns we will keep for identification/context
ID_COLS = [
    "player_id",
    "player_name",
    "recent_team",
    "position",
    "season",
    "week",
    "season_type",
]

# Target columns to extract (ALL)
TARGET_COLS = ["targets", "target_share"]


# ---------- Helpers ----------
def fail(msg: str) -> None:
    """
    Print error and exit with non-zero.
    """
    print(msg.strip(), file=sys.stderr)
    sys.exit(1)


def require_file(path: Path) -> None:
    """
    Ensure the input file exists.
    """
    if not path.exists():
        fail(
            "INSUFFICIENT INFORMATION: missing input file "
            f"'{path.as_posix()}'. Provide data/features/weekly_all.csv.gz."
        )


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    """
    Ensure the required columns exist.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        fail(
            "INSUFFICIENT INFORMATION: missing required columns in "
            f"'data/features/weekly_all.csv.gz': {missing}. "
            "Supply a file that includes these columns."
        )


# ---------- Main ----------
def main() -> None:
    # Validate inputs
    require_file(INPUT_FILE)

    # Load data (gzip CSV)
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        fail(f"CANNOT COMPLY: failed to read '{INPUT_FILE.as_posix()}': {e}")

    # Validate required columns
    require_columns(df, ID_COLS + TARGET_COLS)

    # Select and order columns
    keep_cols = ID_COLS + TARGET_COLS
    out = df[keep_cols].copy()

    # Basic cleaning: standardize dtypes
    # 'targets' should be numeric (integer-like)
    if "targets" in out.columns:
        out["targets"] = pd.to_numeric(out["targets"], errors="coerce")

    # 'target_share' should be numeric float between 0 and 1 (if given as percent, keep as provided)
    if "target_share" in out.columns:
        out["target_share"] = pd.to_numeric(out["target_share"], errors="coerce")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write combined file
    try:
        out.to_csv(OUTPUT_ALL, index=False)
    except Exception as e:
        fail(f"CANNOT COMPLY: failed to write '{OUTPUT_ALL.as_posix()}': {e}")

    # Write split files (count and share), keeping same ID columns
    try:
        out[ID_COLS + ["targets"]].to_csv(OUTPUT_COUNT, index=False)
        out[ID_COLS + ["target_share"]].to_csv(OUTPUT_SHARE, index=False)
    except Exception as e:
        fail(f"CANNOT COMPLY: failed to write split outputs: {e}")

    # Success message
    print(
        "Targets built.\n"
        f"- {OUTPUT_ALL.as_posix()}\n"
        f"- {OUTPUT_COUNT.as_posix()}\n"
        f"- {OUTPUT_SHARE.as_posix()}"
    )


if __name__ == "__main__":
    main()
