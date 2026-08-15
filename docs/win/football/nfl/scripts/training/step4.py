#!/usr/bin/env python3
"""
Step 4: remove outcome-mined enrichment features from the historical NFL table.

This step is intentionally a sanitization gate. The former Step 4 applied the
static moneyline/spread/totals rule masters to every historical row. Those rule
masters contain hit rates, lifts, game counts, and rule identities learned from
multi-season outcomes, so applying them before a chronological split leaks
future information into historical model features.

READS / WRITES:
  docs/win/football/nfl/training/historical_core_2021_2025.csv

The input file is rewritten in place with every ml_*, ats_*, and totals_*
enrichment column removed. No rule-master file is read and no outcome-derived
feature is created.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pandas as pd


NFL_ROOT = Path("docs/win/football/nfl")
TRAINING_PATH = NFL_ROOT / "training/historical_core_2021_2025.csv"
LEAKY_PREFIXES = ("ml_", "ats_", "totals_")


def fail(message: str) -> None:
    raise RuntimeError(message)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def main() -> int:
    if not TRAINING_PATH.is_file():
        fail(f"Missing training file: {TRAINING_PATH}")

    df = pd.read_csv(
        TRAINING_PATH,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8-sig",
        low_memory=False,
    )
    if df.empty:
        fail(f"{TRAINING_PATH}: no data rows")
    if len(df.columns) != len(set(df.columns)):
        fail(f"{TRAINING_PATH}: duplicate column names")

    required = {"game_id", "season", "week", "margin", "total_points"}
    missing = sorted(required - set(df.columns))
    if missing:
        fail(f"{TRAINING_PATH}: missing required columns: {missing}")

    leaky_columns = [
        column for column in df.columns if column.startswith(LEAKY_PREFIXES)
    ]
    clean = df.drop(columns=leaky_columns, errors="ignore")

    remaining = [
        column for column in clean.columns if column.startswith(LEAKY_PREFIXES)
    ]
    if remaining:
        fail(f"Leakage sanitization failed; columns remain: {remaining}")

    if clean["game_id"].astype(str).str.strip().eq("").any():
        fail("Blank game_id values detected")
    if clean["game_id"].duplicated().any():
        fail("Duplicate game_id values detected")

    atomic_write_csv(clean, TRAINING_PATH)

    print("Step 4 leakage sanitization complete.")
    print(f"Rows: {len(clean)}")
    print(f"Removed outcome-derived enrichment columns: {len(leaky_columns)}")
    if leaky_columns:
        print("Removed: " + ", ".join(leaky_columns))
    print(f"Wrote: {TRAINING_PATH}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
