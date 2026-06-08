#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full script path: /mnt/data/football_for_mat-main/scripts/05_publish_schedules.py

Purpose:
  Publish validated, normalized schedules as canonical "latest" outputs.

Inputs:
  - /mnt/data/football_for_mat-main/data/processed/schedules/schedules_normalized.csv

Outputs:
  - /mnt/data/football_for_mat-main/data/final/schedules_latest.csv
  - /mnt/data/football_for_mat-main/data/final/schedules_latest.parquet

Rules:
  - No network calls.
  - No assumptions: only rows with non-empty canonical_game_id are published.
  - Column order preserved from normalized file.
"""
from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NORM_CSV = ROOT / "data" / "processed" / "schedules" / "schedules_normalized.csv"

FINAL_DIR = ROOT / "data" / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = FINAL_DIR / "schedules_latest.csv"
OUT_PARQUET = FINAL_DIR / "schedules_latest.parquet"


def main() -> int:
    if not NORM_CSV.exists():
        # Write empty outputs to keep pipeline deterministic
        pd.DataFrame().to_csv(OUT_CSV, index=False)
        try:
            pd.DataFrame().to_parquet(OUT_PARQUET, index=False)
        except Exception:
            pass
        print(f"Missing input: {NORM_CSV}. Wrote empty publish artifacts.")
        return 0

    df = pd.read_csv(NORM_CSV, dtype=str).fillna("")

    # Only publish rows with canonical_game_id
    if "canonical_game_id" in df.columns:
        df = df[df["canonical_game_id"].astype(str).str.strip() != ""].copy()

    # Stable sort for deterministic outputs
    for col in ["season", "season_type", "week", "game_date_utc", "game_time_utc", "canonical_game_id"]:
        if col not in df.columns:
            df[col] = ""
    df["__wk"] = pd.to_numeric(df["week"], errors="coerce")
    df = df.sort_values(
        by=["season", "season_type", "__wk", "game_date_utc", "game_time_utc", "canonical_game_id"],
        kind="mergesort",
        na_position="last"
    ).drop(columns="__wk")

    # Write outputs
    df.to_csv(OUT_CSV, index=False)
    try:
        df.to_parquet(OUT_PARQUET, index=False)
    except Exception:
        pass

    print(f"Published:\n  {OUT_CSV}\n  {OUT_PARQUET if OUT_PARQUET.exists() else '(parquet skipped)'}\n  rows={len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
