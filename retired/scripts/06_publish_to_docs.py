#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full path: /football_for_mat-main/scripts/06_publish_to_docs.py

Purpose:
  Mirror canonical publish artifacts to docs/ for front-end use.

Inputs:
  - /football_for_mat-main/data/final/schedules_latest.csv
  - /football_for_mat-main/data/final/schedules_latest.parquet

Outputs:
  - /football_for_mat-main/docs/data/schedules_latest.csv
  - /football_for_mat-main/docs/data/schedules_latest.parquet
  - /football_for_mat-main/docs/data/.gitignore  (tracks the two files above)

Behavior:
  - If a source file is missing, writes an empty placeholder so the site build does not break.
  - No network calls.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SRC_CSV = ROOT / "data" / "final" / "schedules_latest.csv"
SRC_PARQUET = ROOT / "data" / "final" / "schedules_latest.parquet"

DOCS_DIR = ROOT / "docs" / "data"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

DST_CSV = DOCS_DIR / "schedules_latest.csv"
DST_PARQUET = DOCS_DIR / "schedules_latest.parquet"
DOCS_GITIGNORE = DOCS_DIR / ".gitignore"


def _ensure_gitignore():
    content = "*\n!.gitignore\n!schedules_latest.csv\n!schedules_latest.parquet\n"
    if not DOCS_GITIGNORE.exists() or DOCS_GITIGNORE.read_text(encoding="utf-8") != content:
        DOCS_GITIGNORE.write_text(content, encoding="utf-8")


def _copy_or_empty(src: Path, dst: Path, fmt: str):
    if src.exists() and src.is_file():
        # Use pandas for copy to normalize line endings and avoid permissions quirks
        if fmt == "csv":
            df = pd.read_csv(src, dtype=str).fillna("")
            df.to_csv(dst, index=False)
        elif fmt == "parquet":
            try:
                import pyarrow as pa  # noqa: F401
                import pyarrow.parquet as pq  # noqa: F401
                df = pd.read_parquet(src)
                df.to_parquet(dst, index=False)
            except Exception:
                # If pyarrow not present, fallback to raw copy
                shutil.copyfile(src, dst)
        else:
            shutil.copyfile(src, dst)
    else:
        # Write empty placeholder
        if fmt == "csv":
            pd.DataFrame().to_csv(dst, index=False)
        elif fmt == "parquet":
            try:
                pd.DataFrame().to_parquet(dst, index=False)
            except Exception:
                # Create an empty file to satisfy site loaders
                dst.write_bytes(b"")


def main() -> int:
    _ensure_gitignore()
    _copy_or_empty(SRC_CSV, DST_CSV, "csv")
    _copy_or_empty(SRC_PARQUET, DST_PARQUET, "parquet")
    print(f"Docs publish complete:\n  {DST_CSV}\n  {DST_PARQUET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
