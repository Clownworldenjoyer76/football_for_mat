#!/usr/bin/env python3
# Shared strict CSV contract reader for NFL pipeline scripts.

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Never


def read_csv_contract(
    path: Path,
    *,
    label: str,
    fail: Callable[[str], Never],
    required_columns: list[str] | None = None,
    exact_columns: list[str] | None = None,
    allow_empty: bool = False,
) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        fail(f"Missing {label}: {path}")

    if path.stat().st_size == 0:
        fail(f"Zero-byte {label}: {path}")

    try:
        with path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
    except Exception as exc:
        fail(
            f"Could not read {label} {path}: "
            f"{type(exc).__name__}: {exc}"
        )

    if not fieldnames:
        fail(f"{label} has no CSV header: {path}")

    if len(fieldnames) != len(set(fieldnames)):
        fail(f"{label} contains duplicate CSV columns: {path}")

    if required_columns:
        missing = [
            column
            for column in required_columns
            if column not in fieldnames
        ]
        if missing:
            fail(f"{label} missing expected columns: {missing}")

    if exact_columns is not None and fieldnames != exact_columns:
        fail(
            f"{label} has unexpected column order/schema. "
            f"Expected={exact_columns} actual={fieldnames}"
        )

    if not rows and not allow_empty:
        fail(f"{label} contains no data rows: {path}")

    return fieldnames, rows
