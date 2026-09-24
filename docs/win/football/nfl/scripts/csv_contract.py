#!/usr/bin/env python3
# Shared strict CSV contract reader for NFL pipeline scripts.

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Callable, Never


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



def read_validated_csv_header(
    path: Path,
    *,
    label: str,
    clean: Callable[[Any], str],
    fail: Callable[[str], Any],
) -> list[str]:
    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            header = next(csv.reader(handle), None)
    except UnicodeDecodeError as exc:
        fail(
            f"{label}: invalid UTF-8 CSV: "
            f"{path}: {exc}"
        )

    if not header:
        fail(f"{label}: missing CSV header: {path}")

    normalized = [clean(column) for column in header]

    if any(not column for column in normalized):
        fail(f"{label}: blank CSV column name found: {path}")

    counts: dict[str, int] = {}
    for column in normalized:
        counts[column] = counts.get(column, 0) + 1

    duplicates = sorted(
        column
        for column, count in counts.items()
        if count > 1
    )
    if duplicates:
        fail(
            f"{label}: duplicate CSV column names: "
            f"{duplicates}"
        )

    return header


def validate_csv_header_names(
    path: Path,
    *,
    label: str,
    clean: Callable[[Any], str],
    fail: Callable[[str], Any],
) -> list[str]:
    header = read_validated_csv_header(
        path,
        label=label,
        clean=clean,
        fail=fail,
    )
    return [clean(column) for column in header]


def normalize_csv_rows(
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str],
    clean: Callable[[Any], str],
) -> list[dict[str, str]]:
    return [
        {
            column: clean(row.get(column))
            for column in fieldnames
        }
        for row in rows
    ]


def write_csv_contract(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str],
    clean: Callable[[Any], str] | None = None,
    mkdir: bool = False,
    extrasaction: str = "raise",
    lineterminator: str | None = None,
    encoding: str = "utf-8",
) -> None:
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)

    writer_options: dict[str, Any] = {
        "fieldnames": fieldnames,
        "extrasaction": extrasaction,
    }
    if lineterminator is not None:
        writer_options["lineterminator"] = lineterminator

    output_rows: Any = rows
    if clean is not None:
        output_rows = (
            {
                column: clean(row.get(column))
                for column in fieldnames
            }
            for row in rows
        )

    with path.open(
        "w",
        newline="",
        encoding=encoding,
    ) as handle:
        writer = csv.DictWriter(handle, **writer_options)
        writer.writeheader()
        writer.writerows(output_rows)
        handle.flush()
        os.fsync(handle.fileno())
