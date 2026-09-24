#!/usr/bin/env python3
"""Shared helpers for NFL graded-bet reports and dashboard validation."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Never

EPSILON = 1e-9


def fail(message: str) -> Never:
    raise RuntimeError(message)


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "nat", "<na>"}:
        return ""
    return text


def to_float(value: Any) -> float | None:
    try:
        result = float(str(value).strip())
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def units_won(odds: Any, result: Any) -> float | None:
    result = str(result).strip().title()
    if result == "Push": return 0.0
    if result == "Loss": return -1.0
    if result != "Win": return None
    odds_num = to_float(odds)
    if odds_num is None or odds_num == 0: return None
    return odds_num / 100.0 if odds_num > 0 else 100.0 / abs(odds_num)


def parse_positive_int(value: Any, *, label: str) -> int:
    number = to_float(value)
    if number is None or not number.is_integer() or number <= 0:
        fail(f"{label} must be a positive integer; found {value!r}")
    return int(number)


def parse_nonnegative_int(value: Any, *, label: str) -> int:
    number = to_float(value)
    if number is None or not number.is_integer() or number < 0:
        fail(f"{label} must be a nonnegative integer; found {value!r}")
    return int(number)


def require_finite_if_present(value: Any, *, label: str) -> float | None:
    text = clean(value)
    if not text:
        return None
    number = to_float(text)
    if number is None:
        fail(f"{label} must be finite when present; found {value!r}")
    return number


def validate_csv_header(path: Path, *, expected: list[str], label: str) -> None:
    if not path.exists():
        fail(f"{label}: file not found: {path}")
    if not path.is_file():
        fail(f"{label}: not a regular file: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
    except Exception as exc:
        fail(f"{label}: unable to read CSV header: {type(exc).__name__}: {exc}")
    if header is None:
        fail(f"{label}: CSV is empty")
    if any(not str(column).strip() for column in header):
        fail(f"{label}: blank CSV header name")
    seen: set[str] = set()
    duplicates: list[str] = []
    for column in header:
        if column in seen and column not in duplicates:
            duplicates.append(column)
        seen.add(column)
    if duplicates:
        fail(f"{label}: duplicate header columns: {duplicates}")
    if header != expected:
        fail(f"{label}: column contract failed")


def metric_row(sub: Any) -> dict[str, Any]:
    import pandas as pd
    wins = int((sub["bet_result"] == "Win").sum())
    losses = int((sub["bet_result"] == "Loss").sum())
    pushes = int((sub["bet_result"] == "Push").sum())
    excl = wins + losses
    incl = wins + losses + pushes

    unit_vals = pd.to_numeric(sub["bet_units"], errors="coerce").dropna()
    units = round(float(unit_vals.sum()), 4) if not unit_vals.empty else 0.0

    ev_vals = pd.to_numeric(sub.get("ev", pd.Series(dtype=float)), errors="coerce").dropna()
    odds_vals = pd.to_numeric(sub.get("odds_american", pd.Series(dtype=float)), errors="coerce").dropna()
    prob_vals = pd.to_numeric(sub.get("model_prob", pd.Series(dtype=float)), errors="coerce").dropna()

    return {
        "Win": wins,
        "Loss": losses,
        "Push": pushes,
        "Total": incl,
        "bets_excluding_pushes": excl,
        "bets_including_pushes": incl,
        "Win_Pct": round(wins / excl, 4) if excl else 0.0,
        "Win_Pct_All_Bets": round(wins / incl, 4) if incl else 0.0,
        "units": units,
        "ROI_Excluding_Pushes": round(units / excl, 4) if excl else 0.0,
        "ROI_Including_Pushes": round(units / incl, 4) if incl else 0.0,
        "avg_ev": round(float(ev_vals.mean()), 4) if not ev_vals.empty else None,
        "avg_odds": round(float(odds_vals.mean()), 1) if not odds_vals.empty else None,
        "avg_model_prob": round(float(prob_vals.mean()), 4) if not prob_vals.empty else None,
    }


def validate_metric_frame(frame: Any, *, label: str) -> None:
    for index, row in frame.iterrows():
        row_number = index + 2
        wins = parse_nonnegative_int(row["Win"], label=f"{label} row {row_number} Win")
        losses = parse_nonnegative_int(row["Loss"], label=f"{label} row {row_number} Loss")
        pushes = parse_nonnegative_int(row["Push"], label=f"{label} row {row_number} Push")
        total = parse_nonnegative_int(row["Total"], label=f"{label} row {row_number} Total")
        excl = parse_nonnegative_int(
            row["bets_excluding_pushes"], label=f"{label} row {row_number} bets_excluding_pushes"
        )
        incl = parse_nonnegative_int(
            row["bets_including_pushes"], label=f"{label} row {row_number} bets_including_pushes"
        )
        if total != wins + losses + pushes or excl != wins + losses or incl != total:
            fail(f"{label} row {row_number}: count contract failed")

        win_pct = to_float(row["Win_Pct"])
        win_pct_all = to_float(row["Win_Pct_All_Bets"])
        units = to_float(row["units"])
        roi_excl = to_float(row["ROI_Excluding_Pushes"])
        roi_incl = to_float(row["ROI_Including_Pushes"])
        if None in {win_pct, win_pct_all, units, roi_excl, roi_incl}:
            fail(f"{label} row {row_number}: required metric is non-finite")
        if not 0.0 <= win_pct <= 1.0 or not 0.0 <= win_pct_all <= 1.0:
            fail(f"{label} row {row_number}: win percentage is outside [0, 1]")

        expected_win_pct = round(wins / excl, 4) if excl else 0.0
        expected_win_pct_all = round(wins / incl, 4) if incl else 0.0
        expected_roi_excl = round(units / excl, 4) if excl else 0.0
        expected_roi_incl = round(units / incl, 4) if incl else 0.0
        if abs(win_pct - expected_win_pct) > EPSILON:
            fail(f"{label} row {row_number}: Win_Pct contract failed")
        if abs(win_pct_all - expected_win_pct_all) > EPSILON:
            fail(f"{label} row {row_number}: Win_Pct_All_Bets contract failed")
        if abs(roi_excl - expected_roi_excl) > EPSILON:
            fail(f"{label} row {row_number}: ROI_Excluding_Pushes contract failed")
        if abs(roi_incl - expected_roi_incl) > EPSILON:
            fail(f"{label} row {row_number}: ROI_Including_Pushes contract failed")

        for column in ("avg_ev", "avg_odds", "avg_model_prob"):
            require_finite_if_present(row[column], label=f"{label} row {row_number} {column}")


def extend_market_report_paths(
    paths: list[Path],
    reports_root: Path,
    specs: dict[str, Any],
) -> None:
    for spec in specs.values():
        market_dir = reports_root / spec["directory"]
        for dimension in spec["dimensions"]:
            base_name = f"nfl_{spec['file_key']}_by_{dimension}"
            paths.append(market_dir / f"{base_name}.csv")
            paths.append(market_dir / f"{base_name}_side_summary.csv")
