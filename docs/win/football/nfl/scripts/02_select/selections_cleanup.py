#!/usr/bin/env python3
"""
Create compact, grading-ready NFL selection files without renaming source columns.

READS:
  docs/win/football/nfl/02_select/week_{week}_NFL_selected.csv

WRITES:
  docs/win/football/nfl/02_select/clean/week_{week}_moneyline_selected.csv
  docs/win/football/nfl/02_select/clean/week_{week}_spread_selected.csv
  docs/win/football/nfl/02_select/clean/week_{week}_total_selected.csv

Only rows with the corresponding *_selected column equal to 1 are written.
Source selection column names are preserved exactly.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


NFL_ROOT = Path(__file__).resolve().parents[2]

BASE_COLUMNS = [
    "season",
    "week",
    "game_id",
    "away_team",
    "home_team",
]

MONEYLINE_COLUMNS = [
    *BASE_COLUMNS,
    "ml_selection",
    "ml_odds_american",
    "ml_model_probability",
    "ml_implied_probability",
    "ml_edge",
    "ml_ev",
    "ml_kelly",
]

SPREAD_COLUMNS = [
    *BASE_COLUMNS,
    "spread_selection",
    "spread_line",
    "spread_odds_american",
    "spread_model_probability",
    "spread_implied_probability",
    "spread_edge",
    "spread_ev",
    "spread_kelly",
]

TOTAL_COLUMNS = [
    *BASE_COLUMNS,
    "total_selection",
    "total_line",
    "total_odds_american",
    "total_model_probability",
    "total_implied_probability",
    "total_edge",
    "total_ev",
    "total_kelly",
]


def fail(message: str) -> None:
    raise ValueError(message)


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        fail(f"Selection input missing required columns: {missing}")


def write_atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temporary, index=False)
    os.replace(temporary, path)


def clean_market(
    source: pd.DataFrame,
    selected_column: str,
    output_columns: list[str],
) -> pd.DataFrame:
    selected = pd.to_numeric(source[selected_column], errors="coerce").fillna(0)
    output = source.loc[selected.eq(1), output_columns].copy()
    return output.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    input_path = (
        NFL_ROOT
        / "02_select"
        / f"week_{args.week}_NFL_selected.csv"
    )
    output_dir = NFL_ROOT / "02_select" / "clean"

    if not input_path.is_file():
        fail(f"Selection input not found: {input_path}")

    source = pd.read_csv(input_path)

    required = list(
        dict.fromkeys(
            [
                *MONEYLINE_COLUMNS,
                *SPREAD_COLUMNS,
                *TOTAL_COLUMNS,
                "ml_selected",
                "spread_selected",
                "total_selected",
            ]
        )
    )
    require_columns(source, required)

    seasons = set(pd.to_numeric(source["season"], errors="coerce").dropna().astype(int))
    weeks = set(pd.to_numeric(source["week"], errors="coerce").dropna().astype(int))
    if seasons != {args.season}:
        fail(f"Expected season={args.season}; found {sorted(seasons)}")
    if weeks != {args.week}:
        fail(f"Expected week={args.week}; found {sorted(weeks)}")

    if source["game_id"].isna().any() or source["game_id"].duplicated().any():
        fail("Selection input game_id values must be nonblank and unique")

    moneyline = clean_market(source, "ml_selected", MONEYLINE_COLUMNS)
    spread = clean_market(source, "spread_selected", SPREAD_COLUMNS)
    total = clean_market(source, "total_selected", TOTAL_COLUMNS)

    outputs = [
        (
            moneyline,
            output_dir / f"week_{args.week}_moneyline_selected.csv",
            "moneyline",
        ),
        (
            spread,
            output_dir / f"week_{args.week}_spread_selected.csv",
            "spread",
        ),
        (
            total,
            output_dir / f"week_{args.week}_total_selected.csv",
            "total",
        ),
    ]

    for frame, path, label in outputs:
        write_atomic_csv(frame, path)
        print(f"{label}_rows={len(frame)}")
        print(f"Wrote: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
