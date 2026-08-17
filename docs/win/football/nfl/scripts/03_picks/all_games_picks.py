#!/usr/bin/env python3
"""
Build compact all-game NFL projection picks output.

READS:
  docs/win/football/nfl/02_select/week_{week}_NFL_selected.csv

WRITES:
  docs/win/football/nfl/03_picks/all_games/all_week_{week}_NFL_picks.csv

OUTPUT COLUMNS:
  season
  week
  game_id
  away_team
  home_team
  predicted_away_score
  predicted_home_score
  predicted_total
  predicted_home_spread
  predicted_away_spread

The projected score columns are rounded to the nearest 0.5 point.

The projected total is recalculated from the rounded projected scores so the
displayed score, total, and spreads remain mathematically consistent.

Spread definitions:
  predicted_home_spread =
      predicted_away_score - predicted_home_score

  predicted_away_spread =
      predicted_home_score - predicted_away_score
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
NFL_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_INPUT_DIR = NFL_ROOT / "02_select"
DEFAULT_OUTPUT_DIR = NFL_ROOT / "03_picks" / "all_games"

OUTPUT_COLUMNS = [
    "season",
    "week",
    "game_id",
    "away_team",
    "home_team",
    "predicted_away_score",
    "predicted_home_score",
    "predicted_total",
    "predicted_home_spread",
    "predicted_away_spread",
]

REQUIRED_INPUT_COLUMNS = [
    "season",
    "week",
    "game_id",
    "away_team",
    "home_team",
    "predicted_away_score",
    "predicted_home_score",
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def clean(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if text.casefold() in {
        "",
        "nan",
        "none",
        "null",
        "<na>",
        "nat",
    }:
        return ""

    return text


def parse_float(
    value: Any,
    *,
    column: str,
    row_number: int,
) -> float:
    text = clean(value)

    if not text:
        fail(
            f"Row {row_number}: "
            f"{column} is blank"
        )

    try:
        number = float(text)
    except (TypeError, ValueError):
        fail(
            f"Row {row_number}: "
            f"{column} is not numeric: "
            f"{value!r}"
        )

    if not math.isfinite(number):
        fail(
            f"Row {row_number}: "
            f"{column} is non-finite: "
            f"{value!r}"
        )

    return number


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        fail(
            f"{label}: missing required columns: "
            f"{missing}"
        )


def validate_game_ids(
    df: pd.DataFrame,
    label: str,
) -> None:
    game_ids = df["game_id"].map(clean)

    if (game_ids == "").any():
        fail(
            f"{label}: blank game_id found"
        )

    duplicates = (
        game_ids[
            game_ids.duplicated(
                keep=False
            )
        ]
        .drop_duplicates()
        .tolist()
    )

    if duplicates:
        fail(
            f"{label}: duplicate game_id values: "
            f"{duplicates[:10]}"
        )


def round_to_half(
    value: float,
) -> float:
    return round(value * 2.0) / 2.0


def format_half_point(
    value: float,
) -> str:
    return f"{value:.1f}"


def build_output(
    source: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for index, row in source.iterrows():
        row_number = index + 2

        away_score = parse_float(
            row["predicted_away_score"],
            column="predicted_away_score",
            row_number=row_number,
        )

        home_score = parse_float(
            row["predicted_home_score"],
            column="predicted_home_score",
            row_number=row_number,
        )

        away_score_rounded = round_to_half(
            away_score
        )

        home_score_rounded = round_to_half(
            home_score
        )

        predicted_total = (
            away_score_rounded
            + home_score_rounded
        )

        predicted_home_spread = (
            away_score_rounded
            - home_score_rounded
        )

        predicted_away_spread = (
            home_score_rounded
            - away_score_rounded
        )

        rows.append(
            {
                "season": clean(
                    row["season"]
                ),
                "week": clean(
                    row["week"]
                ),
                "game_id": clean(
                    row["game_id"]
                ),
                "away_team": clean(
                    row["away_team"]
                ),
                "home_team": clean(
                    row["home_team"]
                ),
                "predicted_away_score": (
                    format_half_point(
                        away_score_rounded
                    )
                ),
                "predicted_home_score": (
                    format_half_point(
                        home_score_rounded
                    )
                ),
                "predicted_total": (
                    format_half_point(
                        predicted_total
                    )
                ),
                "predicted_home_spread": (
                    format_half_point(
                        predicted_home_spread
                    )
                ),
                "predicted_away_spread": (
                    format_half_point(
                        predicted_away_spread
                    )
                ),
            }
        )

    output = pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )

    return output


def write_atomic_csv(
    df: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    df.to_csv(
        temporary,
        index=False,
    )

    os.replace(
        temporary,
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--week",
        type=int,
        required=True,
        help="NFL week number",
    )

    args = parser.parse_args()

    if args.week <= 0:
        fail(
            "--week must be greater than 0"
        )

    input_path = (
        DEFAULT_INPUT_DIR
        / f"week_{args.week}_NFL_selected.csv"
    )

    output_path = (
        DEFAULT_OUTPUT_DIR
        / f"all_week_{args.week}_NFL_picks.csv"
    )

    if not input_path.is_file():
        fail(
            f"Input file not found: "
            f"{input_path}"
        )

    source = pd.read_csv(
        input_path,
        dtype=str,
        keep_default_na=False,
    )

    require_columns(
        source,
        REQUIRED_INPUT_COLUMNS,
        str(input_path),
    )

    validate_game_ids(
        source,
        str(input_path),
    )

    output = build_output(
        source
    )

    if len(output) != len(source):
        fail(
            "Output row count does not match "
            "input row count"
        )

    if list(output.columns) != OUTPUT_COLUMNS:
        fail(
            "Output column integrity check failed"
        )

    if (
        output["game_id"].tolist()
        != source["game_id"].map(clean).tolist()
    ):
        fail(
            "game_id order changed during processing"
        )

    write_atomic_csv(
        output,
        output_path,
    )

    print(
        f"WROTE {output_path} | "
        f"games={len(output)}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
