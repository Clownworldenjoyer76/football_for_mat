#!/usr/bin/env python3
"""
Build NFL survivor picks ranked from largest favorite to smallest favorite.

READS:
  docs/win/football/nfl/03_picks/all_games/all_week_{week}_NFL_picks.csv

WRITES:
  docs/win/football/nfl/03_picks/survivor/{week}_survivor_picks.csv

OUTPUT COLUMNS:
  week
  game_id
  pick
  pt_diff
  away_team
  home_team

A negative spread is the favorite.

Rows are sorted from largest favorite to smallest favorite.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
NFL_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_INPUT_DIR = NFL_ROOT / "03_picks" / "all_games"
DEFAULT_OUTPUT_DIR = NFL_ROOT / "03_picks" / "survivor"

INPUT_PATTERN = "all_week_*_NFL_picks.csv"
FILENAME_PATTERN = re.compile(
    r"^all_week_(\d+)_NFL_picks\.csv$"
)

OUTPUT_COLUMNS = [
    "week",
    "game_id",
    "pick",
    "pt_diff",
    "away_team",
    "home_team",
]

REQUIRED_INPUT_COLUMNS = [
    "week",
    "game_id",
    "away_team",
    "home_team",
    "predicted_home_spread",
    "predicted_away_spread",
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


def validate_week(
    df: pd.DataFrame,
    expected_week: int,
    label: str,
) -> None:
    weeks = df["week"].map(clean)

    if (weeks == "").any():
        fail(
            f"{label}: blank week value found"
        )

    invalid = sorted(
        {
            value
            for value in weeks
            if value != str(expected_week)
        }
    )

    if invalid:
        fail(
            f"{label}: expected week {expected_week}, "
            f"found week values: {invalid}"
        )


def format_one_decimal(value: float) -> str:
    return f"{value:.1f}"


def build_output(
    source: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for index, row in source.iterrows():
        row_number = index + 2

        away_team = clean(
            row["away_team"]
        )
        home_team = clean(
            row["home_team"]
        )

        if not away_team:
            fail(
                f"Row {row_number}: away_team is blank"
            )

        if not home_team:
            fail(
                f"Row {row_number}: home_team is blank"
            )

        home_spread = parse_float(
            row["predicted_home_spread"],
            column="predicted_home_spread",
            row_number=row_number,
        )

        away_spread = parse_float(
            row["predicted_away_spread"],
            column="predicted_away_spread",
            row_number=row_number,
        )

        home_is_favorite = home_spread < 0
        away_is_favorite = away_spread < 0

        if home_is_favorite == away_is_favorite:
            fail(
                f"Row {row_number}: expected exactly one negative spread, "
                f"got predicted_home_spread={home_spread} and "
                f"predicted_away_spread={away_spread}"
            )

        if home_is_favorite:
            pick = home_team
            favorite_spread = home_spread
        else:
            pick = away_team
            favorite_spread = away_spread

        rows.append(
            {
                "week": clean(
                    row["week"]
                ),
                "game_id": clean(
                    row["game_id"]
                ),
                "pick": pick,
                "pt_diff_numeric": abs(
                    favorite_spread
                ),
                "away_team": away_team,
                "home_team": home_team,
            }
        )

    output = pd.DataFrame(rows)

    if output.empty:
        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    output = output.sort_values(
        by="pt_diff_numeric",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)

    output["pt_diff"] = output[
        "pt_diff_numeric"
    ].map(format_one_decimal)

    return output[OUTPUT_COLUMNS]


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
    input_paths = sorted(
        DEFAULT_INPUT_DIR.glob(
            INPUT_PATTERN
        )
    )

    if not input_paths:
        fail(
            f"No input files found: "
            f"{DEFAULT_INPUT_DIR / INPUT_PATTERN}"
        )

    total_games = 0

    for input_path in input_paths:
        match = FILENAME_PATTERN.match(
            input_path.name
        )

        if not match:
            continue

        week = int(
            match.group(1)
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

        validate_week(
            source,
            week,
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

        output_path = (
            DEFAULT_OUTPUT_DIR
            / f"{week}_survivor_picks.csv"
        )

        write_atomic_csv(
            output,
            output_path,
        )

        total_games += len(output)

        print(
            f"WROTE {output_path} | "
            f"games={len(output)}"
        )

    print(
        f"COMPLETE | "
        f"files={len(input_paths)} | "
        f"games={total_games}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
