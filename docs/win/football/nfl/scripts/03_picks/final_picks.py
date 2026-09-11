#!/usr/bin/env python3
"""
Build final NFL selected-bet and weekly projection outputs.

READS:
  docs/win/football/nfl/03_picks/week_{week}_NFL_picks.csv

WRITES:
  docs/win/football/nfl/03_picks/selected/
      week_{week}_NFL_select_picks.csv

  docs/win/football/nfl/03_picks/projection/
      week_{week}_NFL_projection.csv

Selected output:
- Includes a game when any of these equal 1:
    ml_selected
    spread_selected
    total_selected

Projection output:
- Includes every game for the week.

edt_time:
- Converts commence_time from UTC to America/New_York.
- Output format is HH:MM.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
NFL_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_INPUT_DIR = NFL_ROOT / "03_picks"
SELECTED_OUTPUT_DIR = NFL_ROOT / "03_picks" / "selected"
PROJECTION_OUTPUT_DIR = NFL_ROOT / "03_picks" / "projection"

EASTERN_TZ = ZoneInfo("America/New_York")


SELECTED_OUTPUT_COLUMNS = [
    "week",
    "game_id",
    "game_date",
    "edt_time",
    "away_team",
    "home_team",
    "ml_selection",
    "ml_odds_american",
    "ml_model_probability",
    "spread_selection",
    "spread_line",
    "spread_odds_american",
    "total_selection",
    "total_line",
    "total_odds_american",
    "total_model_probability",
    "season",
    "season_type",
    "ml_selected",
    "spread_selected",
    "total_selected",
    "game_time",
    "commence_time",
]


PROJECTION_OUTPUT_COLUMNS = [
    "week",
    "game_id",
    "game_date",
    "edt_time",
    "away_team",
    "home_team",
    "predicted_away_score",
    "predicted_home_score",
    "predicted_total",
    "season",
    "season_type",
    "game_time",
    "commence_time",
]


REQUIRED_INPUT_COLUMNS = sorted(
    set(
        SELECTED_OUTPUT_COLUMNS
        + PROJECTION_OUTPUT_COLUMNS
        + [
            "ml_selected",
            "spread_selected",
            "total_selected",
        ]
    )
    - {"edt_time"}
)


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


def selection_flag(value: Any) -> bool:
    text = clean(value)

    if not text:
        return False

    try:
        return float(text) == 1.0
    except (TypeError, ValueError):
        return False


def convert_to_eastern_time(
    value: Any,
    *,
    row_number: int,
) -> str:
    text = clean(value)

    if not text:
        fail(
            f"Row {row_number}: commence_time is blank"
        )

    try:
        timestamp = pd.to_datetime(
            text,
            utc=True,
            errors="raise",
        )
    except Exception as exc:
        fail(
            f"Row {row_number}: invalid commence_time "
            f"{value!r}: {exc}"
        )

    return (
        timestamp
        .tz_convert(EASTERN_TZ)
        .strftime("%H:%M")
    )


def add_edt_time(
    df: pd.DataFrame,
) -> pd.DataFrame:
    output = df.copy()

    output["edt_time"] = [
        convert_to_eastern_time(
            value,
            row_number=index + 2,
        )
        for index, value in enumerate(
            output["commence_time"]
        )
    ]

    return output


def build_selected_output(
    source: pd.DataFrame,
) -> pd.DataFrame:
    selected_mask = (
        source["ml_selected"].map(selection_flag)
        | source["spread_selected"].map(selection_flag)
        | source["total_selected"].map(selection_flag)
    )

    selected = source.loc[
        selected_mask
    ].copy()

    selected = add_edt_time(selected)

    return selected[
        SELECTED_OUTPUT_COLUMNS
    ].copy()


def build_projection_output(
    source: pd.DataFrame,
) -> pd.DataFrame:
    projection = add_edt_time(
        source.copy()
    )

    return projection[
        PROJECTION_OUTPUT_COLUMNS
    ].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build final NFL selected picks and "
            "weekly projection CSVs."
        )
    )

    parser.add_argument(
        "--week",
        required=True,
        type=int,
        help="NFL week number",
    )

    args = parser.parse_args()

    week = args.week

    input_path = (
        DEFAULT_INPUT_DIR
        / f"week_{week}_NFL_picks.csv"
    )

    selected_output_path = (
        SELECTED_OUTPUT_DIR
        / f"week_{week}_NFL_select_picks.csv"
    )

    projection_output_path = (
        PROJECTION_OUTPUT_DIR
        / f"week_{week}_NFL_projection.csv"
    )

    if not input_path.exists():
        fail(
            f"Input file not found: {input_path}"
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

    selected_output = build_selected_output(
        source
    )

    projection_output = build_projection_output(
        source
    )

    SELECTED_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROJECTION_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_output.to_csv(
        selected_output_path,
        index=False,
        lineterminator="\n",
    )

    projection_output.to_csv(
        projection_output_path,
        index=False,
        lineterminator="\n",
    )

    print(
        f"WROTE {selected_output_path} "
        f"| rows={len(selected_output)}"
    )

    print(
        f"WROTE {projection_output_path} "
        f"| rows={len(projection_output)}"
    )


if __name__ == "__main__":
    main()
