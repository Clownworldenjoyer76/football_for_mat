#!/usr/bin/env python3
"""Organize weekly prop-selection CSVs into final prop-pick files."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path


PROP_ENGINE_ROOT = Path("docs/win/football/nfl/prop_engine")
OUTPUT_ROOT = PROP_ENGINE_ROOT / "output"
FINAL_ROOT = PROP_ENGINE_ROOT / "prop_picks_final"


SIMPLE_OUTPUTS = {
    "defense": {
        "actual": "actual_prop_total_tackles",
        "engine": "prop_engine_tackles",
        "low": "prop_engine_tackles_low",
        "high": "prop_engine_tackles_high",
    },
    "kicking": {
        "actual": "actual_prop_total_kicking_points",
        "engine": "prop_engine_kicking_points",
        "low": "prop_engine_kicking_points_low",
        "high": "prop_engine_kicking_points_high",
    },
    "passing": {
        "actual": "actual_prop_total_passing_yards",
        "engine": "prop_engine_passing_yards",
        "low": "prop_engine_passing_yards_low",
        "high": "prop_engine_passing_yards_high",
    },
    "receiving": {
        "actual": "actual_prop_total_receiving_yards",
        "engine": "prop_engine_receiving_yards",
        "low": "prop_engine_receiving_yards_low",
        "high": "prop_engine_receiving_yards_high",
    },
    "rushing": {
        "actual": "actual_prop_total_rushing_yards",
        "engine": "prop_engine_rushing_yards",
        "low": "prop_engine_rushing_yards_low",
        "high": "prop_engine_rushing_yards_high",
    },
}


IDENTITY_COLUMNS = [
    "game_date",
    "game_id",
    "player_name",
    "espn_player_id",
    "prop_engine_player_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organize weekly NFL prop-selection CSVs."
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--week",
        type=int,
        required=True,
    )
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input file: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames or [])


def require_columns(path: Path, fieldnames: list[str], required: list[str]) -> None:
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise RuntimeError(
            f"{path} is missing required columns: {', '.join(missing)}"
        )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_values(left: str, right: str) -> str:
    left_text = str(left or "").strip()
    right_text = str(right or "").strip()

    if not left_text or not right_text:
        return ""

    try:
        total = Decimal(left_text) + Decimal(right_text)
    except InvalidOperation as exc:
        raise ValueError(
            f"Cannot add numeric values {left_text!r} and {right_text!r}"
        ) from exc

    text = format(total, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def simple_input_path(season: int, week: int, category: str) -> Path:
    return (
        OUTPUT_ROOT
        / str(season)
        / f"week_{week}_props"
        / "selections"
        / category
        / f"week_{week}_props_select.csv"
    )


def simple_output_path(season: int, week: int, category: str) -> Path:
    return (
        FINAL_ROOT
        / str(season)
        / "stage_1"
        / f"week_{week}"
        / category
        / f"week_{week}_{category}.csv"
    )


def build_simple_output(season: int, week: int, category: str) -> None:
    config = SIMPLE_OUTPUTS[category]
    input_path = simple_input_path(season, week, category)
    rows, fieldnames = read_csv(input_path)

    output_columns = [
        "game_date",
        "game_id",
        "player_name",
        config["actual"],
        config["engine"],
        config["low"],
        config["high"],
        "espn_player_id",
        "prop_engine_player_id",
    ]

    require_columns(input_path, fieldnames, output_columns)

    output_rows = [
        {column: row.get(column, "") for column in output_columns}
        for row in rows
        if str(row.get(config["actual"], "") or "").strip()
    ]

    write_csv(
        simple_output_path(season, week, category),
        output_columns,
        output_rows,
    )


def build_combo_outputs(season: int, week: int) -> None:
    input_path = simple_input_path(season, week, "combo")
    rows, fieldnames = read_csv(input_path)

    required = [
        *IDENTITY_COLUMNS,
        "actual_prop_total_passing_plus_rushing_yards",
        "actual_prop_total_rushing_plus_receiving_yards",
        "prop_engine_passing_yards",
        "prop_engine_passing_yards_low",
        "prop_engine_passing_yards_high",
        "prop_engine_rushing_yards",
        "prop_engine_rushing_yards_low",
        "prop_engine_rushing_yards_high",
        "prop_engine_receiving_yards",
        "prop_engine_receiving_yards_low",
        "prop_engine_receiving_yards_high",
    ]
    require_columns(input_path, fieldnames, required)

    pass_rush_columns = [
        "game_date",
        "game_id",
        "player_name",
        "actual_prop_total_passing_plus_rushing_yards",
        "prop_engine_pr",
        "prop_engine_pr_low",
        "prop_engine_pr_high",
        "espn_player_id",
        "prop_engine_player_id",
    ]

    pass_rush_rows: list[dict[str, str]] = []
    for row in rows:
        if not str(
            row.get("actual_prop_total_passing_plus_rushing_yards", "") or ""
        ).strip():
            continue

        pass_rush_rows.append(
            {
                "game_date": row.get("game_date", ""),
                "game_id": row.get("game_id", ""),
                "player_name": row.get("player_name", ""),
                "actual_prop_total_passing_plus_rushing_yards": row.get(
                    "actual_prop_total_passing_plus_rushing_yards", ""
                ),
                "prop_engine_pr": add_values(
                    row.get("prop_engine_passing_yards", ""),
                    row.get("prop_engine_rushing_yards", ""),
                ),
                "prop_engine_pr_low": add_values(
                    row.get("prop_engine_passing_yards_low", ""),
                    row.get("prop_engine_rushing_yards_low", ""),
                ),
                "prop_engine_pr_high": add_values(
                    row.get("prop_engine_passing_yards_high", ""),
                    row.get("prop_engine_rushing_yards_high", ""),
                ),
                "espn_player_id": row.get("espn_player_id", ""),
                "prop_engine_player_id": row.get("prop_engine_player_id", ""),
            }
        )

    pass_rush_path = (
        FINAL_ROOT
        / str(season)
        / "stage_1"
        / f"week_{week}"
        / "combo"
        / "pass_rush_yds"
        / f"week_{week}_pass_rush_yds.csv"
    )
    write_csv(pass_rush_path, pass_rush_columns, pass_rush_rows)

    rec_rush_columns = [
        "game_date",
        "game_id",
        "player_name",
        "actual_prop_total_rushing_plus_receiving_yards",
        "prop_engine_rr",
        "prop_engine_rr_low",
        "prop_engine_rr_high",
        "espn_player_id",
        "prop_engine_player_id",
    ]

    rec_rush_rows: list[dict[str, str]] = []
    for row in rows:
        if not str(
            row.get("actual_prop_total_rushing_plus_receiving_yards", "") or ""
        ).strip():
            continue

        rec_rush_rows.append(
            {
                "game_date": row.get("game_date", ""),
                "game_id": row.get("game_id", ""),
                "player_name": row.get("player_name", ""),
                "actual_prop_total_rushing_plus_receiving_yards": row.get(
                    "actual_prop_total_rushing_plus_receiving_yards", ""
                ),
                "prop_engine_rr": add_values(
                    row.get("prop_engine_receiving_yards", ""),
                    row.get("prop_engine_rushing_yards", ""),
                ),
                "prop_engine_rr_low": add_values(
                    row.get("prop_engine_receiving_yards_low", ""),
                    row.get("prop_engine_rushing_yards_low", ""),
                ),
                "prop_engine_rr_high": add_values(
                    row.get("prop_engine_receiving_yards_high", ""),
                    row.get("prop_engine_rushing_yards_high", ""),
                ),
                "espn_player_id": row.get("espn_player_id", ""),
                "prop_engine_player_id": row.get("prop_engine_player_id", ""),
            }
        )

    rec_rush_path = (
        FINAL_ROOT
        / str(season)
        / "stage_1"
        / f"week_{week}"
        / "combo"
        / "rec_rush_yds"
        / f"week_{week}_rec_rush_yds.csv"
    )
    write_csv(rec_rush_path, rec_rush_columns, rec_rush_rows)


def main() -> None:
    args = parse_args()

    if args.week < 1:
        raise ValueError("week must be >= 1")

    for category in (
        "defense",
        "kicking",
        "passing",
        "receiving",
        "rushing",
    ):
        build_simple_output(args.season, args.week, category)

    build_combo_outputs(args.season, args.week)


if __name__ == "__main__":
    main()
