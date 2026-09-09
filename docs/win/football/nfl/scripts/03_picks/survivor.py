#!/usr/bin/env python3

import csv
import math
import os
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
NFL_ROOT = SCRIPT_DIR.parents[1]

INPUT_DIR = NFL_ROOT / "03_picks" / "all_games"
OUTPUT_DIR = NFL_ROOT / "03_picks" / "survivor"

INPUT_PATTERN = "all_week_*_NFL_picks.csv"
FILENAME_PATTERN = re.compile(r"^all_week_(\d+)_NFL_picks\.csv$")

OUTPUT_COLUMNS = [
    "week",
    "game_id",
    "pick",
    "pt_diff",
    "away_team",
    "home_team",
]

REQUIRED_COLUMNS = [
    "week",
    "game_id",
    "away_team",
    "home_team",
    "predicted_home_spread",
    "predicted_away_spread",
]


def fail(message):
    raise RuntimeError(message)


def clean(value):
    if value is None:
        return ""
    return str(value).strip()


def parse_float(value, column, row_number):
    text = clean(value)

    if not text:
        fail(f"Row {row_number}: {column} is blank")

    try:
        number = float(text)
    except ValueError:
        fail(
            f"Row {row_number}: "
            f"{column} is not numeric: {value!r}"
        )

    if not math.isfinite(number):
        fail(
            f"Row {row_number}: "
            f"{column} is non-finite: {value!r}"
        )

    return number


def read_input(path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            fail(f"{path}: missing header row")

        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in reader.fieldnames
        ]

        if missing:
            fail(
                f"{path}: missing required columns: "
                f"{missing}"
            )

        return list(reader)


def build_output(rows, week, input_path):
    output = []
    game_ids = set()

    for index, row in enumerate(rows, start=2):
        row_week = clean(row["week"])
        game_id = clean(row["game_id"])
        away_team = clean(row["away_team"])
        home_team = clean(row["home_team"])

        if row_week != str(week):
            fail(
                f"Row {index}: expected week {week}, "
                f"found {row_week!r}"
            )

        if not game_id:
            fail(f"Row {index}: game_id is blank")

        if game_id in game_ids:
            fail(
                f"{input_path}: duplicate game_id "
                f"{game_id}"
            )

        game_ids.add(game_id)

        if not away_team:
            fail(f"Row {index}: away_team is blank")

        if not home_team:
            fail(f"Row {index}: home_team is blank")

        home_spread = parse_float(
            row["predicted_home_spread"],
            "predicted_home_spread",
            index,
        )

        away_spread = parse_float(
            row["predicted_away_spread"],
            "predicted_away_spread",
            index,
        )

        home_favorite = home_spread < 0
        away_favorite = away_spread < 0

        if home_favorite == away_favorite:
            fail(
                f"Row {index}: expected exactly one "
                f"negative spread"
            )

        if home_favorite:
            pick = home_team
            favorite_spread = home_spread
        else:
            pick = away_team
            favorite_spread = away_spread

        output.append(
            {
                "week": row_week,
                "game_id": game_id,
                "pick": pick,
                "pt_diff": abs(favorite_spread),
                "away_team": away_team,
                "home_team": home_team,
            }
        )

    output.sort(
        key=lambda row: row["pt_diff"],
        reverse=True,
    )

    return output


def write_output(rows, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_COLUMNS,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "week": row["week"],
                    "game_id": row["game_id"],
                    "pick": row["pick"],
                    "pt_diff": f"{row['pt_diff']:.1f}",
                    "away_team": row["away_team"],
                    "home_team": row["home_team"],
                }
            )

    os.replace(
        temporary_path,
        path,
    )


def main():
    input_paths = sorted(
        INPUT_DIR.glob(INPUT_PATTERN)
    )

    if not input_paths:
        fail(
            f"No input files found: "
            f"{INPUT_DIR / INPUT_PATTERN}"
        )

    for input_path in input_paths:
        match = FILENAME_PATTERN.match(
            input_path.name
        )

        if match is None:
            continue

        week = int(match.group(1))

        rows = read_input(input_path)

        output = build_output(
            rows,
            week,
            input_path,
        )

        output_path = (
            OUTPUT_DIR
            / f"{week}_survivor_picks.csv"
        )

        write_output(
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
