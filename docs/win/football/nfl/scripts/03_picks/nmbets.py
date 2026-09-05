#!/usr/bin/env python3

import argparse
import csv
import re
import sys
from pathlib import Path


PICKS_DIR = Path("docs/win/football/nfl/03_picks")
OUTPUT_DIR = PICKS_DIR / "nmbets"
INPUT_PATTERN = "week_*_NFL_picks.csv"

OUTPUT_FIELDS = [
    "Date",
    "Time",
    "Away_Team",
    "Home_Team",
    "Projected_Score",
    "Predicted_Margin",
    "Predicted_Total",
]

REQUIRED_FIELDS = {
    "week",
    "game_date",
    "game_time",
    "away_team",
    "home_team",
    "predicted_margin",
    "predicted_total",
    "predicted_home_score",
    "predicted_away_score",
}

TEAM_SCORE_LABELS = {
    "Arizona Cardinals": "Arizona",
    "Atlanta Falcons": "Atlanta",
    "Baltimore Ravens": "Baltimore",
    "Buffalo Bills": "Buffalo",
    "Carolina Panthers": "Carolina",
    "Chicago Bears": "Chicago",
    "Cincinnati Bengals": "Cincinnati",
    "Cleveland Browns": "Cleveland",
    "Dallas Cowboys": "Dallas",
    "Denver Broncos": "Denver",
    "Detroit Lions": "Detroit",
    "Green Bay Packers": "Green Bay",
    "Houston Texans": "Houston",
    "Indianapolis Colts": "Indianapolis",
    "Jacksonville Jaguars": "Jacksonville",
    "Kansas City Chiefs": "Kansas City",
    "Las Vegas Raiders": "Las Vegas",
    "Los Angeles Chargers": "LA Chargers",
    "Los Angeles Rams": "LA Rams",
    "Miami Dolphins": "Miami",
    "Minnesota Vikings": "Minnesota",
    "New England Patriots": "New England",
    "New Orleans Saints": "New Orleans",
    "New York Giants": "NY Giants",
    "New York Jets": "NY Jets",
    "Philadelphia Eagles": "Philadelphia",
    "Pittsburgh Steelers": "Pittsburgh",
    "San Francisco 49ers": "San Francisco",
    "Seattle Seahawks": "Seattle",
    "Tampa Bay Buccaneers": "Tampa Bay",
    "Tennessee Titans": "Tennessee",
    "Washington Commanders": "Washington",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create simplified NM NFL weekly picks files from "
            "week_*_NFL_picks.csv files."
        )
    )
    parser.add_argument(
        "--week",
        type=int,
        default=None,
        help="Optional single week to process. Default: process all weekly picks files.",
    )
    return parser.parse_args()


def team_score_label(team_name):
    team_name = (team_name or "").strip()

    if team_name in TEAM_SCORE_LABELS:
        return TEAM_SCORE_LABELS[team_name]

    parts = team_name.rsplit(" ", 1)
    if len(parts) == 2:
        return parts[0]

    return team_name


def parse_number(value, field_name, source_file, row_number):
    value = (value or "").strip()

    if value == "":
        raise ValueError(
            f"{source_file}: row {row_number}: "
            f"missing required numeric value for {field_name}"
        )

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"{source_file}: row {row_number}: "
            f"invalid numeric value for {field_name}: {value!r}"
        ) from exc


def format_number(value):
    if abs(value) < 0.005:
        value = 0.0

    return f"{value:.2f}"


def format_margin(value):
    if abs(value) < 0.005:
        value = 0.0

    return f"{value:+.2f} Home"


def week_from_filename(path):
    match = re.fullmatch(r"week_(\d+)_NFL_picks\.csv", path.name)

    if not match:
        raise ValueError(f"Unable to determine week from filename: {path}")

    return int(match.group(1))


def validate_headers(fieldnames, source_file):
    if not fieldnames:
        raise ValueError(f"{source_file}: missing CSV header")

    missing = sorted(REQUIRED_FIELDS - set(fieldnames))

    if missing:
        raise ValueError(
            f"{source_file}: missing required columns: {', '.join(missing)}"
        )


def build_output_row(row, source_file, row_number):
    away_team = (row.get("away_team") or "").strip()
    home_team = (row.get("home_team") or "").strip()

    if not away_team:
        raise ValueError(
            f"{source_file}: row {row_number}: missing away_team"
        )

    if not home_team:
        raise ValueError(
            f"{source_file}: row {row_number}: missing home_team"
        )

    home_score = parse_number(
        row.get("predicted_home_score"),
        "predicted_home_score",
        source_file,
        row_number,
    )

    away_score = parse_number(
        row.get("predicted_away_score"),
        "predicted_away_score",
        source_file,
        row_number,
    )

    predicted_margin = parse_number(
        row.get("predicted_margin"),
        "predicted_margin",
        source_file,
        row_number,
    )

    predicted_total = parse_number(
        row.get("predicted_total"),
        "predicted_total",
        source_file,
        row_number,
    )

    home_label = team_score_label(home_team)
    away_label = team_score_label(away_team)

    projected_score = (
        f"{home_label} {format_number(home_score)} "
        f"– {away_label} {format_number(away_score)}"
    )

    return {
        "Date": (row.get("game_date") or "").strip(),
        "Time": (row.get("game_time") or "").strip(),
        "Away_Team": away_team,
        "Home_Team": home_team,
        "Projected_Score": projected_score,
        "Predicted_Margin": format_margin(predicted_margin),
        "Predicted_Total": format_number(predicted_total),
    }


def process_file(source_file):
    filename_week = week_from_filename(source_file)

    output_rows = []
    row_weeks = set()

    with source_file.open("r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)

        validate_headers(reader.fieldnames, source_file)

        for row_number, row in enumerate(reader, start=2):
            raw_week = (row.get("week") or "").strip()

            if not raw_week:
                raise ValueError(
                    f"{source_file}: row {row_number}: missing week"
                )

            try:
                row_week = int(float(raw_week))
            except ValueError as exc:
                raise ValueError(
                    f"{source_file}: row {row_number}: "
                    f"invalid week value: {raw_week!r}"
                ) from exc

            row_weeks.add(row_week)

            if row_week != filename_week:
                raise ValueError(
                    f"{source_file}: row {row_number}: "
                    f"week={row_week} does not match filename week={filename_week}"
                )

            output_rows.append(
                build_output_row(
                    row=row,
                    source_file=source_file,
                    row_number=row_number,
                )
            )

    if len(row_weeks) > 1:
        raise ValueError(
            f"{source_file}: contains multiple weeks: {sorted(row_weeks)}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = (
        OUTPUT_DIR
        / f"week_{filename_week}_NM_NFL_picks.csv"
    )

    with output_file.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=OUTPUT_FIELDS,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(output_rows)

    print(
        f"week={filename_week} "
        f"rows={len(output_rows)} "
        f"input={source_file} "
        f"output={output_file}"
    )

    return output_file


def main():
    args = parse_args()

    if not PICKS_DIR.exists():
        print(
            f"ERROR: picks directory does not exist: {PICKS_DIR}",
            file=sys.stderr,
        )
        return 1

    source_files = sorted(
        PICKS_DIR.glob(INPUT_PATTERN),
        key=week_from_filename,
    )

    if args.week is not None:
        source_files = [
            path
            for path in source_files
            if week_from_filename(path) == args.week
        ]

    if not source_files:
        if args.week is not None:
            print(
                f"ERROR: no input file found for week {args.week}: "
                f"{PICKS_DIR / f'week_{args.week}_NFL_picks.csv'}",
                file=sys.stderr,
            )
        else:
            print(
                f"ERROR: no input files found matching "
                f"{PICKS_DIR / INPUT_PATTERN}",
                file=sys.stderr,
            )

        return 1

    try:
        for source_file in source_files:
            process_file(source_file)

    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
