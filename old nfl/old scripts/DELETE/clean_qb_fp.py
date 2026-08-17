#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Optional


RAW_DIR = Path("docs/win/football/nfl/data/qb_data/raw_qb")
OUT_DIR = Path("docs/win/football/nfl/data/qb_data/clean_qb")

REQUIRED_HEADERS = [
    "RK",
    "Player",
    "G",
    "COMP",
    "ATT",
    "PCT",
    "YDS",
    "Y/A",
    "AIR",
    "AIR/A",
    "10+ YDS",
    "20+ YDS",
    "30+ YDS",
    "40+ YDS",
    "50+ YDS",
    "PKT TIME",
    "SACK",
    "KNCK",
    "HRRY",
    "BLITZ",
    "POOR",
    "DROP",
    "RZ ATT",
    "RTG",
]

OUTPUT_HEADERS = [
    "season",
    "team",
    "qb_name",
    "fantasypros_player_raw",
    "rank",
    "games",
    "completions",
    "attempts",
    "completion_pct",
    "passing_yards",
    "yards_per_attempt",
    "air_yards",
    "air_yards_per_attempt",
    "passes_10_plus_yards",
    "passes_20_plus_yards",
    "passes_30_plus_yards",
    "passes_40_plus_yards",
    "passes_50_plus_yards",
    "pocket_time",
    "sacks",
    "knockdowns",
    "hurries",
    "blitzes",
    "poor_throws",
    "drops",
    "red_zone_attempts",
    "passer_rating",
    "adjusted_completion_pct",
    "pressure_to_sack_rate",
]

HEADER_MAP = {
    "RK": "rank",
    "G": "games",
    "COMP": "completions",
    "ATT": "attempts",
    "PCT": "completion_pct",
    "YDS": "passing_yards",
    "Y/A": "yards_per_attempt",
    "AIR": "air_yards",
    "AIR/A": "air_yards_per_attempt",
    "10+ YDS": "passes_10_plus_yards",
    "20+ YDS": "passes_20_plus_yards",
    "30+ YDS": "passes_30_plus_yards",
    "40+ YDS": "passes_40_plus_yards",
    "50+ YDS": "passes_50_plus_yards",
    "PKT TIME": "pocket_time",
    "SACK": "sacks",
    "KNCK": "knockdowns",
    "HRRY": "hurries",
    "BLITZ": "blitzes",
    "POOR": "poor_throws",
    "DROP": "drops",
    "RZ ATT": "red_zone_attempts",
    "RTG": "passer_rating",
}


def clean_number(value: object) -> Optional[float]:
    if value is None:
        return None

    text = str(value).strip()
    if text == "" or text == "-":
        return None

    text = text.replace(",", "")
    text = text.replace("%", "")

    try:
        return float(text)
    except ValueError:
        return None


def format_value(value: Optional[float]) -> str:
    if value is None:
        return ""

    if float(value).is_integer():
        return str(int(value))

    return f"{value:.6f}".rstrip("0").rstrip(".")


def safe_rate(numerator: Optional[float], denominator: Optional[float]) -> str:
    if numerator is None or denominator is None or denominator == 0:
        return ""

    return f"{numerator / denominator:.6f}".rstrip("0").rstrip(".")


def split_player(raw_player: str) -> tuple[str, str]:
    text = raw_player.strip()

    match = re.match(r"^(?P<name>.*?)(?:\s{2,}|\t+)(?P<team>[A-Z]{2,3})$", text)
    if match:
        return match.group("name").strip(), match.group("team").strip()

    match = re.match(r"^(?P<name>.*)\s+(?P<team>[A-Z]{2,3})$", text)
    if match:
        return match.group("name").strip(), match.group("team").strip()

    return text, ""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")

        missing = [h for h in REQUIRED_HEADERS if h not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} missing required headers: {missing}")

        return list(reader)


def clean_row(row: dict[str, str], season: int) -> dict[str, str]:
    raw_player = row.get("Player", "").strip()
    qb_name, team = split_player(raw_player)

    cleaned: dict[str, str] = {
        "season": str(season),
        "team": team,
        "qb_name": qb_name,
        "fantasypros_player_raw": raw_player,
    }

    numeric_values: dict[str, Optional[float]] = {}

    for source_header, output_header in HEADER_MAP.items():
        value = clean_number(row.get(source_header))
        numeric_values[output_header] = value
        cleaned[output_header] = format_value(value)

    completions = numeric_values.get("completions")
    drops = numeric_values.get("drops")
    sacks = numeric_values.get("sacks")
    knockdowns = numeric_values.get("knockdowns")
    hurries = numeric_values.get("hurries")

    adjusted_denominator = None
    if completions is not None and drops is not None:
        adjusted_denominator = completions + drops

    pressure_denominator = None
    if sacks is not None and knockdowns is not None and hurries is not None:
        pressure_denominator = sacks + knockdowns + hurries

    cleaned["adjusted_completion_pct"] = safe_rate(completions, adjusted_denominator)
    cleaned["pressure_to_sack_rate"] = safe_rate(sacks, pressure_denominator)

    return {header: cleaned.get(header, "") for header in OUTPUT_HEADERS}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True, type=int)
    args = parser.parse_args()

    raw_files = sorted(RAW_DIR.glob("*.csv"))
    if not raw_files:
        print(f"No raw QB CSV files found in {RAW_DIR}", file=sys.stderr)
        return 1

    output_rows: list[dict[str, str]] = []

    for raw_file in raw_files:
        rows = read_csv(raw_file)
        for row in rows:
            output_rows.append(clean_row(row, args.season))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / f"{args.season}_fp_qb.csv"

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADERS)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"input_files={len(raw_files)}")
    print(f"output_rows={len(output_rows)}")
    print(f"output={output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
