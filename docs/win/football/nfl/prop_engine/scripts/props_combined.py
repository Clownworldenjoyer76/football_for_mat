#!/usr/bin/env python3
"""Combine filtered Stage 2 NFL prop CSV files into one Stage 3 CSV per week."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import yaml


PROP_ENGINE_ROOT = Path("docs/win/football/nfl/prop_engine")
FINAL_ROOT = PROP_ENGINE_ROOT / "prop_picks_final"
MARKETS_PATH = PROP_ENGINE_ROOT / "config" / "markets.yaml"

PROP_LINE_COLUMNS = {
    "kicking_points": "actual_prop_total_kicking_points",
    "passing_rushing_yards": "actual_prop_total_passing_plus_rushing_yards",
    "passing_yards": "actual_prop_total_passing_yards",
    "receiving_yards": "actual_prop_total_receiving_yards",
    "rushing_receiving_yards": "actual_prop_total_rushing_plus_receiving_yards",
    "rushing_yards": "actual_prop_total_rushing_yards",
    "tackles": "actual_prop_total_tackles",
}

FILTER_KEYS = {
    "actual_prop_total_*",
    "pick_prob",
    "over_prob",
    "under_prob",
}

PICK_DIRECTIONS = {
    "over",
    "under",
}


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
    }:
        return ""

    return text


def number(value: Any, label: str) -> float:
    text = clean(value)

    if not text:
        fail(f"{label} is required")

    try:
        result = float(text)
    except (TypeError, ValueError):
        fail(f"{label} must be numeric; found {value!r}")

    if not math.isfinite(result):
        fail(f"{label} must be finite; found {value!r}")

    return result


def boolean(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value

    text = clean(value).casefold()

    if text in {"true", "yes", "y", "1", "on"}:
        return True

    if text in {"false", "no", "n", "0", "off"}:
        return False

    fail(f"{label} must be true/false; found {value!r}")


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{label} must be a YAML mapping")

    return value


def reject_unknown(
    mapping: dict[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    unknown = sorted(set(mapping) - allowed)

    if unknown:
        fail(f"{label} contains unsupported keys: {unknown}")


def parse_bands(
    value: Any,
    label: str,
) -> list[tuple[float, float]]:
    if not isinstance(value, list) or not value:
        fail(f"{label} must be a non-empty list of [min, max] bands")

    result: list[tuple[float, float]] = []

    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            fail(f"{label}[{index}] must be [min, max]")

        low = number(item[0], f"{label}[{index}][0]")
        high = number(item[1], f"{label}[{index}][1]")

        if low > high:
            fail(f"{label}[{index}] has min greater than max")

        result.append((low, high))

    return result


def matches_bands(
    value: float,
    bands: list[tuple[float, float]],
) -> bool:
    return any(
        low <= value <= high
        for low, high in bands
    )


def load_markets() -> dict[str, Any]:
    if not MARKETS_PATH.is_file():
        fail(f"Missing markets config: {MARKETS_PATH}")

    with MARKETS_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw = require_mapping(raw, "markets.yaml")

    reject_unknown(
        raw,
        set(PROP_LINE_COLUMNS),
        "markets.yaml",
    )

    missing_categories = sorted(
        set(PROP_LINE_COLUMNS) - set(raw)
    )

    if missing_categories:
        fail(
            "markets.yaml missing prop categories: "
            f"{missing_categories}"
        )

    normalized: dict[str, Any] = {}

    for category in PROP_LINE_COLUMNS:
        market_label = f"markets.yaml.{category}"
        market_raw = require_mapping(
            raw.get(category),
            market_label,
        )

        reject_unknown(
            market_raw,
            {
                "enabled",
                "pick_preference",
                "over",
                "under",
            },
            market_label,
        )

        preference = clean(
            market_raw.get("pick_preference")
        ).casefold()

        if preference != "best_prob":
            fail(
                f"{market_label}.pick_preference must be best_prob"
            )

        market_config: dict[str, Any] = {
            "enabled": boolean(
                market_raw.get("enabled"),
                f"{market_label}.enabled",
            ),
            "pick_preference": preference,
            "sides": {},
        }

        for side in ("over", "under"):
            side_label = f"{market_label}.{side}"
            side_raw = require_mapping(
                market_raw.get(side),
                side_label,
            )

            reject_unknown(
                side_raw,
                {"enabled"} | FILTER_KEYS,
                side_label,
            )

            filters: dict[str, list[tuple[float, float]]] = {}

            for filter_key in FILTER_KEYS:
                if filter_key not in side_raw:
                    fail(
                        f"{side_label} missing required filter: "
                        f"{filter_key}"
                    )

                filters[filter_key] = parse_bands(
                    side_raw[filter_key],
                    f"{side_label}.{filter_key}",
                )

            market_config["sides"][side] = {
                "enabled": boolean(
                    side_raw.get("enabled"),
                    f"{side_label}.enabled",
                ),
                "filters": filters,
            }

        normalized[category] = market_config

    return normalized


def discover_stage_2_weeks() -> list[tuple[str, Path]]:
    discovered: list[tuple[str, Path]] = []

    if not FINAL_ROOT.is_dir():
        return discovered

    for season_path in sorted(FINAL_ROOT.iterdir()):
        if not season_path.is_dir():
            continue

        if not season_path.name.isdigit():
            continue

        stage_2_path = season_path / "stage_2"

        if not stage_2_path.is_dir():
            continue

        for week_path in sorted(stage_2_path.iterdir()):
            if not week_path.is_dir():
                continue

            if not week_path.name.startswith("week_"):
                continue

            week_number = week_path.name.removeprefix("week_")

            if not week_number.isdigit():
                continue

            discovered.append((season_path.name, week_path))

    return discovered


def discover_csv_files(week_path: Path) -> list[Path]:
    return sorted(
        path
        for path in week_path.rglob("*")
        if path.is_file() and path.suffix.lower() == ".csv"
    )


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]

    return fieldnames, rows


def combine_csv_files(csv_files: list[Path]) -> tuple[list[str], list[dict[str, str]]]:
    combined_fieldnames: list[str] = []
    combined_rows: list[dict[str, str]] = []

    for csv_path in csv_files:
        fieldnames, rows = read_csv(csv_path)

        for fieldname in fieldnames:
            if fieldname not in combined_fieldnames:
                combined_fieldnames.append(fieldname)

        combined_rows.extend(rows)

    return combined_fieldnames, combined_rows


def detect_category(
    row: dict[str, str],
) -> tuple[str, str]:
    found = [
        (category, line_column)
        for category, line_column in PROP_LINE_COLUMNS.items()
        if clean(row.get(line_column))
    ]

    if not found:
        fail(
            "Stage 2 row has no recognized prop line: "
            f"game_id={clean(row.get('game_id'))} "
            f"player_name={clean(row.get('player_name'))}"
        )

    if len(found) > 1:
        fail(
            "Stage 2 row has multiple recognized prop lines: "
            f"game_id={clean(row.get('game_id'))} "
            f"player_name={clean(row.get('player_name'))} "
            f"categories={[category for category, _ in found]}"
        )

    return found[0]


def row_passes_filters(
    row: dict[str, str],
    markets: dict[str, Any],
) -> bool:
    category, line_column = detect_category(row)
    market = markets[category]

    if not market["enabled"]:
        return False

    pick = clean(row.get("pick")).casefold()

    if pick not in PICK_DIRECTIONS:
        return False

    side = market["sides"][pick]

    if not side["enabled"]:
        return False

    value_columns = {
        "actual_prop_total_*": line_column,
        "pick_prob": "pick_prob",
        "over_prob": "over_prob",
        "under_prob": "under_prob",
    }

    for filter_key, column in value_columns.items():
        value = number(
            row.get(column),
            (
                f"{category}.{pick}.{filter_key} "
                f"for game_id={clean(row.get('game_id'))} "
                f"player_name={clean(row.get('player_name'))}"
            ),
        )

        if not matches_bands(
            value,
            side["filters"][filter_key],
        ):
            return False

    return True


def filter_rows(
    rows: list[dict[str, str]],
    markets: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row_passes_filters(row, markets)
    ]


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    fieldname: row.get(fieldname, "")
                    for fieldname in fieldnames
                }
            )


def process_week(
    season: str,
    week_path: Path,
    markets: dict[str, Any],
) -> None:
    week_number = week_path.name.removeprefix("week_")
    csv_files = discover_csv_files(week_path)

    if not csv_files:
        return

    fieldnames, rows = combine_csv_files(csv_files)

    if not fieldnames:
        return

    filtered_rows = filter_rows(
        rows,
        markets,
    )

    output_path = (
        FINAL_ROOT
        / season
        / "stage_3"
        / f"{season}_{week_number}_all_props.csv"
    )

    write_csv(
        output_path,
        fieldnames,
        filtered_rows,
    )


def main() -> None:
    markets = load_markets()
    stage_2_weeks = discover_stage_2_weeks()

    if not stage_2_weeks:
        raise FileNotFoundError(
            f"No Stage 2 week folders found under {FINAL_ROOT}"
        )

    for season, week_path in stage_2_weeks:
        process_week(
            season,
            week_path,
            markets,
        )


if __name__ == "__main__":
    main()
