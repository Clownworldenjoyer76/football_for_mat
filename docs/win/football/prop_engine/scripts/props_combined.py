#!/usr/bin/env python3
"""Combine filtered Stage 2 NFL prop CSV files into one Stage 3 CSV per week."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Never
from zoneinfo import ZoneInfo

import yaml

import common
from pipeline_reporter import PipelineReporter


REPO_ROOT = common.repo_root().resolve()
PROP_ENGINE_ROOT = common.prop_root().resolve()
FINAL_ROOT = PROP_ENGINE_ROOT / "prop_picks_final"
LOCKED_ROOT = FINAL_ROOT / "locked"
MARKETS_PATH = PROP_ENGINE_ROOT / "config" / "markets.yaml"

EASTERN_TZ = ZoneInfo("America/New_York")

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
}

PICK_DIRECTIONS = {
    "over",
    "under",
}

STAGE2_FILES = (
    (Path("combo/pass_rush_yds"), "pass_rush_yds"),
    (Path("combo/rec_rush_yds"), "rec_rush_yds"),
    (Path("defense"), "defense"),
    (Path("kicking"), "kicking"),
    (Path("passing"), "passing"),
    (Path("receiving"), "receiving"),
    (Path("rushing"), "rushing"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine filtered Stage 2 NFL prop CSV files "
            "into Stage 3 final selections."
        )
    )
    parser.add_argument(
        "--season",
        type=int,
    )
    parser.add_argument(
        "--week",
        type=int,
    )

    args = parser.parse_args()

    if (args.season is None) != (args.week is None):
        parser.error("--season and --week must be supplied together")

    if args.week is not None and args.week < 1:
        parser.error("--week must be >= 1")

    return args


def fail(message: str) -> Never:
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


def expected_stage_2_files(
    week_path: Path,
) -> list[Path]:
    week_number = week_path.name.removeprefix("week_")

    return [
        (
            week_path
            / relative_path
            / f"week_{week_number}_{filename_suffix}.csv"
        )
        for relative_path, filename_suffix in STAGE2_FILES
    ]


def inspect_stage_2_files(
    week_path: Path,
) -> tuple[list[Path], list[Path], list[Path]]:
    expected = expected_stage_2_files(week_path)
    expected_set = {
        path.resolve()
        for path in expected
    }

    available = [
        path
        for path in expected
        if path.is_file()
    ]
    missing = [
        path
        for path in expected
        if not path.is_file()
    ]
    unexpected = sorted(
        path
        for path in week_path.rglob("*.csv")
        if path.is_file()
        and path.resolve() not in expected_set
    )

    return available, missing, unexpected


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


def validate_stage3_output(
    path: Path,
    expected_row_count: int,
    markets: dict[str, Any],
) -> None:
    _, written_rows = read_csv(path)

    if len(written_rows) != expected_row_count:
        fail(
            "Stage 3 output row-count mismatch: "
            f"path={path} "
            f"expected={expected_row_count} "
            f"actual={len(written_rows)}"
        )

    for row in written_rows:
        if row_passes_filters(row, markets):
            continue

        fail(
            "Stage 3 output contains a selection that violates "
            "the active markets.yaml: "
            f"path={path} "
            f"game_id={clean(row.get('game_id'))} "
            f"player_name={clean(row.get('player_name'))}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def write_locked_snapshot(
    source_path: Path,
    season: str,
    week_number: str,
) -> Path:
    if (
        not source_path.is_file()
        or source_path.stat().st_size == 0
    ):
        fail(
            "Cannot lock missing/empty Stage 3 prop file: "
            f"{source_path}"
        )

    LOCKED_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        EASTERN_TZ
    ).strftime("%Y%m%d_%H%M%S")

    locked_path = (
        LOCKED_ROOT
        / (
            f"{season}_{week_number}_all_props_"
            f"{timestamp}.csv"
        )
    )

    if locked_path.exists():
        fail(
            "Refusing to overwrite existing locked prop snapshot: "
            f"{locked_path}"
        )

    temp_path = LOCKED_ROOT / f".{locked_path.name}.tmp"

    if temp_path.exists():
        temp_path.unlink()

    try:
        shutil.copy2(
            source_path,
            temp_path,
        )

        source_hash = sha256_file(source_path)
        locked_hash = sha256_file(temp_path)

        if source_hash != locked_hash:
            fail(
                "Locked prop snapshot hash verification failed: "
                f"{locked_path}"
            )

        os.replace(temp_path, locked_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    if sha256_file(locked_path) != sha256_file(source_path):
        fail(
            "Locked prop snapshot verification failed after publish: "
            f"{locked_path}"
        )

    return locked_path


def process_week(
    season: str,
    week_path: Path,
    markets: dict[str, Any],
    *,
    lock_snapshot: bool,
    reporter: PipelineReporter,
) -> dict[str, Any]:
    week_number = week_path.name.removeprefix("week_")
    csv_files, missing_files, unexpected_files = inspect_stage_2_files(
        week_path
    )

    if missing_files and lock_snapshot:
        fail(
            "Target Stage 2 week is incomplete; missing expected files: "
            + ", ".join(
                path.relative_to(REPO_ROOT).as_posix()
                for path in missing_files
            )
        )

    if missing_files:
        reporter.warning(
            "Historical Stage 2 week is incomplete.",
            season=season,
            week=week_number,
            missing_files=[
                path.relative_to(REPO_ROOT).as_posix()
                for path in missing_files
            ],
        )

    if unexpected_files:
        reporter.warning(
            "Unexpected Stage 2 CSV files were ignored.",
            season=season,
            week=week_number,
            unexpected_files=[
                path.relative_to(REPO_ROOT).as_posix()
                for path in unexpected_files
            ],
        )

    if not csv_files:
        return {
            "season": season,
            "week": week_number,
            "status": "skipped",
            "input_files": [],
            "missing_files": [
                path.relative_to(REPO_ROOT).as_posix()
                for path in missing_files
            ],
            "unexpected_files": [
                path.relative_to(REPO_ROOT).as_posix()
                for path in unexpected_files
            ],
            "input_rows": 0,
            "selected_rows": 0,
            "selected_by_category_side": {},
            "locked_snapshot": "",
        }

    for csv_path in csv_files:
        reporter.add_input(csv_path)

    fieldnames, rows = combine_csv_files(csv_files)

    if not fieldnames:
        if lock_snapshot:
            fail(
                "Target Stage 2 files contain no CSV headers: "
                f"season={season} week={week_number}"
            )

        reporter.warning(
            "Historical Stage 2 files contain no CSV headers.",
            season=season,
            week=week_number,
        )

        return {
            "season": season,
            "week": week_number,
            "status": "skipped",
            "input_files": [
                path.relative_to(REPO_ROOT).as_posix()
                for path in csv_files
            ],
            "missing_files": [
                path.relative_to(REPO_ROOT).as_posix()
                for path in missing_files
            ],
            "unexpected_files": [
                path.relative_to(REPO_ROOT).as_posix()
                for path in unexpected_files
            ],
            "input_rows": int(len(rows)),
            "selected_rows": 0,
            "selected_by_category_side": {},
            "locked_snapshot": "",
        }

    filtered_rows = filter_rows(
        rows,
        markets,
    )

    selected_by_category_side: dict[str, dict[str, int]] = {}

    for row in filtered_rows:
        category, _ = detect_category(row)
        pick = clean(row.get("pick")).casefold()

        category_counts = selected_by_category_side.setdefault(
            category,
            {
                "over": 0,
                "under": 0,
            },
        )
        category_counts[pick] += 1

    output_path = (
        FINAL_ROOT
        / season
        / "stage_3"
        / f"{season}_{week_number}_all_props.csv"
    )

    common.write_filtered_csv_dict_rows_atomic(
        output_path,
        fieldnames,
        filtered_rows,
    )
    reporter.add_output(output_path)

    validate_stage3_output(
        output_path,
        len(filtered_rows),
        markets,
    )

    locked_path: Path | None = None

    if lock_snapshot:
        locked_path = write_locked_snapshot(
            output_path,
            season,
            week_number,
        )
        reporter.add_output(locked_path)

    return {
        "season": season,
        "week": week_number,
        "status": "processed",
        "input_files": [
            path.relative_to(REPO_ROOT).as_posix()
            for path in csv_files
        ],
        "missing_files": [
            path.relative_to(REPO_ROOT).as_posix()
            for path in missing_files
        ],
        "unexpected_files": [
            path.relative_to(REPO_ROOT).as_posix()
            for path in unexpected_files
        ],
        "output_file": output_path.relative_to(REPO_ROOT).as_posix(),
        "input_rows": int(len(rows)),
        "selected_rows": int(len(filtered_rows)),
        "selected_by_category_side": selected_by_category_side,
        "locked_snapshot": (
            locked_path.relative_to(REPO_ROOT).as_posix()
            if locked_path is not None
            else ""
        ),
    }


def _run(reporter: PipelineReporter) -> None:
    args = parse_args()
    reporter.add_input(MARKETS_PATH)

    markets = load_markets()
    stage_2_weeks = discover_stage_2_weeks()

    if not stage_2_weeks:
        raise FileNotFoundError(
            f"No Stage 2 week folders found under {FINAL_ROOT}"
        )

    lock_snapshot = (
        args.season is not None
        and args.week is not None
    )

    if lock_snapshot:
        target_season = str(args.season)
        target_week = f"week_{args.week}"

        stage_2_weeks = [
            (season, week_path)
            for season, week_path in stage_2_weeks
            if (
                season == target_season
                and week_path.name == target_week
            )
        ]

        if not stage_2_weeks:
            raise FileNotFoundError(
                "No Stage 2 folder found for "
                f"season={args.season} week={args.week}"
            )

    week_stats: dict[str, dict[str, Any]] = {}

    for season, week_path in stage_2_weeks:
        stats = process_week(
            season,
            week_path,
            markets,
            lock_snapshot=lock_snapshot,
            reporter=reporter,
        )
        week_stats[f"{season}/{week_path.name}"] = stats

    processed = [
        stats
        for stats in week_stats.values()
        if stats["status"] == "processed"
    ]

    total_input_rows = sum(
        int(stats["input_rows"])
        for stats in processed
    )
    total_selected_rows = sum(
        int(stats["selected_rows"])
        for stats in processed
    )
    missing_file_count = sum(
        len(stats["missing_files"])
        for stats in week_stats.values()
    )
    unexpected_file_count = sum(
        len(stats["unexpected_files"])
        for stats in week_stats.values()
    )

    selected_totals: dict[str, dict[str, int]] = {}

    for stats in processed:
        category_counts = stats["selected_by_category_side"]
        if not isinstance(category_counts, dict):
            raise RuntimeError("Invalid Stage 3 selection statistics.")

        for category, counts in category_counts.items():
            if not isinstance(counts, dict):
                raise RuntimeError(
                    "Invalid Stage 3 category selection statistics."
                )

            totals = selected_totals.setdefault(
                str(category),
                {
                    "over": 0,
                    "under": 0,
                },
            )
            totals["over"] += int(counts.get("over", 0))
            totals["under"] += int(counts.get("under", 0))

    reporter.set_rows(
        rows_in=int(total_input_rows),
        rows_out=int(total_selected_rows),
    )
    reporter.update_details(
        {
            "lock_snapshot": bool(lock_snapshot),
            "target_season": (
                int(args.season)
                if args.season is not None
                else None
            ),
            "target_week": (
                int(args.week)
                if args.week is not None
                else None
            ),
            "weeks_discovered": int(len(stage_2_weeks)),
            "weeks_processed": int(len(processed)),
            "missing_expected_stage_2_files": int(
                missing_file_count
            ),
            "ignored_unexpected_stage_2_files": int(
                unexpected_file_count
            ),
            "selected_by_category_side": selected_totals,
            "week_stats": week_stats,
        }
    )

    print(
        "PROPS COMBINED: PASS "
        f"weeks={len(processed)} "
        f"rows_in={total_input_rows} "
        f"selected={total_selected_rows} "
        f"missing={missing_file_count} "
        f"ignored={unexpected_file_count} "
        f"locked={str(lock_snapshot).lower()}"
    )


def main() -> None:
    with PipelineReporter(
        script=Path(__file__).name,
        stage="props",
        report_root=common.prop_root() / "logs" / "pipeline_reports",
    ) as reporter:
        _run(reporter)


if __name__ == "__main__":
    main()
