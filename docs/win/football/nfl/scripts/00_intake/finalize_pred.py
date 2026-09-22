#!/usr/bin/env python3
"""
Finalize one complete season of cleaned ESPN predictions.

Inputs:
    docs/win/football/nfl/00_intake/predictions/clean/
        {season}_{season_type}_{week}_predictions.csv
    docs/win/football/nfl/00_intake/schedule/{season}_schedule.csv
    docs/win/football/nfl/00_intake/schedule/weekly/
        week_{week}_NFL_weekly_schedule.csv

Output:
    docs/win/football/nfl/00_intake/predictions/final/
        {season}_{season_type}_{week}_clean_predictions.csv

Schedule date/time are required for every game. Projected score fields are
populated only when a valid weekly total exists; weeks without a weekly odds
file and games with legitimately unavailable totals remain blank.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import re
import shutil
import sys
import tempfile
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import ModuleType
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter

CLEANER_PATH = SCRIPT_PATH.with_name("clean_e_pred.py")
WEEKLY_BUILDER_PATH = SCRIPT_PATH.with_name(
    "build_weekly_schedule.py"
)

CLEAN_DIR = (
    NFL_ROOT
    / "00_intake"
    / "predictions"
    / "clean"
)
WEEKLY_DIR = (
    NFL_ROOT
    / "00_intake"
    / "schedule"
    / "weekly"
)
OUT_DIR = (
    NFL_ROOT
    / "00_intake"
    / "predictions"
    / "final"
)
LOG_PATH = (
    NFL_ROOT
    / "errors"
    / "00_intake"
    / "finalize_pred.txt"
)
REPORT_ROOT = NFL_ROOT / "errors"

OUT_HEADERS = [
    "game_id",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "matchupQuality",
    "home_prob",
    "away_prob",
    "tie_prob",
    "away_projected_pts",
    "home_projected_pts",
    "total_projected_pts",
    "home_PtDiff",
    "away_PtDiff",
    "home_rating",
    "away_rating",
    "game_name",
    "season",
    "season_type",
    "week",
    "sport",
    "league",
]

FINALIZED_FIELDS = {
    "game_date",
    "game_time",
    "away_projected_pts",
    "home_projected_pts",
    "total_projected_pts",
}

WEEKLY_FILE_RE = re.compile(
    r"week_(\d+)_NFL_weekly_schedule\.csv"
)


class FinalizePredictionError(RuntimeError):
    pass


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def fail(message: str) -> None:
    raise FinalizePredictionError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize one complete season of cleaned ESPN "
            "predictions."
        )
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
    )
    args = parser.parse_args()

    if args.season < 2000 or args.season > 2100:
        parser.error("--season must be between 2000 and 2100")

    return args


def load_module(
    *,
    path: Path,
    module_name: str,
) -> ModuleType:
    if not path.is_file():
        fail(f"Missing dependency module: {path}")

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    if spec is None or spec.loader is None:
        fail(f"Could not load dependency module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(
    path: Path,
    *,
    label: str,
    exact_columns: list[str],
) -> list[dict[str, str]]:
    if not path.is_file():
        fail(f"Missing {label}: {path}")

    if path.stat().st_size == 0:
        fail(f"Zero-byte {label}: {path}")

    try:
        with path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
    except Exception as exc:
        fail(
            f"Could not read {label} {path}: "
            f"{type(exc).__name__}: {exc}"
        )

    if fieldnames != exact_columns:
        fail(
            f"{label} has unexpected column order/schema. "
            f"Expected={exact_columns} actual={fieldnames}"
        )

    if not rows:
        fail(f"{label} contains no data rows: {path}")

    return rows


def finite_decimal(
    raw: Any,
    *,
    label: str,
) -> Decimal:
    text = clean(raw)
    if not text:
        fail(f"{label} is blank")

    try:
        value = Decimal(text)
    except InvalidOperation:
        fail(
            f"{label} must be numeric; received={text!r}"
        )

    if not value.is_finite():
        fail(
            f"{label} must be finite; received={text!r}"
        )

    return value


def optional_decimal(
    raw: Any,
    *,
    label: str,
) -> Decimal | None:
    text = clean(raw)
    if not text:
        return None

    return finite_decimal(
        text,
        label=label,
    )


def fmt2(value: Decimal) -> str:
    if not value.is_finite():
        fail(
            f"Cannot format non-finite Decimal value={value}"
        )

    try:
        return str(
            value.quantize(
                Decimal("0.01")
            )
        )
    except InvalidOperation as exc:
        fail(
            f"Could not quantize value={value}: {exc}"
        )


def expected_final_paths(
    clean_paths: dict[
        tuple[str, str, str],
        Path,
    ],
) -> dict[
    tuple[str, str, str],
    Path,
]:
    return {
        key: (
            OUT_DIR
            / (
                f"{key[0]}_{key[1]}_{key[2]}"
                "_clean_predictions.csv"
            )
        )
        for key in clean_paths
    }


def load_and_validate_clean_generation(
    *,
    season: int,
    cleaner: ModuleType,
    reporter: PipelineReporter,
) -> tuple[
    list[dict[str, str]],
    dict[
        tuple[str, str, str],
        list[dict[str, str]],
    ],
    dict[
        tuple[str, str, str],
        Path,
    ],
]:
    producer = cleaner.load_producer_module()

    (
        schedule_rows,
        source_rows_by_group,
        producer_paths,
    ) = cleaner.load_and_validate_sources(
        season=season,
        producer=producer,
        reporter=reporter,
    )

    clean_paths = cleaner.expected_clean_paths(
        producer_paths
    )

    expected_names = {
        path.name
        for path in clean_paths.values()
    }

    if not CLEAN_DIR.is_dir():
        fail(
            f"Missing clean prediction directory: "
            f"{CLEAN_DIR}"
        )

    actual_paths = sorted(
        CLEAN_DIR.glob(
            f"{season}_*_predictions.csv"
        )
    )
    actual_names = {
        path.name
        for path in actual_paths
    }

    if actual_names != expected_names:
        fail(
            "Clean prediction file set mismatch "
            f"missing={sorted(expected_names - actual_names)} "
            f"extra={sorted(actual_names - expected_names)}"
        )

    clean_rows_by_group: dict[
        tuple[str, str, str],
        list[dict[str, str]],
    ] = {}

    for key, path in sorted(
        clean_paths.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            int(item[0][2]),
        ),
    ):
        reporter.add_input(path)

        rows = read_csv(
            path,
            label="clean ESPN prediction input",
            exact_columns=list(cleaner.OUT_HEADERS),
        )

        cleaner.validate_clean_rows(
            rows,
            source_rows=source_rows_by_group[key],
            key=key,
            path=path,
        )

        expected_rows = cleaner.transform_source_file(
            source_rows_by_group[key],
            path=(
                cleaner.IN_DIR
                / (
                    f"{key[0]}_{key[1]}_{key[2]}"
                    "_e_predictions.csv"
                )
            ),
        )

        if (
            cleaner.normalize_rows(rows)
            != cleaner.normalize_rows(expected_rows)
        ):
            fail(
                "Clean prediction input differs from the "
                f"validated cleaner transform: {path}"
            )

        clean_rows_by_group[key] = (
            cleaner.normalize_rows(rows)
        )

    return (
        schedule_rows,
        clean_rows_by_group,
        clean_paths,
    )


def build_schedule_index(
    schedule_rows: list[dict[str, str]],
    *,
    season: int,
) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}

    for line_number, row in enumerate(
        schedule_rows,
        start=2,
    ):
        if clean(row.get("season")) != str(season):
            fail(
                f"Season schedule line {line_number} has "
                f"wrong season={row.get('season')!r}"
            )

        game_id = clean(row.get("game_id"))
        game_date = clean(row.get("game_date"))
        game_time = clean(row.get("game_time"))

        if not game_id:
            fail(
                f"Season schedule line {line_number} "
                "has blank game_id"
            )

        if game_id in index:
            fail(
                f"Season schedule contains duplicate "
                f"game_id={game_id}"
            )

        if not game_date or not game_time:
            fail(
                f"Season schedule game_id={game_id} has "
                "blank game_date/game_time"
            )

        index[game_id] = row

    return index


def validate_weekly_file(
    *,
    path: Path,
    rows: list[dict[str, str]],
    season: int,
    schedule_rows: list[dict[str, str]],
    schedule_index: dict[str, dict[str, str]],
    weekly: ModuleType,
) -> tuple[
    int | None,
    dict[str, dict[str, str]],
]:
    match = WEEKLY_FILE_RE.fullmatch(path.name)
    if match is None:
        fail(
            f"Unexpected weekly schedule filename: {path}"
        )

    file_week = int(match.group(1))

    row_seasons = {
        clean(row.get("season"))
        for row in rows
    }

    if len(row_seasons) != 1:
        fail(
            f"{path} contains multiple seasons: "
            f"{sorted(row_seasons)}"
        )

    row_season = next(iter(row_seasons))

    if row_season != str(season):
        return None, {}

    row_types = {
        clean(row.get("season_type"))
        for row in rows
    }
    row_weeks = {
        clean(row.get("week"))
        for row in rows
    }

    if len(row_types) != 1 or len(row_weeks) != 1:
        fail(
            f"{path} contains multiple season_type/week "
            "targets"
        )

    season_type = next(iter(row_types))
    row_week = next(iter(row_weeks))

    if not season_type:
        fail(f"{path} has blank season_type")

    if row_week != str(file_week):
        fail(
            f"{path} row week={row_week!r} does not match "
            f"filename week={file_week}"
        )

    target_schedule_rows = weekly.validate_schedule(
        schedule_rows,
        season=season,
        season_type=season_type,
        week=file_week,
    )

    expected_ids = {
        clean(row.get("game_id"))
        for row in target_schedule_rows
    }
    actual_ids: set[str] = set()
    lookup: dict[str, dict[str, str]] = {}

    for line_number, row in enumerate(rows, start=2):
        game_id = clean(row.get("game_id"))

        if not game_id:
            fail(
                f"{path} line {line_number} has blank game_id"
            )

        if game_id in actual_ids:
            fail(
                f"{path} contains duplicate game_id={game_id}"
            )

        if game_id not in schedule_index:
            fail(
                f"{path} game_id={game_id} is absent from "
                f"{season}_schedule.csv"
            )

        schedule_row = schedule_index[game_id]
        actual_target = (
            clean(row.get("season")),
            clean(row.get("season_type")),
            clean(row.get("week")),
        )
        expected_target = (
            clean(schedule_row.get("season")),
            clean(schedule_row.get("season_type")),
            clean(schedule_row.get("week")),
        )

        if actual_target != expected_target:
            fail(
                f"{path} game_id={game_id} target mismatch "
                f"weekly={actual_target} schedule={expected_target}"
            )

        for field in (
            "away_team",
            "home_team",
        ):
            if clean(row.get(field)) != clean(
                schedule_row.get(field)
            ):
                fail(
                    f"{path} game_id={game_id} "
                    f"{field} does not match season schedule"
                )

        odds_available = clean(
            row.get("odds_available")
        )
        missing_reason = clean(
            row.get("odds_missing_reason")
        )
        provider_game_id = clean(
            row.get("odds_provider_game_id")
        )
        bookmaker = clean(
            row.get("bookmaker")
        )
        total_text = clean(
            row.get("total")
        )

        if odds_available not in {"0", "1"}:
            fail(
                f"{path} game_id={game_id} has invalid "
                f"odds_available={odds_available!r}"
            )

        if odds_available == "1":
            if (
                not provider_game_id
                or missing_reason
                or not bookmaker
            ):
                fail(
                    f"{path} game_id={game_id} has invalid "
                    "available-odds state"
                )

            if total_text:
                finite_decimal(
                    total_text,
                    label=(
                        f"{path} game_id={game_id} total"
                    ),
                )

        elif missing_reason == "no_odds_returned":
            if not provider_game_id:
                fail(
                    f"{path} game_id={game_id} has invalid "
                    "no_odds_returned state"
                )

            if total_text:
                fail(
                    f"{path} game_id={game_id} has a total "
                    "while odds_available=0"
                )

        elif missing_reason == "no_odds_event_match":
            if provider_game_id:
                fail(
                    f"{path} game_id={game_id} has provider ID "
                    "with no_odds_event_match"
                )

            if total_text:
                fail(
                    f"{path} game_id={game_id} has a total "
                    "while odds_available=0"
                )

        else:
            fail(
                f"{path} game_id={game_id} has invalid "
                f"odds_missing_reason={missing_reason!r}"
            )

        actual_ids.add(game_id)
        lookup[game_id] = row

    if actual_ids != expected_ids:
        fail(
            f"{path} game IDs do not exactly match "
            f"{season}/{season_type}/week {file_week} schedule"
        )

    return file_week, lookup


def load_weekly_odds(
    *,
    season: int,
    schedule_rows: list[dict[str, str]],
    schedule_index: dict[str, dict[str, str]],
    weekly: ModuleType,
    reporter: PipelineReporter,
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, int],
]:
    metrics = {
        "weekly_files_scanned": 0,
        "target_season_weekly_files": 0,
        "games_with_valid_total": 0,
        "games_available_without_total": 0,
        "games_odds_unavailable": 0,
    }

    if not WEEKLY_DIR.is_dir():
        return {}, metrics

    lookup: dict[str, dict[str, str]] = {}

    for path in sorted(
        WEEKLY_DIR.glob(
            "week_*_NFL_weekly_schedule.csv"
        )
    ):
        metrics["weekly_files_scanned"] += 1

        rows = read_csv(
            path,
            label="weekly schedule/odds input",
            exact_columns=list(weekly.OUTPUT_COLUMNS),
        )

        file_week, file_lookup = validate_weekly_file(
            path=path,
            rows=rows,
            season=season,
            schedule_rows=schedule_rows,
            schedule_index=schedule_index,
            weekly=weekly,
        )

        if file_week is None:
            continue

        reporter.add_input(path)
        metrics["target_season_weekly_files"] += 1

        for game_id, row in file_lookup.items():
            if game_id in lookup:
                fail(
                    "Target-season weekly odds contain duplicate "
                    f"game_id across files: {game_id}"
                )

            lookup[game_id] = row

            if clean(row.get("odds_available")) == "0":
                metrics["games_odds_unavailable"] += 1
            elif clean(row.get("total")):
                metrics["games_with_valid_total"] += 1
            else:
                metrics[
                    "games_available_without_total"
                ] += 1

    return lookup, metrics


def finalize_group(
    *,
    key: tuple[str, str, str],
    clean_rows: list[dict[str, str]],
    schedule_index: dict[str, dict[str, str]],
    weekly_lookup: dict[str, dict[str, str]],
    log_lines: list[str],
) -> tuple[
    list[dict[str, str]],
    dict[str, int],
]:
    metrics = {
        "rows_finalized": 0,
        "rows_without_weekly_file": 0,
        "rows_odds_unavailable": 0,
        "rows_available_without_total": 0,
        "rows_with_total": 0,
        "projection_check_failures": 0,
    }

    output_rows: list[dict[str, str]] = []

    for line_number, source in enumerate(
        clean_rows,
        start=2,
    ):
        rec = {
            header: clean(source.get(header))
            for header in OUT_HEADERS
        }
        game_id = rec["game_id"]

        if not game_id:
            fail(
                f"Clean input target={key} line={line_number} "
                "has blank game_id"
            )

        schedule_row = schedule_index.get(game_id)
        if schedule_row is None:
            fail(
                f"Clean input game_id={game_id} is absent "
                "from season schedule"
            )

        source_target = (
            rec["season"],
            rec["season_type"],
            rec["week"],
        )
        schedule_target = (
            clean(schedule_row.get("season")),
            clean(schedule_row.get("season_type")),
            clean(schedule_row.get("week")),
        )

        if source_target != key:
            fail(
                f"Clean input game_id={game_id} target="
                f"{source_target}; expected={key}"
            )

        if source_target != schedule_target:
            fail(
                f"Clean input game_id={game_id} target "
                "does not match season schedule"
            )

        rec["game_date"] = clean(
            schedule_row.get("game_date")
        )
        rec["game_time"] = clean(
            schedule_row.get("game_time")
        )

        if not rec["game_date"] or not rec["game_time"]:
            fail(
                f"Season schedule game_id={game_id} has "
                "blank game_date/game_time"
            )

        home_pt_diff = finite_decimal(
            rec["home_PtDiff"],
            label=(
                f"game_id={game_id} home_PtDiff"
            ),
        )
        away_pt_diff = finite_decimal(
            rec["away_PtDiff"],
            label=(
                f"game_id={game_id} away_PtDiff"
            ),
        )

        weekly_row = weekly_lookup.get(game_id)

        if weekly_row is None:
            rec["total_projected_pts"] = ""
            rec["home_projected_pts"] = ""
            rec["away_projected_pts"] = ""
            metrics[
                "rows_without_weekly_file"
            ] += 1
            log_lines.append(
                "NO WEEKLY ODDS FILE "
                f"game_id={game_id} target={key} "
                "(projected pts blank)"
            )

        elif clean(
            weekly_row.get("odds_available")
        ) == "0":
            rec["total_projected_pts"] = ""
            rec["home_projected_pts"] = ""
            rec["away_projected_pts"] = ""
            metrics["rows_odds_unavailable"] += 1
            log_lines.append(
                "NO TOTAL MATCH "
                f"game_id={game_id} target={key} "
                f"reason={clean(weekly_row.get('odds_missing_reason'))} "
                "(projected pts blank)"
            )

        else:
            total = optional_decimal(
                weekly_row.get("total"),
                label=(
                    f"game_id={game_id} weekly total"
                ),
            )

            if total is None:
                rec["total_projected_pts"] = ""
                rec["home_projected_pts"] = ""
                rec["away_projected_pts"] = ""
                metrics[
                    "rows_available_without_total"
                ] += 1
                log_lines.append(
                    "NO TOTAL MARKET "
                    f"game_id={game_id} target={key} "
                    "(projected pts blank)"
                )

            else:
                half = total / Decimal(2)
                home_projected = (
                    half + home_pt_diff
                )
                away_projected = (
                    half + away_pt_diff
                )

                rec["total_projected_pts"] = fmt2(
                    total
                )
                rec["home_projected_pts"] = fmt2(
                    home_projected
                )
                rec["away_projected_pts"] = fmt2(
                    away_projected
                )

                diff = abs(
                    home_projected
                    + away_projected
                    - total
                )

                if diff > Decimal("1.0"):
                    metrics[
                        "projection_check_failures"
                    ] += 1
                    fail(
                        "Projected-score consistency check "
                        f"failed game_id={game_id} "
                        f"home+away={fmt2(home_projected + away_projected)} "
                        f"total={fmt2(total)} diff={fmt2(diff)}"
                    )

                metrics["rows_with_total"] += 1

        output_rows.append(rec)
        metrics["rows_finalized"] += 1

    return output_rows, metrics


def validate_final_rows(
    rows: list[dict[str, str]],
    *,
    clean_rows: list[dict[str, str]],
    key: tuple[str, str, str],
    schedule_index: dict[str, dict[str, str]],
    weekly_lookup: dict[str, dict[str, str]],
    path: Path,
) -> None:
    if len(rows) != len(clean_rows):
        fail(
            f"{path} row count mismatch "
            f"expected={len(clean_rows)} actual={len(rows)}"
        )

    clean_by_id = {
        clean(row.get("game_id")): row
        for row in clean_rows
    }
    seen_ids: set[str] = set()

    for line_number, row in enumerate(
        rows,
        start=2,
    ):
        game_id = clean(row.get("game_id"))

        if not game_id:
            fail(
                f"{path} line {line_number} has blank game_id"
            )

        if game_id in seen_ids:
            fail(
                f"{path} contains duplicate game_id={game_id}"
            )
        seen_ids.add(game_id)

        source = clean_by_id.get(game_id)
        if source is None:
            fail(
                f"{path} contains unexpected game_id={game_id}"
            )

        actual_target = (
            clean(row.get("season")),
            clean(row.get("season_type")),
            clean(row.get("week")),
        )
        if actual_target != key:
            fail(
                f"{path} game_id={game_id} target="
                f"{actual_target}; expected={key}"
            )

        for field in OUT_HEADERS:
            if field in FINALIZED_FIELDS:
                continue

            if clean(row.get(field)) != clean(
                source.get(field)
            ):
                fail(
                    f"{path} game_id={game_id} changed "
                    f"clean-source field={field}"
                )

        schedule_row = schedule_index.get(game_id)
        if schedule_row is None:
            fail(
                f"{path} game_id={game_id} absent from "
                "season schedule"
            )

        expected_date = clean(
            schedule_row.get("game_date")
        )
        expected_time = clean(
            schedule_row.get("game_time")
        )

        if (
            clean(row.get("game_date"))
            != expected_date
            or clean(row.get("game_time"))
            != expected_time
        ):
            fail(
                f"{path} game_id={game_id} date/time "
                "does not match season schedule"
            )

        weekly_row = weekly_lookup.get(game_id)

        should_have_total = (
            weekly_row is not None
            and clean(
                weekly_row.get("odds_available")
            ) == "1"
            and bool(
                clean(weekly_row.get("total"))
            )
        )

        if not should_have_total:
            for field in (
                "total_projected_pts",
                "home_projected_pts",
                "away_projected_pts",
            ):
                if clean(row.get(field)):
                    fail(
                        f"{path} game_id={game_id} field="
                        f"{field} must be blank without "
                        "a valid weekly total"
                    )
            continue

        total = finite_decimal(
            weekly_row.get("total"),
            label=(
                f"{path} game_id={game_id} weekly total"
            ),
        )
        home_pt_diff = finite_decimal(
            source.get("home_PtDiff"),
            label=(
                f"{path} game_id={game_id} home_PtDiff"
            ),
        )
        away_pt_diff = finite_decimal(
            source.get("away_PtDiff"),
            label=(
                f"{path} game_id={game_id} away_PtDiff"
            ),
        )

        half = total / Decimal(2)
        expected_total = fmt2(total)
        expected_home = fmt2(
            half + home_pt_diff
        )
        expected_away = fmt2(
            half + away_pt_diff
        )

        actual_projected = (
            clean(row.get("total_projected_pts")),
            clean(row.get("home_projected_pts")),
            clean(row.get("away_projected_pts")),
        )
        expected_projected = (
            expected_total,
            expected_home,
            expected_away,
        )

        if actual_projected != expected_projected:
            fail(
                f"{path} game_id={game_id} projected "
                f"points mismatch expected={expected_projected} "
                f"actual={actual_projected}"
            )

        diff = abs(
            (half + home_pt_diff)
            + (half + away_pt_diff)
            - total
        )
        if diff > Decimal("1.0"):
            fail(
                f"{path} game_id={game_id} projected-score "
                f"consistency diff={fmt2(diff)} exceeds 1.00"
            )

    if seen_ids != set(clean_by_id):
        fail(
            f"{path} final/clean game universe mismatch "
            f"missing={sorted(set(clean_by_id) - seen_ids)[:10]} "
            f"extra={sorted(seen_ids - set(clean_by_id))[:10]}"
        )


def normalize_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            header: clean(row.get(header))
            for header in OUT_HEADERS
        }
        for row in rows
    ]


def write_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUT_HEADERS,
        )
        writer.writeheader()
        writer.writerows(
            normalize_rows(rows)
        )
        handle.flush()
        os.fsync(handle.fileno())


def build_staged_root(
    *,
    season: int,
    final_rows_by_group: dict[
        tuple[str, str, str],
        list[dict[str, str]],
    ],
    clean_rows_by_group: dict[
        tuple[str, str, str],
        list[dict[str, str]],
    ],
    final_paths: dict[
        tuple[str, str, str],
        Path,
    ],
    schedule_index: dict[str, dict[str, str]],
    weekly_lookup: dict[str, dict[str, str]],
) -> Path:
    OUT_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stage_root = Path(
        tempfile.mkdtemp(
            prefix=".final_predictions_stage_",
            dir=OUT_DIR.parent,
        )
    )

    try:
        if OUT_DIR.exists():
            shutil.copytree(
                OUT_DIR,
                stage_root,
                dirs_exist_ok=True,
            )

        for stale in stage_root.glob(
            f"{season}_*_clean_predictions.csv"
        ):
            stale.unlink()

        for key, production_path in sorted(
            final_paths.items(),
            key=lambda item: (
                item[0][0],
                item[0][1],
                int(item[0][2]),
            ),
        ):
            staged_path = (
                stage_root
                / production_path.name
            )
            rows = final_rows_by_group[key]

            write_csv(
                staged_path,
                rows,
            )

            staged_rows = read_csv(
                staged_path,
                label="staged final ESPN prediction output",
                exact_columns=OUT_HEADERS,
            )

            validate_final_rows(
                staged_rows,
                clean_rows=clean_rows_by_group[key],
                key=key,
                schedule_index=schedule_index,
                weekly_lookup=weekly_lookup,
                path=staged_path,
            )

            if staged_rows != normalize_rows(rows):
                fail(
                    "Staged final prediction file differs from "
                    f"validated in-memory generation: {staged_path}"
                )

        expected_names = {
            path.name
            for path in final_paths.values()
        }
        actual_names = {
            path.name
            for path in stage_root.glob(
                f"{season}_*_clean_predictions.csv"
            )
        }

        if actual_names != expected_names:
            fail(
                "Staged final prediction file set mismatch "
                f"expected={sorted(expected_names)} "
                f"actual={sorted(actual_names)}"
            )

        return stage_root

    except Exception:
        shutil.rmtree(
            stage_root,
            ignore_errors=True,
        )
        raise


def publish_staged_root(
    stage_root: Path,
    *,
    reporter: PipelineReporter,
) -> None:
    backup_root = (
        OUT_DIR.parent
        / (
            f".{OUT_DIR.name}_backup_"
            f"{uuid.uuid4().hex}"
        )
    )

    try:
        if OUT_DIR.exists():
            os.replace(
                OUT_DIR,
                backup_root,
            )

        os.replace(
            stage_root,
            OUT_DIR,
        )

    except Exception:
        if OUT_DIR.exists():
            shutil.rmtree(
                OUT_DIR,
                ignore_errors=True,
            )

        if backup_root.exists():
            os.replace(
                backup_root,
                OUT_DIR,
            )

        raise

    if backup_root.exists():
        try:
            shutil.rmtree(backup_root)
        except Exception as exc:
            reporter.warning(
                "Final predictions were published but the "
                "temporary backup directory could not be removed",
                backup_path=str(backup_root),
                error_type=type(exc).__name__,
                error=str(exc),
            )


def write_legacy_log(
    log_lines: list[str],
) -> None:
    LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with LOG_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "\n".join(log_lines) + "\n"
        )


def safe_write_legacy_log(
    log_lines: list[str],
    *,
    reporter: PipelineReporter,
) -> None:
    try:
        write_legacy_log(log_lines)
        reporter.add_output(LOG_PATH)
        reporter.set_detail(
            "legacy_log_written",
            True,
        )
    except Exception as exc:
        reporter.warning(
            "Legacy finalize_pred text log could not be written",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        reporter.set_detail(
            "legacy_log_written",
            False,
        )


def run(
    reporter: PipelineReporter,
    *,
    season: int,
    log_lines: list[str],
) -> None:
    cleaner = load_module(
        path=CLEANER_PATH,
        module_name="_finalize_pred_cleaner_contract",
    )
    weekly = load_module(
        path=WEEKLY_BUILDER_PATH,
        module_name="_finalize_pred_weekly_contract",
    )

    if list(cleaner.OUT_HEADERS) != OUT_HEADERS:
        fail(
            "Cleaner/finalizer 22-column schemas differ"
        )

    (
        schedule_rows,
        clean_rows_by_group,
        clean_paths,
    ) = load_and_validate_clean_generation(
        season=season,
        cleaner=cleaner,
        reporter=reporter,
    )

    schedule_index = build_schedule_index(
        schedule_rows,
        season=season,
    )

    weekly_lookup, weekly_metrics = (
        load_weekly_odds(
            season=season,
            schedule_rows=schedule_rows,
            schedule_index=schedule_index,
            weekly=weekly,
            reporter=reporter,
        )
    )

    final_paths = expected_final_paths(
        clean_paths
    )
    final_rows_by_group: dict[
        tuple[str, str, str],
        list[dict[str, str]],
    ] = {}

    aggregate_metrics = {
        "rows_finalized": 0,
        "rows_without_weekly_file": 0,
        "rows_odds_unavailable": 0,
        "rows_available_without_total": 0,
        "rows_with_total": 0,
        "projection_check_failures": 0,
    }

    for key, clean_rows in sorted(
        clean_rows_by_group.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            int(item[0][2]),
        ),
    ):
        rows, metrics = finalize_group(
            key=key,
            clean_rows=clean_rows,
            schedule_index=schedule_index,
            weekly_lookup=weekly_lookup,
            log_lines=log_lines,
        )

        validate_final_rows(
            rows,
            clean_rows=clean_rows,
            key=key,
            schedule_index=schedule_index,
            weekly_lookup=weekly_lookup,
            path=final_paths[key],
        )

        final_rows_by_group[key] = rows

        for metric, value in metrics.items():
            aggregate_metrics[metric] += value

    source_rows_total = sum(
        len(rows)
        for rows in clean_rows_by_group.values()
    )
    final_rows_total = sum(
        len(rows)
        for rows in final_rows_by_group.values()
    )

    if final_rows_total != len(schedule_rows):
        fail(
            "Final prediction row count does not match "
            f"season schedule games: final={final_rows_total} "
            f"schedule={len(schedule_rows)}"
        )

    reporter.set_rows(
        rows_in=source_rows_total,
        rows_out=0,
    )
    reporter.update_details(
        {
            "season": season,
            "refresh_scope": "full season",
            "schedule_games": len(schedule_rows),
            "clean_files": len(clean_paths),
            "clean_rows": source_rows_total,
            "final_files": len(final_paths),
            "expected_final_rows": final_rows_total,
            "weekly_files_scanned": weekly_metrics[
                "weekly_files_scanned"
            ],
            "target_season_weekly_files": weekly_metrics[
                "target_season_weekly_files"
            ],
            "games_with_valid_total": weekly_metrics[
                "games_with_valid_total"
            ],
            "games_available_without_total": weekly_metrics[
                "games_available_without_total"
            ],
            "games_odds_unavailable": weekly_metrics[
                "games_odds_unavailable"
            ],
            **aggregate_metrics,
            "rounding": (
                "Decimal.quantize(0.01)_default_half_even"
            ),
            "projection_consistency_tolerance": "1.0",
            "publication_mode": (
                "validated_directory_swap_with_rollback"
            ),
            "publication_completed": False,
            "staged_roundtrip_verified": False,
        }
    )

    if (
        weekly_metrics[
            "games_available_without_total"
        ]
        > 0
    ):
        reporter.warning(
            "One or more games have an available odds event "
            "but no totals market; projected score fields "
            "remain blank for those games",
            count=weekly_metrics[
                "games_available_without_total"
            ],
        )

    stage_root: Path | None = None

    try:
        stage_root = build_staged_root(
            season=season,
            final_rows_by_group=final_rows_by_group,
            clean_rows_by_group=clean_rows_by_group,
            final_paths=final_paths,
            schedule_index=schedule_index,
            weekly_lookup=weekly_lookup,
        )

        reporter.set_detail(
            "staged_roundtrip_verified",
            True,
        )

        publish_staged_root(
            stage_root,
            reporter=reporter,
        )
        stage_root = None

    finally:
        if (
            stage_root is not None
            and stage_root.exists()
        ):
            shutil.rmtree(
                stage_root,
                ignore_errors=True,
            )

    for key, path in sorted(
        final_paths.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            int(item[0][2]),
        ),
    ):
        reporter.add_output(path)
        log_lines.append(
            f"wrote {len(final_rows_by_group[key])} "
            f"rows to {path}"
        )

    reporter.set_rows(
        rows_in=source_rows_total,
        rows_out=final_rows_total,
    )
    reporter.update_details(
        {
            "rows_published": final_rows_total,
            "files_published": len(final_paths),
            "publication_completed": True,
        }
    )

    no_total_match = (
        aggregate_metrics[
            "rows_without_weekly_file"
        ]
        + aggregate_metrics[
            "rows_odds_unavailable"
        ]
        + aggregate_metrics[
            "rows_available_without_total"
        ]
    )

    summary = (
        f"SUMMARY: pred_files={len(clean_paths)} "
        f"files_written={len(final_paths)} "
        f"rows_written={final_rows_total} "
        "no_schedule_match=0 "
        f"no_total_match={no_total_match} "
        "check_failures=0"
    )
    print(summary)
    log_lines.append(summary)


def main() -> int:
    args = parse_args()
    log_lines: list[str] = []

    try:
        with PipelineReporter(
            script=SCRIPT_PATH,
            stage="00_intake",
            report_root=REPORT_ROOT,
            pipeline="NFL",
            league="NFL",
            season=args.season,
            extra_context={
                "component": "ESPN prediction finalizer",
                "refresh_scope": "full season",
            },
        ) as reporter:
            try:
                run(
                    reporter,
                    season=args.season,
                    log_lines=log_lines,
                )
            except Exception as exc:
                log_lines.append(
                    f"FATAL: {type(exc).__name__}: {exc}"
                )
                safe_write_legacy_log(
                    log_lines,
                    reporter=reporter,
                )
                raise

            safe_write_legacy_log(
                log_lines,
                reporter=reporter,
            )

        return 0

    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
