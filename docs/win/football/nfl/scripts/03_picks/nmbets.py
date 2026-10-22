#!/usr/bin/env python3
"""Create simplified NM NFL weekly picks files from root weekly picks outputs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
SCRIPTS_DIR = SCRIPT_DIR.parent
NFL_ROOT = SCRIPT_DIR.parents[1]
REPORT_ROOT = NFL_ROOT / "errors"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter


PICKS_DIR = NFL_ROOT / "03_picks"
OUTPUT_DIR = PICKS_DIR / "nmbets"
INPUT_PATTERN = "week_*_NFL_picks.csv"
INPUT_FILENAME_PATTERN = re.compile(r"^week_(\d+)_NFL_picks\.csv$")
OUTPUT_FILENAME_PATTERN = re.compile(r"^week_(\d+)_NM_NFL_picks\.csv$")
TIME_PATTERN = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
EPSILON = 1e-9

OUTPUT_FIELDS = [
    "Date",
    "Time",
    "Away_Team",
    "Home_Team",
    "Projected_Score",
    "Predicted_Margin",
    "Predicted_Total",
]

REQUIRED_FIELDS = [
    "season",
    "week",
    "game_id",
    "game_date",
    "game_time",
    "away_team",
    "home_team",
    "predicted_margin",
    "predicted_total",
    "predicted_home_score",
    "predicted_away_score",
]

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


def fail(message: str) -> None:
    raise RuntimeError(message)


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"}:
        return ""
    return text


def parse_args() -> argparse.Namespace:
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


def team_score_label(team_name: Any) -> str:
    team_name = clean(team_name)

    if team_name in TEAM_SCORE_LABELS:
        return TEAM_SCORE_LABELS[team_name]

    parts = team_name.rsplit(" ", 1)
    if len(parts) == 2:
        return parts[0]

    return team_name


def parse_number(
    value: Any,
    field_name: str,
    source_file: Path,
    row_number: int,
) -> float:
    text = clean(value)

    if not text:
        fail(
            f"{source_file}: row {row_number}: "
            f"missing required numeric value for {field_name}"
        )

    try:
        number = float(text)
    except (TypeError, ValueError):
        fail(
            f"{source_file}: row {row_number}: "
            f"invalid numeric value for {field_name}: {value!r}"
        )

    if not math.isfinite(number):
        fail(
            f"{source_file}: row {row_number}: "
            f"non-finite numeric value for {field_name}: {value!r}"
        )

    return number


def parse_positive_int(
    value: Any,
    field_name: str,
    source_file: Path,
    row_number: int,
) -> int:
    number = parse_number(
        value,
        field_name,
        source_file,
        row_number,
    )
    if not number.is_integer() or number <= 0:
        fail(
            f"{source_file}: row {row_number}: "
            f"{field_name} must be a positive integer; found {value!r}"
        )
    return int(number)


def format_number(value: float) -> str:
    if abs(value) < 0.005:
        value = 0.0

    return f"{value:.2f}"


def format_margin(value: float) -> str:
    if abs(value) < 0.005:
        value = 0.0

    return f"{value:+.2f} Home"


def week_from_filename(path: Path) -> int:
    match = INPUT_FILENAME_PATTERN.fullmatch(path.name)

    if not match:
        fail(f"Unable to determine week from filename: {path}")

    week = int(match.group(1))
    if week <= 0:
        fail(f"Input filename week must be greater than 0: {path}")

    return week


def output_week_from_filename(path: Path) -> int:
    match = OUTPUT_FILENAME_PATTERN.fullmatch(path.name)

    if not match:
        fail(f"Invalid managed NM output filename: {path}")

    week = int(match.group(1))
    if week <= 0:
        fail(f"NM output filename week must be greater than 0: {path}")

    return week


def validate_headers(fieldnames: list[str] | None, source_file: Path) -> None:
    if fieldnames is None:
        fail(f"{source_file}: missing CSV header")

    normalized = [clean(field) for field in fieldnames]
    if any(not field for field in normalized):
        fail(f"{source_file}: blank CSV column name found")

    duplicates = sorted(
        {
            field
            for field in normalized
            if normalized.count(field) > 1
        }
    )
    if duplicates:
        fail(f"{source_file}: duplicate CSV columns: {duplicates}")

    missing = [
        field
        for field in REQUIRED_FIELDS
        if field not in normalized
    ]
    if missing:
        fail(
            f"{source_file}: missing required columns: "
            f"{missing}"
        )


def read_source(source_file: Path) -> list[dict[str, str]]:
    if not source_file.is_file():
        fail(f"NM source file not found: {source_file}")

    try:
        with source_file.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as infile:
            reader = csv.DictReader(infile)
            validate_headers(reader.fieldnames, source_file)
            rows = list(reader)
    except UnicodeDecodeError as exc:
        fail(f"{source_file}: invalid UTF-8 CSV: {exc}")
    except csv.Error as exc:
        fail(f"{source_file}: CSV read failed: {exc}")

    if not rows:
        fail(f"{source_file}: contains no rows")

    return rows


def validate_game_date(
    value: Any,
    source_file: Path,
    row_number: int,
) -> str:
    text = clean(value)
    if not DATE_PATTERN.fullmatch(text):
        fail(
            f"{source_file}: row {row_number}: "
            f"game_date must use YYYY-MM-DD; found {value!r}"
        )
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        fail(
            f"{source_file}: row {row_number}: "
            f"invalid game_date: {value!r}"
        )
    if parsed.isoformat() != text:
        fail(
            f"{source_file}: row {row_number}: "
            f"game_date must use YYYY-MM-DD; found {value!r}"
        )
    return text


def validate_game_time(
    value: Any,
    source_file: Path,
    row_number: int,
) -> str:
    text = clean(value)
    if not TIME_PATTERN.fullmatch(text):
        fail(
            f"{source_file}: row {row_number}: "
            f"game_time must use HH:MM or HH:MM:SS; found {value!r}"
        )
    return text


def validate_source_rows(
    rows: list[dict[str, str]],
    source_file: Path,
    filename_week: int,
) -> dict[str, Any]:
    game_ids: set[str] = set()
    seasons: set[int] = set()

    for row_number, row in enumerate(rows, start=2):
        season = parse_positive_int(
            row.get("season"),
            "season",
            source_file,
            row_number,
        )
        row_week = parse_positive_int(
            row.get("week"),
            "week",
            source_file,
            row_number,
        )
        seasons.add(season)

        if row_week != filename_week:
            fail(
                f"{source_file}: row {row_number}: "
                f"week={row_week} does not match filename week={filename_week}"
            )

        game_id = clean(row.get("game_id"))
        if not game_id:
            fail(f"{source_file}: row {row_number}: missing game_id")
        if game_id in game_ids:
            fail(
                f"{source_file}: duplicate game_id {game_id!r}"
            )
        game_ids.add(game_id)

        away_team = clean(row.get("away_team"))
        home_team = clean(row.get("home_team"))
        if not away_team:
            fail(f"{source_file}: row {row_number}: missing away_team")
        if not home_team:
            fail(f"{source_file}: row {row_number}: missing home_team")
        if away_team == home_team:
            fail(
                f"{source_file}: row {row_number}: "
                "away_team and home_team must differ"
            )

        validate_game_date(
            row.get("game_date"),
            source_file,
            row_number,
        )
        validate_game_time(
            row.get("game_time"),
            source_file,
            row_number,
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

        expected_total = home_score + away_score
        expected_margin = home_score - away_score

        if not math.isclose(
            predicted_total,
            expected_total,
            rel_tol=0.0,
            abs_tol=EPSILON,
        ):
            fail(
                f"{source_file}: row {row_number}: predicted_total "
                "does not equal predicted_home_score + predicted_away_score"
            )

        if not math.isclose(
            predicted_margin,
            expected_margin,
            rel_tol=0.0,
            abs_tol=EPSILON,
        ):
            fail(
                f"{source_file}: row {row_number}: predicted_margin "
                "does not equal predicted_home_score - predicted_away_score"
            )

    return {
        "week": filename_week,
        "seasons": sorted(seasons),
        "rows": len(rows),
    }


def build_output_row(
    row: dict[str, str],
    source_file: Path,
    row_number: int,
) -> dict[str, str]:
    away_team = clean(row.get("away_team"))
    home_team = clean(row.get("home_team"))

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
        f"\u2013 {away_label} {format_number(away_score)}"
    )

    return {
        "Date": clean(row.get("game_date")),
        "Time": clean(row.get("game_time")),
        "Away_Team": away_team,
        "Home_Team": home_team,
        "Projected_Score": projected_score,
        "Predicted_Margin": format_margin(predicted_margin),
        "Predicted_Total": format_number(predicted_total),
    }


def build_output_rows(
    rows: list[dict[str, str]],
    source_file: Path,
) -> list[dict[str, str]]:
    return [
        build_output_row(
            row,
            source_file,
            row_number,
        )
        for row_number, row in enumerate(rows, start=2)
    ]


def validate_output_rows(
    output_rows: list[dict[str, str]],
    source_rows: list[dict[str, str]],
    source_file: Path,
    *,
    label: str,
) -> None:
    if len(output_rows) != len(source_rows):
        fail(
            f"{label}: output row count {len(output_rows)} "
            f"does not match source row count {len(source_rows)}"
        )

    for index, (output_row, source_row) in enumerate(
        zip(output_rows, source_rows, strict=True),
        start=2,
    ):
        if list(output_row) != OUTPUT_FIELDS:
            fail(f"{label}: row {index}: output column contract failed")

        expected = build_output_row(
            source_row,
            source_file,
            index,
        )
        if output_row != expected:
            mismatches = [
                field
                for field in OUTPUT_FIELDS
                if clean(output_row.get(field))
                != clean(expected.get(field))
            ]
            fail(
                f"{label}: row {index}: output differs from source "
                f"calculation for fields={mismatches}"
            )


def write_staged_output(
    output_rows: list[dict[str, str]],
    stage_path: Path,
) -> None:
    stage_path.parent.mkdir(parents=True, exist_ok=True)

    with stage_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=OUTPUT_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)
        outfile.flush()
        os.fsync(outfile.fileno())


def read_nm_output(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        fail(f"NM output not found: {path}")

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as infile:
            reader = csv.DictReader(infile)
            if reader.fieldnames is None:
                fail(f"{path}: missing CSV header")
            if list(reader.fieldnames) != OUTPUT_FIELDS:
                fail(f"{path}: NM output column contract failed")
            rows = list(reader)
    except UnicodeDecodeError as exc:
        fail(f"{path}: invalid UTF-8 CSV: {exc}")
    except csv.Error as exc:
        fail(f"{path}: CSV read failed: {exc}")

    return rows


def validate_serialized_output(
    path: Path,
    source_rows: list[dict[str, str]],
    source_file: Path,
    *,
    label: str,
) -> list[dict[str, str]]:
    rows = read_nm_output(path)
    validate_output_rows(
        rows,
        source_rows,
        source_file,
        label=label,
    )
    return rows


def discover_source_files(
    picks_dir: Path,
    requested_week: int | None,
) -> list[tuple[int, Path]]:
    if not picks_dir.is_dir():
        fail(f"Picks directory does not exist: {picks_dir}")

    candidates = sorted(
        path
        for path in picks_dir.glob(INPUT_PATTERN)
        if path.is_file()
    )

    if not candidates:
        fail(f"No input files found matching {picks_dir / INPUT_PATTERN}")

    discovered: list[tuple[int, Path]] = []
    weeks: set[int] = set()

    for path in candidates:
        week = week_from_filename(path)
        if week in weeks:
            fail(f"Duplicate managed source week discovered: week={week}")
        weeks.add(week)
        discovered.append((week, path.resolve()))

    discovered.sort(key=lambda item: item[0])

    if requested_week is None:
        return discovered

    selected = [
        item
        for item in discovered
        if item[0] == requested_week
    ]
    if not selected:
        fail(
            f"No input file found for week {requested_week}: "
            f"{picks_dir / f'week_{requested_week}_NFL_picks.csv'}"
        )

    return selected


def discover_managed_outputs(output_dir: Path) -> dict[int, Path]:
    if not output_dir.exists():
        return {}

    managed: dict[int, Path] = {}
    for path in sorted(output_dir.glob("week_*_NM_NFL_picks.csv")):
        if not path.is_file():
            continue
        week = output_week_from_filename(path)
        if week in managed:
            fail(f"Duplicate managed NM output week discovered: week={week}")
        managed[week] = path.resolve()
    return managed


def stage_output_set(
    entries: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_root = Path(
        tempfile.mkdtemp(
            prefix=".nmbets_stage_",
            dir=str(output_dir),
        )
    )

    try:
        for entry in entries:
            stage_path = stage_root / entry["output_path"].name
            write_staged_output(
                entry["output_rows"],
                stage_path,
            )
            validate_serialized_output(
                stage_path,
                entry["source_rows"],
                entry["source_path"],
                label=f"staged NM output {entry['output_path']}",
            )
            entry["staged_path"] = stage_path
        return stage_root
    except Exception:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise


def validate_published_set(
    entries: list[dict[str, Any]],
    stale_paths: list[Path],
    *,
    sync_full_set: bool,
    output_dir: Path,
) -> None:
    for entry in entries:
        validate_serialized_output(
            entry["output_path"],
            entry["source_rows"],
            entry["source_path"],
            label=f"published NM output {entry['output_path']}",
        )

    remaining_stale = [
        str(path)
        for path in stale_paths
        if path.exists()
    ]
    if remaining_stale:
        fail(
            "Stale managed NM outputs remain after publication: "
            f"{remaining_stale}"
        )

    if sync_full_set:
        expected = {
            entry["output_path"].resolve()
            for entry in entries
        }
        actual = set(discover_managed_outputs(output_dir).values())
        if actual != expected:
            missing = sorted(str(path) for path in expected - actual)
            extra = sorted(str(path) for path in actual - expected)
            fail(
                "Published NM output set mismatch: "
                f"missing={missing} extra={extra}"
            )


def publish_output_set(
    entries: list[dict[str, Any]],
    stale_paths: list[Path],
    stage_root: Path,
    *,
    sync_full_set: bool,
    output_dir: Path,
    reporter: PipelineReporter,
) -> None:
    managed_paths = [
        entry["output_path"]
        for entry in entries
    ] + list(stale_paths)

    backup_root = Path(
        tempfile.mkdtemp(
            prefix=".nmbets_backup_",
            dir=str(output_dir),
        )
    )
    backups: dict[Path, Path] = {}
    publication_started = False
    rollback_failed = False

    reporter.update_details(
        {
            "publication_mode": "transactional_multi_file_atomic_replace_with_rollback",
            "publication_completed": False,
            "post_publish_validation": False,
            "rollback_performed": False,
        }
    )

    try:
        for live_path in managed_paths:
            if not live_path.exists():
                continue
            backup_path = backup_root / live_path.name
            shutil.copy2(live_path, backup_path)
            backups[live_path] = backup_path

        publication_started = True

        for entry in entries:
            os.replace(
                entry["staged_path"],
                entry["output_path"],
            )

        for stale_path in stale_paths:
            stale_path.unlink(missing_ok=True)

        validate_published_set(
            entries,
            stale_paths,
            sync_full_set=sync_full_set,
            output_dir=output_dir,
        )

        reporter.update_details(
            {
                "publication_completed": True,
                "post_publish_validation": True,
            }
        )
    except Exception as publish_exc:
        if publication_started:
            try:
                for live_path in managed_paths:
                    backup_path = backups.get(live_path)
                    if (
                        backup_path is not None
                        and backup_path.exists()
                    ):
                        restore_fd, restore_raw = tempfile.mkstemp(
                            prefix=f".{live_path.name}.restore.",
                            suffix=".tmp",
                            dir=str(output_dir),
                        )
                        os.close(restore_fd)
                        restore_path = Path(restore_raw)
                        try:
                            shutil.copy2(
                                backup_path,
                                restore_path,
                            )
                            os.replace(
                                restore_path,
                                live_path,
                            )
                        finally:
                            restore_path.unlink(missing_ok=True)
                    else:
                        live_path.unlink(missing_ok=True)

                reporter.update_details(
                    {
                        "publication_completed": False,
                        "post_publish_validation": False,
                        "rollback_performed": True,
                    }
                )
            except Exception as rollback_exc:
                rollback_failed = True
                reporter.update_details(
                    {
                        "publication_completed": False,
                        "post_publish_validation": False,
                        "rollback_performed": False,
                        "rollback_error_type": type(rollback_exc).__name__,
                        "rollback_error": str(rollback_exc),
                    }
                )
                raise RuntimeError(
                    "NM publication failed and rollback also failed: "
                    f"publication_error={publish_exc}; "
                    f"rollback_error={rollback_exc}"
                ) from rollback_exc
        raise
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)
        if not rollback_failed:
            try:
                shutil.rmtree(backup_root, ignore_errors=False)
            except Exception as cleanup_exc:
                reporter.warning(
                    "Temporary NM backup cleanup failed",
                    backup_root=str(backup_root),
                    error_type=type(cleanup_exc).__name__,
                    error=str(cleanup_exc),
                )


def run(
    args: argparse.Namespace,
    reporter: PipelineReporter,
) -> None:
    requested_week = args.week
    if requested_week is not None and requested_week <= 0:
        fail("--week must be greater than 0")

    picks_dir = PICKS_DIR.resolve()
    output_dir = OUTPUT_DIR.resolve()
    sync_full_set = requested_week is None

    source_files = discover_source_files(
        picks_dir,
        requested_week,
    )

    existing_outputs = discover_managed_outputs(output_dir)

    entries: list[dict[str, Any]] = []
    total_rows = 0
    per_file: list[dict[str, Any]] = []

    reporter.update_details(
        {
            "requested_week": requested_week,
            "managed_set_sync": sync_full_set,
            "all_outputs_staged": False,
            "staged_roundtrip_verified": False,
            "publication_completed": False,
            "post_publish_validation": False,
            "rollback_performed": False,
        }
    )

    for week, source_path in source_files:
        reporter.add_input(source_path)

        source_rows = read_source(source_path)
        source_meta = validate_source_rows(
            source_rows,
            source_path,
            week,
        )
        output_rows = build_output_rows(
            source_rows,
            source_path,
        )
        validate_output_rows(
            output_rows,
            source_rows,
            source_path,
            label=f"in-memory NM output week={week}",
        )

        output_path = (
            output_dir
            / f"week_{week}_NM_NFL_picks.csv"
        ).resolve()
        reporter.add_output(output_path)

        total_rows += len(source_rows)
        entries.append(
            {
                "week": week,
                "source_path": source_path,
                "source_rows": source_rows,
                "source_meta": source_meta,
                "output_rows": output_rows,
                "output_path": output_path,
            }
        )
        per_file.append(
            {
                "week": week,
                "input": str(source_path),
                "output": str(output_path),
                "rows": len(source_rows),
                "seasons": source_meta["seasons"],
            }
        )

    expected_weeks = {
        entry["week"]
        for entry in entries
    }
    stale_paths: list[Path] = []
    if sync_full_set:
        stale_paths = sorted(
            path
            for week, path in existing_outputs.items()
            if week not in expected_weeks
        )

    reporter.update_details(
        {
            "files_processed": len(entries),
            "processed_weeks": sorted(expected_weeks),
            "per_file": per_file,
            "stale_managed_outputs": [
                str(path)
                for path in stale_paths
            ],
            "stale_managed_output_count": len(stale_paths),
        }
    )
    reporter.set_rows(
        rows_in=total_rows,
        rows_out=total_rows,
    )

    stage_root = stage_output_set(
        entries,
        output_dir,
    )
    reporter.update_details(
        {
            "all_outputs_staged": True,
            "staged_roundtrip_verified": True,
        }
    )

    publish_output_set(
        entries,
        stale_paths,
        stage_root,
        sync_full_set=sync_full_set,
        output_dir=output_dir,
        reporter=reporter,
    )

    for entry in entries:
        print(
            f"week={entry['week']} "
            f"rows={len(entry['output_rows'])} "
            f"input={entry['source_path']} "
            f"output={entry['output_path']}"
        )


def main() -> int:
    try:
        with PipelineReporter(
            script=SCRIPT_PATH,
            stage="03_picks",
            report_root=REPORT_ROOT,
            pipeline="NFL",
            league="NFL",
            extra_context={
                "component": "NM simplified weekly picks",
            },
        ) as reporter:
            args = parse_args()
            run(
                args,
                reporter,
            )
        return 0
    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
