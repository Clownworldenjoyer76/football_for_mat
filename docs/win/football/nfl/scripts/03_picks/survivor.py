#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
import os
import re
import shutil
import sys
import tempfile
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


INPUT_DIR = NFL_ROOT / "03_picks" / "all_games"
OUTPUT_DIR = NFL_ROOT / "03_picks" / "survivor"

INPUT_PATTERN = "all_week_*_NFL_picks.csv"
FILENAME_PATTERN = re.compile(r"^all_week_(\d+)_NFL_picks\.csv$")
OUTPUT_FILENAME_PATTERN = re.compile(r"^(\d+)_survivor_picks\.csv$")

INPUT_COLUMNS = [
    "season",
    "week",
    "game_id",
    "away_team",
    "home_team",
    "predicted_away_score",
    "predicted_home_score",
    "predicted_total",
    "predicted_home_spread",
    "predicted_away_spread",
]

OUTPUT_COLUMNS = [
    "week",
    "game_id",
    "pick",
    "pt_diff",
    "away_team",
    "home_team",
]

PROJECTION_COLUMNS = [
    "predicted_away_score",
    "predicted_home_score",
    "predicted_total",
    "predicted_home_spread",
    "predicted_away_spread",
]

EPSILON = 1e-9
DISPLAY_RECONCILIATION_TOLERANCE = 0.150000001


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
    except (TypeError, ValueError):
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


def parse_positive_int(value: Any, column: str, row_number: int) -> int:
    number = parse_float(value, column, row_number)

    if not float(number).is_integer() or number <= 0:
        fail(
            f"Row {row_number}: {column} must be a "
            f"positive integer: {value!r}"
        )

    return int(number)


def validate_header(path: Path, expected: list[str], label: str) -> None:
    if not path.is_file():
        fail(f"{label} not found: {path}")

    try:
        with path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            header = next(csv.reader(handle), None)
    except Exception as exc:
        fail(
            f"{label}: unable to read CSV header: "
            f"{type(exc).__name__}: {exc}"
        )

    if header is None:
        fail(f"{label}: missing header row")

    if any(not clean(column) for column in header):
        fail(f"{label}: blank CSV header name")

    duplicates = sorted(
        {
            column
            for column in header
            if header.count(column) > 1
        }
    )

    if duplicates:
        fail(
            f"{label}: duplicate CSV header names: "
            f"{duplicates}"
        )

    if header != expected:
        fail(
            f"{label}: column contract failed; "
            f"expected={expected} actual={header}"
        )


def read_input(path):
    validate_header(
        path,
        INPUT_COLUMNS,
        f"survivor input {path}",
    )

    try:
        with path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            reader = csv.DictReader(file)
            rows = list(reader)
    except Exception as exc:
        fail(
            f"{path}: unable to read CSV: "
            f"{type(exc).__name__}: {exc}"
        )

    if not rows:
        fail(f"{path}: input contains no rows")

    for row_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            fail(
                f"{path}: row {row_number} has malformed "
                "CSV field count"
            )

    return rows


def discover_inputs() -> list[tuple[int, Path]]:
    if not INPUT_DIR.is_dir():
        fail(f"Input directory not found: {INPUT_DIR}")

    candidates = sorted(
        path
        for path in INPUT_DIR.glob(INPUT_PATTERN)
        if path.is_file()
    )

    if not candidates:
        fail(
            f"No input files found: "
            f"{INPUT_DIR / INPUT_PATTERN}"
        )

    discovered: list[tuple[int, Path]] = []
    seen_weeks: set[int] = set()

    for path in candidates:
        match = FILENAME_PATTERN.fullmatch(path.name)

        if match is None:
            fail(f"Invalid managed input filename: {path.name}")

        week = int(match.group(1))

        if week <= 0:
            fail(f"Invalid managed input week: {path.name}")

        if week in seen_weeks:
            fail(f"Duplicate managed input week: {week}")

        seen_weeks.add(week)
        discovered.append((week, path))

    discovered.sort(key=lambda item: item[0])
    return discovered


def validate_input_rows(
    rows: list[dict[str, str]],
    week: int,
    input_path: Path,
) -> set[int]:
    game_ids: set[str] = set()
    seasons: set[int] = set()

    for index, row in enumerate(rows, start=2):
        season = parse_positive_int(
            row["season"],
            "season",
            index,
        )
        row_week = parse_positive_int(
            row["week"],
            "week",
            index,
        )

        seasons.add(season)

        if row_week != week:
            fail(
                f"Row {index}: expected week {week}, "
                f"found {row_week!r}"
            )

        game_id = clean(row["game_id"])
        away_team = clean(row["away_team"])
        home_team = clean(row["home_team"])

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

        if away_team == home_team:
            fail(
                f"Row {index}: away_team and home_team "
                "must differ"
            )

        values: dict[str, float] = {}

        for column in PROJECTION_COLUMNS:
            text = clean(row[column])

            if not re.fullmatch(r"-?\d+\.\d", text):
                fail(
                    f"Row {index}: {column} must use "
                    f"exactly one decimal place: {text!r}"
                )

            values[column] = parse_float(
                row[column],
                column,
                index,
            )

        away_score = values["predicted_away_score"]
        home_score = values["predicted_home_score"]
        predicted_total = values["predicted_total"]
        home_spread = values["predicted_home_spread"]
        away_spread = values["predicted_away_spread"]

        if abs(
            (away_score + home_score) - predicted_total
        ) > DISPLAY_RECONCILIATION_TOLERANCE:
            fail(
                f"Row {index}: displayed projected scores "
                "do not reconcile to predicted_total within "
                "the producer's one-decimal rounding tolerance"
            )

        expected_home_spread = round(
            away_score - home_score,
            1,
        )
        expected_away_spread = round(
            home_score - away_score,
            1,
        )

        if abs(home_spread - expected_home_spread) > EPSILON:
            fail(
                f"Row {index}: predicted_home_spread does "
                "not match displayed projected scores"
            )

        if abs(away_spread - expected_away_spread) > EPSILON:
            fail(
                f"Row {index}: predicted_away_spread does "
                "not match displayed projected scores"
            )

        if abs(home_spread + away_spread) > EPSILON:
            fail(
                f"Row {index}: home/away predicted spreads "
                "are not exact opposites"
            )

        home_favorite = home_spread < 0
        away_favorite = away_spread < 0

        if home_favorite == away_favorite:
            fail(
                f"Row {index}: expected exactly one "
                f"negative spread"
            )

    return seasons


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


def expected_output_rows(
    rows: list[dict[str, str]],
    week: int,
) -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=2):
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

        if home_spread < 0:
            pick = clean(row["home_team"])
            favorite_spread = home_spread
        else:
            pick = clean(row["away_team"])
            favorite_spread = away_spread

        expected.append(
            {
                "week": str(week),
                "game_id": clean(row["game_id"]),
                "pick": pick,
                "pt_diff": abs(favorite_spread),
                "away_team": clean(row["away_team"]),
                "home_team": clean(row["home_team"]),
            }
        )

    expected.sort(
        key=lambda row: row["pt_diff"],
        reverse=True,
    )
    return expected


def validate_output_rows(
    output: list[dict[str, Any]],
    source_rows: list[dict[str, str]],
    week: int,
    *,
    label: str,
) -> None:
    if len(output) != len(source_rows):
        fail(
            f"{label}: output row count does not match "
            "input row count"
        )

    expected = expected_output_rows(
        source_rows,
        week,
    )

    seen_game_ids: set[str] = set()

    for index, (actual, wanted) in enumerate(
        zip(output, expected, strict=True),
        start=2,
    ):
        if list(actual.keys()) != OUTPUT_COLUMNS:
            fail(
                f"{label}: row {index} output column "
                "contract failed"
            )

        game_id = clean(actual["game_id"])

        if game_id in seen_game_ids:
            fail(
                f"{label}: duplicate game_id {game_id}"
            )
        seen_game_ids.add(game_id)

        for column in (
            "week",
            "game_id",
            "pick",
            "away_team",
            "home_team",
        ):
            if clean(actual[column]) != clean(wanted[column]):
                fail(
                    f"{label}: row {index} {column} "
                    "contract failed"
                )

        actual_diff = parse_float(
            actual["pt_diff"],
            "pt_diff",
            index,
        )
        wanted_diff = float(wanted["pt_diff"])

        if abs(actual_diff - wanted_diff) > EPSILON:
            fail(
                f"{label}: row {index} pt_diff "
                "contract failed"
            )


def write_staged_output(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
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

    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def read_output(path: Path) -> list[dict[str, str]]:
    validate_header(
        path,
        OUTPUT_COLUMNS,
        f"survivor output {path}",
    )

    try:
        with path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        fail(
            f"{path}: unable to read survivor output: "
            f"{type(exc).__name__}: {exc}"
        )

    if not rows:
        fail(f"{path}: survivor output contains no rows")

    for row_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            fail(
                f"{path}: row {row_number} has malformed "
                "CSV field count"
            )

    return rows


def validate_serialized_output(
    path: Path,
    source_rows: list[dict[str, str]],
    week: int,
    *,
    label: str,
) -> None:
    rows = read_output(path)

    normalized: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=2):
        pt_diff_text = clean(row["pt_diff"])

        if not re.fullmatch(r"\d+\.\d", pt_diff_text):
            fail(
                f"{label}: row {index} pt_diff must use "
                f"exactly one decimal place: {pt_diff_text!r}"
            )

        normalized.append(
            {
                "week": clean(row["week"]),
                "game_id": clean(row["game_id"]),
                "pick": clean(row["pick"]),
                "pt_diff": parse_float(
                    row["pt_diff"],
                    "pt_diff",
                    index,
                ),
                "away_team": clean(row["away_team"]),
                "home_team": clean(row["home_team"]),
            }
        )

    validate_output_rows(
        normalized,
        source_rows,
        week,
        label=label,
    )


def stage_output_set(
    outputs: dict[int, list[dict[str, Any]]],
    sources: dict[int, list[dict[str, str]]],
) -> Path:
    OUTPUT_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stage_root = Path(
        tempfile.mkdtemp(
            prefix=".survivor_stage_",
            dir=str(OUTPUT_DIR.parent),
        )
    )

    try:
        if set(outputs) != set(sources):
            fail(
                "Survivor staging week set does not match "
                "source week set"
            )

        for week in sorted(outputs):
            stage_path = (
                stage_root
                / f"{week}_survivor_picks.csv"
            )

            validate_output_rows(
                outputs[week],
                sources[week],
                week,
                label=(
                    f"in-memory survivor output week={week}"
                ),
            )

            write_staged_output(
                outputs[week],
                stage_path,
            )

            validate_serialized_output(
                stage_path,
                sources[week],
                week,
                label=(
                    f"staged survivor output week={week}"
                ),
            )

        return stage_root
    except Exception:
        shutil.rmtree(
            stage_root,
            ignore_errors=True,
        )
        raise


def managed_output_paths() -> set[Path]:
    if not OUTPUT_DIR.exists():
        return set()

    managed: set[Path] = set()

    for path in OUTPUT_DIR.glob(
        "*_survivor_picks.csv"
    ):
        if not path.is_file():
            continue

        match = OUTPUT_FILENAME_PATTERN.fullmatch(
            path.name
        )

        if match is None:
            fail(
                f"Invalid managed survivor output "
                f"filename: {path.name}"
            )

        week = int(match.group(1))

        if week <= 0:
            fail(
                f"Invalid managed survivor output "
                f"week: {path.name}"
            )

        managed.add(path.resolve())

    return managed


def expected_output_paths(
    weeks: set[int],
) -> dict[int, Path]:
    return {
        week: (
            OUTPUT_DIR
            / f"{week}_survivor_picks.csv"
        ).resolve()
        for week in weeks
    }


def validate_live_output_set(
    sources: dict[int, list[dict[str, str]]],
) -> None:
    expected = expected_output_paths(
        set(sources)
    )

    for week, path in sorted(
        expected.items()
    ):
        validate_serialized_output(
            path,
            sources[week],
            week,
            label=(
                f"published survivor output week={week}"
            ),
        )

    existing = managed_output_paths()
    expected_set = set(expected.values())

    if existing != expected_set:
        missing = sorted(
            str(path)
            for path in expected_set - existing
        )
        extra = sorted(
            str(path)
            for path in existing - expected_set
        )
        fail(
            "Published survivor output set mismatch: "
            f"missing={missing} extra={extra}"
        )


def publish_output_set(
    stage_root: Path,
    sources: dict[int, list[dict[str, str]]],
    *,
    reporter: PipelineReporter,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected = expected_output_paths(
        set(sources)
    )
    existing = managed_output_paths()
    expected_set = set(expected.values())
    managed_union = existing | expected_set
    stale = existing - expected_set

    backup_root = Path(
        tempfile.mkdtemp(
            prefix=".survivor_backup_",
            dir=str(OUTPUT_DIR.parent),
        )
    )
    backed_up: set[Path] = set()
    publication_started = False

    reporter.update_details(
        {
            "publication_mode": (
                "transactional_multi_week_"
                "atomic_replace_with_rollback"
            ),
            "publication_completed": False,
            "post_publish_validation": False,
            "rollback_performed": False,
            "stale_managed_output_count": len(stale),
            "stale_managed_outputs": [
                str(path)
                for path in sorted(stale)
            ],
        }
    )

    try:
        for live_path in sorted(
            existing,
            key=str,
        ):
            backup_path = (
                backup_root
                / live_path.name
            )
            shutil.copy2(
                live_path,
                backup_path,
            )
            backed_up.add(live_path)

        publication_started = True

        for week, live_path in sorted(
            expected.items()
        ):
            stage_path = (
                stage_root
                / f"{week}_survivor_picks.csv"
            )
            os.replace(
                stage_path,
                live_path,
            )

        for stale_path in sorted(
            stale,
            key=str,
        ):
            stale_path.unlink(
                missing_ok=True
            )

        validate_live_output_set(
            sources
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
                for live_path in sorted(
                    managed_union,
                    key=str,
                    reverse=True,
                ):
                    if live_path in backed_up:
                        backup_path = (
                            backup_root
                            / live_path.name
                        )
                        restore_fd, restore_raw = (
                            tempfile.mkstemp(
                                prefix=(
                                    f".{live_path.name}."
                                    "restore."
                                ),
                                suffix=".tmp",
                                dir=str(
                                    OUTPUT_DIR
                                ),
                            )
                        )
                        os.close(restore_fd)
                        restore_path = Path(
                            restore_raw
                        )

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
                            restore_path.unlink(
                                missing_ok=True
                            )
                    else:
                        live_path.unlink(
                            missing_ok=True
                        )

                reporter.update_details(
                    {
                        "publication_completed": False,
                        "post_publish_validation": False,
                        "rollback_performed": True,
                    }
                )
            except Exception as rollback_exc:
                reporter.update_details(
                    {
                        "publication_completed": False,
                        "post_publish_validation": False,
                        "rollback_performed": False,
                        "rollback_error_type": (
                            type(rollback_exc).__name__
                        ),
                        "rollback_error": str(
                            rollback_exc
                        ),
                    }
                )

                raise RuntimeError(
                    "NFL survivor publication failed and "
                    "rollback also failed: "
                    f"publication_error={publish_exc}; "
                    f"rollback_error={rollback_exc}"
                ) from rollback_exc

        raise
    finally:
        shutil.rmtree(
            stage_root,
            ignore_errors=True,
        )

        try:
            shutil.rmtree(
                backup_root,
                ignore_errors=False,
            )
        except Exception as cleanup_exc:
            reporter.warning(
                "Temporary NFL survivor backup cleanup "
                "failed",
                backup_root=str(
                    backup_root
                ),
                error_type=(
                    type(cleanup_exc).__name__
                ),
                error=str(
                    cleanup_exc
                ),
            )


def run(
    reporter: PipelineReporter,
) -> None:
    discovered = discover_inputs()

    sources: dict[
        int,
        list[dict[str, str]],
    ] = {}
    outputs: dict[
        int,
        list[dict[str, Any]],
    ] = {}
    seasons: set[int] = set()

    for week, input_path in discovered:
        reporter.add_input(
            input_path
        )

        rows = read_input(
            input_path
        )
        seasons.update(
            validate_input_rows(
                rows,
                week,
                input_path,
            )
        )

        output = build_output(
            rows,
            week,
            input_path,
        )

        validate_output_rows(
            output,
            rows,
            week,
            label=(
                f"in-memory survivor output week={week}"
            ),
        )

        sources[week] = rows
        outputs[week] = output

    expected = expected_output_paths(
        set(sources)
    )

    for path in expected.values():
        reporter.add_output(path)

    if len(seasons) == 1:
        reporter.season = next(
            iter(seasons)
        )

    total_input_rows = sum(
        len(rows)
        for rows in sources.values()
    )
    total_output_rows = sum(
        len(rows)
        for rows in outputs.values()
    )

    reporter.set_rows(
        rows_in=total_input_rows,
        rows_out=total_output_rows,
    )

    reporter.update_details(
        {
            "resolved_weeks": sorted(
                sources
            ),
            "resolved_seasons": sorted(
                seasons
            ),
            "input_file_count": len(
                sources
            ),
            "output_file_count": len(
                outputs
            ),
            "input_rows_validated": (
                total_input_rows
            ),
            "output_rows_validated": (
                total_output_rows
            ),
            "output_columns": list(
                OUTPUT_COLUMNS
            ),
            "staged_roundtrip_verified": False,
            "publication_completed": False,
            "post_publish_validation": False,
            "rollback_performed": False,
        }
    )

    stage_root = stage_output_set(
        outputs,
        sources,
    )

    reporter.set_detail(
        "staged_roundtrip_verified",
        True,
    )

    publish_output_set(
        stage_root,
        sources,
        reporter=reporter,
    )

    for week in sorted(outputs):
        print(
            f"WROTE {expected[week]} | "
            f"games={len(outputs[week])}"
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
                "component": (
                    "survivor projection picks"
                ),
            },
        ) as reporter:
            run(
                reporter
            )

        return 0
    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: "
            f"{exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
