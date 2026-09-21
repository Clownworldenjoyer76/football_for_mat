#!/usr/bin/env python3
"""
Build compact all-game NFL projection picks output.

READS:
  docs/win/football/nfl/02_select/week_{week}_NFL_selected.csv

WRITES:
  docs/win/football/nfl/03_picks/all_games/all_week_{week}_NFL_picks.csv

OUTPUT COLUMNS:
  season
  week
  game_id
  away_team
  home_team
  predicted_away_score
  predicted_home_score
  predicted_total
  predicted_home_spread
  predicted_away_spread

The projected away score, home score, and total use the original model
projection values and are displayed to exactly 1 decimal place.

The projected spreads are calculated from the displayed 1-decimal projected
scores.

Spread definitions:
  predicted_home_spread =
      predicted_away_score - predicted_home_score

  predicted_away_spread =
      predicted_home_score - predicted_away_score
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import sys
import tempfile
import uuid
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


DEFAULT_INPUT_DIR = NFL_ROOT / "02_select"
DEFAULT_OUTPUT_DIR = NFL_ROOT / "03_picks" / "all_games"

SOURCE_FILENAME_PATTERN = re.compile(
    r"^week_(\d+)_NFL_selected\.csv$"
)
OUTPUT_FILENAME_PATTERN = re.compile(
    r"^all_week_(\d+)_NFL_picks\.csv$"
)

OUTPUT_COLUMNS = [
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

REQUIRED_INPUT_COLUMNS = [
    "season",
    "week",
    "game_id",
    "away_team",
    "home_team",
    "predicted_away_score",
    "predicted_home_score",
    "predicted_total",
]


def load_runtime_dependencies(
    reporter: PipelineReporter,
) -> None:
    global pd

    try:
        import pandas as pd_module
    except Exception:
        reporter.set_detail(
            "dependency_imports_ok",
            False,
        )
        raise

    pd = pd_module

    reporter.set_detail(
        "dependency_imports_ok",
        True,
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


def parse_float(
    value: Any,
    *,
    column: str,
    row_number: int,
) -> float:
    text = clean(value)

    if not text:
        fail(
            f"Row {row_number}: "
            f"{column} is blank"
        )

    try:
        number = float(text)
    except (TypeError, ValueError):
        fail(
            f"Row {row_number}: "
            f"{column} is not numeric: "
            f"{value!r}"
        )

    if not math.isfinite(number):
        fail(
            f"Row {row_number}: "
            f"{column} is non-finite: "
            f"{value!r}"
        )

    return number


def parse_positive_int(
    value: Any,
    *,
    column: str,
    row_number: int,
) -> int:
    number = parse_float(
        value,
        column=column,
        row_number=row_number,
    )

    if (
        not float(number).is_integer()
        or number <= 0
    ):
        fail(
            f"Row {row_number}: "
            f"{column} must be a positive integer; "
            f"found {value!r}"
        )

    return int(number)


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


def validate_game_ids(
    df: pd.DataFrame,
    label: str,
) -> None:
    game_ids = df["game_id"].map(clean)

    if (game_ids == "").any():
        fail(
            f"{label}: blank game_id found"
        )

    duplicates = (
        game_ids[
            game_ids.duplicated(
                keep=False
            )
        ]
        .drop_duplicates()
        .tolist()
    )

    if duplicates:
        fail(
            f"{label}: duplicate game_id values: "
            f"{duplicates[:10]}"
        )


def validate_csv_header(
    path: Path,
    label: str,
) -> None:
    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
    except UnicodeDecodeError as exc:
        fail(
            f"{label}: invalid UTF-8 CSV: "
            f"{path}: {exc}"
        )

    if not header:
        fail(
            f"{label}: missing CSV header: "
            f"{path}"
        )

    normalized = [
        clean(column)
        for column in header
    ]

    if any(
        not column
        for column in normalized
    ):
        fail(
            f"{label}: blank CSV column name "
            f"found: {path}"
        )

    duplicates = sorted(
        {
            column
            for column in normalized
            if normalized.count(column) > 1
        }
    )

    if duplicates:
        fail(
            f"{label}: duplicate CSV column "
            f"names: {duplicates}"
        )


def read_csv(
    path: Path,
    label: str,
) -> pd.DataFrame:
    if not path.is_file():
        fail(
            f"{label} not found: {path}"
        )

    validate_csv_header(
        path,
        label,
    )

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8-sig",
        low_memory=False,
    )

    if df.empty:
        fail(
            f"{label} contains no rows: "
            f"{path}"
        )

    return df


def validate_source(
    source: pd.DataFrame,
    input_path: Path,
    requested_week: int,
) -> int:
    label = str(input_path)

    require_columns(
        source,
        REQUIRED_INPUT_COLUMNS,
        label,
    )

    validate_game_ids(
        source,
        label,
    )

    seasons: set[int] = set()
    weeks: set[int] = set()

    for index, row in source.iterrows():
        row_number = index + 2

        season = parse_positive_int(
            row["season"],
            column="season",
            row_number=row_number,
        )
        week = parse_positive_int(
            row["week"],
            column="week",
            row_number=row_number,
        )

        seasons.add(season)
        weeks.add(week)

        if week != requested_week:
            fail(
                f"Row {row_number}: source week "
                f"{week} does not match requested "
                f"--week {requested_week}"
            )

        away_team = clean(
            row["away_team"]
        )
        home_team = clean(
            row["home_team"]
        )

        if not away_team:
            fail(
                f"Row {row_number}: "
                "away_team is blank"
            )

        if not home_team:
            fail(
                f"Row {row_number}: "
                "home_team is blank"
            )

        away_score = parse_float(
            row["predicted_away_score"],
            column="predicted_away_score",
            row_number=row_number,
        )
        home_score = parse_float(
            row["predicted_home_score"],
            column="predicted_home_score",
            row_number=row_number,
        )
        predicted_total = parse_float(
            row["predicted_total"],
            column="predicted_total",
            row_number=row_number,
        )

        if not math.isclose(
            away_score + home_score,
            predicted_total,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            fail(
                f"Row {row_number}: predicted "
                "away/home scores do not "
                "reconcile to predicted_total"
            )

    if len(seasons) != 1:
        fail(
            f"{label}: expected one season; "
            f"found {sorted(seasons)}"
        )

    if weeks != {
        requested_week
    }:
        fail(
            f"{label}: expected only "
            f"week={requested_week}; "
            f"found {sorted(weeks)}"
        )

    return next(iter(seasons))


def round_one_decimal(
    value: float,
) -> float:
    return round(value, 1)


def format_one_decimal(
    value: float,
) -> str:
    return f"{value:.1f}"


def build_output(
    source: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for index, row in source.iterrows():
        row_number = index + 2

        away_score = parse_float(
            row["predicted_away_score"],
            column="predicted_away_score",
            row_number=row_number,
        )

        home_score = parse_float(
            row["predicted_home_score"],
            column="predicted_home_score",
            row_number=row_number,
        )

        predicted_total = parse_float(
            row["predicted_total"],
            column="predicted_total",
            row_number=row_number,
        )

        away_score_display = round_one_decimal(
            away_score
        )

        home_score_display = round_one_decimal(
            home_score
        )

        total_display = round_one_decimal(
            predicted_total
        )

        predicted_home_spread = round_one_decimal(
            away_score_display
            - home_score_display
        )

        predicted_away_spread = round_one_decimal(
            home_score_display
            - away_score_display
        )

        rows.append(
            {
                "season": clean(
                    row["season"]
                ),
                "week": clean(
                    row["week"]
                ),
                "game_id": clean(
                    row["game_id"]
                ),
                "away_team": clean(
                    row["away_team"]
                ),
                "home_team": clean(
                    row["home_team"]
                ),
                "predicted_away_score": (
                    format_one_decimal(
                        away_score_display
                    )
                ),
                "predicted_home_score": (
                    format_one_decimal(
                        home_score_display
                    )
                ),
                "predicted_total": (
                    format_one_decimal(
                        total_display
                    )
                ),
                "predicted_home_spread": (
                    format_one_decimal(
                        predicted_home_spread
                    )
                ),
                "predicted_away_spread": (
                    format_one_decimal(
                        predicted_away_spread
                    )
                ),
            }
        )

    output = pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )

    return output


def validate_output(
    output: pd.DataFrame,
    source: pd.DataFrame,
    *,
    label: str,
) -> None:
    if list(output.columns) != OUTPUT_COLUMNS:
        fail(
            f"{label}: output column "
            "integrity check failed"
        )

    if len(output) != len(source):
        fail(
            f"{label}: output row count "
            "does not match input row count"
        )

    validate_game_ids(
        output,
        label,
    )

    expected = build_output(
        source
    )

    if len(expected) != len(output):
        fail(
            f"{label}: internal expected "
            "row-count mismatch"
        )

    for column in OUTPUT_COLUMNS:
        actual_values = [
            clean(value)
            for value in output[
                column
            ].tolist()
        ]
        expected_values = [
            clean(value)
            for value in expected[
                column
            ].tolist()
        ]

        if actual_values != expected_values:
            mismatch_index = next(
                index
                for index, (
                    actual_value,
                    expected_value,
                )
                in enumerate(
                    zip(
                        actual_values,
                        expected_values,
                        strict=True,
                    )
                )
                if actual_value
                != expected_value
            )

            game_id = expected.iloc[
                mismatch_index
            ]["game_id"]

            fail(
                f"{label}: column "
                f"{column!r} differs from "
                "the documented calculation "
                f"for game_id={game_id}"
            )

    for column in (
        "predicted_away_score",
        "predicted_home_score",
        "predicted_total",
        "predicted_home_spread",
        "predicted_away_spread",
    ):
        for index, value in enumerate(
            output[column].tolist()
        ):
            text = clean(
                value
            )

            if not re.fullmatch(
                r"-?\d+\.\d",
                text,
            ):
                game_id = output.iloc[
                    index
                ]["game_id"]

                fail(
                    f"{label}: {column} "
                    "must use exactly one "
                    "decimal place for "
                    f"game_id={game_id}; "
                    f"found {value!r}"
                )


def stage_output(
    output: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, raw_path = (
        tempfile.mkstemp(
            prefix=(
                f".{output_path.name}."
                "stage."
            ),
            suffix=".csv",
            dir=str(
                output_path.parent
            ),
        )
    )
    os.close(descriptor)

    staged_path = Path(
        raw_path
    )

    try:
        output.to_csv(
            staged_path,
            index=False,
            encoding="utf-8",
        )

        if (
            not staged_path.is_file()
            or staged_path.stat().st_size
            == 0
        ):
            fail(
                "Staged all-games output "
                f"was not written: "
                f"{staged_path}"
            )

        return staged_path
    except Exception:
        staged_path.unlink(
            missing_ok=True
        )
        raise


def validate_serialized_output(
    path: Path,
    source: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    serialized = read_csv(
        path,
        label,
    )

    validate_output(
        serialized,
        source,
        label=label,
    )

    return serialized


def source_weeks() -> set[int]:
    if not DEFAULT_INPUT_DIR.is_dir():
        fail(
            "Missing selected input "
            f"directory: {DEFAULT_INPUT_DIR}"
        )

    weeks: set[int] = set()

    for path in DEFAULT_INPUT_DIR.glob(
        "week_*_NFL_selected.csv"
    ):
        if not path.is_file():
            continue

        match = (
            SOURCE_FILENAME_PATTERN
            .fullmatch(
                path.name
            )
        )

        if match is None:
            continue

        week = int(
            match.group(1)
        )

        if week <= 0:
            fail(
                "Invalid selected-source "
                f"week filename: {path.name}"
            )

        if week in weeks:
            fail(
                "Duplicate selected-source "
                f"week discovered: {week}"
            )

        weeks.add(
            week
        )

    if not weeks:
        fail(
            "No managed selected-source "
            f"files found in {DEFAULT_INPUT_DIR}"
        )

    return weeks


def stale_output_paths(
    managed_weeks: set[int],
) -> list[Path]:
    if not DEFAULT_OUTPUT_DIR.exists():
        return []

    stale: list[Path] = []

    for path in DEFAULT_OUTPUT_DIR.glob(
        "all_week_*_NFL_picks.csv"
    ):
        if not path.is_file():
            continue

        match = (
            OUTPUT_FILENAME_PATTERN
            .fullmatch(
                path.name
            )
        )

        if match is None:
            continue

        week = int(
            match.group(1)
        )

        if week not in managed_weeks:
            stale.append(
                path.resolve()
            )

    return sorted(
        stale
    )


def publish_transaction(
    staged_path: Path,
    output_path: Path,
    stale_paths: list[Path],
    source: pd.DataFrame,
    *,
    reporter: PipelineReporter,
) -> None:
    managed_paths = [
        output_path,
        *stale_paths,
    ]

    backups: dict[
        Path,
        Path,
    ] = {}

    live_modified = False
    rollback_failed = False

    reporter.update_details(
        {
            "publication_mode": (
                "transactional_atomic_"
                "replace_with_backup_rollback"
            ),
            "publication_completed": False,
            "post_publish_validation": False,
            "rollback_performed": False,
        }
    )

    try:
        for path in managed_paths:
            if not path.exists():
                continue

            backup = (
                path.parent
                / (
                    f".{path.name}."
                    f"backup.{uuid.uuid4().hex}"
                )
            )

            shutil.copy2(
                path,
                backup,
            )
            backups[
                path
            ] = backup

        os.replace(
            staged_path,
            output_path,
        )
        live_modified = True

        for stale_path in stale_paths:
            if stale_path.exists():
                stale_path.unlink()
                live_modified = True

        validate_serialized_output(
            output_path,
            source,
            label=(
                "published all-games output"
            ),
        )

        remaining_stale = [
            str(path)
            for path in stale_paths
            if path.exists()
        ]

        if remaining_stale:
            fail(
                "Stale managed all-games "
                "outputs remain after "
                f"publication: {remaining_stale}"
            )

        reporter.update_details(
            {
                "publication_completed": True,
                "post_publish_validation": True,
            }
        )
    except Exception as publish_exc:
        if live_modified:
            try:
                for path in managed_paths:
                    backup = backups.get(
                        path
                    )

                    if (
                        backup is not None
                        and backup.exists()
                    ):
                        os.replace(
                            backup,
                            path,
                        )
                    elif path.exists():
                        path.unlink()

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
                        "rollback_error_type": (
                            type(
                                rollback_exc
                            ).__name__
                        ),
                        "rollback_error": str(
                            rollback_exc
                        ),
                    }
                )

                raise RuntimeError(
                    "All-games publication "
                    "failed and rollback also "
                    "failed: "
                    f"publication_error="
                    f"{publish_exc}; "
                    f"rollback_error="
                    f"{rollback_exc}"
                ) from rollback_exc

        raise
    finally:
        staged_path.unlink(
            missing_ok=True
        )

        if not rollback_failed:
            for backup in backups.values():
                if not backup.exists():
                    continue

                try:
                    backup.unlink()
                except Exception as exc:
                    reporter.warning(
                        "Temporary all-games "
                        "backup cleanup failed",
                        backup_path=str(
                            backup
                        ),
                        error_type=(
                            type(exc).__name__
                        ),
                        error=str(
                            exc
                        ),
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--week",
        type=int,
        required=True,
        help="NFL week number",
    )

    return parser.parse_args()


def run(
    args: argparse.Namespace,
    reporter: PipelineReporter,
) -> None:
    load_runtime_dependencies(
        reporter
    )

    if args.week <= 0:
        fail(
            "--week must be greater than 0"
        )

    input_path = (
        DEFAULT_INPUT_DIR
        / f"week_{args.week}_NFL_selected.csv"
    ).resolve()

    output_path = (
        DEFAULT_OUTPUT_DIR
        / f"all_week_{args.week}_NFL_picks.csv"
    ).resolve()

    reporter.add_input(
        input_path
    )
    reporter.add_output(
        output_path
    )

    reporter.update_details(
        {
            "requested_week": args.week,
            "staged_roundtrip_verified": False,
            "publication_completed": False,
            "post_publish_validation": False,
            "rollback_performed": False,
        }
    )

    source = read_csv(
        input_path,
        "selected input",
    )

    season = validate_source(
        source,
        input_path,
        args.week,
    )

    reporter.season = season
    reporter.week = args.week

    reporter.set_rows(
        rows_in=len(
            source
        )
    )

    output = build_output(
        source
    )

    validate_output(
        output,
        source,
        label=(
            "in-memory all-games output"
        ),
    )

    managed_weeks = (
        source_weeks()
    )

    if args.week not in managed_weeks:
        fail(
            "Requested week is not in "
            "the managed selected-source "
            f"set: week={args.week}"
        )

    stale_paths = (
        stale_output_paths(
            managed_weeks
        )
    )

    reporter.update_details(
        {
            "resolved_season": season,
            "resolved_week": args.week,
            "input_path": str(
                input_path
            ),
            "output_path": str(
                output_path
            ),
            "output_columns": list(
                OUTPUT_COLUMNS
            ),
            "managed_source_weeks": sorted(
                managed_weeks
            ),
            "stale_managed_outputs": [
                str(path)
                for path in stale_paths
            ],
            "stale_managed_output_count": len(
                stale_paths
            ),
        }
    )

    staged_path = stage_output(
        output,
        output_path,
    )

    try:
        validate_serialized_output(
            staged_path,
            source,
            label=(
                "staged all-games output"
            ),
        )

        reporter.set_detail(
            "staged_roundtrip_verified",
            True,
        )

        publish_transaction(
            staged_path,
            output_path,
            stale_paths,
            source,
            reporter=reporter,
        )
    finally:
        staged_path.unlink(
            missing_ok=True
        )

    reporter.set_rows(
        rows_out=len(
            output
        )
    )

    print(
        f"WROTE {output_path} | "
        f"games={len(output)}"
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
                    "all-games projection picks"
                ),
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
            f"ERROR: {type(exc).__name__}: "
            f"{exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())