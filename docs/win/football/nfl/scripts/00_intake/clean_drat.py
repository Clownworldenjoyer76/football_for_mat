#!/usr/bin/env python3

from __future__ import annotations

import csv
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter


DRAT_RAW_DIR = NFL_ROOT / "00_intake" / "predictions" / "drat" / "raw"
DRAT_CLEAN_DIR = NFL_ROOT / "00_intake" / "predictions" / "drat" / "clean"
SCHEDULE_WEEKLY_DIR = NFL_ROOT / "00_intake" / "schedule" / "weekly"
REPORT_ROOT = NFL_ROOT / "errors"
LEGACY_LOG_PATH = REPORT_ROOT / "00_intake" / "clean_drat.txt"

HISTORICAL_FILENAME_RE = re.compile(
    r"^(?P<season>\d{4})_wk(?P<week>\d{2})_odds\.csv$",
    re.IGNORECASE,
)
CLEAN_HISTORICAL_FILENAME_RE = re.compile(
    r"^(?P<season>\d{4})_week_(?P<week>\d+)_drat\.csv$",
    re.IGNORECASE,
)

EXPECTED_INPUT_HEADERS = [
    "season",
    "week",
    "game_id",
    "commence_time_utc",
    "home_team",
    "away_team",
    "book",
    "spread_home",
    "spread_away",
    "total",
    "moneyline_home",
    "moneyline_away",
    "updated_at_utc",
    "is_consensus",
    "game_date",
    "game_time",
    "home_prob",
    "away_prob",
    "spread_home_odds",
    "spread_away_odds",
    "total_over",
    "total_under",
    "total_odds_over",
    "total_odds_under",
    "away_projected_score",
    "home_projected_score",
    "total_projected_score",
]

OUTPUT_HEADERS = [
    "season",
    "week",
    "game_id",
    "commence_time_utc",
    "home_team",
    "away_team",
    "spread_home",
    "spread_away",
    "total",
    "moneyline_home",
    "moneyline_away",
    "updated_at_utc",
    "game_date",
    "game_time",
    "home_prob",
    "away_prob",
    "spread_home_odds",
    "spread_away_odds",
    "total_over",
    "total_under",
    "total_odds_over",
    "total_odds_under",
    "away_projected_score",
    "home_projected_score",
    "total_projected_score",
]

SCHEDULE_REQUIRED_HEADERS = [
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
]

MatchKey = tuple[str, str, str, str]
ScheduleIndex = dict[MatchKey, set[str]]


class RunLog:
    def __init__(self, reporter: PipelineReporter) -> None:
        self.reporter = reporter
        self.info_lines: list[str] = []
        self.warning_lines: list[str] = []
        self.error_lines: list[str] = []

    def info(self, message: str) -> None:
        self.info_lines.append(message)
        print(f"INFO: {message}")

    def warning(self, message: str, **details: object) -> None:
        self.warning_lines.append(message)
        self.reporter.warning(message, **details)
        print(f"WARNING: {message}")

    def error(self, message: str, **details: object) -> None:
        self.error_lines.append(message)
        self.reporter.error(message, **details)
        print(f"ERROR: {message}", file=sys.stderr)

    @property
    def has_errors(self) -> bool:
        return bool(self.error_lines)

    def write_legacy(
        self,
        *,
        schedule_files: int,
        schedule_rows: int,
        schedule_keys: int,
        historical_found: int,
        historical_ignored_2025: int,
        historical_succeeded: int,
        historical_failed: int,
        latest_succeeded: int,
        latest_failed: int,
        rows_read: int,
        rows_written: int,
        publication_completed: bool,
        stale_managed_outputs_removed: int,
    ) -> None:
        LEGACY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        written_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        lines = [
            "clean_drat.py",
            "=" * 80,
            f"Log written UTC: {written_at}",
            "",
            "Paths",
            "-" * 80,
            f"DRAT raw:       {DRAT_RAW_DIR}",
            f"DRAT clean:     {DRAT_CLEAN_DIR}",
            f"Schedule input: {SCHEDULE_WEEKLY_DIR}",
            f"JSON report:    {self.reporter.report_path}",
            f"Legacy log:     {LEGACY_LOG_PATH}",
            "",
            "Summary",
            "-" * 80,
            f"Schedule files read:            {schedule_files}",
            f"Schedule rows read:             {schedule_rows}",
            f"Schedule match keys loaded:     {schedule_keys}",
            f"Historical DRAT files found:    {historical_found}",
            f"2025 historical files ignored: {historical_ignored_2025}",
            f"Historical files succeeded:    {historical_succeeded}",
            f"Historical files failed:       {historical_failed}",
            f"latest.csv succeeded:           {latest_succeeded}",
            f"latest.csv failed:              {latest_failed}",
            f"DRAT rows read:                 {rows_read}",
            f"Rows written:                   {rows_written}",
            f"Publication completed:          {publication_completed}",
            f"Stale managed outputs removed:  {stale_managed_outputs_removed}",
            f"Warnings:                       {len(self.warning_lines)}",
            f"Errors:                         {len(self.error_lines)}",
            "",
        ]

        if self.info_lines:
            lines.extend(
                [
                    "Details",
                    "-" * 80,
                    *self.info_lines,
                    "",
                ]
            )

        if self.warning_lines:
            lines.extend(
                [
                    "Warnings",
                    "-" * 80,
                    *self.warning_lines,
                    "",
                ]
            )

        if self.error_lines:
            lines.extend(
                [
                    "Errors",
                    "-" * 80,
                    *self.error_lines,
                    "",
                ]
            )

        if self.has_errors:
            result = "FAILED"
        elif self.warning_lines:
            result = "WARNING"
        else:
            result = "SUCCESS"

        lines.extend(
            [
                "Result",
                "-" * 80,
                result,
                "",
            ]
        )

        temporary_path = LEGACY_LOG_PATH.with_name(
            f".{LEGACY_LOG_PATH.name}.tmp"
        )

        try:
            with temporary_path.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                handle.write("\n".join(lines))
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temporary_path, LEGACY_LOG_PATH)
        finally:
            temporary_path.unlink(missing_ok=True)


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_number(value: object) -> str:
    """
    Normalize integer-like season/week values for matching only.

    Examples:
        "01"   -> "1"
        "1"    -> "1"
        "2026" -> "2026"

    Non-integer values are returned stripped and unchanged.
    """
    text = clean_text(value)

    if not text:
        return ""

    try:
        return str(int(text))
    except ValueError:
        return text


def make_match_key(row: dict[str, str]) -> MatchKey:
    return (
        normalize_number(row.get("season")),
        normalize_number(row.get("week")),
        clean_text(row.get("home_team")),
        clean_text(row.get("away_team")),
    )


def describe_match_key(key: MatchKey) -> str:
    season, week, home_team, away_team = key
    return (
        f"season={season}, week={week}, "
        f"home_team={home_team!r}, away_team={away_team!r}"
    )


def read_csv_rows(
    path: Path,
    required_headers: Iterable[str],
    *,
    require_rows: bool = False,
) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")

        fieldnames = [clean_text(field) for field in reader.fieldnames]
        required = list(required_headers)

        missing = [header for header in required if header not in fieldnames]
        if missing:
            raise ValueError(
                "Missing required header(s): " + ", ".join(missing)
            )

        rows = list(reader)

    if require_rows and not rows:
        raise ValueError("CSV contains no data rows.")

    return fieldnames, rows


def build_schedule_index(
    log: RunLog,
    reporter: PipelineReporter,
) -> tuple[ScheduleIndex | None, int, int]:
    if not SCHEDULE_WEEKLY_DIR.exists():
        log.error(
            f"Schedule directory does not exist: {SCHEDULE_WEEKLY_DIR}"
        )
        return None, 0, 0

    schedule_files = sorted(SCHEDULE_WEEKLY_DIR.glob("*.csv"))

    if not schedule_files:
        log.error(
            f"No schedule CSV files found in: {SCHEDULE_WEEKLY_DIR}"
        )
        return None, 0, 0

    schedule_index: ScheduleIndex = {}
    schedule_rows_read = 0
    load_failed = False

    for schedule_path in schedule_files:
        reporter.add_input(schedule_path)

        try:
            _, rows = read_csv_rows(
                schedule_path,
                SCHEDULE_REQUIRED_HEADERS,
                require_rows=True,
            )
        except Exception as exc:
            log.error(
                f"Could not read schedule file {schedule_path}: {exc}",
                source=str(schedule_path),
                error_type=type(exc).__name__,
            )
            load_failed = True
            continue

        file_rows_loaded = 0

        for row_number, row in enumerate(rows, start=2):
            schedule_rows_read += 1

            key = make_match_key(row)
            game_id = clean_text(row.get("game_id"))

            season, week, home_team, away_team = key

            if not season or not week or not home_team or not away_team:
                log.error(
                    f"Invalid schedule row with incomplete match fields: "
                    f"{schedule_path} row {row_number}; "
                    f"{describe_match_key(key)}",
                    source=str(schedule_path),
                    row_number=row_number,
                )
                load_failed = True
                continue

            if not game_id:
                log.error(
                    f"Invalid schedule row with blank game_id: "
                    f"{schedule_path} row {row_number}; "
                    f"{describe_match_key(key)}",
                    source=str(schedule_path),
                    row_number=row_number,
                )
                load_failed = True
                continue

            schedule_index.setdefault(key, set()).add(game_id)
            file_rows_loaded += 1

        log.info(
            f"Schedule loaded: {schedule_path.name} "
            f"({file_rows_loaded} usable rows)"
        )

    if load_failed:
        log.error(
            "Schedule index could not be trusted because one or more "
            "schedule files failed validation."
        )
        return None, len(schedule_files), schedule_rows_read

    if not schedule_index:
        log.error("Schedule index contains no usable game records.")
        return None, len(schedule_files), schedule_rows_read

    ambiguous_keys = {
        key: game_ids
        for key, game_ids in schedule_index.items()
        if len(game_ids) > 1
    }

    if ambiguous_keys:
        log.warning(
            f"Schedule data contains {len(ambiguous_keys)} match key(s) "
            "with multiple game_id values. A DRAT row using one of those "
            "keys will fail rather than choosing a game_id arbitrarily.",
            ambiguous_key_count=len(ambiguous_keys),
        )

    return schedule_index, len(schedule_files), schedule_rows_read


def write_clean_csv(
    output_path: Path,
    rows: list[dict[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_HEADERS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    header: clean_text(row.get(header))
                    for header in OUTPUT_HEADERS
                }
            )

        handle.flush()
        os.fsync(handle.fileno())


def transform_drat_rows(
    *,
    source_path: Path,
    rows: list[dict[str, str]],
    schedule_index: ScheduleIndex,
    log: RunLog,
    expected_season: str | None = None,
    expected_week: str | None = None,
) -> list[dict[str, str]] | None:
    output_rows: list[dict[str, str]] = []
    file_errors: list[str] = []
    seen_match_keys: dict[MatchKey, int] = {}
    seen_game_ids: dict[str, int] = {}

    normalized_expected_season = (
        normalize_number(expected_season)
        if expected_season is not None
        else None
    )
    normalized_expected_week = (
        normalize_number(expected_week)
        if expected_week is not None
        else None
    )

    for row_number, row in enumerate(rows, start=2):
        key = make_match_key(row)
        season, week, home_team, away_team = key

        if not season or not week or not home_team or not away_team:
            file_errors.append(
                f"{source_path} row {row_number}: "
                "missing one or more schedule-match values; "
                f"{describe_match_key(key)}"
            )
            continue

        if (
            normalized_expected_season is not None
            and season != normalized_expected_season
        ):
            file_errors.append(
                f"{source_path} row {row_number}: "
                f"row season={season!r} does not match filename "
                f"season={normalized_expected_season!r}."
            )
            continue

        if (
            normalized_expected_week is not None
            and week != normalized_expected_week
        ):
            file_errors.append(
                f"{source_path} row {row_number}: "
                f"row week={week!r} does not match filename "
                f"week={normalized_expected_week!r}."
            )
            continue

        prior_match_row = seen_match_keys.get(key)
        if prior_match_row is not None:
            file_errors.append(
                f"{source_path} row {row_number}: duplicate DRAT matchup; "
                f"first seen at row {prior_match_row}; "
                f"{describe_match_key(key)}."
            )
            continue

        seen_match_keys[key] = row_number

        schedule_game_ids = schedule_index.get(key)

        if not schedule_game_ids:
            file_errors.append(
                f"{source_path} row {row_number}: "
                "no schedule match found for "
                f"{describe_match_key(key)}."
            )
            continue

        if len(schedule_game_ids) != 1:
            file_errors.append(
                f"{source_path} row {row_number}: "
                "schedule match is ambiguous for "
                f"{describe_match_key(key)}; "
                f"game_id values={sorted(schedule_game_ids)!r}."
            )
            continue

        schedule_game_id = next(iter(schedule_game_ids))

        prior_game_row = seen_game_ids.get(schedule_game_id)
        if prior_game_row is not None:
            file_errors.append(
                f"{source_path} row {row_number}: duplicate resulting "
                f"schedule game_id={schedule_game_id!r}; first seen at "
                f"row {prior_game_row}."
            )
            continue

        seen_game_ids[schedule_game_id] = row_number

        clean_row = {
            header: clean_text(row.get(header))
            for header in OUTPUT_HEADERS
        }

        # Always replace the imported DRAT game_id with the schedule game_id.
        clean_row["game_id"] = schedule_game_id

        output_rows.append(clean_row)

    if file_errors:
        for message in file_errors:
            log.error(
                message,
                source=str(source_path),
            )

        log.error(
            f"{source_path.name}: clean output was not staged because "
            f"{len(file_errors)} row error(s) were found.",
            source=str(source_path),
            row_error_count=len(file_errors),
        )
        return None

    return output_rows


def process_historical_file(
    source_path: Path,
    schedule_index: ScheduleIndex,
    staging_dir: Path,
    log: RunLog,
) -> tuple[bool, int, int, Path | None]:
    match = HISTORICAL_FILENAME_RE.fullmatch(source_path.name)

    if match is None:
        log.error(
            f"Historical filename does not match expected pattern: "
            f"{source_path.name}"
        )
        return False, 0, 0, None

    season = match.group("season")
    week_text = match.group("week")
    week_number = int(week_text)

    output_path = (
        staging_dir
        / f"{season}_week_{week_number}_drat.csv"
    )

    try:
        fieldnames, rows = read_csv_rows(
            source_path,
            EXPECTED_INPUT_HEADERS,
            require_rows=True,
        )
    except Exception as exc:
        log.error(
            f"Could not read {source_path}: {exc}",
            source=str(source_path),
            error_type=type(exc).__name__,
        )
        return False, 0, 0, None

    extra_headers = [
        header
        for header in fieldnames
        if header not in EXPECTED_INPUT_HEADERS
    ]

    if extra_headers:
        log.warning(
            f"{source_path.name}: extra input header(s) will be ignored: "
            + ", ".join(extra_headers),
            source=str(source_path),
            extra_headers=extra_headers,
        )

    clean_rows = transform_drat_rows(
        source_path=source_path,
        rows=rows,
        schedule_index=schedule_index,
        log=log,
        expected_season=season,
        expected_week=str(week_number),
    )

    if clean_rows is None:
        return False, len(rows), 0, None

    try:
        write_clean_csv(output_path, clean_rows)
    except Exception as exc:
        log.error(
            f"Could not stage clean output {output_path}: {exc}",
            source=str(source_path),
            output=str(output_path),
            error_type=type(exc).__name__,
        )
        return False, len(rows), 0, None

    log.info(
        f"Historical DRAT staged: {source_path.name} -> "
        f"{output_path.name} ({len(clean_rows)} rows)"
    )

    return True, len(rows), len(clean_rows), output_path


def process_latest_file(
    source_path: Path,
    schedule_index: ScheduleIndex,
    staging_dir: Path,
    log: RunLog,
) -> tuple[bool, int, int, Path | None, set[str]]:
    output_path = staging_dir / "latest.csv"

    try:
        fieldnames, rows = read_csv_rows(
            source_path,
            EXPECTED_INPUT_HEADERS,
            require_rows=True,
        )
    except Exception as exc:
        log.error(
            f"Could not read {source_path}: {exc}",
            source=str(source_path),
            error_type=type(exc).__name__,
        )
        return False, 0, 0, None, set()

    extra_headers = [
        header
        for header in fieldnames
        if header not in EXPECTED_INPUT_HEADERS
    ]

    if extra_headers:
        log.warning(
            f"{source_path.name}: extra input header(s) will be ignored: "
            + ", ".join(extra_headers),
            source=str(source_path),
            extra_headers=extra_headers,
        )

    clean_rows = transform_drat_rows(
        source_path=source_path,
        rows=rows,
        schedule_index=schedule_index,
        log=log,
    )

    if clean_rows is None:
        return False, len(rows), 0, None, set()

    active_seasons = {
        normalize_number(row.get("season"))
        for row in clean_rows
        if normalize_number(row.get("season"))
    }

    try:
        write_clean_csv(output_path, clean_rows)
    except Exception as exc:
        log.error(
            f"Could not stage clean output {output_path}: {exc}",
            source=str(source_path),
            output=str(output_path),
            error_type=type(exc).__name__,
        )
        return False, len(rows), 0, None, active_seasons

    log.info(
        f"Latest DRAT staged: {source_path.name} -> "
        f"{output_path.name} ({len(clean_rows)} rows)"
    )

    return True, len(rows), len(clean_rows), output_path, active_seasons


def existing_managed_outputs(active_seasons: set[str]) -> list[Path]:
    if not DRAT_CLEAN_DIR.exists():
        return []

    managed: list[Path] = []

    latest_path = DRAT_CLEAN_DIR / "latest.csv"
    if latest_path.exists():
        managed.append(latest_path)

    for path in sorted(DRAT_CLEAN_DIR.glob("*_week_*_drat.csv")):
        match = CLEAN_HISTORICAL_FILENAME_RE.fullmatch(path.name)
        if match is None:
            continue

        season = normalize_number(match.group("season"))

        # 2025 historical outputs are outside this cleaner's managed set,
        # matching the existing behavior that ignores 2025 historical raw data.
        if season == "2025":
            continue

        if season in active_seasons:
            managed.append(path)

    return managed


def publish_staged_outputs(
    staging_dir: Path,
    active_seasons: set[str],
    log: RunLog,
) -> tuple[list[Path], int]:
    DRAT_CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    staged_files = sorted(
        path
        for path in staging_dir.iterdir()
        if path.is_file()
    )

    if not staged_files:
        raise RuntimeError("No staged DRAT outputs are available to publish.")

    staged_names = {path.name for path in staged_files}
    current_managed = existing_managed_outputs(active_seasons)
    stale_managed_outputs_removed = sum(
        1
        for path in current_managed
        if path.name not in staged_names
    )

    backup_dir = Path(
        tempfile.mkdtemp(
            prefix=".clean_drat_backup_",
            dir=DRAT_CLEAN_DIR.parent,
        )
    )
    published_paths: list[Path] = []

    try:
        for existing_path in current_managed:
            os.replace(
                existing_path,
                backup_dir / existing_path.name,
            )

        try:
            for staged_path in staged_files:
                final_path = DRAT_CLEAN_DIR / staged_path.name
                os.replace(staged_path, final_path)
                published_paths.append(final_path)

        except Exception:
            for final_path in published_paths:
                final_path.unlink(missing_ok=True)

            for backup_path in backup_dir.iterdir():
                os.replace(
                    backup_path,
                    DRAT_CLEAN_DIR / backup_path.name,
                )

            raise

    finally:
        shutil.rmtree(backup_dir, ignore_errors=True)

    log.info(
        f"Published {len(published_paths)} DRAT clean output(s); "
        f"removed {stale_managed_outputs_removed} stale managed output(s)."
    )

    return published_paths, stale_managed_outputs_removed


def main() -> int:
    with PipelineReporter(
        script=SCRIPT_PATH,
        stage="00_intake",
        report_root=REPORT_ROOT,
        pipeline="NFL",
        league="NFL",
        extra_context={
            "component": "DRAT cleaner",
        },
    ) as reporter:
        log = RunLog(reporter)

        reporter.add_input(DRAT_RAW_DIR)
        reporter.add_input(SCHEDULE_WEEKLY_DIR)
        reporter.add_output(LEGACY_LOG_PATH)

        schedule_files = 0
        schedule_rows = 0
        schedule_keys = 0

        historical_found = 0
        historical_ignored_2025 = 0
        historical_succeeded = 0
        historical_failed = 0

        latest_succeeded = 0
        latest_failed = 0
        rows_read = 0
        rows_written = 0

        publication_completed = False
        stale_managed_outputs_removed = 0
        published_paths: list[Path] = []
        active_seasons: set[str] = set()

        if not DRAT_RAW_DIR.exists():
            log.error(
                f"DRAT raw directory does not exist: {DRAT_RAW_DIR}"
            )
        else:
            schedule_index, schedule_files, schedule_rows = (
                build_schedule_index(log, reporter)
            )

            if schedule_index is not None:
                schedule_keys = len(schedule_index)

                raw_csv_files = sorted(DRAT_RAW_DIR.glob("*.csv"))

                historical_files: list[Path] = []
                latest_path: Path | None = None

                for source_path in raw_csv_files:
                    reporter.add_input(source_path)

                    filename_lower = source_path.name.lower()

                    if filename_lower == "latest.csv":
                        latest_path = source_path
                        continue

                    if source_path.name.startswith("2025"):
                        historical_ignored_2025 += 1
                        log.info(
                            f"Ignored 2025 DRAT file: "
                            f"{source_path.name}"
                        )
                        continue

                    match = HISTORICAL_FILENAME_RE.fullmatch(
                        source_path.name
                    )

                    if match:
                        historical_files.append(source_path)
                        active_seasons.add(
                            normalize_number(match.group("season"))
                        )
                        continue

                    log.warning(
                        f"Ignored unrecognized CSV in DRAT raw directory: "
                        f"{source_path.name}",
                        source=str(source_path),
                    )

                historical_found = len(historical_files)

                if not historical_files:
                    log.warning(
                        "No non-2025 historical DRAT files matching "
                        "YYYY_wkNN_odds.csv were found."
                    )

                with tempfile.TemporaryDirectory(
                    prefix=".clean_drat_stage_",
                    dir=DRAT_CLEAN_DIR.parent,
                ) as staging_name:
                    staging_dir = Path(staging_name)

                    for source_path in historical_files:
                        (
                            success,
                            input_count,
                            output_count,
                            _,
                        ) = process_historical_file(
                            source_path,
                            schedule_index,
                            staging_dir,
                            log,
                        )

                        rows_read += input_count

                        if success:
                            historical_succeeded += 1
                            rows_written += output_count
                        else:
                            historical_failed += 1

                    if latest_path is None:
                        latest_failed += 1
                        log.error(
                            f"Required latest.csv was not found at: "
                            f"{DRAT_RAW_DIR / 'latest.csv'}"
                        )
                    else:
                        (
                            success,
                            input_count,
                            output_count,
                            _,
                            latest_seasons,
                        ) = process_latest_file(
                            latest_path,
                            schedule_index,
                            staging_dir,
                            log,
                        )

                        rows_read += input_count
                        active_seasons.update(latest_seasons)

                        if success:
                            latest_succeeded += 1
                            rows_written += output_count
                        else:
                            latest_failed += 1

                    if not log.has_errors:
                        (
                            published_paths,
                            stale_managed_outputs_removed,
                        ) = publish_staged_outputs(
                            staging_dir,
                            active_seasons,
                            log,
                        )
                        publication_completed = True

                        for path in published_paths:
                            reporter.add_output(path)

        reporter.set_rows(
            rows_in=rows_read,
            rows_out=rows_written if publication_completed else 0,
        )
        reporter.update_details(
            {
                "schedule_files_read": schedule_files,
                "schedule_rows_read": schedule_rows,
                "schedule_match_keys_loaded": schedule_keys,
                "historical_drat_files_found": historical_found,
                "historical_2025_files_ignored": (
                    historical_ignored_2025
                ),
                "historical_files_succeeded": historical_succeeded,
                "historical_files_failed": historical_failed,
                "latest_succeeded": latest_succeeded,
                "latest_failed": latest_failed,
                "active_seasons": sorted(active_seasons),
                "publication_completed": publication_completed,
                "published_outputs": [
                    str(path)
                    for path in published_paths
                ],
                "stale_managed_outputs_removed": (
                    stale_managed_outputs_removed
                ),
            }
        )

        log.write_legacy(
            schedule_files=schedule_files,
            schedule_rows=schedule_rows,
            schedule_keys=schedule_keys,
            historical_found=historical_found,
            historical_ignored_2025=historical_ignored_2025,
            historical_succeeded=historical_succeeded,
            historical_failed=historical_failed,
            latest_succeeded=latest_succeeded,
            latest_failed=latest_failed,
            rows_read=rows_read,
            rows_written=(
                rows_written if publication_completed else 0
            ),
            publication_completed=publication_completed,
            stale_managed_outputs_removed=stale_managed_outputs_removed,
        )

        return 1 if log.has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
