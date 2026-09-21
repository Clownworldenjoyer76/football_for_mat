#!/usr/bin/env python3
"""
pull_final_scores.py

Reads game_ids from the existing schedule output and resolves final
score + status for each game from ESPN's core API. Writes one CSV per
season/season_type/week.

Input:
    docs/win/football/nfl/00_intake/schedule/{season}_schedule.csv

Source:
    https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{game_id}/competitions/{game_id}
    (status + per-competitor score resolved automatically)

Output:
    docs/win/football/nfl/04_final_results/results/{season}_{season_type}_{week}.csv

Error/run log:
    docs/win/football/nfl/errors/04_final_results/pull_final_scores.txt
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict
from datetime import datetime, timezone
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


SETTINGS_PATH = NFL_ROOT / "config" / "settings.yaml"
SCHEDULE_DIR = NFL_ROOT / "00_intake" / "schedule"
RESULTS_DIR = NFL_ROOT / "04_final_results" / "results"
ERROR_LOG_PATH = (
    NFL_ROOT
    / "errors"
    / "04_final_results"
    / "pull_final_scores.txt"
)

COMPETITION_URL_TEMPLATE = (
    "https://sports.core.api.espn.com/v2/sports/"
    "football/leagues/nfl/events/{game_id}/"
    "competitions/{game_id}"
)

OUTPUT_HEADER = [
    "season",
    "season_type",
    "week",
    "game_id",
    "game_date",
    "game_time",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
    "status",
]

REQUIRED_SCHEDULE_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "game_date",
    "game_time",
    "away_team",
    "home_team",
]

TRANSIENT_HTTP_CODES = {
    408,
    429,
    500,
    502,
    503,
    504,
}

MAX_FETCH_ATTEMPTS = 3
FETCH_TIMEOUT_SECONDS = 10
RETRY_BASE_DELAY_SECONDS = 0.5


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


def fail(message: str) -> None:
    raise RuntimeError(message)


def parse_positive_int(
    value: Any,
    *,
    label: str,
) -> int:
    text = clean(value)

    if not text:
        fail(f"{label} is blank")

    try:
        number = float(text)
    except (TypeError, ValueError):
        fail(
            f"{label} must be a positive integer; "
            f"found {value!r}"
        )

    if (
        not math.isfinite(number)
        or not number.is_integer()
        or number <= 0
    ):
        fail(
            f"{label} must be a positive integer; "
            f"found {value!r}"
        )

    return int(number)


def parse_score(
    value: Any,
    *,
    label: str,
) -> str:
    text = clean(value)

    if not text:
        fail(f"{label} is blank")

    try:
        number = float(text)
    except (TypeError, ValueError):
        fail(
            f"{label} must be numeric; "
            f"found {value!r}"
        )

    if (
        not math.isfinite(number)
        or number < 0
        or not number.is_integer()
    ):
        fail(
            f"{label} must be a nonnegative "
            f"integer score; found {value!r}"
        )

    return text


def read_configured_season() -> tuple[int, str]:
    environment_value = clean(
        os.environ.get("NFL_SEASON")
    )

    if environment_value:
        season = parse_positive_int(
            environment_value,
            label="NFL_SEASON",
        )

        if not 2000 <= season <= 2100:
            fail(
                "NFL_SEASON must be between "
                "2000 and 2100"
            )

        return season, "environment:NFL_SEASON"

    if not SETTINGS_PATH.is_file():
        fail(
            "NFL_SEASON is not set and "
            f"settings file is missing: "
            f"{SETTINGS_PATH}"
        )

    matches: list[str] = []

    with SETTINGS_PATH.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        for raw_line in handle:
            content = raw_line.split(
                "#",
                1,
            )[0].strip()

            if not content:
                continue

            match = re.fullmatch(
                r"season\s*:\s*(.+?)\s*",
                content,
            )

            if match:
                matches.append(
                    match.group(1)
                )

    if len(matches) != 1:
        fail(
            "settings.yaml must contain exactly "
            "one top-level season entry"
        )

    season = parse_positive_int(
        matches[0],
        label="settings.yaml season",
    )

    if not 2000 <= season <= 2100:
        fail(
            "settings.yaml season must be "
            "between 2000 and 2100"
        )

    return season, "settings.yaml"


def append_legacy_log(
    lines: list[str],
) -> None:
    ERROR_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    with ERROR_LOG_PATH.open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(
            f"--- run {timestamp} ---\n"
        )

        for line in lines:
            handle.write(
                f"{line}\n"
            )

        handle.write("\n")


def validate_csv_header(
    path: Path,
    *,
    label: str,
) -> list[str]:
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

    return normalized


def read_schedule(
    path: Path,
    *,
    expected_season: int,
) -> list[dict[str, str]]:
    if not path.is_file():
        fail(
            f"Schedule file not found: {path}"
        )

    header = validate_csv_header(
        path,
        label="schedule",
    )

    missing = [
        column
        for column in REQUIRED_SCHEDULE_COLUMNS
        if column not in header
    ]

    if missing:
        fail(
            "schedule: missing required "
            f"columns: {missing}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        rows = [
            {
                key: clean(value)
                for key, value in row.items()
                if key is not None
            }
            for row in reader
        ]

    if not rows:
        fail(
            f"Schedule contains no rows: {path}"
        )

    seen_game_ids: set[str] = set()

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        season = parse_positive_int(
            row.get("season"),
            label=(
                f"schedule row {row_number} season"
            ),
        )

        if season != expected_season:
            fail(
                f"schedule row {row_number}: "
                f"season={season}; expected "
                f"{expected_season}"
            )

        season_type = clean(
            row.get("season_type")
        )
        week = parse_positive_int(
            row.get("week"),
            label=(
                f"schedule row {row_number} week"
            ),
        )
        game_id = clean(
            row.get("game_id")
        )
        game_date = clean(
            row.get("game_date")
        )
        game_time = clean(
            row.get("game_time")
        )
        away_team = clean(
            row.get("away_team")
        )
        home_team = clean(
            row.get("home_team")
        )

        if not season_type:
            fail(
                f"schedule row {row_number}: "
                "season_type is blank"
            )

        if not game_id:
            fail(
                f"schedule row {row_number}: "
                "game_id is blank"
            )

        if game_id in seen_game_ids:
            fail(
                f"schedule row {row_number}: "
                f"duplicate game_id={game_id}"
            )

        seen_game_ids.add(game_id)

        for column, value in (
            ("game_date", game_date),
            ("game_time", game_time),
            ("away_team", away_team),
            ("home_team", home_team),
        ):
            if not value:
                fail(
                    f"schedule row {row_number}: "
                    f"{column} is blank"
                )

        if away_team == home_team:
            fail(
                f"schedule row {row_number}: "
                "away_team and home_team "
                "must differ"
            )

        row["season"] = str(season)
        row["season_type"] = season_type
        row["week"] = str(week)
        row["game_id"] = game_id
        row["game_date"] = game_date
        row["game_time"] = game_time
        row["away_team"] = away_team
        row["home_team"] = home_team

    return rows


class FetchCounters:
    def __init__(self) -> None:
        self.attempts = 0
        self.retries = 0
        self.failures = 0


def fetch_json(
    url: str,
    *,
    label: str,
    counters: FetchCounters,
    timeout: int = FETCH_TIMEOUT_SECONDS,
    max_attempts: int = MAX_FETCH_ATTEMPTS,
) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(
        1,
        max_attempts + 1,
    ):
        counters.attempts += 1

        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "football_for_mat/"
                        "pull_final_scores"
                    ),
                    "Accept": "application/json",
                },
            )

            with urllib.request.urlopen(
                request,
                timeout=timeout,
            ) as response:
                payload = json.loads(
                    response.read().decode(
                        "utf-8"
                    )
                )

            if not isinstance(
                payload,
                dict,
            ):
                fail(
                    f"{label}: ESPN response "
                    "is not a JSON object"
                )

            return payload
        except urllib.error.HTTPError as exc:
            last_error = exc

            if (
                exc.code
                not in TRANSIENT_HTTP_CODES
                or attempt == max_attempts
            ):
                break
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
        ) as exc:
            last_error = exc

            if attempt == max_attempts:
                break
        except json.JSONDecodeError as exc:
            last_error = exc
            break

        counters.retries += 1

        time.sleep(
            RETRY_BASE_DELAY_SECONDS
            * attempt
        )

    counters.failures += 1

    fail(
        f"{label}: failed after "
        f"{max_attempts if last_error else 1} "
        f"attempt(s): {last_error}"
    )


def get_score_and_status(
    game_id: str,
    *,
    counters: FetchCounters,
) -> tuple[
    str,
    str,
    bool,
    str,
]:
    competition_url = (
        COMPETITION_URL_TEMPLATE.format(
            game_id=game_id
        )
    )

    competition = fetch_json(
        competition_url,
        label=(
            f"game_id={game_id} competition"
        ),
        counters=counters,
    )

    status_object = competition.get(
        "status"
    )

    if not isinstance(
        status_object,
        dict,
    ):
        fail(
            f"game_id={game_id}: "
            "competition status object "
            "is missing"
        )

    status_ref = clean(
        status_object.get("$ref")
    )

    if not status_ref:
        fail(
            f"game_id={game_id}: "
            "competition status reference "
            "is missing"
        )

    status = fetch_json(
        status_ref,
        label=(
            f"game_id={game_id} status"
        ),
        counters=counters,
    )

    status_type = status.get(
        "type"
    )

    if not isinstance(
        status_type,
        dict,
    ):
        fail(
            f"game_id={game_id}: "
            "status type object is missing"
        )

    completed = status_type.get(
        "completed"
    )

    if not isinstance(
        completed,
        bool,
    ):
        fail(
            f"game_id={game_id}: "
            "status completed flag is "
            f"invalid: {completed!r}"
        )

    status_text = clean(
        status_type.get(
            "description"
        )
    )

    if not status_text:
        fail(
            f"game_id={game_id}: "
            "status description is blank"
        )

    competitors = competition.get(
        "competitors"
    )

    if not isinstance(
        competitors,
        list,
    ):
        fail(
            f"game_id={game_id}: "
            "competitors is not a list"
        )

    by_side: dict[
        str,
        dict[str, Any],
    ] = {}

    for competitor in competitors:
        if not isinstance(
            competitor,
            dict,
        ):
            fail(
                f"game_id={game_id}: "
                "competitor entry is invalid"
            )

        side = clean(
            competitor.get("homeAway")
        )

        if side not in {
            "away",
            "home",
        }:
            fail(
                f"game_id={game_id}: "
                f"invalid competitor "
                f"homeAway={side!r}"
            )

        if side in by_side:
            fail(
                f"game_id={game_id}: "
                f"duplicate {side} competitor"
            )

        by_side[
            side
        ] = competitor

    if set(
        by_side
    ) != {
        "away",
        "home",
    }:
        fail(
            f"game_id={game_id}: "
            "expected exactly one away "
            "and one home competitor"
        )

    scores: dict[
        str,
        str,
    ] = {}

    for side in (
        "away",
        "home",
    ):
        score_object = by_side[
            side
        ].get(
            "score"
        )

        if not isinstance(
            score_object,
            dict,
        ):
            fail(
                f"game_id={game_id}: "
                f"{side} score object "
                "is missing"
            )

        score_ref = clean(
            score_object.get("$ref")
        )

        if not score_ref:
            fail(
                f"game_id={game_id}: "
                f"{side} score reference "
                "is missing"
            )

        score_data = fetch_json(
            score_ref,
            label=(
                f"game_id={game_id} "
                f"{side} score"
            ),
            counters=counters,
        )

        value = score_data.get(
            "displayValue"
        )

        if clean(value) == "":
            value = score_data.get(
                "value"
            )

        scores[
            side
        ] = parse_score(
            value,
            label=(
                f"game_id={game_id} "
                f"{side} score"
            ),
        )

    return (
        scores["away"],
        scores["home"],
        completed,
        status_text,
    )


def build_result_rows(
    schedule_rows: list[
        dict[str, str]
    ],
    *,
    counters: FetchCounters,
) -> tuple[
    dict[
        tuple[str, str, str],
        list[dict[str, str]],
    ],
    dict[str, bool],
    int,
    int,
]:
    rows_by_week: dict[
        tuple[str, str, str],
        list[dict[str, str]],
    ] = defaultdict(list)

    completed_by_game: dict[
        str,
        bool,
    ] = {}

    completed_count = 0
    not_final_count = 0

    for row in schedule_rows:
        game_id = row["game_id"]

        (
            away_score,
            home_score,
            is_final,
            status_text,
        ) = get_score_and_status(
            game_id,
            counters=counters,
        )

        if is_final:
            completed_count += 1
        else:
            not_final_count += 1

        completed_by_game[
            game_id
        ] = is_final

        out_row = {
            "season": row["season"],
            "season_type": (
                row["season_type"]
            ),
            "week": row["week"],
            "game_id": game_id,
            "game_date": row["game_date"],
            "game_time": row["game_time"],
            "away_team": row["away_team"],
            "home_team": row["home_team"],
            "away_score": away_score,
            "home_score": home_score,
            "status": status_text,
        }

        key = (
            out_row["season"],
            out_row["season_type"],
            out_row["week"],
        )

        rows_by_week[
            key
        ].append(
            out_row
        )

    return (
        dict(rows_by_week),
        completed_by_game,
        completed_count,
        not_final_count,
    )


def validate_result_rows(
    rows_by_week: dict[
        tuple[str, str, str],
        list[dict[str, str]],
    ],
    schedule_rows: list[
        dict[str, str]
    ],
    completed_by_game: dict[
        str,
        bool,
    ],
    *,
    expected_season: int,
) -> None:
    schedule_by_group: dict[
        tuple[str, str, str],
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in schedule_rows:
        key = (
            row["season"],
            row["season_type"],
            row["week"],
        )
        schedule_by_group[
            key
        ].append(
            row
        )

    if set(
        rows_by_week
    ) != set(
        schedule_by_group
    ):
        fail(
            "Result groups do not match "
            "schedule groups"
        )

    all_game_ids: set[str] = set()

    for key, expected_rows in (
        schedule_by_group.items()
    ):
        actual_rows = rows_by_week[
            key
        ]

        if len(
            actual_rows
        ) != len(
            expected_rows
        ):
            fail(
                f"Result row-count mismatch "
                f"for group={key}: "
                f"expected={len(expected_rows)} "
                f"actual={len(actual_rows)}"
            )

        for index, (
            expected,
            actual,
        ) in enumerate(
            zip(
                expected_rows,
                actual_rows,
                strict=True,
            ),
            start=2,
        ):
            if set(
                actual
            ) != set(
                OUTPUT_HEADER
            ):
                fail(
                    f"group={key} row={index}: "
                    "output columns are invalid"
                )

            game_id = clean(
                actual["game_id"]
            )

            if not game_id:
                fail(
                    f"group={key} row={index}: "
                    "game_id is blank"
                )

            if game_id in all_game_ids:
                fail(
                    f"Duplicate output "
                    f"game_id={game_id}"
                )

            all_game_ids.add(
                game_id
            )

            for column in (
                "season",
                "season_type",
                "week",
                "game_id",
                "game_date",
                "game_time",
                "away_team",
                "home_team",
            ):
                if (
                    clean(
                        actual[column]
                    )
                    != clean(
                        expected[column]
                    )
                ):
                    fail(
                        f"game_id={game_id}: "
                        f"{column} differs "
                        "from schedule"
                    )

            if (
                parse_positive_int(
                    actual["season"],
                    label=(
                        f"game_id={game_id} "
                        "season"
                    ),
                )
                != expected_season
            ):
                fail(
                    f"game_id={game_id}: "
                    "output season mismatch"
                )

            parse_positive_int(
                actual["week"],
                label=(
                    f"game_id={game_id} week"
                ),
            )

            parse_score(
                actual["away_score"],
                label=(
                    f"game_id={game_id} "
                    "away_score"
                ),
            )
            parse_score(
                actual["home_score"],
                label=(
                    f"game_id={game_id} "
                    "home_score"
                ),
            )

            if not clean(
                actual["status"]
            ):
                fail(
                    f"game_id={game_id}: "
                    "status is blank"
                )

            if (
                game_id
                not in completed_by_game
            ):
                fail(
                    f"game_id={game_id}: "
                    "missing completed-state "
                    "metadata"
                )

    if len(
        all_game_ids
    ) != len(
        schedule_rows
    ):
        fail(
            "Output game_id count does not "
            "match schedule row count"
        )


def result_path_for_group(
    group: tuple[
        str,
        str,
        str,
    ],
) -> Path:
    (
        season,
        season_type,
        week,
    ) = group

    return (
        RESULTS_DIR
        / (
            f"{season}_{season_type}_"
            f"{week}.csv"
        )
    ).resolve()


def stage_result_file(
    rows: list[
        dict[str, str]
    ],
    target_path: Path,
) -> Path:
    target_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, raw_path = (
        tempfile.mkstemp(
            prefix=(
                f".{target_path.name}."
                "stage."
            ),
            suffix=".csv",
            dir=str(
                target_path.parent
            ),
        )
    )
    os.close(
        descriptor
    )

    stage_path = Path(
        raw_path
    )

    try:
        with stage_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=OUTPUT_HEADER,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(
                rows
            )

        if (
            not stage_path.is_file()
            or stage_path.stat().st_size
            == 0
        ):
            fail(
                f"Staged result was not "
                f"written: {stage_path}"
            )

        return stage_path
    except Exception:
        stage_path.unlink(
            missing_ok=True
        )
        raise


def read_result_file(
    path: Path,
    *,
    allow_empty: bool = False,
) -> list[dict[str, str]]:
    header = validate_csv_header(
        path,
        label="result CSV",
    )

    if header != OUTPUT_HEADER:
        fail(
            f"Result CSV header mismatch: "
            f"{path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(
            handle
        )
        rows = [
            {
                column: clean(
                    row.get(
                        column
                    )
                )
                for column in OUTPUT_HEADER
            }
            for row in reader
        ]

    if (
        not rows
        and not allow_empty
    ):
        fail(
            f"Result CSV contains no rows: "
            f"{path}"
        )

    return rows


def validate_serialized_group(
    path: Path,
    expected_rows: list[
        dict[str, str]
    ],
    *,
    expected_group: tuple[
        str,
        str,
        str,
    ],
) -> None:
    actual_rows = read_result_file(
        path
    )

    if len(
        actual_rows
    ) != len(
        expected_rows
    ):
        fail(
            f"{path}: row-count mismatch"
        )

    for actual, expected in zip(
        actual_rows,
        expected_rows,
        strict=True,
    ):
        for column in OUTPUT_HEADER:
            if (
                clean(
                    actual[column]
                )
                != clean(
                    expected[column]
                )
            ):
                fail(
                    f"{path}: game_id="
                    f"{expected['game_id']}: "
                    f"column {column!r} "
                    "changed during "
                    "serialization"
                )

        if (
            actual["season"],
            actual["season_type"],
            actual["week"],
        ) != expected_group:
            fail(
                f"{path}: group identity "
                "changed during "
                "serialization"
            )

        parse_score(
            actual["away_score"],
            label=(
                f"{path} game_id="
                f"{actual['game_id']} "
                "away_score"
            ),
        )
        parse_score(
            actual["home_score"],
            label=(
                f"{path} game_id="
                f"{actual['game_id']} "
                "home_score"
            ),
        )

        if not actual["status"]:
            fail(
                f"{path}: game_id="
                f"{actual['game_id']} "
                "status is blank"
            )


def stale_managed_paths(
    *,
    season: int,
    expected_paths: set[Path],
) -> list[Path]:
    if not RESULTS_DIR.exists():
        return []

    pattern = re.compile(
        rf"^{season}_[^_]+_\d+\.csv$"
    )

    stale: list[Path] = []

    for path in RESULTS_DIR.iterdir():
        if not path.is_file():
            continue

        if not pattern.fullmatch(
            path.name
        ):
            continue

        resolved = path.resolve()

        if (
            resolved
            not in expected_paths
        ):
            stale.append(
                resolved
            )

    return sorted(
        stale
    )


def publish_transaction(
    staged: dict[
        Path,
        Path,
    ],
    expected_rows_by_path: dict[
        Path,
        list[dict[str, str]],
    ],
    group_by_path: dict[
        Path,
        tuple[str, str, str],
    ],
    stale_paths: list[Path],
    *,
    reporter: PipelineReporter,
) -> None:
    managed_paths = [
        *staged.keys(),
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
                "transactional_multi_file_"
                "atomic_replace_with_rollback"
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
                    f"backup."
                    f"{uuid.uuid4().hex}"
                )
            )

            shutil.copy2(
                path,
                backup,
            )

            backups[
                path
            ] = backup

        for live_path, stage_path in (
            staged.items()
        ):
            os.replace(
                stage_path,
                live_path,
            )
            live_modified = True

        for stale_path in stale_paths:
            if stale_path.exists():
                stale_path.unlink()
                live_modified = True

        for live_path, expected_rows in (
            expected_rows_by_path.items()
        ):
            validate_serialized_group(
                live_path,
                expected_rows,
                expected_group=(
                    group_by_path[
                        live_path
                    ]
                ),
            )

        remaining_stale = [
            str(path)
            for path in stale_paths
            if path.exists()
        ]

        if remaining_stale:
            fail(
                "Stale managed result files "
                "remain after publication: "
                f"{remaining_stale}"
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
                    "Final-score publication "
                    "failed and rollback also "
                    "failed: "
                    f"publication_error="
                    f"{publish_exc}; "
                    f"rollback_error="
                    f"{rollback_exc}"
                ) from rollback_exc

        raise
    finally:
        for stage_path in staged.values():
            stage_path.unlink(
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
                        "Temporary final-score "
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


def run(
    reporter: PipelineReporter,
    legacy_lines: list[str],
) -> None:
    season, season_source = (
        read_configured_season()
    )

    schedule_path = (
        SCHEDULE_DIR
        / f"{season}_schedule.csv"
    ).resolve()

    reporter.season = season
    reporter.add_input(
        schedule_path
    )

    reporter.update_details(
        {
            "configured_season": season,
            "season_source": season_source,
            "schedule_path": str(
                schedule_path
            ),
            "fetch_max_attempts": (
                MAX_FETCH_ATTEMPTS
            ),
            "fetch_timeout_seconds": (
                FETCH_TIMEOUT_SECONDS
            ),
            "staged_roundtrip_verified": False,
            "publication_completed": False,
            "post_publish_validation": False,
            "rollback_performed": False,
        }
    )

    legacy_lines.append(
        f"season={season}"
    )
    legacy_lines.append(
        f"season_source={season_source}"
    )

    schedule_rows = read_schedule(
        schedule_path,
        expected_season=season,
    )

    reporter.set_rows(
        rows_in=len(
            schedule_rows
        )
    )

    counters = FetchCounters()

    (
        rows_by_week,
        completed_by_game,
        completed_count,
        not_final_count,
    ) = build_result_rows(
        schedule_rows,
        counters=counters,
    )

    validate_result_rows(
        rows_by_week,
        schedule_rows,
        completed_by_game,
        expected_season=season,
    )

    staged: dict[
        Path,
        Path,
    ] = {}

    expected_rows_by_path: dict[
        Path,
        list[dict[str, str]],
    ] = {}

    group_by_path: dict[
        Path,
        tuple[str, str, str],
    ] = {}

    try:
        for group, rows in sorted(
            rows_by_week.items(),
            key=lambda item: (
                item[0][0],
                item[0][1],
                int(item[0][2]),
            ),
        ):
            output_path = (
                result_path_for_group(
                    group
                )
            )

            reporter.add_output(
                output_path
            )

            stage_path = stage_result_file(
                rows,
                output_path,
            )

            validate_serialized_group(
                stage_path,
                rows,
                expected_group=group,
            )

            staged[
                output_path
            ] = stage_path
            expected_rows_by_path[
                output_path
            ] = rows
            group_by_path[
                output_path
            ] = group

        reporter.set_detail(
            "staged_roundtrip_verified",
            True,
        )

        expected_paths = set(
            staged
        )

        stale_paths = (
            stale_managed_paths(
                season=season,
                expected_paths=(
                    expected_paths
                ),
            )
        )

        reporter.update_details(
            {
                "games_processed": len(
                    schedule_rows
                ),
                "completed_games": (
                    completed_count
                ),
                "not_final_games": (
                    not_final_count
                ),
                "fetch_attempts": (
                    counters.attempts
                ),
                "fetch_retries": (
                    counters.retries
                ),
                "fetch_failures": (
                    counters.failures
                ),
                "result_files": len(
                    staged
                ),
                "stale_managed_files": [
                    str(path)
                    for path in stale_paths
                ],
                "stale_managed_file_count": len(
                    stale_paths
                ),
            }
        )

        publish_transaction(
            staged,
            expected_rows_by_path,
            group_by_path,
            stale_paths,
            reporter=reporter,
        )
    finally:
        for stage_path in staged.values():
            stage_path.unlink(
                missing_ok=True
            )

    reporter.set_rows(
        rows_out=sum(
            len(rows)
            for rows
            in rows_by_week.values()
        )
    )

    for output_path, rows in (
        expected_rows_by_path.items()
    ):
        legacy_lines.append(
            f"wrote {len(rows)} rows "
            f"to {output_path}"
        )

    summary = (
        f"games_processed={len(schedule_rows)} "
        f"completed={completed_count} "
        f"not_final={not_final_count} "
        f"failed={counters.failures} "
        f"files_written="
        f"{len(expected_rows_by_path)} "
        f"fetch_attempts={counters.attempts} "
        f"fetch_retries={counters.retries}"
    )

    print(
        summary
    )
    legacy_lines.append(
        summary
    )


def main() -> int:
    legacy_lines: list[str] = []

    try:
        with PipelineReporter(
            script=SCRIPT_PATH,
            stage="04_final_results",
            report_root=REPORT_ROOT,
            pipeline="NFL",
            league="NFL",
            extra_context={
                "component": (
                    "ESPN final score pull"
                ),
            },
        ) as reporter:
            try:
                run(
                    reporter,
                    legacy_lines,
                )
            except Exception as exc:
                legacy_lines.append(
                    "FAILED "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
                raise
            finally:
                if legacy_lines:
                    try:
                        append_legacy_log(
                            legacy_lines
                        )
                    except Exception as exc:
                        reporter.warning(
                            "Legacy final-score log "
                            "write failed",
                            error_type=(
                                type(exc).__name__
                            ),
                            error=str(
                                exc
                            ),
                            log_path=str(
                                ERROR_LOG_PATH
                            ),
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