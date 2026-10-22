#!/usr/bin/env python3
"""
Pull and validate ESPN predictor data for every game in one NFL season.

Input:
    docs/win/football/nfl/00_intake/schedule/{season}_schedule.csv

Source:
    https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{game_id}/competitions/{game_id}/predictor

Output:
    docs/win/football/nfl/00_intake/predictions/e_predictions/
        {season}_{season_type}_{week}_e_predictions.csv

The refresh is season-wide. Every scheduled game must resolve to one homeTeam
and one awayTeam row before any production prediction file is replaced.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter

SCHEDULE_DIR = NFL_ROOT / "00_intake" / "schedule"
TEAM_MAP_PATH = NFL_ROOT / "config" / "mapping" / "team_map.csv"
OUTPUT_DIR = (
    NFL_ROOT
    / "00_intake"
    / "predictions"
    / "e_predictions"
)
ERROR_LOG_PATH = (
    NFL_ROOT
    / "errors"
    / "00_intake"
    / "pull_e_predictions.txt"
)
REPORT_ROOT = NFL_ROOT / "errors"

PREDICTOR_URL_TEMPLATE = (
    "https://sports.core.api.espn.com/v2/sports/football/"
    "leagues/nfl/events/{game_id}/competitions/{game_id}/predictor"
)

REQUEST_TIMEOUT = 10
REQUEST_ATTEMPTS = 4
REQUEST_SLEEP_SECONDS = 0.10

OUTPUT_HEADER = [
    "season",
    "season_type",
    "week",
    "game_id",
    "game_name",
    "home_away",
    "team_id",
    "gameProjection",
    "matchupQuality",
    "oppSeasonStrengthFbsRank",
    "oppSeasonStrengthRating",
    "teamChanceLoss",
    "teamChanceTie",
    "teamPredPtDiff",
]

SCHEDULE_REQUIRED_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "away_team",
    "home_team",
]

TEAM_MAP_REQUIRED_COLUMNS = [
    "sport",
    "league",
    "team_id",
    "canonical_team",
]

REQUIRED_PREDICTOR_STATS = [
    "gameProjection",
    "matchupQuality",
    "oppSeasonStrengthFbsRank",
    "oppSeasonStrengthRating",
    "teamChanceLoss",
    "teamChanceTie",
    "teamPredPtDiff",
]

SIDES = ("homeTeam", "awayTeam")

RETRYABLE_HTTP_CODES = {
    408,
    425,
    429,
    500,
    502,
    503,
    504,
}


class PredictionPullError(RuntimeError):
    pass


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def fail(message: str) -> None:
    raise PredictionPullError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pull validated ESPN predictor data for one complete "
            "NFL season."
        )
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="NFL season year to refresh.",
    )
    args = parser.parse_args()

    if args.season < 2000 or args.season > 2100:
        parser.error("--season must be between 2000 and 2100")

    return args


def read_csv(
    path: Path,
    *,
    label: str,
    required_columns: list[str] | None = None,
    exact_columns: list[str] | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
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

    if not fieldnames:
        fail(f"{label} has no CSV header: {path}")

    if len(fieldnames) != len(set(fieldnames)):
        fail(f"{label} contains duplicate CSV columns: {path}")

    if required_columns:
        missing = [
            column
            for column in required_columns
            if column not in fieldnames
        ]
        if missing:
            fail(
                f"{label} missing expected columns: {missing}"
            )

    if exact_columns is not None and fieldnames != exact_columns:
        fail(
            f"{label} has unexpected column order/schema. "
            f"Expected={exact_columns} actual={fieldnames}"
        )

    if not rows:
        fail(f"{label} contains no data rows: {path}")

    return fieldnames, rows


def load_team_map() -> dict[str, str]:
    _, rows = read_csv(
        TEAM_MAP_PATH,
        label="NFL team map",
        required_columns=TEAM_MAP_REQUIRED_COLUMNS,
    )

    by_name: dict[str, str] = {}
    by_id: dict[str, str] = {}

    for line_number, row in enumerate(rows, start=2):
        sport = clean(row.get("sport")).casefold()
        league = clean(row.get("league")).casefold()

        if sport != "football" or league != "nfl":
            continue

        team_id = clean(row.get("team_id"))
        canonical_team = clean(
            row.get("canonical_team")
        )

        if not team_id or not canonical_team:
            fail(
                f"{TEAM_MAP_PATH} line {line_number} has "
                "blank team_id/canonical_team"
            )

        previous_id = by_name.get(canonical_team)
        if previous_id and previous_id != team_id:
            fail(
                f"{TEAM_MAP_PATH} has conflicting team IDs for "
                f"canonical_team={canonical_team!r}: "
                f"{previous_id!r} vs {team_id!r}"
            )

        previous_name = by_id.get(team_id)
        if (
            previous_name
            and previous_name != canonical_team
        ):
            fail(
                f"{TEAM_MAP_PATH} has conflicting canonical "
                f"teams for team_id={team_id}: "
                f"{previous_name!r} vs {canonical_team!r}"
            )

        by_name[canonical_team] = team_id
        by_id[team_id] = canonical_team

    if len(by_name) != 32 or len(by_id) != 32:
        fail(
            "NFL team map must resolve exactly 32 canonical "
            f"teams; names={len(by_name)} ids={len(by_id)}"
        )

    return by_name


def load_schedule(
    *,
    season: int,
    team_map: dict[str, str],
) -> tuple[Path, list[dict[str, str]]]:
    path = SCHEDULE_DIR / f"{season}_schedule.csv"

    _, rows = read_csv(
        path,
        label="season schedule",
        required_columns=SCHEDULE_REQUIRED_COLUMNS,
    )

    seen_game_ids: set[str] = set()

    for line_number, row in enumerate(rows, start=2):
        row_season = clean(row.get("season"))
        season_type = clean(row.get("season_type"))
        week = clean(row.get("week"))
        game_id = clean(row.get("game_id"))
        away_team = clean(row.get("away_team"))
        home_team = clean(row.get("home_team"))

        if row_season != str(season):
            fail(
                f"{path} line {line_number} has season="
                f"{row_season!r}; expected={season}"
            )

        if not season_type:
            fail(
                f"{path} line {line_number} has blank season_type"
            )

        if not week:
            fail(
                f"{path} line {line_number} has blank week"
            )

        try:
            week_number = int(week)
        except ValueError:
            fail(
                f"{path} line {line_number} has non-integer "
                f"week={week!r}"
            )

        if week_number < 1 or week_number > 22:
            fail(
                f"{path} line {line_number} has invalid "
                f"week={week_number}"
            )

        if not game_id:
            fail(
                f"{path} line {line_number} has blank game_id"
            )

        if game_id in seen_game_ids:
            fail(
                f"{path} contains duplicate game_id={game_id}"
            )

        if not game_id.isdigit():
            fail(
                f"{path} line {line_number} has nonnumeric "
                f"game_id={game_id!r}"
            )

        if not away_team or not home_team:
            fail(
                f"{path} line {line_number} has blank "
                "away_team/home_team"
            )

        if away_team == home_team:
            fail(
                f"{path} line {line_number} has identical "
                f"away/home team={home_team!r}"
            )

        if away_team not in team_map:
            fail(
                f"{path} line {line_number} has unknown "
                f"away_team={away_team!r}"
            )

        if home_team not in team_map:
            fail(
                f"{path} line {line_number} has unknown "
                f"home_team={home_team!r}"
            )

        seen_game_ids.add(game_id)

    return path, rows


def fetch_json(
    url: str,
    *,
    timeout: int = REQUEST_TIMEOUT,
    attempts: int = REQUEST_ATTEMPTS,
) -> tuple[dict[str, Any] | None, int, str]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "football_for_mat/1.0",
        },
    )

    last_error = ""

    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout,
            ) as response:
                raw = response.read()

            try:
                payload = json.loads(
                    raw.decode("utf-8")
                )
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:
                last_error = (
                    "invalid UTF-8/JSON response: "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                if isinstance(payload, dict):
                    return payload, attempt, ""

                last_error = (
                    "response root was not a JSON object"
                )

        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"

            if exc.code not in RETRYABLE_HTTP_CODES:
                return None, attempt, last_error

        except (
            urllib.error.URLError,
            TimeoutError,
        ) as exc:
            last_error = (
                f"{type(exc).__name__}: {exc}"
            )

        except Exception as exc:
            last_error = (
                f"{type(exc).__name__}: {exc}"
            )
            return None, attempt, last_error

        if attempt < attempts:
            time.sleep(2 ** (attempt - 1))

    return None, attempts, last_error


def extract_team_id(ref_url: Any) -> str:
    ref = clean(ref_url)

    if not ref:
        return ""

    match = re.search(
        r"/teams/(\d+)(?:[/?]|$)",
        ref,
    )

    return match.group(1) if match else ""


def validate_numeric(
    value: Any,
    *,
    label: str,
) -> str:
    text = clean(value)

    if not text:
        fail(f"{label} is blank")

    try:
        number = Decimal(text)
    except InvalidOperation:
        fail(
            f"{label} must be numeric; received={text!r}"
        )

    if not number.is_finite():
        fail(
            f"{label} must be finite; received={text!r}"
        )

    return text


def parse_side(
    predictor: dict[str, Any],
    *,
    side: str,
    schedule_row: dict[str, str],
    team_map: dict[str, str],
) -> dict[str, str]:
    game_id = clean(schedule_row.get("game_id"))
    side_data = predictor.get(side)

    if not isinstance(side_data, dict):
        fail(
            f"game_id={game_id} predictor missing "
            f"{side} object"
        )

    team_data = side_data.get("team")
    if not isinstance(team_data, dict):
        fail(
            f"game_id={game_id} {side} missing team object"
        )

    team_id = extract_team_id(
        team_data.get("$ref")
    )

    if not team_id:
        fail(
            f"game_id={game_id} {side} has invalid team $ref"
        )

    expected_team_name = (
        clean(schedule_row.get("home_team"))
        if side == "homeTeam"
        else clean(schedule_row.get("away_team"))
    )
    expected_team_id = team_map[expected_team_name]

    if team_id != expected_team_id:
        fail(
            f"game_id={game_id} {side} team_id mismatch "
            f"expected={expected_team_id} actual={team_id}"
        )

    statistics = side_data.get("statistics")

    if not isinstance(statistics, list):
        fail(
            f"game_id={game_id} {side} statistics "
            "must be a list"
        )

    stats: dict[str, str] = {}

    for index, stat in enumerate(
        statistics,
        start=1,
    ):
        if not isinstance(stat, dict):
            fail(
                f"game_id={game_id} {side} statistic "
                f"index={index} is not an object"
            )

        name = clean(stat.get("name"))

        if not name:
            continue

        if name not in REQUIRED_PREDICTOR_STATS:
            continue

        if name in stats:
            fail(
                f"game_id={game_id} {side} contains duplicate "
                f"required statistic {name!r}"
            )

        stats[name] = validate_numeric(
            stat.get("value"),
            label=(
                f"game_id={game_id} {side} statistic {name}"
            ),
        )

    missing_stats = [
        name
        for name in REQUIRED_PREDICTOR_STATS
        if name not in stats
    ]

    if missing_stats:
        fail(
            f"game_id={game_id} {side} missing required "
            f"predictor statistics: {missing_stats}"
        )

    return {
        "season": clean(schedule_row.get("season")),
        "season_type": clean(
            schedule_row.get("season_type")
        ),
        "week": clean(schedule_row.get("week")),
        "game_id": game_id,
        "game_name": "",
        "home_away": side,
        "team_id": team_id,
        **stats,
    }


def parse_predictor_response(
    predictor: dict[str, Any],
    *,
    schedule_row: dict[str, str],
    team_map: dict[str, str],
) -> list[dict[str, str]]:
    game_id = clean(schedule_row.get("game_id"))
    away_team = clean(
        schedule_row.get("away_team")
    )
    home_team = clean(
        schedule_row.get("home_team")
    )
    expected_game_name = (
        f"{away_team} at {home_team}"
    )
    game_name = clean(predictor.get("name"))

    if not game_name:
        fail(
            f"game_id={game_id} predictor has blank name"
        )

    if game_name != expected_game_name:
        fail(
            f"game_id={game_id} predictor name mismatch "
            f"expected={expected_game_name!r} "
            f"actual={game_name!r}"
        )

    rows = [
        parse_side(
            predictor,
            side=side,
            schedule_row=schedule_row,
            team_map=team_map,
        )
        for side in SIDES
    ]

    for row in rows:
        row["game_name"] = game_name

    return rows


def pull_all_predictions(
    schedule_rows: list[dict[str, str]],
    *,
    team_map: dict[str, str],
    reporter: PipelineReporter,
    log_lines: list[str],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []

    metrics = {
        "games_attempted": 0,
        "games_resolved": 0,
        "games_failed": 0,
        "provider_request_attempts": 0,
    }

    for schedule_row in schedule_rows:
        game_id = clean(
            schedule_row.get("game_id")
        )
        metrics["games_attempted"] += 1

        url = PREDICTOR_URL_TEMPLATE.format(
            game_id=game_id
        )

        payload, attempts, fetch_error = fetch_json(
            url
        )
        metrics[
            "provider_request_attempts"
        ] += attempts

        if payload is None:
            metrics["games_failed"] += 1
            failure = {
                "game_id": game_id,
                "error": fetch_error,
            }
            failures.append(failure)
            log_lines.append(
                f"ERROR: game_id={game_id} predictor "
                f"unavailable after {attempts} attempt(s): "
                f"{fetch_error}"
            )
            continue

        try:
            game_rows = parse_predictor_response(
                payload,
                schedule_row=schedule_row,
                team_map=team_map,
            )
        except Exception as exc:
            metrics["games_failed"] += 1
            failure = {
                "game_id": game_id,
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }
            failures.append(failure)
            log_lines.append(
                f"ERROR: game_id={game_id} predictor "
                f"validation failed: {type(exc).__name__}: "
                f"{exc}"
            )
            continue

        rows.extend(game_rows)
        metrics["games_resolved"] += 1

        if REQUEST_SLEEP_SECONDS:
            time.sleep(REQUEST_SLEEP_SECONDS)

    if failures:
        reporter.error(
            "One or more scheduled games did not produce a "
            "complete validated ESPN predictor response",
            count=len(failures),
            examples=failures[:20],
        )
        fail(
            "ESPN predictor refresh incomplete: "
            f"{len(failures)} of "
            f"{len(schedule_rows)} games failed. "
            "No prediction files were published."
        )

    return rows, metrics


def validate_generation(
    rows: list[dict[str, str]],
    *,
    schedule_rows: list[dict[str, str]],
    team_map: dict[str, str],
) -> None:
    schedule_by_id = {
        clean(row.get("game_id")): row
        for row in schedule_rows
    }

    expected_rows = len(schedule_rows) * 2

    if len(rows) != expected_rows:
        fail(
            "Prediction generation row count mismatch "
            f"expected={expected_rows} actual={len(rows)}"
        )

    by_game: dict[
        str,
        dict[str, dict[str, str]],
    ] = defaultdict(dict)

    for line_number, row in enumerate(rows, start=2):
        for field in OUTPUT_HEADER:
            if not clean(row.get(field)):
                fail(
                    "Prediction row has blank required field "
                    f"line={line_number} field={field}"
                )

        game_id = clean(row.get("game_id"))
        side = clean(row.get("home_away"))

        if side not in SIDES:
            fail(
                f"Prediction line {line_number} has invalid "
                f"home_away={side!r}"
            )

        if game_id not in schedule_by_id:
            fail(
                "Prediction output contains game absent from "
                f"schedule game_id={game_id}"
            )

        if side in by_game[game_id]:
            fail(
                "Prediction output contains duplicate "
                f"(game_id, home_away)=({game_id}, {side})"
            )

        schedule_row = schedule_by_id[game_id]

        expected_target = (
            clean(schedule_row.get("season")),
            clean(schedule_row.get("season_type")),
            clean(schedule_row.get("week")),
        )
        actual_target = (
            clean(row.get("season")),
            clean(row.get("season_type")),
            clean(row.get("week")),
        )

        if actual_target != expected_target:
            fail(
                "Prediction target mismatch "
                f"game_id={game_id} "
                f"expected={expected_target} "
                f"actual={actual_target}"
            )

        expected_name = (
            f"{clean(schedule_row.get('away_team'))} at "
            f"{clean(schedule_row.get('home_team'))}"
        )

        if clean(row.get("game_name")) != expected_name:
            fail(
                "Prediction game_name mismatch "
                f"game_id={game_id}"
            )

        expected_team_name = (
            clean(schedule_row.get("home_team"))
            if side == "homeTeam"
            else clean(schedule_row.get("away_team"))
        )
        expected_team_id = team_map[
            expected_team_name
        ]

        if clean(row.get("team_id")) != expected_team_id:
            fail(
                "Prediction team_id mismatch "
                f"game_id={game_id} side={side}"
            )

        for stat_name in REQUIRED_PREDICTOR_STATS:
            validate_numeric(
                row.get(stat_name),
                label=(
                    f"prediction game_id={game_id} "
                    f"side={side} stat={stat_name}"
                ),
            )

        by_game[game_id][side] = row

    if set(by_game) != set(schedule_by_id):
        missing = sorted(
            set(schedule_by_id) - set(by_game)
        )
        extra = sorted(
            set(by_game) - set(schedule_by_id)
        )
        fail(
            "Prediction game universe mismatch "
            f"missing={missing[:10]} extra={extra[:10]}"
        )

    incomplete = [
        game_id
        for game_id, sides in by_game.items()
        if set(sides) != set(SIDES)
    ]

    if incomplete:
        fail(
            "Prediction games missing a side: "
            f"{incomplete[:10]}"
        )


def group_rows(
    rows: list[dict[str, str]],
) -> dict[
    tuple[str, str, str],
    list[dict[str, str]],
]:
    grouped: dict[
        tuple[str, str, str],
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in rows:
        key = (
            clean(row.get("season")),
            clean(row.get("season_type")),
            clean(row.get("week")),
        )
        grouped[key].append(row)

    return dict(grouped)


def expected_output_paths(
    schedule_rows: list[dict[str, str]],
) -> dict[
    tuple[str, str, str],
    Path,
]:
    paths: dict[
        tuple[str, str, str],
        Path,
    ] = {}

    for row in schedule_rows:
        key = (
            clean(row.get("season")),
            clean(row.get("season_type")),
            clean(row.get("week")),
        )
        season, season_type, week = key

        if key not in paths:
            paths[key] = (
                OUTPUT_DIR
                / (
                    f"{season}_{season_type}_{week}"
                    "_e_predictions.csv"
                )
            )

    return paths


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
            fieldnames=OUTPUT_HEADER,
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def normalize_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            column: clean(row.get(column))
            for column in OUTPUT_HEADER
        }
        for row in rows
    ]


def validate_staged_file(
    path: Path,
    *,
    expected_rows_for_group: list[dict[str, str]],
    schedule_rows_for_group: list[dict[str, str]],
    team_map: dict[str, str],
) -> None:
    staged_header, staged_rows = read_csv(
        path,
        label="staged ESPN prediction file",
        exact_columns=OUTPUT_HEADER,
    )

    if staged_header != OUTPUT_HEADER:
        fail(
            f"Staged prediction schema changed: {path}"
        )

    validate_generation(
        staged_rows,
        schedule_rows=schedule_rows_for_group,
        team_map=team_map,
    )

    if staged_rows != normalize_rows(
        expected_rows_for_group
    ):
        fail(
            "Staged prediction file differs from validated "
            f"in-memory generation: {path}"
        )


def build_staged_root(
    rows: list[dict[str, str]],
    *,
    schedule_rows: list[dict[str, str]],
    season: int,
    team_map: dict[str, str],
) -> tuple[
    Path,
    dict[tuple[str, str, str], Path],
]:
    OUTPUT_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stage_root = Path(
        tempfile.mkdtemp(
            prefix=".e_predictions_stage_",
            dir=OUTPUT_DIR.parent,
        )
    )

    try:
        if OUTPUT_DIR.exists():
            shutil.copytree(
                OUTPUT_DIR,
                stage_root,
                dirs_exist_ok=True,
            )

        for stale in stage_root.glob(
            f"{season}_*_e_predictions.csv"
        ):
            stale.unlink()

        rows_by_group = group_rows(rows)
        output_paths = expected_output_paths(
            schedule_rows
        )

        schedule_by_group: dict[
            tuple[str, str, str],
            list[dict[str, str]],
        ] = defaultdict(list)

        for schedule_row in schedule_rows:
            key = (
                clean(schedule_row.get("season")),
                clean(
                    schedule_row.get("season_type")
                ),
                clean(schedule_row.get("week")),
            )
            schedule_by_group[key].append(
                schedule_row
            )

        if set(rows_by_group) != set(output_paths):
            fail(
                "Prediction output groups do not match schedule "
                "groups"
            )

        staged_output_paths: dict[
            tuple[str, str, str],
            Path,
        ] = {}

        for key, production_path in sorted(
            output_paths.items(),
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
            write_csv(
                staged_path,
                rows_by_group[key],
            )
            validate_staged_file(
                staged_path,
                expected_rows_for_group=rows_by_group[key],
                schedule_rows_for_group=schedule_by_group[
                    key
                ],
                team_map=team_map,
            )
            staged_output_paths[
                key
            ] = staged_path

        expected_names = {
            path.name
            for path in output_paths.values()
        }
        actual_names = {
            path.name
            for path in stage_root.glob(
                f"{season}_*_e_predictions.csv"
            )
        }

        if actual_names != expected_names:
            fail(
                "Staged season prediction file set mismatch "
                f"expected={sorted(expected_names)} "
                f"actual={sorted(actual_names)}"
            )

        return stage_root, staged_output_paths

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
        OUTPUT_DIR.parent
        / (
            f".{OUTPUT_DIR.name}_backup_"
            f"{uuid.uuid4().hex}"
        )
    )
    had_existing = OUTPUT_DIR.exists()

    try:
        if had_existing:
            os.replace(
                OUTPUT_DIR,
                backup_root,
            )

        os.replace(
            stage_root,
            OUTPUT_DIR,
        )

    except Exception:
        if OUTPUT_DIR.exists():
            shutil.rmtree(
                OUTPUT_DIR,
                ignore_errors=True,
            )

        if backup_root.exists():
            os.replace(
                backup_root,
                OUTPUT_DIR,
            )

        raise

    if backup_root.exists():
        try:
            shutil.rmtree(backup_root)
        except Exception as exc:
            reporter.warning(
                "Prediction files were published but the "
                "temporary backup directory could not be removed",
                backup_path=str(backup_root),
                error_type=type(exc).__name__,
                error=str(exc),
            )


def append_legacy_log(
    log_lines: list[str],
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
    ) as handle:
        handle.write(
            f"--- run {timestamp} ---\n"
        )

        for line in log_lines:
            handle.write(line + "\n")

        handle.write("\n")


def safe_append_legacy_log(
    log_lines: list[str],
    *,
    reporter: PipelineReporter,
) -> None:
    try:
        append_legacy_log(log_lines)
        reporter.add_output(ERROR_LOG_PATH)
        reporter.set_detail(
            "legacy_log_written",
            True,
        )
    except Exception as exc:
        reporter.warning(
            "Legacy ESPN prediction log could not be appended",
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
    team_map = load_team_map()
    schedule_path, schedule_rows = load_schedule(
        season=season,
        team_map=team_map,
    )

    output_paths = expected_output_paths(
        schedule_rows
    )

    reporter.add_input(schedule_path)
    reporter.add_input(TEAM_MAP_PATH)
    reporter.update_details(
        {
            "season": season,
            "schedule_path": str(schedule_path),
            "team_map_path": str(TEAM_MAP_PATH),
            "provider_endpoint_template": (
                PREDICTOR_URL_TEMPLATE
            ),
            "scheduled_games": len(schedule_rows),
            "schedule_groups": len(output_paths),
            "expected_prediction_rows": (
                len(schedule_rows) * 2
            ),
            "expected_output_columns": len(
                OUTPUT_HEADER
            ),
            "request_attempts": REQUEST_ATTEMPTS,
            "request_timeout_seconds": REQUEST_TIMEOUT,
            "publication_mode": (
                "validated_directory_swap_with_rollback"
            ),
            "publication_completed": False,
            "staged_roundtrip_verified": False,
        }
    )

    reporter.set_rows(
        rows_in=len(schedule_rows),
        rows_out=0,
    )

    log_lines.append(
        f"season={season}"
    )
    log_lines.append(
        f"schedule_games={len(schedule_rows)}"
    )

    rows, metrics = pull_all_predictions(
        schedule_rows,
        team_map=team_map,
        reporter=reporter,
        log_lines=log_lines,
    )

    validate_generation(
        rows,
        schedule_rows=schedule_rows,
        team_map=team_map,
    )

    rows_by_group = group_rows(rows)

    reporter.update_details(
        {
            **metrics,
            "prediction_rows": len(rows),
            "output_files": len(output_paths),
            "weekly_row_counts": {
                (
                    f"{key[0]}_{key[1]}_{key[2]}"
                ): len(group_rows)
                for key, group_rows
                in sorted(
                    rows_by_group.items(),
                    key=lambda item: (
                        item[0][0],
                        item[0][1],
                        int(item[0][2]),
                    ),
                )
            },
        }
    )

    stage_root: Path | None = None

    try:
        (
            stage_root,
            staged_output_paths,
        ) = build_staged_root(
            rows,
            schedule_rows=schedule_rows,
            season=season,
            team_map=team_map,
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

    for key, production_path in sorted(
        output_paths.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            int(item[0][2]),
        ),
    ):
        reporter.add_output(production_path)
        log_lines.append(
            f"wrote {len(rows_by_group[key])} rows to "
            f"{production_path}"
        )

    reporter.set_rows(
        rows_in=len(schedule_rows),
        rows_out=len(rows),
    )
    reporter.update_details(
        {
            "rows_published": len(rows),
            "files_published": len(output_paths),
            "publication_completed": True,
        }
    )

    summary = (
        f"games_processed={len(schedule_rows)} "
        f"resolved={metrics['games_resolved']} "
        f"failed={metrics['games_failed']} "
        f"files_written={len(output_paths)}"
    )
    print(summary)
    log_lines.append(summary)


def main() -> int:
    args = parse_args()

    try:
        with PipelineReporter(
            script=SCRIPT_PATH,
            stage="00_intake",
            report_root=REPORT_ROOT,
            pipeline="NFL",
            league="NFL",
            season=args.season,
            extra_context={
                "component": "ESPN predictor pull",
                "refresh_scope": "full season",
            },
        ) as reporter:
            log_lines: list[str] = []

            try:
                run(
                    reporter,
                    season=args.season,
                    log_lines=log_lines,
                )
            except Exception as exc:
                log_lines.append(
                    f"ERROR: {type(exc).__name__}: {exc}"
                )
                safe_append_legacy_log(
                    log_lines,
                    reporter=reporter,
                )
                raise

            safe_append_legacy_log(
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
