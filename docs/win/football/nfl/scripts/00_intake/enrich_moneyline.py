#!/usr/bin/env python3
"""
NFL weekly historical moneyline enrichment.

The rule definitions, bucket boundaries, historical rates, and action
directions remain data-driven from moneyline_enrichment.csv.
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
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter

MASTER_PATH = (
    NFL_ROOT
    / "config"
    / "prediction_enrichment"
    / "moneyline_enrichment.csv"
)
SCHEDULE_DIR = (
    NFL_ROOT
    / "00_intake"
    / "schedule"
    / "weekly"
)
EPRED_DIR = (
    NFL_ROOT
    / "00_intake"
    / "predictions"
    / "final"
)
DRAT_DIR = (
    NFL_ROOT
    / "00_intake"
    / "predictions"
    / "drat"
    / "clean"
)
ODDS_DIR = NFL_ROOT / "00_intake" / "odds"
OUTPUT_DIR = (
    NFL_ROOT
    / "00_intake"
    / "predictions"
    / "enriched"
    / "moneyline"
)
REPORT_ROOT = NFL_ROOT / "errors"

WEEKLY_FILENAME_RE = re.compile(
    r"week_(\d+)_NFL_weekly_schedule\.csv"
)

WEEKLY_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "odds_provider_game_id",
    "game_date",
    "game_time",
    "commence_time",
    "away_team",
    "home_team",
    "odds_away_team",
    "odds_home_team",
    "neutral_site",
    "stadium",
    "roof",
    "surface",
    "home_timezone",
    "away_timezone",
    "game_timezone",
    "bookmaker",
    "home_moneyline_american",
    "away_moneyline_american",
    "home_spread",
    "away_spread",
    "home_spread_american",
    "away_spread_american",
    "total",
    "over_american",
    "under_american",
    "odds_last_update",
    "odds_available",
    "odds_missing_reason",
]

DRAT_HEADERS = [
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

EPRED_HEADERS = [
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

ODDS_HEADERS = [
    "snapshot_id",
    "snapshot_fetched_at",
    "game_id",
    "commence_time",
    "home_team",
    "away_team",
    "bookmaker",
    "market_type",
    "bet_side",
    "line",
    "odds_american",
    "odds_decimal",
    "last_update",
    "home_moneyline_american",
    "away_moneyline_american",
    "home_spread",
    "away_spread",
    "home_spread_american",
    "away_spread_american",
    "total",
    "over_american",
    "under_american",
]

EXPECTED_MARKET_SIDES = {
    ("h2h", "home"),
    ("h2h", "away"),
    ("spreads", "home"),
    ("spreads", "away"),
    ("totals", "over"),
    ("totals", "under"),
}

MASTER_REQUIRED_HEADERS = [
    "rule_id",
    "active",
    "pipeline_supported",
    "family",
    "source_condition",
    "condition_count",
    "condition_1_test_feature",
    "condition_1_formula_code",
    "condition_1_match_type",
    "condition_1_min_inclusive",
    "condition_1_max_exclusive",
    "condition_1_equals_value",
    "condition_2_test_feature",
    "condition_2_formula_code",
    "condition_2_match_type",
    "condition_2_min_inclusive",
    "condition_2_max_exclusive",
    "condition_2_equals_value",
    "games",
    "historical_win_rate_pct",
    "lift_vs_family_pct_points",
    "action_direction",
]

SUPPORTED_FAMILIES = {
    "DRAT",
    "EPRED",
    "MARKET",
    "DRAT_EPRED_CONSENSUS",
    "ALL3_CONSENSUS",
}

SUPPORTED_FORMULAS = {
    "USE_FAMILY_SELECTED_PROB",
    "MARKET_ROLE_FOR_FAMILY_SELECTED_SIDE",
    "SPREAD_FOR_FAMILY_SELECTED_SIDE",
    "EPRED_RATING_SELECTED_MINUS_OPPONENT",
    "RAW_EPRED_MATCHUP_QUALITY",
    "RAW_WEEK",
    "RAW_MARKET_TOTAL",
    "COMPARE_DRAT_PICK_TO_EPRED_PICK",
    "COMPARE_FAMILY_PICK_TO_MARKET_PICK",
    "ABS_DRAT_HOME_PROB_MINUS_EPRED_NORMALIZED_HOME_PROB_X100",
    "FAMILY_SELECTED_PROB_MINUS_MARKET_SELECTED_PROB_X100",
    "UNAVAILABLE",
}

SUPPORTED_MATCH_TYPES = {
    "IS_NULL",
    "TEXT_EQUALS",
    "NUMERIC_RANGE",
}

APPENDED_FIELDS = [
    "drat_home_prob",
    "drat_away_prob",
    "epred_home_prob_raw",
    "epred_away_prob_raw",
    "epred_home_prob",
    "epred_away_prob",
    "epred_home_rating",
    "epred_away_rating",
    "epred_matchupQuality",
    "market_bookmaker",
    "market_last_update",
    "market_home_moneyline_american",
    "market_away_moneyline_american",
    "market_home_spread",
    "market_away_spread",
    "market_total",
    "market_home_prob_novig",
    "market_away_prob_novig",
    "drat_pick",
    "epred_pick",
    "market_pick",
    "drat_epred_agree",
    "drat_market_agree",
    "epred_market_agree",
    "all_three_agree",
    "epred_rating_gap_home",
    "drat_epred_prob_diff_pp",
    "drat_market_edge_home_pp",
    "epred_market_edge_home_pp",
    "matched_rule_count",
    "matched_positive_rule_count",
    "matched_negative_rule_count",
    "matched_rule_ids",
    "matched_rule_conditions",
    "home_matched_rule_count",
    "home_matched_rule_ids",
    "home_strongest_positive_rule_id",
    "home_strongest_positive_hist_win_rate_pct",
    "home_strongest_positive_lift_pp",
    "home_strongest_positive_games",
    "home_strongest_negative_rule_id",
    "home_strongest_negative_hist_win_rate_pct",
    "home_strongest_negative_lift_pp",
    "home_strongest_negative_games",
    "away_matched_rule_count",
    "away_matched_rule_ids",
    "away_strongest_positive_rule_id",
    "away_strongest_positive_hist_win_rate_pct",
    "away_strongest_positive_lift_pp",
    "away_strongest_positive_games",
    "away_strongest_negative_rule_id",
    "away_strongest_negative_hist_win_rate_pct",
    "away_strongest_negative_lift_pp",
    "away_strongest_negative_games",
    "drat_matched_rule_count",
    "drat_matched_rule_ids",
    "epred_matched_rule_count",
    "epred_matched_rule_ids",
    "market_matched_rule_count",
    "market_matched_rule_ids",
    "drat_epred_consensus_matched_rule_count",
    "drat_epred_consensus_matched_rule_ids",
    "all3_consensus_matched_rule_count",
    "all3_consensus_matched_rule_ids",
]

OUTPUT_HEADERS = WEEKLY_COLUMNS + [
    column
    for column in APPENDED_FIELDS
    if column not in WEEKLY_COLUMNS
]


class MoneylineEnrichmentError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise MoneylineEnrichmentError(message)


def s(value: Any) -> str:
    return "" if value is None else str(value).strip()


def num(value: Any):
    text = s(value)
    if text == "":
        return None
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def require_finite_number(
    value: Any,
    *,
    label: str,
) -> float:
    parsed = num(value)
    if parsed is None:
        fail(
            f"{label} must be a finite numeric value; "
            f"received={s(value)!r}"
        )
    return parsed


def parse_int_text(
    value: Any,
    *,
    label: str,
) -> int:
    text = s(value)
    if not text:
        fail(f"{label} is blank")

    try:
        parsed = float(text)
    except ValueError:
        fail(
            f"{label} must be an integer; received={text!r}"
        )

    if (
        not math.isfinite(parsed)
        or not parsed.is_integer()
    ):
        fail(
            f"{label} must be an integer; received={text!r}"
        )

    return int(parsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build NFL historical moneyline enrichment for "
            "the requested season's available weekly schedules."
        )
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
    )
    args = parser.parse_args()

    if args.season < 2000 or args.season > 2100:
        parser.error(
            "--season must be between 2000 and 2100"
        )

    return args


def read_csv_table(
    path: Path,
) -> tuple[
    list[str],
    list[dict[str, str]],
]:
    if not path.is_file():
        fail(f"Input file not found: {path}")

    if path.stat().st_size == 0:
        fail(f"Input file is zero bytes: {path}")

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            rows = list(reader)
    except Exception as exc:
        fail(
            f"Could not read {path}: "
            f"{type(exc).__name__}: {exc}"
        )

    if not headers:
        fail(f"CSV has no header: {path}")

    if len(headers) != len(set(headers)):
        duplicates = sorted(
            {
                column
                for column in headers
                if headers.count(column) > 1
            }
        )
        fail(
            f"CSV contains duplicate columns {duplicates}: "
            f"{path}"
        )

    return headers, rows


def read_csv(path: Path) -> list[dict[str, str]]:
    return read_csv_table(path)[1]


def require_exact_headers(
    headers: list[str],
    expected: list[str],
    *,
    label: str,
) -> None:
    if headers != expected:
        fail(
            f"{label} schema/order mismatch. "
            f"Expected={expected} actual={headers}"
        )


def require_columns(
    headers: list[str],
    required: list[str],
    *,
    label: str,
) -> None:
    missing = [
        column
        for column in required
        if column not in headers
    ]
    if missing:
        fail(
            f"{label} is missing required columns: "
            + ", ".join(missing)
        )


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def same_text(a: Any, b: Any) -> bool:
    return s(a).casefold() == s(b).casefold()


def team_key(value: Any) -> str:
    return " ".join(
        s(value).casefold().split()
    )


def game_team_key(
    season: Any,
    week: Any,
    home: Any,
    away: Any,
):
    return (
        str(
            parse_int_text(
                season,
                label="game-team-key season",
            )
        ),
        str(
            parse_int_text(
                week,
                label="game-team-key week",
            )
        ),
        team_key(home),
        team_key(away),
    )


def parse_iso_dt(
    value: Any,
) -> datetime | None:
    text = s(value)
    if not text:
        return None

    try:
        return datetime.fromisoformat(
            text.replace("Z", "+00:00")
        ).replace(tzinfo=None)
    except ValueError:
        return None


def iso_dt(value: Any) -> datetime:
    parsed = parse_iso_dt(value)
    return (
        parsed
        if parsed is not None
        else datetime.min
    )


def american_implied(odds: Any):
    value = num(odds)
    if value is None or value == 0:
        return None
    if value > 0:
        return 100.0 / (value + 100.0)
    return (-value) / ((-value) + 100.0)


def no_vig_probs(
    home_ml: Any,
    away_ml: Any,
):
    home = american_implied(home_ml)
    away = american_implied(away_ml)

    if home is None or away is None:
        return None, None

    total = home + away
    if total <= 0:
        return None, None

    return home / total, away / total


def schedule_identity(
    rows: list[dict[str, str]],
    path: Path,
) -> tuple[int, str, int]:
    if not rows:
        fail(
            f"Weekly schedule contains no data rows: {path}"
        )

    values: set[tuple[str, str, str]] = set()

    for line_number, row in enumerate(
        rows,
        start=2,
    ):
        season = s(row.get("season"))
        season_type = s(row.get("season_type"))
        week = s(row.get("week"))

        if not season or not season_type or not week:
            fail(
                f"{path} line {line_number} has blank "
                "season/season_type/week"
            )

        values.add(
            (
                season,
                season_type,
                week,
            )
        )

    if len(values) != 1:
        fail(
            f"{path.name}: expected exactly one "
            "season/season_type/week combination, "
            f"found {sorted(values)}"
        )

    season_text, season_type, week_text = next(
        iter(values)
    )

    season = parse_int_text(
        season_text,
        label=f"{path.name} season",
    )
    week = parse_int_text(
        week_text,
        label=f"{path.name} week",
    )

    if season_type not in {
        "pre",
        "reg",
        "post",
    }:
        fail(
            f"{path.name}: unsupported "
            f"season_type={season_type!r}"
        )

    if week < 1 or week > 25:
        fail(
            f"{path.name}: week outside 1..25: {week}"
        )

    return season, season_type, week


def validate_weekly_schedule(
    rows: list[dict[str, str]],
    *,
    path: Path,
    season: int,
    season_type: str,
    week: int,
) -> None:
    match = WEEKLY_FILENAME_RE.fullmatch(
        path.name
    )
    if match is None:
        fail(
            f"Unexpected weekly schedule filename: {path}"
        )

    filename_week = int(match.group(1))
    if filename_week != week:
        fail(
            f"{path.name}: filename week={filename_week} "
            f"but row week={week}"
        )

    seen_ids: set[str] = set()

    for line_number, row in enumerate(
        rows,
        start=2,
    ):
        target = (
            parse_int_text(
                row.get("season"),
                label=(
                    f"{path.name} line {line_number} season"
                ),
            ),
            s(row.get("season_type")),
            parse_int_text(
                row.get("week"),
                label=(
                    f"{path.name} line {line_number} week"
                ),
            ),
        )

        if target != (
            season,
            season_type,
            week,
        ):
            fail(
                f"{path.name} line {line_number} "
                f"target={target}; expected="
                f"{(season, season_type, week)}"
            )

        game_id = s(row.get("game_id"))
        home_team = s(row.get("home_team"))
        away_team = s(row.get("away_team"))

        if not game_id:
            fail(
                f"{path.name} line {line_number} "
                "has blank game_id"
            )

        if game_id in seen_ids:
            fail(
                f"{path.name} contains duplicate "
                f"game_id={game_id}"
            )
        seen_ids.add(game_id)

        if (
            not home_team
            or not away_team
            or same_text(
                home_team,
                away_team,
            )
        ):
            fail(
                f"{path.name} game_id={game_id} has "
                "invalid home/away team identity"
            )

        odds_available = s(
            row.get("odds_available")
        )

        if odds_available not in {"0", "1"}:
            fail(
                f"{path.name} game_id={game_id} has "
                f"invalid odds_available={odds_available!r}"
            )


def load_target_schedules(
    *,
    season: int,
    reporter: PipelineReporter,
) -> dict[
    int,
    tuple[
        Path,
        str,
        list[dict[str, str]],
    ],
]:
    if not SCHEDULE_DIR.is_dir():
        fail(
            f"Weekly schedule directory not found: "
            f"{SCHEDULE_DIR}"
        )

    schedule_paths = sorted(
        SCHEDULE_DIR.glob(
            "week_*_NFL_weekly_schedule.csv"
        )
    )

    if not schedule_paths:
        fail(
            "No weekly schedule files found in "
            f"{SCHEDULE_DIR}"
        )

    target: dict[
        int,
        tuple[
            Path,
            str,
            list[dict[str, str]],
        ],
    ] = {}

    for path in schedule_paths:
        headers, rows = read_csv_table(path)
        require_exact_headers(
            headers,
            WEEKLY_COLUMNS,
            label=f"weekly schedule {path.name}",
        )

        row_season, season_type, week = (
            schedule_identity(
                rows,
                path,
            )
        )

        if row_season != season:
            continue

        validate_weekly_schedule(
            rows,
            path=path,
            season=season,
            season_type=season_type,
            week=week,
        )

        if week in target:
            fail(
                f"More than one target-season weekly "
                f"schedule exists for week={week}"
            )

        target[week] = (
            path,
            season_type,
            rows,
        )
        reporter.add_input(path)

    if not target:
        fail(
            f"No weekly schedules found for season={season}"
        )

    return target


def load_drat(
    *,
    season: int,
    week: int,
    schedule_rows: list[dict[str, str]],
    reporter: PipelineReporter,
) -> tuple[
    Path,
    list[dict[str, str]],
    dict[
        tuple[str, str, str, str],
        dict[str, str],
    ],
]:
    path = (
        DRAT_DIR
        / f"{season}_week_{week}_drat.csv"
    )
    headers, rows = read_csv_table(path)
    require_exact_headers(
        headers,
        DRAT_HEADERS,
        label=f"DRAT {path.name}",
    )
    reporter.add_input(path)

    schedule_by_id = {
        s(row.get("game_id")): row
        for row in schedule_rows
    }

    seen_ids: set[str] = set()
    by_teams: dict[
        tuple[str, str, str, str],
        dict[str, str],
    ] = {}

    for line_number, row in enumerate(
        rows,
        start=2,
    ):
        row_season = parse_int_text(
            row.get("season"),
            label=(
                f"{path.name} line {line_number} season"
            ),
        )
        row_week = parse_int_text(
            row.get("week"),
            label=(
                f"{path.name} line {line_number} week"
            ),
        )

        if (
            row_season != season
            or row_week != week
        ):
            fail(
                f"{path.name} line {line_number} "
                f"target={(row_season, row_week)}; "
                f"expected={(season, week)}"
            )

        game_id = s(row.get("game_id"))
        if not game_id:
            fail(
                f"{path.name} line {line_number} "
                "has blank game_id"
            )

        if game_id in seen_ids:
            fail(
                f"{path.name} contains duplicate "
                f"game_id={game_id}"
            )
        seen_ids.add(game_id)

        schedule_row = schedule_by_id.get(
            game_id
        )
        if schedule_row is None:
            fail(
                f"{path.name} contains unexpected "
                f"game_id={game_id}"
            )

        for field in (
            "home_team",
            "away_team",
        ):
            if not same_text(
                row.get(field),
                schedule_row.get(field),
            ):
                fail(
                    f"{path.name} game_id={game_id} "
                    f"{field} does not match weekly schedule"
                )

        home_prob = require_finite_number(
            row.get("home_prob"),
            label=(
                f"{path.name} game_id={game_id} "
                "home_prob"
            ),
        )
        away_prob = require_finite_number(
            row.get("away_prob"),
            label=(
                f"{path.name} game_id={game_id} "
                "away_prob"
            ),
        )

        if (
            home_prob < 0
            or home_prob > 1
            or away_prob < 0
            or away_prob > 1
            or home_prob + away_prob <= 0
        ):
            fail(
                f"{path.name} game_id={game_id} "
                "has invalid DRAT probabilities"
            )

        key = game_team_key(
            row.get("season"),
            row.get("week"),
            row.get("home_team"),
            row.get("away_team"),
        )

        if key in by_teams:
            fail(
                f"{path.name} contains duplicate "
                f"DRAT team key={key}"
            )

        by_teams[key] = row

    expected_ids = set(schedule_by_id)
    if seen_ids != expected_ids:
        fail(
            f"{path.name} DRAT/schedule game universe "
            f"mismatch missing="
            f"{sorted(expected_ids - seen_ids)} "
            f"extra={sorted(seen_ids - expected_ids)}"
        )

    return path, rows, by_teams


def load_epred(
    *,
    season: int,
    season_type: str,
    week: int,
    schedule_rows: list[dict[str, str]],
    reporter: PipelineReporter,
) -> tuple[
    Path,
    list[dict[str, str]],
    dict[str, dict[str, str]],
]:
    path = (
        EPRED_DIR
        / (
            f"{season}_{season_type}_{week}"
            "_clean_predictions.csv"
        )
    )
    headers, rows = read_csv_table(path)
    require_exact_headers(
        headers,
        EPRED_HEADERS,
        label=f"EPRED {path.name}",
    )
    reporter.add_input(path)

    schedule_by_id = {
        s(row.get("game_id")): row
        for row in schedule_rows
    }

    by_game: dict[
        str,
        dict[str, str],
    ] = {}

    for line_number, row in enumerate(
        rows,
        start=2,
    ):
        target = (
            parse_int_text(
                row.get("season"),
                label=(
                    f"{path.name} line {line_number} season"
                ),
            ),
            s(row.get("season_type")),
            parse_int_text(
                row.get("week"),
                label=(
                    f"{path.name} line {line_number} week"
                ),
            ),
        )

        if target != (
            season,
            season_type,
            week,
        ):
            fail(
                f"{path.name} line {line_number} "
                f"target={target}; expected="
                f"{(season, season_type, week)}"
            )

        game_id = s(row.get("game_id"))
        if not game_id:
            fail(
                f"{path.name} line {line_number} "
                "has blank game_id"
            )

        if game_id in by_game:
            fail(
                f"{path.name} contains duplicate "
                f"game_id={game_id}"
            )

        schedule_row = schedule_by_id.get(
            game_id
        )
        if schedule_row is None:
            fail(
                f"{path.name} contains unexpected "
                f"game_id={game_id}"
            )

        for field in (
            "home_team",
            "away_team",
        ):
            if not same_text(
                row.get(field),
                schedule_row.get(field),
            ):
                fail(
                    f"{path.name} game_id={game_id} "
                    f"{field} does not match weekly schedule"
                )

        home_prob = require_finite_number(
            row.get("home_prob"),
            label=(
                f"{path.name} game_id={game_id} "
                "home_prob"
            ),
        )
        away_prob = require_finite_number(
            row.get("away_prob"),
            label=(
                f"{path.name} game_id={game_id} "
                "away_prob"
            ),
        )

        if (
            home_prob < 0
            or home_prob > 1
            or away_prob < 0
            or away_prob > 1
            or home_prob + away_prob <= 0
        ):
            fail(
                f"{path.name} game_id={game_id} "
                "has invalid EPRED probabilities"
            )

        for field in (
            "home_rating",
            "away_rating",
            "matchupQuality",
        ):
            require_finite_number(
                row.get(field),
                label=(
                    f"{path.name} game_id={game_id} "
                    f"{field}"
                ),
            )

        by_game[game_id] = row

    expected_ids = set(schedule_by_id)
    actual_ids = set(by_game)

    if actual_ids != expected_ids:
        fail(
            f"{path.name} EPRED/schedule game universe "
            f"mismatch missing="
            f"{sorted(expected_ids - actual_ids)} "
            f"extra={sorted(actual_ids - expected_ids)}"
        )

    return path, rows, by_game


def validate_master(
    headers: list[str],
    rows: list[dict[str, str]],
) -> set[str]:
    require_columns(
        headers,
        MASTER_REQUIRED_HEADERS,
        label="moneyline enrichment master",
    )

    if not rows:
        fail(
            "Moneyline enrichment master contains no rows"
        )

    seen_ids: set[str] = set()
    active_supported_ids: set[str] = set()

    for line_number, rule in enumerate(
        rows,
        start=2,
    ):
        rule_id = s(rule.get("rule_id"))
        if not rule_id:
            fail(
                f"Moneyline master line {line_number} "
                "has blank rule_id"
            )

        if rule_id in seen_ids:
            fail(
                f"Moneyline master contains duplicate "
                f"rule_id={rule_id}"
            )
        seen_ids.add(rule_id)

        active = s(rule.get("active"))
        supported = s(
            rule.get("pipeline_supported")
        )

        if active not in {"0", "1"}:
            fail(
                f"Moneyline master rule_id={rule_id} "
                f"has invalid active={active!r}"
            )

        if supported not in {"0", "1"}:
            fail(
                f"Moneyline master rule_id={rule_id} "
                "has invalid pipeline_supported="
                f"{supported!r}"
            )

        if active != "1" or supported != "1":
            continue

        active_supported_ids.add(rule_id)

        family = s(rule.get("family"))
        if family not in SUPPORTED_FAMILIES:
            fail(
                f"Moneyline master rule_id={rule_id} "
                f"has unsupported family={family!r}"
            )

        condition_count = parse_int_text(
            rule.get("condition_count"),
            label=(
                f"Moneyline master rule_id={rule_id} "
                "condition_count"
            ),
        )

        if condition_count < 0 or condition_count > 2:
            fail(
                f"Moneyline master rule_id={rule_id} "
                "condition_count must be between 0 and 2"
            )

        direction = s(
            rule.get("action_direction")
        )
        if direction not in {
            "POSITIVE",
            "NEGATIVE",
        }:
            fail(
                f"Moneyline master rule_id={rule_id} "
                f"has invalid action_direction={direction!r}"
            )

        if not s(rule.get("source_condition")):
            fail(
                f"Moneyline master rule_id={rule_id} "
                "has blank source_condition"
            )

        for metric in (
            "historical_win_rate_pct",
            "lift_vs_family_pct_points",
            "games",
        ):
            require_finite_number(
                rule.get(metric),
                label=(
                    f"Moneyline master rule_id={rule_id} "
                    f"{metric}"
                ),
            )

        for number in range(
            1,
            condition_count + 1,
        ):
            prefix = f"condition_{number}_"
            formula = s(
                rule.get(
                    prefix + "formula_code"
                )
            )
            match_type = s(
                rule.get(
                    prefix + "match_type"
                )
            )

            if formula not in SUPPORTED_FORMULAS:
                fail(
                    f"Moneyline master rule_id={rule_id} "
                    f"has unsupported formula_code={formula!r}"
                )

            if (
                match_type
                not in SUPPORTED_MATCH_TYPES
            ):
                fail(
                    f"Moneyline master rule_id={rule_id} "
                    f"has unsupported match_type={match_type!r}"
                )

            if match_type == "NUMERIC_RANGE":
                lower = s(
                    rule.get(
                        prefix + "min_inclusive"
                    )
                )
                upper = s(
                    rule.get(
                        prefix + "max_exclusive"
                    )
                )

                if not lower and not upper:
                    fail(
                        f"Moneyline master rule_id={rule_id} "
                        f"condition {number} numeric range "
                        "has no bound"
                    )

                if lower:
                    require_finite_number(
                        lower,
                        label=(
                            f"Moneyline master rule_id={rule_id} "
                            f"condition {number} lower bound"
                        ),
                    )

                if upper:
                    require_finite_number(
                        upper,
                        label=(
                            f"Moneyline master rule_id={rule_id} "
                            f"condition {number} upper bound"
                        ),
                    )

            if (
                match_type == "TEXT_EQUALS"
                and not s(
                    rule.get(
                        prefix + "equals_value"
                    )
                )
            ):
                fail(
                    f"Moneyline master rule_id={rule_id} "
                    f"condition {number} has blank equals_value"
                )

    if not active_supported_ids:
        fail(
            "Moneyline enrichment master has no active "
            "pipeline-supported rules"
        )

    return active_supported_ids


def select_latest_odds_file(
    *,
    reporter: PipelineReporter,
) -> tuple[Path, int]:
    if not ODDS_DIR.is_dir():
        fail(
            f"Odds directory not found: {ODDS_DIR}"
        )

    paths = sorted(
        ODDS_DIR.glob("*_NFL_odds.csv")
    )

    if not paths:
        fail(
            f"No *_NFL_odds.csv files found in "
            f"{ODDS_DIR}"
        )

    candidates: list[
        tuple[datetime, str, Path]
    ] = []
    skipped = 0

    for path in paths:
        try:
            headers, rows = read_csv_table(path)
        except Exception as exc:
            skipped += 1
            reporter.warning(
                "Skipped unreadable odds compatibility file "
                "while selecting the latest capture",
                path=str(path),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            continue

        if "last_update" not in headers:
            skipped += 1
            reporter.warning(
                "Skipped odds compatibility file without "
                "last_update while selecting latest capture",
                path=str(path),
            )
            continue

        valid_updates = [
            parsed
            for parsed in (
                parse_iso_dt(
                    row.get("last_update")
                )
                for row in rows
            )
            if parsed is not None
        ]

        if not valid_updates:
            skipped += 1
            reporter.warning(
                "Skipped odds compatibility file with no "
                "valid last_update while selecting latest capture",
                path=str(path),
            )
            continue

        candidates.append(
            (
                max(valid_updates),
                path.name,
                path,
            )
        )

    if not candidates:
        fail(
            "No direct-child *_NFL_odds.csv file contains "
            "a valid last_update value"
        )

    candidates.sort()
    return candidates[-1][2], skipped


def validate_selected_odds(
    *,
    path: Path,
    headers: list[str],
    rows: list[dict[str, str]],
) -> None:
    require_exact_headers(
        headers,
        ODDS_HEADERS,
        label=f"selected current odds {path.name}",
    )

    if not rows:
        fail(
            f"Selected current odds file has no rows: "
            f"{path}"
        )

    snapshot_ids = {
        s(row.get("snapshot_id"))
        for row in rows
    }
    fetched_values = {
        s(row.get("snapshot_fetched_at"))
        for row in rows
    }

    if (
        "" in snapshot_ids
        or len(snapshot_ids) != 1
    ):
        fail(
            f"{path.name} must contain exactly one "
            "nonblank snapshot_id"
        )

    if (
        "" in fetched_values
        or len(fetched_values) != 1
    ):
        fail(
            f"{path.name} must contain exactly one "
            "nonblank snapshot_fetched_at"
        )

    fetched = next(iter(fetched_values))
    if parse_iso_dt(fetched) is None:
        fail(
            f"{path.name} has invalid "
            f"snapshot_fetched_at={fetched!r}"
        )

    seen_keys: set[
        tuple[str, str, str, str]
    ] = set()
    market_pairs_by_group: dict[
        tuple[str, str],
        set[tuple[str, str]],
    ] = {}

    numeric_fields = (
        "home_moneyline_american",
        "away_moneyline_american",
        "home_spread",
        "away_spread",
        "home_spread_american",
        "away_spread_american",
        "total",
        "over_american",
        "under_american",
    )

    for line_number, row in enumerate(
        rows,
        start=2,
    ):
        game_id = s(row.get("game_id"))
        bookmaker = s(row.get("bookmaker"))
        market_type = s(row.get("market_type"))
        bet_side = s(row.get("bet_side"))
        home_team = s(row.get("home_team"))
        away_team = s(row.get("away_team"))
        last_update = s(row.get("last_update"))

        if (
            not game_id
            or not bookmaker
            or not market_type
            or not bet_side
            or not home_team
            or not away_team
        ):
            fail(
                f"{path.name} line {line_number} has "
                "blank required odds identity field"
            )

        pair = (
            market_type,
            bet_side,
        )
        if pair not in EXPECTED_MARKET_SIDES:
            fail(
                f"{path.name} game_id={game_id} "
                f"has invalid market pair={pair}"
            )

        if parse_iso_dt(last_update) is None:
            fail(
                f"{path.name} game_id={game_id} "
                f"has invalid last_update={last_update!r}"
            )

        key = (
            game_id,
            bookmaker.casefold(),
            market_type,
            bet_side,
        )

        if key in seen_keys:
            fail(
                f"{path.name} contains duplicate "
                f"market row key={key}"
            )
        seen_keys.add(key)

        group = (
            game_id,
            bookmaker.casefold(),
        )
        market_pairs_by_group.setdefault(
            group,
            set(),
        ).add(pair)

        for field in numeric_fields:
            text = s(row.get(field))
            if text:
                require_finite_number(
                    text,
                    label=(
                        f"{path.name} game_id={game_id} "
                        f"{field}"
                    ),
                )

    for group, pairs in (
        market_pairs_by_group.items()
    ):
        if pairs != EXPECTED_MARKET_SIDES:
            fail(
                f"{path.name} group={group} has "
                f"market pairs={sorted(pairs)}; expected="
                f"{sorted(EXPECTED_MARKET_SIDES)}"
            )


def aggregate_latest_odds(
    rows: list[dict[str, str]],
):
    fields = [
        "home_moneyline_american",
        "away_moneyline_american",
        "home_spread",
        "away_spread",
        "home_spread_american",
        "away_spread_american",
        "total",
        "over_american",
        "under_american",
    ]
    groups = {}

    for row in rows:
        game_id = s(row.get("game_id"))
        bookmaker = s(row.get("bookmaker"))

        if not game_id:
            continue

        key = (
            game_id,
            bookmaker.casefold(),
        )
        group = groups.setdefault(
            key,
            {
                "game_id": game_id,
                "bookmaker": bookmaker,
                "last_update": "",
                "__last_dt": datetime.min,
                "__field_dt": {
                    field: datetime.min
                    for field in fields
                },
            },
        )

        updated = iso_dt(
            row.get("last_update")
        )

        if updated >= group["__last_dt"]:
            group["__last_dt"] = updated
            group["last_update"] = s(
                row.get("last_update")
            )

        for field in fields:
            value = s(row.get(field))
            if (
                value != ""
                and updated
                >= group["__field_dt"][field]
            ):
                group[field] = value
                group["__field_dt"][field] = (
                    updated
                )

    return groups


def choose_odds_record(
    groups,
    provider_game_id: Any,
    preferred_bookmaker: Any,
):
    game_id = s(provider_game_id)
    if not game_id:
        return None

    preferred = s(
        preferred_bookmaker
    ).casefold()

    if (
        preferred
        and (
            game_id,
            preferred,
        )
        in groups
    ):
        return groups[
            (
                game_id,
                preferred,
            )
        ]

    matches = [
        group
        for (
            candidate_game_id,
            _,
        ), group in groups.items()
        if candidate_game_id == game_id
    ]

    if not matches:
        return None

    return max(
        matches,
        key=lambda group: group["__last_dt"],
    )


def build_family_contexts(g):
    contexts = {}

    dh = num(g.get("drat_home_prob"))
    da = num(g.get("drat_away_prob"))
    eh = num(g.get("epred_home_prob"))
    ea = num(g.get("epred_away_prob"))
    mh = num(g.get("market_home_prob_novig"))
    ma = num(g.get("market_away_prob_novig"))
    hs = num(g.get("market_home_spread"))
    aws = num(g.get("market_away_spread"))

    drat_side = None
    drat_prob = None
    if dh is not None and da is not None:
        drat_side = (
            "Home"
            if dh >= da
            else "Away"
        )
        drat_prob = (
            dh
            if drat_side == "Home"
            else da
        )

    epred_side = None
    epred_prob = None
    if eh is not None and ea is not None:
        epred_side = (
            "Home"
            if eh >= ea
            else "Away"
        )
        epred_prob = (
            eh
            if epred_side == "Home"
            else ea
        )

    market_side = None
    market_prob = None
    if mh is not None and ma is not None:
        market_side = (
            "Home"
            if mh >= ma
            else "Away"
        )
        market_prob = (
            mh
            if market_side == "Home"
            else ma
        )
    elif hs is not None and aws is not None:
        if hs < 0:
            market_side = "Home"
        elif aws < 0:
            market_side = "Away"

    contexts["DRAT"] = {
        "eligible": drat_side is not None,
        "side": drat_side,
        "prob": drat_prob,
    }
    contexts["EPRED"] = {
        "eligible": epred_side is not None,
        "side": epred_side,
        "prob": epred_prob,
    }
    contexts["MARKET"] = {
        "eligible": market_side is not None,
        "side": market_side,
        "prob": market_prob,
    }

    de_ok = (
        drat_side is not None
        and epred_side is not None
        and drat_side == epred_side
    )
    contexts["DRAT_EPRED_CONSENSUS"] = {
        "eligible": de_ok,
        "side": (
            drat_side
            if de_ok
            else None
        ),
        "prob": (
            (drat_prob + epred_prob) / 2.0
            if (
                de_ok
                and drat_prob is not None
                and epred_prob is not None
            )
            else None
        ),
    }

    all_ok = (
        de_ok
        and market_side is not None
        and drat_side == market_side
    )
    contexts["ALL3_CONSENSUS"] = {
        "eligible": all_ok,
        "side": (
            drat_side
            if all_ok
            else None
        ),
        "prob": (
            (
                drat_prob
                + epred_prob
                + market_prob
            )
            / 3.0
            if (
                all_ok
                and market_prob is not None
            )
            else None
        ),
    }

    g["drat_pick_side"] = (
        drat_side or ""
    )
    g["epred_pick_side"] = (
        epred_side or ""
    )
    g["market_pick_side"] = (
        market_side or ""
    )

    g["drat_pick"] = (
        g["home_team"]
        if drat_side == "Home"
        else (
            g["away_team"]
            if drat_side == "Away"
            else ""
        )
    )
    g["epred_pick"] = (
        g["home_team"]
        if epred_side == "Home"
        else (
            g["away_team"]
            if epred_side == "Away"
            else ""
        )
    )
    g["market_pick"] = (
        g["home_team"]
        if market_side == "Home"
        else (
            g["away_team"]
            if market_side == "Away"
            else ""
        )
    )

    def agreement(a, b):
        if a is None or b is None:
            return "Unknown"
        return (
            "Agree"
            if a == b
            else "Disagree"
        )

    g["drat_epred_agree"] = agreement(
        drat_side,
        epred_side,
    )
    g["drat_market_agree"] = agreement(
        drat_side,
        market_side,
    )
    g["epred_market_agree"] = agreement(
        epred_side,
        market_side,
    )
    g["all_three_agree"] = (
        "Yes"
        if all_ok
        else (
            "No"
            if (
                drat_side
                and epred_side
                and market_side
            )
            else "Unknown"
        )
    )

    return contexts


def market_role_for_side(g, side):
    if side not in ("Home", "Away"):
        return None

    probability = num(
        g.get(
            "market_home_prob_novig"
            if side == "Home"
            else "market_away_prob_novig"
        )
    )

    if probability is not None:
        if probability > 0.5:
            return "Market Favorite"
        if probability < 0.5:
            return "Market Underdog"
        return "Market Even"

    spread = num(
        g.get(
            "market_home_spread"
            if side == "Home"
            else "market_away_spread"
        )
    )

    if spread is None:
        return None
    if spread < 0:
        return "Market Favorite"
    if spread > 0:
        return "Market Underdog"
    return "Market Even"


def feature_value(
    formula_code,
    g,
    family_ctx,
):
    side = family_ctx["side"]

    if formula_code == "USE_FAMILY_SELECTED_PROB":
        return family_ctx["prob"]

    if (
        formula_code
        == "MARKET_ROLE_FOR_FAMILY_SELECTED_SIDE"
    ):
        return market_role_for_side(
            g,
            side,
        )

    if (
        formula_code
        == "SPREAD_FOR_FAMILY_SELECTED_SIDE"
    ):
        if side == "Home":
            return num(
                g.get("market_home_spread")
            )
        if side == "Away":
            return num(
                g.get("market_away_spread")
            )
        return None

    if (
        formula_code
        == "EPRED_RATING_SELECTED_MINUS_OPPONENT"
    ):
        home_rating = num(
            g.get("epred_home_rating")
        )
        away_rating = num(
            g.get("epred_away_rating")
        )

        if (
            home_rating is None
            or away_rating is None
            or side is None
        ):
            return None

        return (
            home_rating - away_rating
            if side == "Home"
            else away_rating - home_rating
        )

    if (
        formula_code
        == "RAW_EPRED_MATCHUP_QUALITY"
    ):
        return num(
            g.get("epred_matchupQuality")
        )

    if formula_code == "RAW_WEEK":
        return num(g.get("week"))

    if formula_code == "RAW_MARKET_TOTAL":
        return num(g.get("market_total"))

    if (
        formula_code
        == "COMPARE_DRAT_PICK_TO_EPRED_PICK"
    ):
        drat = g.get("drat_pick_side")
        epred = g.get("epred_pick_side")

        if not drat or not epred:
            return None

        return (
            "Agree"
            if drat == epred
            else "Disagree"
        )

    if (
        formula_code
        == "COMPARE_FAMILY_PICK_TO_MARKET_PICK"
    ):
        market = g.get("market_pick_side")

        if not side or not market:
            return None

        return (
            "Agree"
            if side == market
            else "Disagree"
        )

    if (
        formula_code
        == "ABS_DRAT_HOME_PROB_MINUS_EPRED_NORMALIZED_HOME_PROB_X100"
    ):
        drat = num(
            g.get("drat_home_prob")
        )
        epred = num(
            g.get("epred_home_prob")
        )

        if drat is None or epred is None:
            return None

        return abs(drat - epred) * 100.0

    if (
        formula_code
        == "FAMILY_SELECTED_PROB_MINUS_MARKET_SELECTED_PROB_X100"
    ):
        family_probability = (
            family_ctx["prob"]
        )

        if side == "Home":
            market_probability = num(
                g.get(
                    "market_home_prob_novig"
                )
            )
        elif side == "Away":
            market_probability = num(
                g.get(
                    "market_away_prob_novig"
                )
            )
        else:
            market_probability = None

        if (
            family_probability is None
            or market_probability is None
        ):
            return None

        return (
            family_probability
            - market_probability
        ) * 100.0

    if formula_code == "UNAVAILABLE":
        return None

    fail(
        "Unsupported formula_code in master: "
        f"{formula_code}"
    )


def condition_matches(
    rule,
    number,
    value,
):
    prefix = f"condition_{number}_"
    match_type = s(
        rule.get(
            prefix + "match_type"
        )
    )

    if not match_type:
        return True

    if match_type == "IS_NULL":
        return (
            value is None
            or s(value) == ""
        )

    if match_type == "TEXT_EQUALS":
        return (
            s(value)
            == s(
                rule.get(
                    prefix + "equals_value"
                )
            )
        )

    if match_type == "NUMERIC_RANGE":
        numeric = num(value)
        if numeric is None:
            return False

        lower = num(
            rule.get(
                prefix + "min_inclusive"
            )
        )
        upper = num(
            rule.get(
                prefix + "max_exclusive"
            )
        )
        test_feature = s(
            rule.get(
                prefix + "test_feature"
            )
        )

        if test_feature == "SpreadBucket":
            if (
                lower is not None
                and numeric <= lower
            ):
                return False
            if (
                upper is not None
                and numeric > upper
            ):
                return False
            return True

        if (
            lower is not None
            and numeric < lower
        ):
            return False
        if (
            upper is not None
            and numeric >= upper
        ):
            return False
        return True

    fail(
        "Unsupported match_type in master: "
        f"{match_type}"
    )


def match_rules(
    g,
    master_rows,
    contexts,
):
    matches = []

    for rule in master_rows:
        if s(rule.get("active")) != "1":
            continue

        if (
            s(rule.get("pipeline_supported"))
            != "1"
        ):
            continue

        family = s(rule.get("family"))
        family_ctx = contexts.get(family)

        if (
            not family_ctx
            or not family_ctx["eligible"]
        ):
            continue

        condition_count = parse_int_text(
            rule.get("condition_count"),
            label=(
                "moneyline master rule "
                f"{s(rule.get('rule_id'))} "
                "condition_count"
            ),
        )
        matched = True

        for number in range(
            1,
            condition_count + 1,
        ):
            formula = s(
                rule.get(
                    f"condition_{number}_formula_code"
                )
            )
            value = feature_value(
                formula,
                g,
                family_ctx,
            )

            if not condition_matches(
                rule,
                number,
                value,
            ):
                matched = False
                break

        if matched:
            matches.append(
                {
                    "rule_id": s(
                        rule.get("rule_id")
                    ),
                    "family": family,
                    "side": family_ctx["side"],
                    "condition": s(
                        rule.get(
                            "source_condition"
                        )
                    ),
                    "historical_win_rate_pct": num(
                        rule.get(
                            "historical_win_rate_pct"
                        )
                    ),
                    "lift_pp": num(
                        rule.get(
                            "lift_vs_family_pct_points"
                        )
                    ),
                    "direction": s(
                        rule.get(
                            "action_direction"
                        )
                    ),
                    "games": num(
                        rule.get("games")
                    ),
                }
            )

    return matches


def strongest(
    matches,
    side,
    direction,
):
    candidates = [
        match
        for match in matches
        if (
            match["side"] == side
            and match["direction"] == direction
            and match["lift_pp"] is not None
        )
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda match: abs(
            match["lift_pp"]
        ),
    )


def family_matches(
    matches,
    family,
):
    return [
        match
        for match in matches
        if match["family"] == family
    ]


def join_text(values):
    return ";".join(
        s(value)
        for value in values
        if s(value)
    )


def build_summary_fields(
    g,
    matches,
):
    positive = [
        match
        for match in matches
        if match["direction"] == "POSITIVE"
    ]
    negative = [
        match
        for match in matches
        if match["direction"] == "NEGATIVE"
    ]

    g["matched_rule_count"] = len(matches)
    g["matched_positive_rule_count"] = (
        len(positive)
    )
    g["matched_negative_rule_count"] = (
        len(negative)
    )
    g["matched_rule_ids"] = join_text(
        match["rule_id"]
        for match in matches
    )
    g["matched_rule_conditions"] = join_text(
        (
            f'{match["rule_id"]}:'
            f'{match["family"]}:'
            f'{match["side"]}:'
            f'{match["condition"]}'
        )
        for match in matches
    )

    for side_name, side in [
        ("home", "Home"),
        ("away", "Away"),
    ]:
        side_matches = [
            match
            for match in matches
            if match["side"] == side
        ]
        g[
            f"{side_name}_matched_rule_count"
        ] = len(side_matches)
        g[
            f"{side_name}_matched_rule_ids"
        ] = join_text(
            match["rule_id"]
            for match in side_matches
        )

        positive_item = strongest(
            matches,
            side,
            "POSITIVE",
        )
        negative_item = strongest(
            matches,
            side,
            "NEGATIVE",
        )

        for label, item in [
            (
                "strongest_positive",
                positive_item,
            ),
            (
                "strongest_negative",
                negative_item,
            ),
        ]:
            g[
                f"{side_name}_{label}_rule_id"
            ] = (
                item["rule_id"]
                if item
                else ""
            )
            g[
                f"{side_name}_{label}_hist_win_rate_pct"
            ] = (
                item["historical_win_rate_pct"]
                if item
                else ""
            )
            g[
                f"{side_name}_{label}_lift_pp"
            ] = (
                item["lift_pp"]
                if item
                else ""
            )
            g[
                f"{side_name}_{label}_games"
            ] = (
                item["games"]
                if item
                else ""
            )

    for family, prefix in [
        ("DRAT", "drat"),
        ("EPRED", "epred"),
        ("MARKET", "market"),
        (
            "DRAT_EPRED_CONSENSUS",
            "drat_epred_consensus",
        ),
        (
            "ALL3_CONSENSUS",
            "all3_consensus",
        ),
    ]:
        family_items = family_matches(
            matches,
            family,
        )
        g[
            f"{prefix}_matched_rule_count"
        ] = len(family_items)
        g[
            f"{prefix}_matched_rule_ids"
        ] = join_text(
            match["rule_id"]
            for match in family_items
        )

    return g


def process_week(
    *,
    season: int,
    season_type: str,
    week: int,
    schedule_rows: list[dict[str, str]],
    drat_by_teams,
    epred_by_game,
    current_odds,
    master_rows,
) -> tuple[
    list[dict[str, object]],
    dict[str, int],
]:
    output_rows: list[
        dict[str, object]
    ] = []
    metrics = {
        "current_odds_matches": 0,
        "weekly_schedule_market_fallbacks": 0,
    }

    for base in schedule_rows:
        game_id = s(base.get("game_id"))
        g = dict(base)

        epred = epred_by_game.get(game_id)
        if epred is None:
            fail(
                f"Missing EPRED join for game_id={game_id}"
            )

        drat_key = game_team_key(
            base.get("season"),
            base.get("week"),
            base.get("home_team"),
            base.get("away_team"),
        )
        drat = drat_by_teams.get(
            drat_key
        )
        if drat is None:
            fail(
                f"Missing DRAT join for game_id={game_id}"
            )

        odds_record = choose_odds_record(
            current_odds,
            base.get(
                "odds_provider_game_id"
            ),
            base.get("bookmaker"),
        )

        if odds_record is None:
            metrics[
                "weekly_schedule_market_fallbacks"
            ] += 1
        else:
            metrics[
                "current_odds_matches"
            ] += 1

        g["drat_home_prob"] = num(
            drat.get("home_prob")
        )
        g["drat_away_prob"] = num(
            drat.get("away_prob")
        )

        epred_home_raw = num(
            epred.get("home_prob")
        )
        epred_away_raw = num(
            epred.get("away_prob")
        )

        g["epred_home_prob_raw"] = (
            epred_home_raw
            if epred_home_raw is not None
            else ""
        )
        g["epred_away_prob_raw"] = (
            epred_away_raw
            if epred_away_raw is not None
            else ""
        )

        epred_sum = (
            epred_home_raw
            + epred_away_raw
            if (
                epred_home_raw is not None
                and epred_away_raw is not None
            )
            else None
        )

        g["epred_home_prob"] = (
            epred_home_raw / epred_sum
            if (
                epred_sum is not None
                and epred_sum > 0
            )
            else ""
        )
        g["epred_away_prob"] = (
            epred_away_raw / epred_sum
            if (
                epred_sum is not None
                and epred_sum > 0
            )
            else ""
        )
        g["epred_home_rating"] = num(
            epred.get("home_rating")
        )
        g["epred_away_rating"] = num(
            epred.get("away_rating")
        )
        g["epred_matchupQuality"] = num(
            epred.get("matchupQuality")
        )

        def market_value(field):
            if (
                odds_record
                and s(
                    odds_record.get(field)
                )
                != ""
            ):
                return odds_record.get(field)
            return base.get(field, "")

        g["market_bookmaker"] = (
            odds_record.get("bookmaker")
            if odds_record
            else base.get("bookmaker", "")
        )
        g["market_last_update"] = (
            odds_record.get("last_update")
            if odds_record
            else ""
        )
        g[
            "market_home_moneyline_american"
        ] = market_value(
            "home_moneyline_american"
        )
        g[
            "market_away_moneyline_american"
        ] = market_value(
            "away_moneyline_american"
        )
        g["market_home_spread"] = (
            market_value("home_spread")
        )
        g["market_away_spread"] = (
            market_value("away_spread")
        )
        g["market_total"] = market_value(
            "total"
        )

        market_home, market_away = (
            no_vig_probs(
                g[
                    "market_home_moneyline_american"
                ],
                g[
                    "market_away_moneyline_american"
                ],
            )
        )

        g["market_home_prob_novig"] = (
            market_home
            if market_home is not None
            else ""
        )
        g["market_away_prob_novig"] = (
            market_away
            if market_away is not None
            else ""
        )

        contexts = build_family_contexts(g)

        home_rating = num(
            g.get("epred_home_rating")
        )
        away_rating = num(
            g.get("epred_away_rating")
        )
        g["epred_rating_gap_home"] = (
            home_rating - away_rating
            if (
                home_rating is not None
                and away_rating is not None
            )
            else ""
        )

        drat_home = num(
            g.get("drat_home_prob")
        )
        epred_home = num(
            g.get("epred_home_prob")
        )

        g["drat_epred_prob_diff_pp"] = (
            abs(
                drat_home - epred_home
            )
            * 100.0
            if (
                drat_home is not None
                and epred_home is not None
            )
            else ""
        )
        g["drat_market_edge_home_pp"] = (
            (
                drat_home
                - market_home
            )
            * 100.0
            if (
                drat_home is not None
                and market_home is not None
            )
            else ""
        )
        g["epred_market_edge_home_pp"] = (
            (
                epred_home
                - market_home
            )
            * 100.0
            if (
                epred_home is not None
                and market_home is not None
            )
            else ""
        )

        matches = match_rules(
            g,
            master_rows,
            contexts,
        )
        build_summary_fields(
            g,
            matches,
        )

        output_rows.append(g)

    if not output_rows:
        fail(
            f"No rows generated for "
            f"season={season} "
            f"season_type={season_type} "
            f"week={week}"
        )

    return output_rows, metrics


def split_rule_ids(
    value: Any,
) -> list[str]:
    text = s(value)
    if not text:
        return []
    return [
        item
        for item in text.split(";")
        if item
    ]


def validate_rule_count(
    row: dict[str, str],
    *,
    count_field: str,
    ids_field: str,
    active_rule_ids: set[str],
    label: str,
) -> None:
    count = parse_int_text(
        row.get(count_field),
        label=f"{label} {count_field}",
    )

    if count < 0:
        fail(
            f"{label} {count_field} cannot be negative"
        )

    ids = split_rule_ids(
        row.get(ids_field)
    )

    if len(ids) != count:
        fail(
            f"{label} {count_field}={count} but "
            f"{ids_field} contains {len(ids)} IDs"
        )

    if len(ids) != len(set(ids)):
        fail(
            f"{label} {ids_field} contains duplicate rule IDs"
        )

    unknown = sorted(
        set(ids) - active_rule_ids
    )
    if unknown:
        fail(
            f"{label} {ids_field} contains unknown or "
            f"inactive rule IDs: {unknown}"
        )


def validate_output_rows(
    rows: list[dict[str, str]],
    *,
    schedule_rows: list[dict[str, str]],
    active_rule_ids: set[str],
    path: Path,
) -> None:
    if len(rows) != len(schedule_rows):
        fail(
            f"{path.name} row count mismatch "
            f"expected={len(schedule_rows)} "
            f"actual={len(rows)}"
        )

    schedule_by_id = {
        s(row.get("game_id")): row
        for row in schedule_rows
    }
    output_by_id: dict[
        str,
        dict[str, str],
    ] = {}

    count_pairs = [
        (
            "matched_rule_count",
            "matched_rule_ids",
        ),
        (
            "home_matched_rule_count",
            "home_matched_rule_ids",
        ),
        (
            "away_matched_rule_count",
            "away_matched_rule_ids",
        ),
        (
            "drat_matched_rule_count",
            "drat_matched_rule_ids",
        ),
        (
            "epred_matched_rule_count",
            "epred_matched_rule_ids",
        ),
        (
            "market_matched_rule_count",
            "market_matched_rule_ids",
        ),
        (
            "drat_epred_consensus_matched_rule_count",
            "drat_epred_consensus_matched_rule_ids",
        ),
        (
            "all3_consensus_matched_rule_count",
            "all3_consensus_matched_rule_ids",
        ),
    ]

    for line_number, row in enumerate(
        rows,
        start=2,
    ):
        game_id = s(row.get("game_id"))
        if not game_id:
            fail(
                f"{path.name} line {line_number} "
                "has blank game_id"
            )

        if game_id in output_by_id:
            fail(
                f"{path.name} contains duplicate "
                f"game_id={game_id}"
            )

        schedule_row = schedule_by_id.get(
            game_id
        )
        if schedule_row is None:
            fail(
                f"{path.name} contains unexpected "
                f"game_id={game_id}"
            )

        for field in WEEKLY_COLUMNS:
            if s(row.get(field)) != s(
                schedule_row.get(field)
            ):
                fail(
                    f"{path.name} game_id={game_id} "
                    f"changed weekly schedule field={field}"
                )

        for field in (
            "drat_home_prob",
            "drat_away_prob",
            "epred_home_prob_raw",
            "epred_away_prob_raw",
            "epred_home_prob",
            "epred_away_prob",
            "epred_home_rating",
            "epred_away_rating",
            "epred_matchupQuality",
            "epred_rating_gap_home",
            "drat_epred_prob_diff_pp",
        ):
            require_finite_number(
                row.get(field),
                label=(
                    f"{path.name} game_id={game_id} "
                    f"{field}"
                ),
            )

        for field in (
            "drat_home_prob",
            "drat_away_prob",
            "epred_home_prob_raw",
            "epred_away_prob_raw",
            "epred_home_prob",
            "epred_away_prob",
        ):
            value = require_finite_number(
                row.get(field),
                label=(
                    f"{path.name} game_id={game_id} "
                    f"{field}"
                ),
            )
            if value < 0 or value > 1:
                fail(
                    f"{path.name} game_id={game_id} "
                    f"{field} outside 0..1"
                )

        epred_home = require_finite_number(
            row.get("epred_home_prob"),
            label=(
                f"{path.name} game_id={game_id} "
                "epred_home_prob"
            ),
        )
        epred_away = require_finite_number(
            row.get("epred_away_prob"),
            label=(
                f"{path.name} game_id={game_id} "
                "epred_away_prob"
            ),
        )

        if abs(
            (
                epred_home
                + epred_away
            )
            - 1.0
        ) > 1e-12:
            fail(
                f"{path.name} game_id={game_id} "
                "normalized EPRED probabilities "
                "do not sum to 1"
            )

        for count_field, ids_field in (
            count_pairs
        ):
            validate_rule_count(
                row,
                count_field=count_field,
                ids_field=ids_field,
                active_rule_ids=active_rule_ids,
                label=(
                    f"{path.name} game_id={game_id}"
                ),
            )

        total_count = parse_int_text(
            row.get("matched_rule_count"),
            label=(
                f"{path.name} game_id={game_id} "
                "matched_rule_count"
            ),
        )
        positive_count = parse_int_text(
            row.get(
                "matched_positive_rule_count"
            ),
            label=(
                f"{path.name} game_id={game_id} "
                "matched_positive_rule_count"
            ),
        )
        negative_count = parse_int_text(
            row.get(
                "matched_negative_rule_count"
            ),
            label=(
                f"{path.name} game_id={game_id} "
                "matched_negative_rule_count"
            ),
        )

        if (
            positive_count
            + negative_count
            != total_count
        ):
            fail(
                f"{path.name} game_id={game_id} "
                "positive+negative matched counts "
                "do not equal total"
            )

        home_ids = set(
            split_rule_ids(
                row.get(
                    "home_matched_rule_ids"
                )
            )
        )
        away_ids = set(
            split_rule_ids(
                row.get(
                    "away_matched_rule_ids"
                )
            )
        )
        all_ids = set(
            split_rule_ids(
                row.get("matched_rule_ids")
            )
        )

        if home_ids | away_ids != all_ids:
            fail(
                f"{path.name} game_id={game_id} "
                "home/away rule ID union does not "
                "equal all matched rule IDs"
            )

        if home_ids & away_ids:
            fail(
                f"{path.name} game_id={game_id} "
                "same rule ID appears on both sides"
            )

        for side_name, side_ids in (
            ("home", home_ids),
            ("away", away_ids),
        ):
            for polarity in (
                "positive",
                "negative",
            ):
                strongest_id = s(
                    row.get(
                        f"{side_name}_strongest_"
                        f"{polarity}_rule_id"
                    )
                )

                if (
                    strongest_id
                    and strongest_id
                    not in side_ids
                ):
                    fail(
                        f"{path.name} game_id={game_id} "
                        f"{side_name} strongest {polarity} "
                        "rule is absent from side matches"
                    )

        output_by_id[game_id] = row

    if set(output_by_id) != set(
        schedule_by_id
    ):
        fail(
            f"{path.name} output/schedule game "
            "universe mismatch"
        )


def normalize_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            header: s(row.get(header))
            for header in OUTPUT_HEADERS
        }
        for row in rows
    ]


def build_staged_root(
    *,
    week_outputs: dict[
        int,
        tuple[
            list[dict[str, object]],
            list[dict[str, str]],
        ],
    ],
    active_rule_ids: set[str],
) -> Path:
    OUTPUT_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stage_root = Path(
        tempfile.mkdtemp(
            prefix=".moneyline_enrichment_stage_",
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
            "week_*_NFL_enriched.csv"
        ):
            stale.unlink()

        expected_names: set[str] = set()

        for week, (
            output_rows,
            schedule_rows,
        ) in sorted(
            week_outputs.items()
        ):
            name = (
                f"week_{week}_NFL_enriched.csv"
            )
            expected_names.add(name)
            path = stage_root / name

            write_csv(
                path,
                output_rows,
                OUTPUT_HEADERS,
            )

            headers, staged_rows = (
                read_csv_table(path)
            )
            require_exact_headers(
                headers,
                OUTPUT_HEADERS,
                label=(
                    "staged moneyline enrichment "
                    f"{name}"
                ),
            )

            validate_output_rows(
                staged_rows,
                schedule_rows=schedule_rows,
                active_rule_ids=active_rule_ids,
                path=path,
            )

            if (
                normalize_rows(staged_rows)
                != normalize_rows(
                    output_rows
                )
            ):
                fail(
                    "Staged moneyline enrichment differs "
                    f"from validated in-memory output: {path}"
                )

        actual_names = {
            path.name
            for path in stage_root.glob(
                "week_*_NFL_enriched.csv"
            )
        }

        if actual_names != expected_names:
            fail(
                "Staged moneyline managed file set mismatch "
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
        OUTPUT_DIR.parent
        / (
            f".{OUTPUT_DIR.name}_backup_"
            f"{uuid.uuid4().hex}"
        )
    )

    try:
        if OUTPUT_DIR.exists():
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
            shutil.rmtree(
                backup_root
            )
        except Exception as exc:
            reporter.warning(
                "Moneyline enrichment published but "
                "temporary backup cleanup failed",
                backup_path=str(backup_root),
                error_type=type(exc).__name__,
                error=str(exc),
            )


def run(
    reporter: PipelineReporter,
    *,
    season: int,
) -> None:
    master_headers, master_rows = (
        read_csv_table(MASTER_PATH)
    )
    reporter.add_input(MASTER_PATH)
    active_rule_ids = validate_master(
        master_headers,
        master_rows,
    )

    odds_path, skipped_odds_candidates = (
        select_latest_odds_file(
            reporter=reporter,
        )
    )
    odds_headers, odds_rows = (
        read_csv_table(odds_path)
    )
    validate_selected_odds(
        path=odds_path,
        headers=odds_headers,
        rows=odds_rows,
    )
    reporter.add_input(odds_path)

    current_odds = aggregate_latest_odds(
        odds_rows
    )

    schedules = load_target_schedules(
        season=season,
        reporter=reporter,
    )

    week_outputs: dict[
        int,
        tuple[
            list[dict[str, object]],
            list[dict[str, str]],
        ],
    ] = {}
    completed = []
    total_rows = 0
    current_odds_matches = 0
    weekly_fallbacks = 0

    for week, (
        schedule_path,
        season_type,
        schedule_rows,
    ) in sorted(
        schedules.items()
    ):
        (
            drat_path,
            _,
            drat_by_teams,
        ) = load_drat(
            season=season,
            week=week,
            schedule_rows=schedule_rows,
            reporter=reporter,
        )

        (
            epred_path,
            _,
            epred_by_game,
        ) = load_epred(
            season=season,
            season_type=season_type,
            week=week,
            schedule_rows=schedule_rows,
            reporter=reporter,
        )

        output_rows, metrics = process_week(
            season=season,
            season_type=season_type,
            week=week,
            schedule_rows=schedule_rows,
            drat_by_teams=drat_by_teams,
            epred_by_game=epred_by_game,
            current_odds=current_odds,
            master_rows=master_rows,
        )

        output_path = (
            OUTPUT_DIR
            / f"week_{week}_NFL_enriched.csv"
        )

        validate_output_rows(
            normalize_rows(output_rows),
            schedule_rows=schedule_rows,
            active_rule_ids=active_rule_ids,
            path=output_path,
        )

        week_outputs[week] = (
            output_rows,
            schedule_rows,
        )

        total_rows += len(output_rows)
        current_odds_matches += metrics[
            "current_odds_matches"
        ]
        weekly_fallbacks += metrics[
            "weekly_schedule_market_fallbacks"
        ]

        completed.append(
            {
                "season": season,
                "season_type": season_type,
                "week": week,
                "schedule": schedule_path.name,
                "drat": drat_path.name,
                "epred": epred_path.name,
                "output": str(output_path),
                "games": len(output_rows),
                "missing_epred": 0,
                "missing_drat": 0,
            }
        )

    reporter.set_rows(
        rows_in=total_rows,
        rows_out=0,
    )
    reporter.update_details(
        {
            "season": season,
            "weeks_enriched": len(completed),
            "games_enriched": total_rows,
            "active_supported_rules": len(
                active_rule_ids
            ),
            "master_rows": len(master_rows),
            "latest_odds_file": str(
                odds_path
            ),
            "latest_odds_rows": len(
                odds_rows
            ),
            "odds_candidates_skipped": (
                skipped_odds_candidates
            ),
            "current_odds_matches": (
                current_odds_matches
            ),
            "weekly_schedule_market_fallbacks": (
                weekly_fallbacks
            ),
            "output_columns": len(
                OUTPUT_HEADERS
            ),
            "publication_mode": (
                "validated_directory_swap_with_rollback"
            ),
            "publication_completed": False,
            "staged_roundtrip_verified": False,
        }
    )

    stage_root: Path | None = None

    try:
        stage_root = build_staged_root(
            week_outputs=week_outputs,
            active_rule_ids=active_rule_ids,
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

    for result in completed:
        reporter.add_output(
            Path(result["output"])
        )

    reporter.set_rows(
        rows_in=total_rows,
        rows_out=total_rows,
    )
    reporter.update_details(
        {
            "files_published": len(completed),
            "rows_published": total_rows,
            "publication_completed": True,
        }
    )

    print(
        f"Historical moneyline master: "
        f"{MASTER_PATH}"
    )
    print(
        f"Latest odds file: {odds_path}"
    )
    print(
        f"Weeks enriched: {len(completed)}"
    )

    for result in completed:
        print(
            f"week {result['week']} -> "
            f"{result['output']} "
            f"(games={result['games']}, "
            "missing_epred=0, missing_drat=0)"
        )


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
                "component": (
                    "historical moneyline enrichment"
                ),
                "refresh_scope": (
                    "requested season available weeks"
                ),
            },
        ) as reporter:
            run(
                reporter,
                season=args.season,
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
    sys.exit(main())
