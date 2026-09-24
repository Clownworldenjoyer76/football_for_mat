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
from typing import Any, Never

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter

# QODANA_SHARED_ENRICHMENT_CORE_IMPORTS_V1
from functools import partial as _enrichment_partial
from enrichment_contract import (
    enrichment_aggregate_latest_odds as aggregate_latest_odds,
    enrichment_american_implied as american_implied,
    enrichment_build_family_contexts as build_family_contexts,
    enrichment_build_side_summary_fields as _enrichment_build_side_summary_fields,
    enrichment_choose_odds_record as choose_odds_record,
    enrichment_clean_text as s,
    enrichment_family_matches as family_matches,
    enrichment_feature_value as _enrichment_feature_value,
    enrichment_game_team_key as _enrichment_game_team_key,
    enrichment_iso_dt as iso_dt,
    enrichment_join_text as join_text,
    enrichment_market_role_for_side as market_role_for_side,
    enrichment_match_rules_side as _enrichment_match_rules_side,
    enrichment_no_vig_probs as no_vig_probs,
    enrichment_normalize_rows as _enrichment_normalize_rows,
    enrichment_number as num,
    enrichment_parse_int_text as _enrichment_parse_int_text,
    enrichment_parse_iso_dt as parse_iso_dt,
    enrichment_process_week as _enrichment_process_week,
    enrichment_read_csv as _enrichment_read_csv,
    enrichment_read_csv_table as _enrichment_read_csv_table,
    enrichment_require_columns as _enrichment_require_columns,
    enrichment_require_exact_headers as _enrichment_require_exact_headers,
    enrichment_require_finite_number as _enrichment_require_finite_number,
    enrichment_same_text as same_text,
    enrichment_schedule_identity as _enrichment_schedule_identity,
    enrichment_select_latest_odds_file as _enrichment_select_latest_odds_file,
    enrichment_split_rule_ids as split_rule_ids,
    enrichment_team_key as team_key,
    enrichment_validate_master_side as _enrichment_validate_master_side,
    enrichment_validate_rule_count as _enrichment_validate_rule_count,
    enrichment_validate_selected_odds as _enrichment_validate_selected_odds,
    enrichment_write_csv as write_csv,
)

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


def fail(message: str) -> Never:
    raise MoneylineEnrichmentError(message)

read_csv_table = _enrichment_partial(_enrichment_read_csv_table, fail=fail)
read_csv = _enrichment_partial(_enrichment_read_csv, fail=fail)
require_exact_headers = _enrichment_partial(_enrichment_require_exact_headers, fail=fail)
require_columns = _enrichment_partial(_enrichment_require_columns, fail=fail)
require_finite_number = _enrichment_partial(_enrichment_require_finite_number, fail=fail)
parse_int_text = _enrichment_partial(_enrichment_parse_int_text, fail=fail)
game_team_key = _enrichment_partial(_enrichment_game_team_key, fail=fail)
schedule_identity = _enrichment_partial(_enrichment_schedule_identity, fail=fail)
feature_value = _enrichment_partial(_enrichment_feature_value, fail=fail)
validate_rule_count = _enrichment_partial(_enrichment_validate_rule_count, fail=fail)
normalize_rows = _enrichment_partial(_enrichment_normalize_rows, output_headers=OUTPUT_HEADERS)
select_latest_odds_file = _enrichment_partial(
    _enrichment_select_latest_odds_file,
    odds_dir=ODDS_DIR,
    fail=fail,
)
validate_selected_odds = _enrichment_partial(
    _enrichment_validate_selected_odds,
    odds_headers=ODDS_HEADERS,
    expected_market_sides=EXPECTED_MARKET_SIDES,
    fail=fail,
)

validate_master = _enrichment_partial(
    _enrichment_validate_master_side,
    header_label="moneyline enrichment master",
    display_name="Moneyline",
    required_headers=MASTER_REQUIRED_HEADERS,
    supported_families=SUPPORTED_FAMILIES,
    supported_formulas=SUPPORTED_FORMULAS,
    supported_match_types=SUPPORTED_MATCH_TYPES,
    metric_fields=("historical_win_rate_pct", "lift_vs_family_pct_points", "games"),
    fail=fail,
)
match_rules = _enrichment_partial(
    _enrichment_match_rules_side,
    master_label="moneyline",
    rate_source_field="historical_win_rate_pct",
    rate_output_field="historical_win_rate_pct",
    fail=fail,
)
build_summary_fields = _enrichment_partial(
    _enrichment_build_side_summary_fields,
    rate_key="historical_win_rate_pct",
    rate_suffix="hist_win_rate_pct",
)
process_week = _enrichment_partial(
    _enrichment_process_week,
    game_team_key=game_team_key,
    match_rules=match_rules,
    build_summary_fields=build_summary_fields,
    fail=fail,
)



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
