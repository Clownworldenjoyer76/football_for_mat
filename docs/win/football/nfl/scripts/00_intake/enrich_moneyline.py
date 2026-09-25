#!/usr/bin/env python3
"""
NFL weekly historical moneyline enrichment.

The rule definitions, bucket boundaries, historical rates, and action
directions remain data-driven from moneyline_enrichment.csv.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Never

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter
from publication_contract import publish_staged_directory
from odds_contract import (
    EXPECTED_MARKET_SIDES,
    ODDS_OUTPUT_COLUMNS as ODDS_HEADERS,
)

# QODANA_SHARED_ENRICHMENT_CORE_IMPORTS_V1
from functools import partial as _enrichment_partial
from enrichment_contract import (
    ENRICHMENT_WEEKLY_COLUMNS as WEEKLY_COLUMNS,
    enrichment_bind_input_helpers as _enrichment_bind_input_helpers,
    enrichment_bind_run as _enrichment_bind_run,
    enrichment_bind_side_output_validator as _enrichment_bind_side_output_validator,
    enrichment_aggregate_latest_odds as aggregate_latest_odds,
    enrichment_build_side_summary_fields as _enrichment_build_side_summary_fields,
    enrichment_feature_value as _enrichment_feature_value,
    enrichment_game_team_key as _enrichment_game_team_key,
    enrichment_match_rules_side as _enrichment_match_rules_side,
    enrichment_normalize_rows as _enrichment_normalize_rows,
    enrichment_parse_int_text as _enrichment_parse_int_text,
    enrichment_process_week as _enrichment_process_week,
    enrichment_read_csv as _enrichment_read_csv,
    enrichment_read_csv_table as _enrichment_read_csv_table,
    enrichment_require_columns as _enrichment_require_columns,
    enrichment_require_exact_headers as _enrichment_require_exact_headers,
    enrichment_require_finite_number as _enrichment_require_finite_number,
    enrichment_schedule_identity as _enrichment_schedule_identity,
    enrichment_select_latest_odds_file as _enrichment_select_latest_odds_file,
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


(
    validate_weekly_schedule,
    load_target_schedules,
    load_drat,
    load_epred,
) = _enrichment_bind_input_helpers(
    schedule_dir=SCHEDULE_DIR,
    drat_dir=DRAT_DIR,
    epred_dir=EPRED_DIR,
    fail=fail,
)








validate_output_rows = _enrichment_bind_side_output_validator(fail=fail)


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
    publish_staged_directory(
        stage_root,
        output_dir=OUTPUT_DIR,
        reporter=reporter,
        cleanup_warning='Moneyline enrichment published but temporary backup cleanup failed',
    )


run = _enrichment_bind_run(
    globals(),
    market_name="moneyline",
    aggregate_latest_odds=aggregate_latest_odds,
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
