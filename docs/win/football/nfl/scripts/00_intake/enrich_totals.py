#!/usr/bin/env python3
"""
NFL weekly historical totals-prediction enrichment.

The rule definitions, bucket boundaries, historical rates, and action
directions remain data-driven from totals_enrichment.csv.
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
from enrichment_contract import TOTALS_APPENDED_FIELDS

# QODANA_SHARED_ENRICHMENT_CORE_IMPORTS_V1
from functools import partial as _enrichment_partial
from enrichment_contract import (
    ENRICHMENT_WEEKLY_COLUMNS as WEEKLY_COLUMNS,
    enrichment_bind_input_helpers as _enrichment_bind_input_helpers,
    enrichment_bind_run as _enrichment_bind_run,
    enrichment_aggregate_latest_odds as aggregate_latest_odds,
    enrichment_clean_text as s,
    enrichment_family_matches as family_matches,
    enrichment_feature_value as _enrichment_feature_value,
    enrichment_game_team_key as _enrichment_game_team_key,
    enrichment_join_text as join_text,
    enrichment_normalize_rows as _enrichment_normalize_rows,
    enrichment_number as num,
    enrichment_parse_int_text as _enrichment_parse_int_text,
    enrichment_process_week as _enrichment_process_week,
    enrichment_read_csv as _enrichment_read_csv,
    enrichment_read_csv_table as _enrichment_read_csv_table,
    enrichment_require_columns as _enrichment_require_columns,
    enrichment_require_exact_headers as _enrichment_require_exact_headers,
    enrichment_require_finite_number as _enrichment_require_finite_number,
    enrichment_schedule_identity as _enrichment_schedule_identity,
    enrichment_select_latest_odds_file as _enrichment_select_latest_odds_file,
    enrichment_split_rule_ids as split_rule_ids,
    enrichment_validate_rule_count as _enrichment_validate_rule_count,
    enrichment_validate_selected_odds as _enrichment_validate_selected_odds,
    enrichment_write_csv as write_csv,
)

MASTER_PATH = (
    NFL_ROOT
    / "config"
    / "prediction_enrichment"
    / "totals_enrichment.csv"
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
    / "totals"
)
REPORT_ROOT = NFL_ROOT / "errors"







MASTER_REQUIRED_HEADERS = ['rule_id', 'active', 'pipeline_supported', 'family', 'source_condition', 'condition_count', 'condition_1_test_feature', 'condition_1_formula_code', 'condition_1_match_type', 'condition_1_min_inclusive', 'condition_1_max_exclusive', 'condition_1_equals_value', 'condition_2_test_feature', 'condition_2_formula_code', 'condition_2_match_type', 'condition_2_min_inclusive', 'condition_2_max_exclusive', 'condition_2_equals_value', 'games', 'historical_hit_rate_pct', 'lift_vs_family_pct_points', 'action_direction', 'totals_direction']

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

SUPPORTED_MATCH_TYPES = {'', 'IS_NULL', 'TEXT_EQUALS', 'NUMERIC_RANGE'}

APPENDED_FIELDS = list(TOTALS_APPENDED_FIELDS)

OUTPUT_HEADERS = WEEKLY_COLUMNS + [
    column
    for column in APPENDED_FIELDS
    if column not in WEEKLY_COLUMNS
]


class TotalsEnrichmentError(RuntimeError):
    pass


def fail(message: str) -> Never:
    raise TotalsEnrichmentError(message)

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



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build NFL historical totals enrichment for "
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








def validate_master(
    headers: list[str],
    rows: list[dict[str, str]],
) -> set[str]:
    require_columns(
        headers,
        MASTER_REQUIRED_HEADERS,
        label="totals enrichment master",
    )

    if not rows:
        fail(
            "Totals enrichment master contains no rows"
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
                f"Totals master line {line_number} "
                "has blank rule_id"
            )

        if rule_id in seen_ids:
            fail(
                f"Totals master contains duplicate "
                f"rule_id={rule_id}"
            )
        seen_ids.add(rule_id)

        active = s(rule.get("active"))
        supported = s(
            rule.get("pipeline_supported")
        )

        if active not in {"0", "1"}:
            fail(
                f"Totals master rule_id={rule_id} "
                f"has invalid active={active!r}"
            )

        if supported not in {"0", "1"}:
            fail(
                f"Totals master rule_id={rule_id} "
                "has invalid pipeline_supported="
                f"{supported!r}"
            )

        if active != "1" or supported != "1":
            continue

        active_supported_ids.add(rule_id)

        family = s(rule.get("family"))
        if family not in SUPPORTED_FAMILIES:
            fail(
                f"Totals master rule_id={rule_id} "
                f"has unsupported family={family!r}"
            )

        totals_direction = s(
            rule.get("totals_direction")
        )
        if totals_direction not in {
            "Over",
            "Under",
        }:
            fail(
                f"Totals master rule_id={rule_id} "
                "has invalid totals_direction="
                f"{totals_direction!r}"
            )

        condition_count = parse_int_text(
            rule.get("condition_count"),
            label=(
                f"Totals master rule_id={rule_id} "
                "condition_count"
            ),
        )

        if condition_count not in {1, 2}:
            fail(
                f"Totals master rule_id={rule_id} "
                "condition_count must be 1 or 2"
            )

        direction = s(
            rule.get("action_direction")
        )
        if direction not in {
            "POSITIVE",
            "NEGATIVE",
        }:
            fail(
                f"Totals master rule_id={rule_id} "
                f"has invalid action_direction={direction!r}"
            )

        if not s(rule.get("source_condition")):
            fail(
                f"Totals master rule_id={rule_id} "
                "has blank source_condition"
            )

        for metric in (
            "historical_hit_rate_pct",
            "lift_vs_family_pct_points",
            "games",
        ):
            require_finite_number(
                rule.get(metric),
                label=(
                    f"Totals master rule_id={rule_id} "
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
                    f"Totals master rule_id={rule_id} "
                    f"has unsupported formula_code={formula!r}"
                )

            if (
                match_type
                not in SUPPORTED_MATCH_TYPES
            ):
                fail(
                    f"Totals master rule_id={rule_id} "
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
                        f"Totals master rule_id={rule_id} "
                        f"condition {number} numeric range "
                        "has no bound"
                    )

                if lower:
                    require_finite_number(
                        lower,
                        label=(
                            f"Totals master rule_id={rule_id} "
                            f"condition {number} lower bound"
                        ),
                    )

                if upper:
                    require_finite_number(
                        upper,
                        label=(
                            f"Totals master rule_id={rule_id} "
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
                    f"Totals master rule_id={rule_id} "
                    f"condition {number} has blank equals_value"
                )

    if not active_supported_ids:
        fail(
            "Totals enrichment master has no active "
            "pipeline-supported rules"
        )

    return active_supported_ids


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
                "totals master rule "
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
                    "totals_direction": s(
                        rule.get("totals_direction")
                    ),
                    "condition": s(
                        rule.get("source_condition")
                    ),
                    "historical_hit_rate_pct": num(
                        rule.get(
                            "historical_hit_rate_pct"
                        )
                    ),
                    "lift_pp": num(
                        rule.get(
                            "lift_vs_family_pct_points"
                        )
                    ),
                    "direction": s(
                        rule.get("action_direction")
                    ),
                    "games": num(
                        rule.get("games")
                    ),
                }
            )

    return matches


def strongest(
    matches,
    totals_direction,
    action_direction,
):
    candidates = [
        match
        for match in matches
        if (
            match["totals_direction"]
            == totals_direction
            and match["direction"]
            == action_direction
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
    g["matched_positive_rule_count"] = len(positive)
    g["matched_negative_rule_count"] = len(negative)
    g["matched_rule_ids"] = join_text(
        match["rule_id"]
        for match in matches
    )
    g["matched_rule_conditions"] = join_text(
        (
            f'{match["rule_id"]}:'
            f'{match["family"]}:'
            f'{match["totals_direction"]}:'
            f'{match["condition"]}'
        )
        for match in matches
    )

    for prefix, totals_direction in [
        ("over", "Over"),
        ("under", "Under"),
    ]:
        direction_matches = [
            match
            for match in matches
            if (
                match["totals_direction"]
                == totals_direction
            )
        ]
        direction_positive = [
            match
            for match in direction_matches
            if match["direction"] == "POSITIVE"
        ]
        direction_negative = [
            match
            for match in direction_matches
            if match["direction"] == "NEGATIVE"
        ]

        g[f"{prefix}_matched_rule_count"] = (
            len(direction_matches)
        )
        g[
            f"{prefix}_matched_positive_rule_count"
        ] = len(direction_positive)
        g[
            f"{prefix}_matched_negative_rule_count"
        ] = len(direction_negative)
        g[f"{prefix}_matched_rule_ids"] = join_text(
            match["rule_id"]
            for match in direction_matches
        )

        positive_item = strongest(
            matches,
            totals_direction,
            "POSITIVE",
        )
        negative_item = strongest(
            matches,
            totals_direction,
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
                f"{prefix}_{label}_rule_id"
            ] = (
                item["rule_id"]
                if item
                else ""
            )
            g[
                f"{prefix}_{label}_hist_hit_rate_pct"
            ] = (
                item["historical_hit_rate_pct"]
                if item
                else ""
            )
            g[
                f"{prefix}_{label}_lift_pp"
            ] = (
                item["lift_pp"]
                if item
                else ""
            )
            g[
                f"{prefix}_{label}_games"
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

process_week = _enrichment_partial(
    _enrichment_process_week,
    game_team_key=game_team_key,
    match_rules=match_rules,
    build_summary_fields=build_summary_fields,
    fail=fail,
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
            "over_matched_rule_count",
            "over_matched_rule_ids",
        ),
        (
            "under_matched_rule_count",
            "under_matched_rule_ids",
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

        over_count = parse_int_text(
            row.get("over_matched_rule_count"),
            label=(
                f"{path.name} game_id={game_id} "
                "over_matched_rule_count"
            ),
        )
        under_count = parse_int_text(
            row.get("under_matched_rule_count"),
            label=(
                f"{path.name} game_id={game_id} "
                "under_matched_rule_count"
            ),
        )

        if over_count + under_count != total_count:
            fail(
                f"{path.name} game_id={game_id} "
                "over+under matched counts do not "
                "equal total"
            )

        all_ids = set(
            split_rule_ids(
                row.get("matched_rule_ids")
            )
        )
        over_ids = set(
            split_rule_ids(
                row.get("over_matched_rule_ids")
            )
        )
        under_ids = set(
            split_rule_ids(
                row.get("under_matched_rule_ids")
            )
        )

        if over_ids | under_ids != all_ids:
            fail(
                f"{path.name} game_id={game_id} "
                "over/under rule ID union does not "
                "equal all matched rule IDs"
            )

        if over_ids & under_ids:
            fail(
                f"{path.name} game_id={game_id} "
                "same rule ID appears in both "
                "Over and Under matches"
            )

        for prefix, direction_ids in (
            ("over", over_ids),
            ("under", under_ids),
        ):
            direction_total = parse_int_text(
                row.get(
                    f"{prefix}_matched_rule_count"
                ),
                label=(
                    f"{path.name} game_id={game_id} "
                    f"{prefix}_matched_rule_count"
                ),
            )
            direction_positive = parse_int_text(
                row.get(
                    f"{prefix}_matched_positive_rule_count"
                ),
                label=(
                    f"{path.name} game_id={game_id} "
                    f"{prefix}_matched_positive_rule_count"
                ),
            )
            direction_negative = parse_int_text(
                row.get(
                    f"{prefix}_matched_negative_rule_count"
                ),
                label=(
                    f"{path.name} game_id={game_id} "
                    f"{prefix}_matched_negative_rule_count"
                ),
            )

            if (
                direction_positive
                + direction_negative
                != direction_total
            ):
                fail(
                    f"{path.name} game_id={game_id} "
                    f"{prefix} positive+negative counts "
                    "do not equal direction total"
                )

            for polarity in (
                "positive",
                "negative",
            ):
                strongest_id = s(
                    row.get(
                        f"{prefix}_strongest_"
                        f"{polarity}_rule_id"
                    )
                )

                if (
                    strongest_id
                    and strongest_id
                    not in direction_ids
                ):
                    fail(
                        f"{path.name} game_id={game_id} "
                        f"{prefix} strongest {polarity} "
                        "rule is absent from direction matches"
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
            prefix=".totals_enrichment_stage_",
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
                    "staged totals enrichment "
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
                    "Staged totals enrichment differs "
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
                "Staged totals managed file set mismatch "
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
        cleanup_warning='Totals enrichment published but temporary backup cleanup failed',
    )


run = _enrichment_bind_run(
    globals(),
    market_name="totals",
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
                    "historical totals enrichment"
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
