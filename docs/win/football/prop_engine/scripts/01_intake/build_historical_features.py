#!/usr/bin/env python3
"""
Assemble the canonical historical Prop Engine feature table.

READS:
    docs/win/football/prop_engine/config/prop_engine.yaml
    docs/win/football/prop_engine/data/historical/universe/player_game_universe.parquet
    docs/win/football/prop_engine/data/historical/targets/player_game_targets.parquet
    docs/win/football/prop_engine/data/historical/features/player_role_history.parquet
    docs/win/football/prop_engine/01_intake/player_form.parquet
    docs/win/football/prop_engine/data/historical/features/team_form.parquet
    docs/win/football/prop_engine/data/historical/features/opponent_form.parquet
    docs/win/football/prop_engine/data/historical/features/environment.parquet
    docs/win/football/prop_engine/data/historical/features/defensive_features.parquet
    docs/win/football/prop_engine/data/historical/features/kicking_features.parquet
    docs/win/football/prop_engine/data/historical/opportunity/position_allowed_week.parquet

WRITES:
    docs/win/football/prop_engine/01_intake/player_game_features.parquet
    docs/win/football/prop_engine/01_intake/feature_manifest.json
    docs/win/football/prop_engine/errors/01_intake/build_historical_features.json

POLICY:
    - Canonical grain is season + week + game_id + player_id.
    - Target-game outcomes are stored only in target_* columns and never enter
      the model feature manifest.
    - Weekly player/team/opponent form inputs are already strictly pregame.
    - Position-allowed realized weekly values are shifted one observed defense
      game before joining to the target row.
    - No sportsbook, market, score-result, same-game snap, same-game
      participation, or played_game_flag field is exposed as a model feature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import gc
import hashlib
import json
import os
import sys
import tempfile
import uuid

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common
from pipeline_reporter import PipelineReporter


GRAIN = ["season", "week", "game_id", "player_id"]

LEADING_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "gameday",
    "kickoff_timestamp",
    "player_id",
    "player_name",
    "team",
    "opponent",
    "position",
    "position_group",
    "home_flag",
]

REQUIRED_MATCHUP_COLUMNS = [
    "matchup_expected_team_plays",
    "matchup_expected_team_dropbacks",
    "matchup_expected_team_rush_attempts",
    "matchup_expected_opponent_plays",
    "matchup_expected_opponent_dropbacks",
    "matchup_player_target_share_x_opp_targets",
    "matchup_player_carry_share_x_opp_rushes",
    "matchup_player_tackle_rate_x_opp_plays",
    "matchup_player_sack_rate_x_opp_plays",
    "matchup_off_epa_vs_def_epa",
    "matchup_pass_rate_vs_opponent",
    "matchup_rush_rate_vs_opponent",
]

AUDIT_COLUMNS = [
    "audit_feature_asof",
    "audit_max_player_source_game",
    "audit_max_team_source_week",
    "audit_depth_snapshot_at",
    "audit_injury_snapshot_at",
    "audit_market_feature_count",
    "audit_row_valid",
]

ROLE_KEYS = set(GRAIN + ["team", "position"])
PLAYER_FORM_KEYS = set(
    GRAIN + ["team", "position", "position_group"]
)
TEAM_FORM_KEYS = {"season", "week", "team"}
PLAYER_HISTORY_COLUMNS = [
    "no_nfl_history_flag",
    "new_team_flag",
    "history_games",
]

DEFENSIVE_ROLE_MAP = {
    "starter_flag": "role_defensive_starter_flag",
    "front7_flag": "role_front7_flag",
    "secondary_flag": "role_secondary_flag",
}

KICKING_ROLE_MAP = {
    "primary_kicker_flag": "role_primary_kicker_flag",
}

KICKING_REDUNDANT_ENVIRONMENT = {
    "temperature",
    "wind",
    "roof",
    "surface",
}

POSITION_ALLOWED_KEYS = [
    "season",
    "week",
    "defense_team",
    "offense_position_group",
]

HISTORICAL_FRANCHISE_ALIASES = {
    "SD": "LAC",
    "OAK": "LV",
    "STL": "LAR",
}

TEAM_FORM_SAFE_SUFFIXES = (
    "_lag1",
    "_roll3_mean",
    "_roll5_mean",
    "_roll8_mean",
    "_ewm3",
    "_ewm5",
    "_season_to_date",
)

VALID_TARGET_TYPES = {
    "continuous_signed",
    "count_nonnegative",
    "derived_count",
}

PLAYER_FORM_METRICS = [
    "pass_attempts",
    "dropbacks",
    "completions",
    "passing_yards",
    "passing_tds",
    "yards_per_attempt",
    "passing_td_rate",
    "passing_air_yards",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "yards_per_carry",
    "carry_share",
    "red_zone_carries",
    "goal_line_carries",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "yards_per_target",
    "catch_rate",
    "target_share",
    "air_yards_share",
    "red_zone_targets",
    "red_zone_target_share",
    "field_goal_attempts",
    "field_goals_made",
    "extra_point_attempts",
    "extra_points_made",
    "tackles",
    "sacks",
    "qb_hits",
    "tackle_rate_per_def_play",
    "sack_rate_per_def_play",
    "qb_hit_rate_per_def_play",
    "offense_snap_pct",
    "defense_snap_pct",
    "offense_participation",
    "defense_participation",
]

PLAYER_FORM_SUFFIXES = [
    "lag1",
    "roll3_mean",
    "roll5_mean",
    "roll8_mean",
    "roll3_median",
    "roll5_std",
    "ewm3",
    "ewm5",
    "season_to_date",
    "career_prior",
]

TEAM_FORM_METRICS = [
    "offensive_plays",
    "drives",
    "dropbacks",
    "pass_attempts",
    "rush_attempts",
    "pass_rate",
    "rush_rate",
    "points_per_drive",
    "red_zone_drives",
    "red_zone_pass_attempts",
    "red_zone_rush_attempts",
    "goal_line_rush_attempts",
    "field_goal_attempts",
    "extra_point_attempts",
    "off_epa_per_play",
    "off_success_rate",
    "yards_per_play",
    "red_zone_td_rate",
    "early_down_epa",
    "third_down_conversion_rate",
]

OPPONENT_FORM_METRICS = [
    "defensive_plays",
    "opponent_dropbacks",
    "opponent_pass_attempts",
    "opponent_rush_attempts",
    "passing_yards_allowed",
    "rushing_yards_allowed",
    "passing_tds_allowed",
    "rushing_tds_allowed",
    "sacks",
    "qb_hits",
    "red_zone_pass_attempts_allowed",
    "red_zone_rush_attempts_allowed",
    "goal_line_rush_attempts_allowed",
    "def_epa_per_play",
    "def_success_rate",
    "yards_per_play_allowed",
    "points_per_drive_allowed",
    "red_zone_td_rate_allowed",
]

ROLE_SOURCE_COLUMNS = GRAIN + [
    "team",
    "position",
    "depth_rank_pregame",
    "depth_starter_flag_pregame",
    "injury_status_pregame",
    "injury_out_flag",
    "injury_doubtful_flag",
    "injury_questionable_flag",
    "prior_offense_snap_pct",
    "prior_defense_snap_pct",
    "snap_pct_roll3",
    "snap_pct_roll5",
    "snap_pct_ewm3",
    "snap_pct_ewm5",
    "prior_offense_participation",
    "prior_defense_participation",
    "participation_roll3",
    "participation_roll5",
    "depth_rank_change",
    "snap_share_change",
    "participation_change",
    "team_change_flag",
    "games_with_current_team_before_game",
    "starter_promotion_flag",
    "starter_demotion_flag",
    "teammate_out_count_position",
    "teammate_unavailable_snap_share_position",
    "role_history_games",
    "role_missing_flag",
]

PLAYER_FORM_SOURCE_COLUMNS = (
    GRAIN
    + ["team", "position", "position_group"]
    + [
        f"{metric}_{suffix}"
        for metric in PLAYER_FORM_METRICS
        for suffix in PLAYER_FORM_SUFFIXES
    ]
    + PLAYER_HISTORY_COLUMNS
)

TEAM_FORM_SOURCE_COLUMNS = (
    ["season", "week", "team"]
    + [
        f"{metric}{suffix}"
        for metric in TEAM_FORM_METRICS
        for suffix in TEAM_FORM_SAFE_SUFFIXES
    ]
)

OPPONENT_FORM_SOURCE_COLUMNS = (
    ["season", "week", "team"]
    + [
        f"{metric}{suffix}"
        for metric in OPPONENT_FORM_METRICS
        for suffix in TEAM_FORM_SAFE_SUFFIXES
    ]
)

DEFENSIVE_SOURCE_COLUMNS = GRAIN + [
    "position",
    "def_snap_pct_lag1",
    "def_snap_pct_roll3",
    "def_participation_lag1",
    "def_participation_roll3",
    "tackles_lag1",
    "tackles_roll3",
    "tackles_roll5",
    "tackle_rate_roll3",
    "tackle_rate_roll5",
    "sacks_lag1",
    "sacks_roll3",
    "sacks_roll5",
    "sack_rate_roll5",
    "qb_hits_roll3",
    "qb_hits_roll5",
    "opponent_plays_roll3",
    "opponent_dropbacks_roll3",
    "opponent_rush_rate_roll3",
    "opponent_pass_rate_roll3",
    "team_def_sack_rate_roll3",
    "starter_flag",
    "front7_flag",
    "secondary_flag",
]

KICKING_SOURCE_COLUMNS = GRAIN + [
    "team",
    "fg_attempts_lag1",
    "fg_attempts_roll3",
    "fg_attempts_roll5",
    "fg_make_pct_career_prior",
    "fg_make_pct_season_prior",
    "pat_attempts_roll3",
    "pat_make_pct_career_prior",
    "team_drives_roll3",
    "team_points_per_drive_roll3",
    "team_red_zone_td_rate_roll3",
    "opponent_points_per_drive_allowed_roll3",
    "opponent_red_zone_td_rate_allowed_roll3",
    "temperature",
    "wind",
    "roof",
    "surface",
    "primary_kicker_flag",
]

POSITION_ALLOWED_VALUE_COLUMNS = [
    "players_faced",
    "targets_allowed",
    "receptions_allowed",
    "receiving_yards_allowed",
    "receiving_tds_allowed",
    "carries_allowed",
    "rushing_yards_allowed",
    "rushing_tds_allowed",
    "passing_yards_allowed",
    "passing_tds_allowed",
    "tackles_generated",
    "raw_rate_sample_size",
    "league_rate",
    "shrunk_rate",
]

POSITION_ALLOWED_SOURCE_COLUMNS = (
    POSITION_ALLOWED_KEYS
    + POSITION_ALLOWED_VALUE_COLUMNS
)

ENVIRONMENT_NUMERIC_COLUMNS = [
    "divisional_game_flag",
    "neutral_site_flag",
    "temperature",
    "wind",
    "home_rest_days",
    "away_rest_days",
    "miles_traveled_away",
    "time_zones_crossed_away",
    "east_to_west_flag",
    "west_to_east_flag",
    "international_flag",
    "weather_missing_flag",
    "travel_missing_flag",
]
FINAL_SCORE_FORBIDDEN_NAMES = {
    "score",
    "home_score",
    "away_score",
    "final_score",
    "score_differential",
    "point_differential",
    "margin",
    "result",
    "win_flag",
    "loss_flag",
}

ENVIRONMENT_DIRECT_COLUMNS = [
    "divisional_game_flag",
    "neutral_site_flag",
    "stadium",
    "stadium_id",
    "roof",
    "surface",
    "temperature",
    "wind",
    "international_flag",
    "weather_missing_flag",
    "travel_missing_flag",
]

ENVIRONMENT_REQUIRED_COLUMNS = [
    "season",
    "week",
    "game_id",
    "gameday",
    "home_team",
    "away_team",
    "divisional_game_flag",
    "neutral_site_flag",
    "stadium",
    "stadium_id",
    "roof",
    "surface",
    "temperature",
    "wind",
    "home_rest_days",
    "away_rest_days",
    "miles_traveled_away",
    "time_zones_crossed_away",
    "east_to_west_flag",
    "west_to_east_flag",
    "international_flag",
    "weather_missing_flag",
    "travel_missing_flag",
]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"}:
        return ""
    return text


def canonical_franchise(value: Any) -> str:
    team = common.normalize_team(value)
    return HISTORICAL_FRANCHISE_ALIASES.get(team, team)



def normalize_position_group(value: Any) -> str:
    return clean_text(value).upper()


def normalize_position(value: Any) -> str:
    return clean_text(value).upper().replace(" ", "")


def validate_target_config(
    config: dict,
) -> tuple[list[str], list[str]]:
    targets = config.get("targets")

    if not isinstance(targets, dict) or not targets:
        raise ValueError(
            "Config section 'targets' must be a non-empty mapping."
        )

    order: list[str] = []

    for raw_name, spec in targets.items():
        name = clean_text(raw_name)

        if not name or name != str(raw_name):
            raise ValueError(
                f"Invalid configured target name: {raw_name!r}"
            )

        if not isinstance(spec, dict):
            raise ValueError(
                f"Configured target {name!r} must be a mapping."
            )

        target_type = clean_text(spec.get("type"))

        if target_type not in VALID_TARGET_TYPES:
            raise ValueError(
                f"Configured target {name!r} has invalid type "
                f"{target_type!r}; expected one of "
                f"{sorted(VALID_TARGET_TYPES)}."
            )

        order.append(name)

    target_columns = [
        f"target_{name}"
        for name in order
    ]

    if len(target_columns) != len(set(target_columns)):
        raise ValueError(
            "Configured target names produce duplicate output columns."
        )

    return order, target_columns


def validate_exact_columns(
    frame: pd.DataFrame,
    expected: list[str],
    label: str,
) -> None:
    duplicate_columns = (
        frame.columns[
            frame.columns.duplicated()
        ]
        .astype(str)
        .tolist()
    )

    if duplicate_columns:
        raise ValueError(
            f"{label}: duplicate source columns: "
            f"{duplicate_columns[:20]}"
        )

    expected_set = set(expected)
    actual_set = set(frame.columns)

    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)

    if missing or unexpected:
        raise ValueError(
            f"{label}: source schema mismatch. "
            f"missing={missing[:30]} "
            f"unexpected={unexpected[:30]}"
        )


def strict_numeric(
    series: pd.Series,
    *,
    label: str | None = None,
) -> pd.Series:
    converted = pd.to_numeric(
        series,
        errors="coerce",
    ).astype("float64")

    invalid = (
        series.map(clean_text).ne("")
        & converted.isna()
    )

    if invalid.any():
        sample = (
            series.loc[invalid]
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"{label or clean_text(series.name) or 'numeric source'} "
            f"contains nonnumeric value(s); sample={sample}"
        )

    infinite = pd.Series(
        np.isinf(
            converted.to_numpy(
                dtype="float64",
                copy=False,
            )
        ),
        index=converted.index,
    )

    if infinite.any():
        sample = (
            series.loc[infinite]
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"{label or clean_text(series.name) or 'numeric source'} "
            f"contains infinite value(s); sample={sample}"
        )

    return converted


def validate_numeric_columns(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    label: str,
) -> None:
    common.require_columns(
        frame,
        columns,
        label,
    )

    for column in columns:
        frame[column] = strict_numeric(
            frame[column],
            label=f"{label}.{column}",
        )


def validate_source_metadata(
    universe: pd.DataFrame,
    source: pd.DataFrame,
    *,
    label: str,
    mappings: list[tuple[str, str, str]],
) -> None:
    source_columns = [
        source_column
        for source_column, _, _ in mappings
    ]
    universe_columns = list(
        dict.fromkeys(
            universe_column
            for _, universe_column, _ in mappings
        )
    )

    common.require_columns(
        source,
        GRAIN + source_columns,
        label,
    )
    common.require_columns(
        universe,
        GRAIN + universe_columns,
        "historical universe metadata",
    )

    left = source[GRAIN + source_columns].copy()
    right = universe[GRAIN + universe_columns].copy()

    for index, (
        source_column,
        universe_column,
        _,
    ) in enumerate(mappings):
        left = left.rename(
            columns={
                source_column: f"__source_{index}",
            }
        )
        right = right.rename(
            columns={
                universe_column: f"__universe_{index}",
            }
        )

    probe = left.merge(
        right,
        on=GRAIN,
        how="left",
        indicator=True,
        validate="one_to_one",
    )

    missing = probe["_merge"].ne("both")

    if missing.any():
        sample = (
            probe.loc[missing, GRAIN]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            f"{label}: metadata row absent from historical universe; "
            f"sample={sample}"
        )

    for index, (
        source_column,
        universe_column,
        kind,
    ) in enumerate(mappings):
        source_values = probe[f"__source_{index}"]
        universe_values = probe[f"__universe_{index}"]

        if kind == "team":
            source_values = source_values.map(
                canonical_franchise
            )
            universe_values = universe_values.map(
                canonical_franchise
            )
        elif kind == "position":
            source_values = source_values.map(
                normalize_position
            )
            universe_values = universe_values.map(
                normalize_position
            )
        elif kind == "position_group":
            source_values = source_values.map(
                normalize_position_group
            )
            universe_values = universe_values.map(
                normalize_position_group
            )
        else:
            raise ValueError(
                f"Unsupported metadata comparison kind: {kind}"
            )

        mismatch = source_values.ne(universe_values)

        if mismatch.any():
            sample = (
                probe.loc[mismatch, GRAIN]
                .head(10)
                .to_dict("records")
            )
            raise ValueError(
                f"{label}: {source_column} disagrees with "
                f"historical universe {universe_column}; "
                f"sample={sample}"
            )


def validate_key_coverage(
    required: pd.DataFrame,
    source: pd.DataFrame,
    *,
    required_columns: list[str],
    source_columns: list[str],
    label: str,
) -> None:
    if len(required_columns) != len(source_columns):
        raise ValueError(
            f"{label}: coverage-key definition length mismatch."
        )

    common.require_columns(
        required,
        required_columns,
        f"{label} required keys",
    )
    common.require_columns(
        source,
        source_columns,
        f"{label} source keys",
    )

    left = (
        required[required_columns]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    right = (
        source[source_columns]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    right.columns = required_columns

    probe = left.merge(
        right,
        on=required_columns,
        how="left",
        indicator=True,
        validate="one_to_one",
    )

    missing = probe["_merge"].ne("both")

    if missing.any():
        sample = (
            probe.loc[missing, required_columns]
            .head(20)
            .to_dict("records")
        )
        raise ValueError(
            f"{label}: missing required historical-universe "
            f"coverage; sample={sample}"
        )


def require_config_path(config: dict, key: str) -> str:
    value = config.get("paths", {}).get(key)
    if not value:
        raise ValueError(
            f"Issue 17 requires config paths.{key}."
        )
    return str(value)


def stable_schema_hash(df: pd.DataFrame) -> str:
    payload = "\n".join(
        f"{column}:{df[column].dtype}"
        for column in df.columns
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()



def safe_mean_pair(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    a = strict_numeric(
        left,
        label=clean_text(left.name) or "mean-left",
    )
    b = strict_numeric(
        right,
        label=clean_text(right.name) or "mean-right",
    )
    return pd.concat(
        [a, b],
        axis=1,
    ).mean(
        axis=1,
        skipna=True,
    )



def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    num = strict_numeric(
        numerator,
        label=clean_text(numerator.name) or "numerator",
    )
    den = strict_numeric(
        denominator,
        label=clean_text(denominator.name) or "denominator",
    )

    result = pd.Series(
        np.nan,
        index=num.index,
        dtype="float64",
    )

    valid = (
        num.notna()
        & den.notna()
        & den.ne(0.0)
    )

    result.loc[valid] = (
        num.loc[valid]
        / den.loc[valid]
    )

    return result



def safe_product(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    a = strict_numeric(
        left,
        label=clean_text(left.name) or "product-left",
    )
    b = strict_numeric(
        right,
        label=clean_text(right.name) or "product-right",
    )

    result = a * b
    result.loc[
        a.isna() | b.isna()
    ] = np.nan

    return result.astype("float64")


def validate_exact_full_grain(
    universe_keys: pd.DataFrame,
    source: pd.DataFrame,
    label: str,
) -> None:
    common.require_columns(source, GRAIN, label)
    common.ensure_unique(source, GRAIN, f"{label} grain")
    left = (
        universe_keys[GRAIN]
        .sort_values(GRAIN, kind="mergesort")
        .reset_index(drop=True)
    )
    right = (
        source[GRAIN]
        .sort_values(GRAIN, kind="mergesort")
        .reset_index(drop=True)
    )
    if len(left) != len(right) or not left.equals(right):
        raise ValueError(
            f"{label}: grain does not exactly match historical universe."
        )


def validate_sparse_grain_subset(
    universe_keys: pd.DataFrame,
    source: pd.DataFrame,
    label: str,
) -> None:
    common.require_columns(source, GRAIN, label)
    common.ensure_unique(source, GRAIN, f"{label} grain")
    probe = source[GRAIN].merge(
        universe_keys[GRAIN],
        on=GRAIN,
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    bad = probe["_merge"].ne("both")
    if bad.any():
        sample = probe.loc[bad, GRAIN].head(10).to_dict("records")
        raise ValueError(
            f"{label}: contains rows outside universe; sample={sample}"
        )




def build_position_allowed_lag(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    validate_exact_columns(
        source,
        POSITION_ALLOWED_SOURCE_COLUMNS,
        "position allowed",
    )

    common.ensure_unique(
        source,
        POSITION_ALLOWED_KEYS,
        "position allowed grain",
    )

    data = source.copy()

    data["season"] = pd.to_numeric(
        data["season"],
        errors="raise",
    ).astype(int)

    data["week"] = pd.to_numeric(
        data["week"],
        errors="raise",
    ).astype(int)

    data["_join_defense"] = data[
        "defense_team"
    ].map(canonical_franchise)

    data["_join_position_group"] = data[
        "offense_position_group"
    ].map(normalize_position_group)

    if (
        data["_join_defense"].eq("").any()
        or data["_join_position_group"].eq("").any()
    ):
        raise ValueError(
            "position allowed contains blank canonical "
            "defense/position group."
        )

    value_columns = list(
        POSITION_ALLOWED_VALUE_COLUMNS
    )

    validate_numeric_columns(
        data,
        value_columns,
        label="position allowed",
    )

    data = data.sort_values(
        [
            "_join_defense",
            "_join_position_group",
            "season",
            "week",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    grouped = data.groupby(
        [
            "_join_defense",
            "_join_position_group",
        ],
        sort=False,
        dropna=False,
    )

    output = data[
        [
            "season",
            "week",
            "_join_defense",
            "_join_position_group",
        ]
    ].copy()

    matchup_columns: list[str] = []

    for column in value_columns:
        output_name = (
            f"matchup_position_allowed_{column}_lag1"
        )
        output[output_name] = (
            grouped[column].shift(1)
        )
        matchup_columns.append(output_name)

    common.ensure_unique(
        output,
        [
            "season",
            "week",
            "_join_defense",
            "_join_position_group",
        ],
        "lagged position allowed join grain",
    )

    return output, matchup_columns



def build_player_audit(
    universe: pd.DataFrame,
) -> pd.DataFrame:
    common.require_columns(
        universe,
        GRAIN
        + [
            "kickoff_timestamp",
            "played_game_flag",
        ],
        "historical universe player audit",
    )

    audit = universe[
        GRAIN
        + [
            "kickoff_timestamp",
            "played_game_flag",
        ]
    ].copy()

    audit["_kickoff_sort"] = pd.to_datetime(
        audit["kickoff_timestamp"],
        errors="raise",
        utc=True,
    )

    audit = audit.sort_values(
        [
            "player_id",
            "_kickoff_sort",
            "game_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    played_numeric = strict_numeric(
        audit["played_game_flag"],
        label="historical universe.played_game_flag",
    )

    invalid_played = (
        played_numeric.notna()
        & ~played_numeric.isin([0.0, 1.0])
    )

    if invalid_played.any():
        raise ValueError(
            "historical universe.played_game_flag "
            "must contain only 0/1/null."
        )

    played = (
        played_numeric
        .fillna(0.0)
        .eq(1.0)
    )

    audit["_played_source_game"] = (
        audit["game_id"]
        .astype("string")
        .where(played)
    )

    audit["audit_max_player_source_game"] = (
        audit.groupby(
            "player_id",
            sort=False,
        )["_played_source_game"]
        .transform(
            lambda series:
            series.ffill().shift(1)
        )
    )

    return audit[
        GRAIN
        + [
            "audit_max_player_source_game",
        ]
    ]


def build_team_source_audit(
    team_form: pd.DataFrame,
) -> pd.DataFrame:
    common.require_columns(
        team_form,
        ["season", "week", "team"],
        "team form audit",
    )

    keys = team_form[["season", "week", "team"]].copy()
    keys["season"] = pd.to_numeric(keys["season"], errors="raise").astype(int)
    keys["week"] = pd.to_numeric(keys["week"], errors="raise").astype(int)
    keys["_join_team"] = keys["team"].map(canonical_franchise)

    common.ensure_unique(
        keys,
        ["season", "week", "_join_team"],
        "team form canonical audit grain",
    )

    keys = keys.sort_values(
        ["_join_team", "season", "week"],
        kind="mergesort",
    ).reset_index(drop=True)

    source_label = (
        keys["season"].astype(str)
        + "-W"
        + keys["week"].astype(str).str.zfill(2)
    )

    keys["audit_max_team_source_week"] = (
        source_label.groupby(keys["_join_team"], sort=False).shift(1)
    )

    return keys[
        [
            "season",
            "week",
            "_join_team",
            "audit_max_team_source_week",
        ]
    ]



def _resolve_prop_destination(
    value: str,
) -> Path:
    destination = Path(value)

    if not destination.is_absolute():
        destination = (
            common.repo_root()
            / destination
        )

    destination = destination.resolve()
    root = common.prop_root().resolve()

    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Output write outside Prop Engine is forbidden: "
            f"{destination}"
        ) from exc

    return destination


def write_output_bundle_atomic(
    frame: pd.DataFrame,
    parquet_path: str,
    manifest: dict,
    manifest_path: str,
) -> None:
    parquet_destination = _resolve_prop_destination(
        parquet_path
    )
    manifest_destination = _resolve_prop_destination(
        manifest_path
    )

    if parquet_destination == manifest_destination:
        raise ValueError(
            "Parquet and manifest destinations must differ."
        )

    for destination in (
        parquet_destination,
        manifest_destination,
    ):
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    parquet_handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{parquet_destination.name}.",
        suffix=".tmp",
        dir=parquet_destination.parent,
        delete=False,
    )
    parquet_temp = Path(parquet_handle.name)
    parquet_handle.close()

    manifest_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{manifest_destination.name}.",
        suffix=".tmp",
        dir=manifest_destination.parent,
        delete=False,
    )
    manifest_temp = Path(manifest_handle.name)

    backups: dict[Path, Path] = {}
    committed: list[Path] = []

    try:
        ordered = common.season_week_sort(
            frame
        )
        ordered.to_parquet(
            parquet_temp,
            index=False,
        )

        if (
            not parquet_temp.is_file()
            or parquet_temp.stat().st_size == 0
        ):
            raise RuntimeError(
                "Staged historical feature parquet is empty."
            )

        with manifest_handle:
            json.dump(
                manifest,
                manifest_handle,
                indent=2,
                sort_keys=False,
                ensure_ascii=False,
                allow_nan=False,
            )
            manifest_handle.write("\n")
            manifest_handle.flush()
            os.fsync(
                manifest_handle.fileno()
            )

        staged_manifest = json.loads(
            manifest_temp.read_text(
                encoding="utf-8",
            )
        )

        if staged_manifest != manifest:
            raise RuntimeError(
                "Staged feature manifest failed "
                "round-trip validation."
            )

        destinations = (
            parquet_destination,
            manifest_destination,
        )

        for destination in destinations:
            if destination.exists():
                backup = destination.with_name(
                    f".{destination.name}."
                    f"{uuid.uuid4().hex}.bak"
                )
                os.replace(
                    destination,
                    backup,
                )
                backups[destination] = backup

        os.replace(
            parquet_temp,
            parquet_destination,
        )
        committed.append(
            parquet_destination
        )

        os.replace(
            manifest_temp,
            manifest_destination,
        )
        committed.append(
            manifest_destination
        )

    except Exception as exc:
        rollback_errors: list[str] = []

        for destination in reversed(committed):
            try:
                destination.unlink(
                    missing_ok=True
                )
            except Exception as rollback_exc:
                rollback_errors.append(
                    f"remove {destination}: "
                    f"{rollback_exc}"
                )

        for destination, backup in backups.items():
            try:
                if backup.exists():
                    os.replace(
                        backup,
                        destination,
                    )
            except Exception as rollback_exc:
                rollback_errors.append(
                    f"restore {destination}: "
                    f"{rollback_exc}"
                )

        if rollback_errors:
            raise RuntimeError(
                "Historical output bundle write failed "
                "and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from exc

        raise

    else:
        for backup in backups.values():
            try:
                backup.unlink(
                    missing_ok=True
                )
            except Exception:
                pass

    finally:
        for temporary in (
            parquet_temp,
            manifest_temp,
        ):
            try:
                temporary.unlink(
                    missing_ok=True
                )
            except Exception:
                pass


def validate_form_column_names(
    columns: list[str],
    label: str,
) -> None:
    bad = [
        column
        for column in columns
        if not column.endswith(TEAM_FORM_SAFE_SUFFIXES)
    ]
    if bad:
        raise ValueError(
            f"{label}: non-lagged form columns detected: {bad[:20]}"
        )


def validate_no_same_game_usage_features(
    feature_columns: list[str],
) -> None:
    usage_tokens = ("snap", "participation")
    safe_tokens = (
        "prior",
        "lag",
        "roll",
        "ewm",
        "change",
        "unavailable",
        "missing",
        "season_to_date",
        "career_prior",
    )

    bad: list[str] = []
    for column in feature_columns:
        lowered = column.casefold()
        if not any(token in lowered for token in usage_tokens):
            continue
        if not any(token in lowered for token in safe_tokens):
            bad.append(column)

    if bad:
        raise ValueError(
            "Potential same-game snap/participation features detected: "
            + ", ".join(bad[:20])
        )


def validate_no_final_score_features(
    feature_columns: list[str],
) -> None:
    bad = [
        column
        for column in feature_columns
        if column.casefold() in FINAL_SCORE_FORBIDDEN_NAMES
        or "final_score" in column.casefold()
    ]
    if bad:
        raise ValueError(
            f"Final-score/result features are forbidden: {bad}"
        )


def classify_feature_columns(
    frame: pd.DataFrame,
    candidate_columns: list[str],
) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []

    for column in candidate_columns:
        dtype = frame[column].dtype

        if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            numeric.append(column)
        elif (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        ):
            categorical.append(column)
        else:
            raise ValueError(
                f"Unclassified model-feature dtype: {column}={dtype}"
            )

    return numeric, categorical


def run(reporter: PipelineReporter) -> int:
    config = common.load_config()

    required_target_order, target_columns = (
        validate_target_config(config)
    )

    paths = {
        "universe": require_config_path(config, "historical_universe"),
        "targets": require_config_path(config, "historical_targets"),
        "role": require_config_path(config, "role_history"),
        "player": require_config_path(config, "player_form"),
        "team": require_config_path(config, "team_form"),
        "opponent": require_config_path(config, "opponent_form"),
        "environment": require_config_path(config, "environment_history"),
        "defensive": require_config_path(config, "defensive_features"),
        "kicking": require_config_path(config, "kicking_features"),
        "position_allowed": require_config_path(config, "position_allowed"),
        "output": require_config_path(config, "historical_features"),
    }

    manifest_path = (
        "docs/win/football/prop_engine/01_intake/"
        "feature_manifest.json"
    )

    reporter.add_input(
        "docs/win/football/prop_engine/config/prop_engine.yaml"
    )

    for key, value in paths.items():
        if key == "output":
            reporter.add_output(value)
        else:
            reporter.add_input(value)

    reporter.add_output(manifest_path)

    universe = common.read_parquet_required(
        paths["universe"],
        LEADING_COLUMNS + ["played_game_flag"],
    )
    common.ensure_unique(universe, GRAIN, "historical universe")

    validate_numeric_columns(
        universe,
        ["home_flag"],
        label="historical universe",
    )

    invalid_home_flag = (
        universe["home_flag"].notna()
        & ~universe["home_flag"].isin([0.0, 1.0])
    )
    if invalid_home_flag.any():
        raise ValueError(
            "historical universe.home_flag must contain only 0/1/null."
        )

    universe_keys = universe[GRAIN].copy()

    universe_row_count = len(universe)
    out = universe[LEADING_COLUMNS].copy()
    out["_join_team"] = out["team"].map(canonical_franchise)
    out["_join_opponent"] = out["opponent"].map(canonical_franchise)
    out["_join_position_group"] = out["position_group"].map(
        normalize_position_group
    )

    if out["_join_team"].eq("").any() or out["_join_opponent"].eq("").any():
        raise ValueError("Universe contains blank canonical team/opponent.")

    role_columns: list[str] = []
    player_columns: list[str] = []
    team_columns: list[str] = []
    opponent_columns: list[str] = []
    matchup_columns: list[str] = []
    environment_columns: list[str] = []
    history_columns: list[str] = []

    # Role history: full-universe one-to-one join.
    role = common.read_parquet_required(paths["role"], GRAIN)

    validate_exact_columns(
        role,
        ROLE_SOURCE_COLUMNS,
        "role history",
    )
    validate_exact_full_grain(
        universe_keys,
        role,
        "role history",
    )
    validate_source_metadata(
        out,
        role,
        label="role history",
        mappings=[
            ("team", "team", "team"),
            ("position", "position", "position"),
        ],
    )

    role_source_columns = [
        column
        for column in role.columns
        if column not in ROLE_KEYS
    ]
    validate_numeric_columns(
        role,
        [
            column
            for column in role_source_columns
            if column != "injury_status_pregame"
        ],
        label="role history",
    )

    role_renamed = {
        column: f"role_{column}"
        for column in role_source_columns
    }
    role_columns.extend(role_renamed.values())

    out = out.merge(
        role[GRAIN + role_source_columns].rename(columns=role_renamed),
        on=GRAIN,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    del role
    gc.collect()

    # Player form: full-universe one-to-one join. History flags get their
    # own family; all rolling/form values get player_*.
    player = common.read_parquet_required(paths["player"], GRAIN)

    validate_exact_columns(
        player,
        PLAYER_FORM_SOURCE_COLUMNS,
        "player form",
    )
    validate_exact_full_grain(
        universe_keys,
        player,
        "player form",
    )
    validate_source_metadata(
        out,
        player,
        label="player form",
        mappings=[
            ("team", "team", "team"),
            ("position", "position", "position"),
            (
                "position_group",
                "position_group",
                "position_group",
            ),
        ],
    )

    for required_history in PLAYER_HISTORY_COLUMNS:
        if required_history not in player.columns:
            raise ValueError(
                f"player form missing required history field {required_history}"
            )

    player_source_columns = [
        column
        for column in player.columns
        if column not in PLAYER_FORM_KEYS
        and column not in PLAYER_HISTORY_COLUMNS
    ]
    validate_numeric_columns(
        player,
        player_source_columns
        + PLAYER_HISTORY_COLUMNS,
        label="player form",
    )

    player_renamed = {
        column: f"player_{column}"
        for column in player_source_columns
    }
    player_columns.extend(player_renamed.values())

    history_renamed = {
        column: f"history_{column}"
        for column in PLAYER_HISTORY_COLUMNS
    }
    history_columns.extend(history_renamed.values())

    out = out.merge(
        player[
            GRAIN + player_source_columns + PLAYER_HISTORY_COLUMNS
        ].rename(columns={**player_renamed, **history_renamed}),
        on=GRAIN,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    del player
    gc.collect()

    # Team form: join player's offense by current game-specific team.
    team = common.read_parquet_required(
        paths["team"],
        ["season", "week", "team"],
    )
    validate_exact_columns(
        team,
        TEAM_FORM_SOURCE_COLUMNS,
        "team form",
    )
    common.ensure_unique(
        team,
        ["season", "week", "team"],
        "team form",
    )
    team_feature_source = [
        column
        for column in team.columns
        if column not in TEAM_FORM_KEYS
    ]
    validate_form_column_names(
        team_feature_source,
        "team form",
    )
    validate_numeric_columns(
        team,
        team_feature_source,
        label="team form",
    )

    team["_join_team"] = team["team"].map(canonical_franchise)
    if team["_join_team"].eq("").any():
        raise ValueError(
            "team form contains blank canonical team."
        )

    common.ensure_unique(
        team,
        ["season", "week", "_join_team"],
        "team form canonical join grain",
    )

    validate_key_coverage(
        out,
        team,
        required_columns=[
            "season",
            "week",
            "_join_team",
        ],
        source_columns=[
            "season",
            "week",
            "_join_team",
        ],
        label="team form",
    )

    team_renamed = {
        column: f"team_{column}"
        for column in team_feature_source
    }
    team_columns.extend(team_renamed.values())

    out = out.merge(
        team[
            ["season", "week", "_join_team"] + team_feature_source
        ].rename(columns=team_renamed),
        on=["season", "week", "_join_team"],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    # Keep a minimal copy for opponent-offense matchup expectations and audit.
    team_internal = team[
        [
            "season",
            "week",
            "_join_team",
            "offensive_plays_roll3_mean",
            "dropbacks_roll3_mean",
        ]
    ].rename(
        columns={
            "_join_team": "_join_opponent",
            "offensive_plays_roll3_mean": "_opp_offensive_plays_roll3",
            "dropbacks_roll3_mean": "_opp_offense_dropbacks_roll3",
        }
    )

    team_audit = build_team_source_audit(team)

    out = out.merge(
        team_internal,
        on=["season", "week", "_join_opponent"],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    out = out.merge(
        team_audit,
        on=["season", "week", "_join_team"],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    del team, team_internal, team_audit
    gc.collect()

    # Opponent form: opponent defensive context for offensive players.
    opponent = common.read_parquet_required(
        paths["opponent"],
        ["season", "week", "team"],
    )
    validate_exact_columns(
        opponent,
        OPPONENT_FORM_SOURCE_COLUMNS,
        "opponent form",
    )
    common.ensure_unique(
        opponent,
        ["season", "week", "team"],
        "opponent form",
    )
    opponent_feature_source = [
        column
        for column in opponent.columns
        if column not in TEAM_FORM_KEYS
    ]
    validate_form_column_names(
        opponent_feature_source,
        "opponent form",
    )
    validate_numeric_columns(
        opponent,
        opponent_feature_source,
        label="opponent form",
    )

    opponent["_join_defense"] = opponent["team"].map(
        canonical_franchise
    )
    if opponent["_join_defense"].eq("").any():
        raise ValueError(
            "opponent form contains blank canonical team."
        )

    common.ensure_unique(
        opponent,
        ["season", "week", "_join_defense"],
        "opponent form canonical join grain",
    )

    validate_key_coverage(
        out,
        opponent,
        required_columns=[
            "season",
            "week",
            "_join_opponent",
        ],
        source_columns=[
            "season",
            "week",
            "_join_defense",
        ],
        label="opponent form",
    )

    opponent_renamed = {
        column: f"opponent_{column}"
        for column in opponent_feature_source
    }
    opponent_columns.extend(opponent_renamed.values())

    opponent_for_join = opponent[
        ["season", "week", "_join_defense"] + opponent_feature_source
    ].rename(columns={"_join_defense": "_join_opponent", **opponent_renamed})

    out = out.merge(
        opponent_for_join,
        on=["season", "week", "_join_opponent"],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    # Minimal player's-team defense copy for opponent-volume expectations.
    team_defense_internal = opponent[
        [
            "season",
            "week",
            "_join_defense",
            "defensive_plays_roll3_mean",
            "opponent_dropbacks_roll3_mean",
        ]
    ].rename(
        columns={
            "_join_defense": "_join_team",
            "defensive_plays_roll3_mean": "_team_defensive_plays_roll3",
            "opponent_dropbacks_roll3_mean": "_team_defense_dropbacks_roll3",
        }
    )

    out = out.merge(
        team_defense_internal,
        on=["season", "week", "_join_team"],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    del opponent, opponent_for_join, team_defense_internal
    gc.collect()

    # Environment: one row per game; expose player/team-relative rest/travel.
    environment = common.read_parquet_required(
        paths["environment"],
        ENVIRONMENT_REQUIRED_COLUMNS,
    )
    validate_numeric_columns(
        environment,
        ENVIRONMENT_NUMERIC_COLUMNS,
        label="environment",
    )
    common.ensure_unique(
        environment,
        ["season", "week", "game_id"],
        "environment",
    )

    env = environment.copy()
    env["_home_join"] = env["home_team"].map(canonical_franchise)
    env["_away_join"] = env["away_team"].map(canonical_franchise)

    direct_rename = {
        column: f"environment_{column}"
        for column in ENVIRONMENT_DIRECT_COLUMNS
    }
    direct_names = list(direct_rename.values())

    env = env.rename(columns=direct_rename)

    out = out.merge(
        env[
            [
                "season",
                "week",
                "game_id",
                "_home_join",
                "_away_join",
                "home_rest_days",
                "away_rest_days",
                "miles_traveled_away",
                "time_zones_crossed_away",
                "east_to_west_flag",
                "west_to_east_flag",
            ]
            + direct_names
        ],
        on=["season", "week", "game_id"],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    home_side = out["_join_team"].eq(out["_home_join"])
    away_side = out["_join_team"].eq(out["_away_join"])
    if (~(home_side | away_side)).any():
        sample = out.loc[
            ~(home_side | away_side),
            GRAIN + ["team", "opponent"],
        ].head(10).to_dict("records")
        raise ValueError(
            f"Environment team/game mismatch; sample={sample}"
        )

    out["environment_team_rest_days"] = np.where(
        home_side,
        out["home_rest_days"],
        out["away_rest_days"],
    )
    out["environment_opponent_rest_days"] = np.where(
        home_side,
        out["away_rest_days"],
        out["home_rest_days"],
    )
    out["environment_team_miles_traveled"] = np.where(
        away_side,
        out["miles_traveled_away"],
        0.0,
    )
    out["environment_opponent_miles_traveled"] = np.where(
        home_side,
        out["miles_traveled_away"],
        0.0,
    )
    out["environment_team_time_zones_crossed"] = np.where(
        away_side,
        out["time_zones_crossed_away"],
        0.0,
    )
    out["environment_opponent_time_zones_crossed"] = np.where(
        home_side,
        out["time_zones_crossed_away"],
        0.0,
    )
    out["environment_team_east_to_west_flag"] = np.where(
        away_side,
        out["east_to_west_flag"],
        0,
    )
    out["environment_opponent_east_to_west_flag"] = np.where(
        home_side,
        out["east_to_west_flag"],
        0,
    )
    out["environment_team_west_to_east_flag"] = np.where(
        away_side,
        out["west_to_east_flag"],
        0,
    )
    out["environment_opponent_west_to_east_flag"] = np.where(
        home_side,
        out["west_to_east_flag"],
        0,
    )

    environment_columns.extend(
        direct_names
        + [
            "environment_team_rest_days",
            "environment_opponent_rest_days",
            "environment_team_miles_traveled",
            "environment_opponent_miles_traveled",
            "environment_team_time_zones_crossed",
            "environment_opponent_time_zones_crossed",
            "environment_team_east_to_west_flag",
            "environment_opponent_east_to_west_flag",
            "environment_team_west_to_east_flag",
            "environment_opponent_west_to_east_flag",
        ]
    )

    out = out.drop(
        columns=[
            "_home_join",
            "_away_join",
            "home_rest_days",
            "away_rest_days",
            "miles_traveled_away",
            "time_zones_crossed_away",
            "east_to_west_flag",
            "west_to_east_flag",
        ]
    )
    del environment, env
    gc.collect()

    # Defensive-specific full-universe features.
    defensive = common.read_parquet_required(
        paths["defensive"],
        GRAIN,
    )
    validate_exact_columns(
        defensive,
        DEFENSIVE_SOURCE_COLUMNS,
        "defensive features",
    )
    validate_exact_full_grain(
        universe_keys,
        defensive,
        "defensive features",
    )
    validate_source_metadata(
        out,
        defensive,
        label="defensive features",
        mappings=[
            ("position", "position", "position"),
        ],
    )

    defensive_source = [
        column
        for column in defensive.columns
        if column not in set(GRAIN + ["position"])
    ]

    validate_numeric_columns(
        defensive,
        defensive_source,
        label="defensive features",
    )

    defensive_rename: dict[str, str] = {}
    for column in defensive_source:
        if column in DEFENSIVE_ROLE_MAP:
            output_name = DEFENSIVE_ROLE_MAP[column]
            role_columns.append(output_name)
        else:
            output_name = f"player_defensive_{column}"
            player_columns.append(output_name)
        defensive_rename[column] = output_name

    out = out.merge(
        defensive[GRAIN + defensive_source].rename(
            columns=defensive_rename
        ),
        on=GRAIN,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    del defensive
    gc.collect()

    # Sparse kicking-specific features; non-kickers remain null.
    kicking = common.read_parquet_required(
        paths["kicking"],
        GRAIN,
    )
    validate_exact_columns(
        kicking,
        KICKING_SOURCE_COLUMNS,
        "kicking features",
    )
    validate_sparse_grain_subset(
        universe_keys,
        kicking,
        "kicking features",
    )
    validate_source_metadata(
        out,
        kicking,
        label="kicking features",
        mappings=[
            ("team", "team", "team"),
        ],
    )

    kicking_source = [
        column
        for column in kicking.columns
        if column not in set(GRAIN + ["team"])
        and column not in KICKING_REDUNDANT_ENVIRONMENT
    ]

    validate_numeric_columns(
        kicking,
        kicking_source,
        label="kicking features",
    )

    kicking_rename: dict[str, str] = {}
    for column in kicking_source:
        if column in KICKING_ROLE_MAP:
            output_name = KICKING_ROLE_MAP[column]
            role_columns.append(output_name)
        else:
            output_name = f"player_kicking_{column}"
            player_columns.append(output_name)
        kicking_rename[column] = output_name

    out = out.merge(
        kicking[GRAIN + kicking_source].rename(
            columns=kicking_rename
        ),
        on=GRAIN,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    del kicking
    gc.collect()

    # Strictly lag position-allowed values before joining.
    position_allowed = common.read_parquet_required(
        paths["position_allowed"],
        POSITION_ALLOWED_KEYS,
    )
    position_lag, position_matchup_columns = build_position_allowed_lag(
        position_allowed
    )
    matchup_columns.extend(REQUIRED_MATCHUP_COLUMNS)
    matchup_columns.extend(position_matchup_columns)

    out = out.merge(
        position_lag,
        left_on=[
            "season",
            "week",
            "_join_opponent",
            "_join_position_group",
        ],
        right_on=[
            "season",
            "week",
            "_join_defense",
            "_join_position_group",
        ],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if "_join_defense" in out.columns:
        out = out.drop(columns=["_join_defense"])
    del position_allowed, position_lag
    gc.collect()

    # Matchup expectations. All inputs are already strictly lagged or shifted.
    out["matchup_expected_team_plays"] = safe_mean_pair(
        out["team_offensive_plays_roll3_mean"],
        out["opponent_defensive_plays_roll3_mean"],
    )
    out["matchup_expected_team_dropbacks"] = safe_mean_pair(
        out["team_dropbacks_roll3_mean"],
        out["opponent_opponent_dropbacks_roll3_mean"],
    )
    out["matchup_expected_team_rush_attempts"] = safe_mean_pair(
        out["team_rush_attempts_roll3_mean"],
        out["opponent_opponent_rush_attempts_roll3_mean"],
    )
    out["matchup_expected_opponent_plays"] = safe_mean_pair(
        out["_opp_offensive_plays_roll3"],
        out["_team_defensive_plays_roll3"],
    )
    out["matchup_expected_opponent_dropbacks"] = safe_mean_pair(
        out["_opp_offense_dropbacks_roll3"],
        out["_team_defense_dropbacks_roll3"],
    )

    out["matchup_player_target_share_x_opp_targets"] = safe_product(
        out["player_target_share_roll3_mean"],
        out["matchup_position_allowed_targets_allowed_lag1"],
    )
    out["matchup_player_carry_share_x_opp_rushes"] = safe_product(
        out["player_carry_share_roll3_mean"],
        out["matchup_position_allowed_carries_allowed_lag1"],
    )
    out["matchup_player_tackle_rate_x_opp_plays"] = safe_product(
        out["player_tackle_rate_per_def_play_roll3_mean"],
        out["matchup_expected_opponent_plays"],
    )
    out["matchup_player_sack_rate_x_opp_plays"] = safe_product(
        out["player_sack_rate_per_def_play_roll5_mean"],
        out["matchup_expected_opponent_plays"],
    )
    out["matchup_off_epa_vs_def_epa"] = (
        strict_numeric(
            out["team_off_epa_per_play_roll3_mean"],
            label="team_off_epa_per_play_roll3_mean",
        )
        - strict_numeric(
            out["opponent_def_epa_per_play_roll3_mean"],
            label="opponent_def_epa_per_play_roll3_mean",
        )
    )

    opponent_pass_rate = safe_divide(
        out["opponent_opponent_pass_attempts_roll3_mean"],
        out["opponent_defensive_plays_roll3_mean"],
    )
    opponent_rush_rate = safe_divide(
        out["opponent_opponent_rush_attempts_roll3_mean"],
        out["opponent_defensive_plays_roll3_mean"],
    )

    out["matchup_pass_rate_vs_opponent"] = (
        strict_numeric(
            out["team_pass_rate_roll3_mean"],
            label="team_pass_rate_roll3_mean",
        )
        - opponent_pass_rate
    )
    out["matchup_rush_rate_vs_opponent"] = (
        strict_numeric(
            out["team_rush_rate_roll3_mean"],
            label="team_rush_rate_roll3_mean",
        )
        - opponent_rush_rate
    )

    out = out.drop(
        columns=[
            "_opp_offensive_plays_roll3",
            "_opp_offense_dropbacks_roll3",
            "_team_defensive_plays_roll3",
            "_team_defense_dropbacks_roll3",
        ]
    )

    # Targets: canonical output target names derive from configured target keys.
    targets = common.read_parquet_required(
        paths["targets"],
        GRAIN + required_target_order,
    )
    validate_exact_full_grain(universe_keys, targets, "historical targets")

    target_rename = {
        name: f"target_{name}"
        for name in required_target_order
    }
    out = out.merge(
        targets[GRAIN + required_target_order].rename(
            columns=target_rename
        ),
        on=GRAIN,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    del targets
    gc.collect()

    # Audit metadata. played_game_flag is used only to identify prior realized
    # player-game provenance and is never exposed as a feature.
    player_audit = build_player_audit(universe)
    out = out.merge(
        player_audit,
        on=GRAIN,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    del player_audit, universe, universe_keys
    gc.collect()

    out["audit_feature_asof"] = out["kickoff_timestamp"]
    out["audit_depth_snapshot_at"] = pd.NaT
    out["audit_injury_snapshot_at"] = pd.NaT
    out["audit_market_feature_count"] = np.int8(0)
    out["audit_row_valid"] = np.int8(1)

    # No join-helper columns may survive.
    helper_columns = [
        column
        for column in out.columns
        if column.startswith("_join_")
    ]
    if helper_columns:
        out = out.drop(columns=helper_columns)

    # Deterministic family order.
    family_lists = [
        role_columns,
        player_columns,
        team_columns,
        opponent_columns,
        matchup_columns,
        environment_columns,
        history_columns,
        target_columns,
        AUDIT_COLUMNS,
    ]

    for family in family_lists:
        duplicates = [
            value
            for value in family
            if family.count(value) > 1
        ]
        if duplicates:
            raise ValueError(
                f"Duplicate output columns inside family: {sorted(set(duplicates))}"
            )

    ordered_columns = (
        LEADING_COLUMNS
        + role_columns
        + player_columns
        + team_columns
        + opponent_columns
        + matchup_columns
        + environment_columns
        + history_columns
        + target_columns
        + AUDIT_COLUMNS
    )

    if len(ordered_columns) != len(set(ordered_columns)):
        seen: set[str] = set()
        duplicates: list[str] = []
        for column in ordered_columns:
            if column in seen:
                duplicates.append(column)
            seen.add(column)
        raise ValueError(
            f"Duplicate final schema columns: {sorted(set(duplicates))}"
        )

    missing_output = [
        column
        for column in ordered_columns
        if column not in out.columns
    ]
    if missing_output:
        raise ValueError(
            f"Missing assembled output columns: {missing_output[:30]}"
        )

    extra_output = [
        column
        for column in out.columns
        if column not in ordered_columns
    ]
    if extra_output:
        raise ValueError(
            f"Unexpected assembled columns: {extra_output[:30]}"
        )

    out = out[ordered_columns]

    common.ensure_unique(out, GRAIN, "historical feature table")
    if len(out) != universe_row_count:
        raise ValueError(
            f"Historical feature row count changed: "
            f"{universe_row_count:,} -> {len(out):,}"
        )

    if list(out.columns[: len(LEADING_COLUMNS)]) != LEADING_COLUMNS:
        raise ValueError("Leading header order mismatch.")

    if [column for column in out.columns if column.startswith("target_")] != target_columns:
        raise ValueError("Target header order mismatch.")

    if [column for column in out.columns if column.startswith("audit_")] != AUDIT_COLUMNS:
        raise ValueError("Audit header order mismatch.")

    # Explicit model-feature candidates: identity/time IDs, target_* and
    # audit_* are not features. Team/opponent/position context is allowed.
    candidate_features = (
        ["home_flag", "team", "opponent", "position", "position_group"]
        + role_columns
        + player_columns
        + team_columns
        + opponent_columns
        + matchup_columns
        + environment_columns
        + history_columns
    )

    if any(column.startswith("target_") for column in candidate_features):
        raise ValueError("Target leakage into candidate features.")

    if any(column.startswith("audit_") for column in candidate_features):
        raise ValueError("Audit columns entered candidate features.")

    if (
        any(
            "played_game_flag" in column.casefold()
            for column in candidate_features
        )
        or "played_game_flag" in out.columns
    ):
        raise ValueError(
            "played_game_flag must not enter assembled schema."
        )

    common.reject_forbidden_feature_columns(candidate_features, config)
    validate_no_same_game_usage_features(candidate_features)
    validate_no_final_score_features(candidate_features)

    numeric_features, categorical_features = classify_feature_columns(
        out,
        candidate_features,
    )

    if set(numeric_features) & set(categorical_features):
        raise ValueError("Feature manifest numeric/categorical overlap.")

    if set(numeric_features + categorical_features) != set(candidate_features):
        raise ValueError("Feature manifest does not cover candidate features exactly.")

    numeric_block = out.select_dtypes(include=[np.number])
    if np.isinf(numeric_block.to_numpy(dtype="float64", copy=False)).any():
        raise ValueError("Historical feature table contains infinity.")

    schema_hash = stable_schema_hash(out)

    manifest = {
        "schema_version": 1,
        "canonical_grain": GRAIN,
        "leading_columns": LEADING_COLUMNS,
        "family_order": [
            "role",
            "player",
            "team",
            "opponent",
            "matchup",
            "environment",
            "history",
            "target",
            "audit",
        ],
        "column_families": {
            "role": role_columns,
            "player": player_columns,
            "team": team_columns,
            "opponent": opponent_columns,
            "matchup": matchup_columns,
            "environment": environment_columns,
            "history": history_columns,
            "target": target_columns,
            "audit": AUDIT_COLUMNS,
        },
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_columns": numeric_features + categorical_features,
        "excluded_from_features": {
            "identity_and_time": [
                "season",
                "season_type",
                "week",
                "game_id",
                "gameday",
                "kickoff_timestamp",
                "player_id",
                "player_name",
            ],
            "targets": target_columns,
            "audit": AUDIT_COLUMNS,
            "outcome_metadata": ["played_game_flag"],
        },
        "target_columns": target_columns,
        "required_matchup_columns": REQUIRED_MATCHUP_COLUMNS,
        "matchup_formulas": {
            "matchup_expected_team_plays": "mean(team offensive_plays roll3, opponent defensive_plays roll3)",
            "matchup_expected_team_dropbacks": "mean(team dropbacks roll3, opponent opponent_dropbacks roll3)",
            "matchup_expected_team_rush_attempts": "mean(team rush_attempts roll3, opponent opponent_rush_attempts roll3)",
            "matchup_expected_opponent_plays": "mean(opponent offensive_plays roll3, team defensive_plays roll3)",
            "matchup_expected_opponent_dropbacks": "mean(opponent dropbacks roll3, team opponent_dropbacks roll3)",
            "matchup_player_target_share_x_opp_targets": "player target_share roll3 * opponent-position targets_allowed lag1",
            "matchup_player_carry_share_x_opp_rushes": "player carry_share roll3 * opponent-position carries_allowed lag1",
            "matchup_player_tackle_rate_x_opp_plays": "player tackle_rate roll3 * expected opponent plays",
            "matchup_player_sack_rate_x_opp_plays": "player sack_rate roll5 * expected opponent plays",
            "matchup_off_epa_vs_def_epa": "team off_epa_per_play roll3 - opponent def_epa_per_play roll3",
            "matchup_pass_rate_vs_opponent": "team pass_rate roll3 - opponent prior pass attempts / opponent prior defensive plays",
            "matchup_rush_rate_vs_opponent": "team rush_rate roll3 - opponent prior rush attempts / opponent prior defensive plays",
        },
        "position_allowed_policy": (
            "Every realized position_allowed_week metric is shifted one "
            "observed defense/position game before target-row use."
        ),
        "audit_policy": {
            "audit_feature_asof": "target kickoff timestamp; all feature sources must be strictly earlier",
            "audit_max_player_source_game": "latest strictly prior played modeled-universe game for player; 2010-2011 prehistory may not have a modeled game_id",
            "audit_max_team_source_week": "latest prior canonical-franchise team-form source week",
            "audit_depth_snapshot_at": "null because upstream role-history output does not retain the source snapshot timestamp; upstream builder validated pregame depth",
            "audit_injury_snapshot_at": "null because upstream role-history output does not retain the source snapshot timestamp; upstream builder validated pregame injury status",
        },
        "source_paths": {
            key: value
            for key, value in paths.items()
            if key != "output"
        },
        "output_path": paths["output"],
        "row_count": int(len(out)),
        "column_count": int(len(out.columns)),
        "feature_count": int(len(candidate_features)),
        "numeric_feature_count": int(len(numeric_features)),
        "categorical_feature_count": int(len(categorical_features)),
        "schema_hash": schema_hash,
        "market_features_used": False,
        "target_columns_in_feature_manifest": False,
    }

    write_output_bundle_atomic(
        out,
        paths["output"],
        manifest,
        manifest_path,
    )

    payload = {
        "status": "passed",
        "rows": int(len(out)),
        "columns": int(len(out.columns)),
        "features": int(len(candidate_features)),
        "numeric_features": int(len(numeric_features)),
        "categorical_features": int(len(categorical_features)),
        "schema_hash": schema_hash,
        "market_feature_count": 0,
        "target_columns_in_manifest": False,
        "position_allowed_lagged": True,
        "output": paths["output"],
        "manifest": manifest_path,
    }
    reporter.set_rows(
        rows_in=int(universe_row_count),
        rows_out=int(len(out)),
    )
    reporter.update_details(
        {
            key: value
            for key, value in payload.items()
            if key != "status"
        }
    )

    print(
        json.dumps(
            {
                "script": Path(__file__).name,
                "payload": payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def main() -> int:
    with PipelineReporter(
        script=__file__,
        stage="01_intake",
        report_root=common.prop_root() / "errors",
        pipeline="nfl_prop_engine",
        league="NFL",
    ) as reporter:
        return run(reporter)

if __name__ == "__main__":
    raise SystemExit(main())
