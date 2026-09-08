#!/usr/bin/env python3
"""
Train deterministic LightGBM opportunity-component models for the NFL Prop Engine.

REQUIRED COMPONENTS
-------------------
qb_pass_attempts
team_pass_attempts
team_rush_attempts
player_carry_share
player_target_share
player_red_zone_target_share
player_goal_line_carry_share
field_goal_attempts
extra_point_attempts
opponent_offensive_plays
opponent_dropbacks
player_defensive_participation

READS
-----
- config/prop_engine.yaml
- config/target_eligibility.yaml
- data/historical/features/player_game_features.parquet
- data/historical/features/feature_manifest.json
- data/historical/opportunity/player_week_opportunity.parquet
- data/historical/opportunity/team_week_opportunity.parquet
- evaluation/backtest_folds.parquet

WRITES, FOR EACH COMPONENT
--------------------------
models/components/{component_name}/model.txt
models/components/{component_name}/feature_manifest.json
models/components/{component_name}/metadata.json

TRAINING POLICY
---------------
- LightGBM regression.
- No random train/test split.
- Model-selection train window ends in 2023.
- Model-selection validation window is 2024.
- Final persisted model is fit through 2024.
- 2025 is untouched: it is not used for feature selection, early stopping,
  hyperparameter selection, metrics, fitting, or metadata-derived choices.
- Share/rate predictions use [0, 1] clipping.
- Volume predictions use a zero floor.
- Player team-share components must be reconciled during current-week
  allocation; this trainer records that requirement but does not perform
  current-week allocation.
- Every production feature is explicitly declared below. There is no
  "all numeric columns" selection.
- Only pregame columns from the canonical Issue 17 feature manifest are used.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    import lightgbm as lgb
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Issue 22 requires LightGBM. Install it in the active Python "
        "environment with: python -m pip install lightgbm"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common



_CONFIG_CONTRACT = common.load_config()
_TRAINING_CONTRACT = _CONFIG_CONTRACT["training"]
SEED = 24024
MODEL_SELECTION_TRAIN_END = int(_TRAINING_CONTRACT["model_selection_train_end_season"])
DEVELOPMENT_VALIDATION_SEASON = int(_TRAINING_CONTRACT["development_validation_season"])
FINAL_TRAIN_END = int(_TRAINING_CONTRACT["final_train_end_season"])
UNTOUCHED_TEST_SEASON = int(_TRAINING_CONTRACT["untouched_test_season"])

FEATURE_MANIFEST_PATH = (
    "docs/win/football/nfl/prop_engine/data/historical/features/"
    "feature_manifest.json"
)
FOLDS_PATH = (
    "docs/win/football/nfl/prop_engine/evaluation/backtest_folds.parquet"
)
ELIGIBILITY_PATH = (
    "docs/win/football/nfl/prop_engine/config/target_eligibility.yaml"
)

GRAIN = ["season", "week", "game_id", "player_id"]
TEAM_GAME_GRAIN = ["season", "week", "game_id", "team"]

HISTORICAL_FRANCHISE_ALIASES = {
    "SD": "LAC",
    "OAK": "LV",
    "STL": "LAR",
}

SHARE_COMPONENTS = {
    "player_carry_share",
    "player_target_share",
    "player_red_zone_target_share",
    "player_goal_line_carry_share",
}

BOUNDED_COMPONENTS = {
    *SHARE_COMPONENTS,
    "player_defensive_participation",
}

VOLUME_COMPONENTS = {
    "qb_pass_attempts",
    "team_pass_attempts",
    "team_rush_attempts",
    "field_goal_attempts",
    "extra_point_attempts",
    "opponent_offensive_plays",
    "opponent_dropbacks",
}

COMPONENT_ORDER = [
    "qb_pass_attempts",
    "team_pass_attempts",
    "team_rush_attempts",
    "player_carry_share",
    "player_target_share",
    "player_red_zone_target_share",
    "player_goal_line_carry_share",
    "field_goal_attempts",
    "extra_point_attempts",
    "opponent_offensive_plays",
    "opponent_dropbacks",
    "player_defensive_participation",
]


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


ROLE_OFFENSE = [
    "home_flag",
    "role_depth_rank_pregame",
    "role_depth_starter_flag_pregame",
    "role_injury_out_flag",
    "role_injury_doubtful_flag",
    "role_injury_questionable_flag",
    "role_prior_offense_snap_pct",
    "role_snap_pct_roll3",
    "role_snap_pct_roll5",
    "role_snap_pct_ewm5",
    "role_prior_offense_participation",
    "role_participation_roll3",
    "role_participation_roll5",
    "role_depth_rank_change",
    "role_snap_share_change",
    "role_participation_change",
    "role_team_change_flag",
    "role_games_with_current_team_before_game",
    "role_starter_promotion_flag",
    "role_starter_demotion_flag",
    "role_teammate_out_count_position",
    "role_teammate_unavailable_snap_share_position",
    "role_role_history_games",
    "role_role_missing_flag",
    "history_no_nfl_history_flag",
    "history_new_team_flag",
    "history_history_games",
]

ROLE_DEFENSE = [
    "home_flag",
    "role_depth_rank_pregame",
    "role_depth_starter_flag_pregame",
    "role_injury_out_flag",
    "role_injury_doubtful_flag",
    "role_injury_questionable_flag",
    "role_prior_defense_snap_pct",
    "role_snap_pct_roll3",
    "role_snap_pct_roll5",
    "role_snap_pct_ewm5",
    "role_prior_defense_participation",
    "role_participation_roll3",
    "role_participation_roll5",
    "role_depth_rank_change",
    "role_snap_share_change",
    "role_participation_change",
    "role_team_change_flag",
    "role_games_with_current_team_before_game",
    "role_starter_promotion_flag",
    "role_starter_demotion_flag",
    "role_teammate_out_count_position",
    "role_teammate_unavailable_snap_share_position",
    "role_role_history_games",
    "role_role_missing_flag",
    "role_defensive_starter_flag",
    "role_front7_flag",
    "role_secondary_flag",
    "history_no_nfl_history_flag",
    "history_new_team_flag",
    "history_history_games",
]

ENVIRONMENT = [
    "environment_divisional_game_flag",
    "environment_neutral_site_flag",
    "environment_temperature",
    "environment_wind",
    "environment_international_flag",
    "environment_weather_missing_flag",
    "environment_travel_missing_flag",
    "environment_team_rest_days",
    "environment_opponent_rest_days",
    "environment_team_miles_traveled",
    "environment_opponent_miles_traveled",
    "environment_team_time_zones_crossed",
    "environment_opponent_time_zones_crossed",
]

TEAM_OFFENSE_CONTEXT = [
    "home_flag",
    "team_offensive_plays_lag1",
    "team_offensive_plays_roll3_mean",
    "team_offensive_plays_roll5_mean",
    "team_offensive_plays_ewm5",
    "team_offensive_plays_season_to_date",
    "team_drives_roll3_mean",
    "team_drives_roll5_mean",
    "team_drives_ewm5",
    "team_drives_season_to_date",
    "team_dropbacks_lag1",
    "team_dropbacks_roll3_mean",
    "team_dropbacks_roll5_mean",
    "team_dropbacks_ewm5",
    "team_dropbacks_season_to_date",
    "team_pass_attempts_lag1",
    "team_pass_attempts_roll3_mean",
    "team_pass_attempts_roll5_mean",
    "team_pass_attempts_ewm5",
    "team_pass_attempts_season_to_date",
    "team_rush_attempts_lag1",
    "team_rush_attempts_roll3_mean",
    "team_rush_attempts_roll5_mean",
    "team_rush_attempts_ewm5",
    "team_rush_attempts_season_to_date",
    "team_pass_rate_lag1",
    "team_pass_rate_roll3_mean",
    "team_pass_rate_roll5_mean",
    "team_pass_rate_ewm5",
    "team_pass_rate_season_to_date",
    "team_rush_rate_lag1",
    "team_rush_rate_roll3_mean",
    "team_rush_rate_roll5_mean",
    "team_rush_rate_ewm5",
    "team_rush_rate_season_to_date",
    "team_points_per_drive_roll3_mean",
    "team_points_per_drive_roll5_mean",
    "team_points_per_drive_ewm5",
    "team_points_per_drive_season_to_date",
    "team_red_zone_drives_roll3_mean",
    "team_red_zone_drives_roll5_mean",
    "team_red_zone_drives_ewm5",
    "team_red_zone_drives_season_to_date",
    "team_red_zone_pass_attempts_roll3_mean",
    "team_red_zone_pass_attempts_roll5_mean",
    "team_red_zone_pass_attempts_ewm5",
    "team_red_zone_pass_attempts_season_to_date",
    "team_red_zone_rush_attempts_roll3_mean",
    "team_red_zone_rush_attempts_roll5_mean",
    "team_red_zone_rush_attempts_ewm5",
    "team_red_zone_rush_attempts_season_to_date",
    "team_goal_line_rush_attempts_roll3_mean",
    "team_goal_line_rush_attempts_roll5_mean",
    "team_goal_line_rush_attempts_ewm5",
    "team_goal_line_rush_attempts_season_to_date",
    "team_field_goal_attempts_lag1",
    "team_field_goal_attempts_roll3_mean",
    "team_field_goal_attempts_roll5_mean",
    "team_field_goal_attempts_ewm5",
    "team_field_goal_attempts_season_to_date",
    "team_extra_point_attempts_lag1",
    "team_extra_point_attempts_roll3_mean",
    "team_extra_point_attempts_roll5_mean",
    "team_extra_point_attempts_ewm5",
    "team_extra_point_attempts_season_to_date",
    "team_off_epa_per_play_roll3_mean",
    "team_off_epa_per_play_roll5_mean",
    "team_off_epa_per_play_ewm5",
    "team_off_epa_per_play_season_to_date",
    "team_off_success_rate_roll3_mean",
    "team_off_success_rate_roll5_mean",
    "team_off_success_rate_ewm5",
    "team_off_success_rate_season_to_date",
    "team_yards_per_play_roll3_mean",
    "team_yards_per_play_roll5_mean",
    "team_yards_per_play_ewm5",
    "team_yards_per_play_season_to_date",
    "team_red_zone_td_rate_roll3_mean",
    "team_red_zone_td_rate_roll5_mean",
    "team_red_zone_td_rate_ewm5",
    "team_red_zone_td_rate_season_to_date",
]

OPPONENT_DEFENSE_CONTEXT = [
    "opponent_defensive_plays_lag1",
    "opponent_defensive_plays_roll3_mean",
    "opponent_defensive_plays_roll5_mean",
    "opponent_defensive_plays_ewm5",
    "opponent_defensive_plays_season_to_date",
    "opponent_opponent_dropbacks_lag1",
    "opponent_opponent_dropbacks_roll3_mean",
    "opponent_opponent_dropbacks_roll5_mean",
    "opponent_opponent_dropbacks_ewm5",
    "opponent_opponent_dropbacks_season_to_date",
    "opponent_opponent_pass_attempts_roll3_mean",
    "opponent_opponent_pass_attempts_roll5_mean",
    "opponent_opponent_pass_attempts_ewm5",
    "opponent_opponent_pass_attempts_season_to_date",
    "opponent_opponent_rush_attempts_roll3_mean",
    "opponent_opponent_rush_attempts_roll5_mean",
    "opponent_opponent_rush_attempts_ewm5",
    "opponent_opponent_rush_attempts_season_to_date",
    "opponent_sacks_roll3_mean",
    "opponent_sacks_roll5_mean",
    "opponent_sacks_ewm5",
    "opponent_sacks_season_to_date",
    "opponent_qb_hits_roll3_mean",
    "opponent_qb_hits_roll5_mean",
    "opponent_qb_hits_ewm5",
    "opponent_qb_hits_season_to_date",
    "opponent_def_epa_per_play_roll3_mean",
    "opponent_def_epa_per_play_roll5_mean",
    "opponent_def_epa_per_play_ewm5",
    "opponent_def_epa_per_play_season_to_date",
    "opponent_def_success_rate_roll3_mean",
    "opponent_def_success_rate_roll5_mean",
    "opponent_def_success_rate_ewm5",
    "opponent_def_success_rate_season_to_date",
    "opponent_yards_per_play_allowed_roll3_mean",
    "opponent_yards_per_play_allowed_roll5_mean",
    "opponent_yards_per_play_allowed_ewm5",
    "opponent_yards_per_play_allowed_season_to_date",
    "opponent_points_per_drive_allowed_roll3_mean",
    "opponent_points_per_drive_allowed_roll5_mean",
    "opponent_points_per_drive_allowed_ewm5",
    "opponent_points_per_drive_allowed_season_to_date",
    "opponent_red_zone_td_rate_allowed_roll3_mean",
    "opponent_red_zone_td_rate_allowed_roll5_mean",
    "opponent_red_zone_td_rate_allowed_ewm5",
    "opponent_red_zone_td_rate_allowed_season_to_date",
]

QB_HISTORY = [
    "player_pass_attempts_lag1",
    "player_pass_attempts_roll3_mean",
    "player_pass_attempts_roll5_mean",
    "player_pass_attempts_roll8_mean",
    "player_pass_attempts_ewm3",
    "player_pass_attempts_ewm5",
    "player_pass_attempts_season_to_date",
    "player_pass_attempts_career_prior",
    "player_dropbacks_lag1",
    "player_dropbacks_roll3_mean",
    "player_dropbacks_roll5_mean",
    "player_dropbacks_roll8_mean",
    "player_dropbacks_ewm3",
    "player_dropbacks_ewm5",
    "player_dropbacks_season_to_date",
    "player_dropbacks_career_prior",
]

CARRY_HISTORY = [
    "player_carries_lag1",
    "player_carries_roll3_mean",
    "player_carries_roll5_mean",
    "player_carries_roll8_mean",
    "player_carries_ewm3",
    "player_carries_ewm5",
    "player_carries_season_to_date",
    "player_carries_career_prior",
    "player_carry_share_lag1",
    "player_carry_share_roll3_mean",
    "player_carry_share_roll5_mean",
    "player_carry_share_roll8_mean",
    "player_carry_share_ewm3",
    "player_carry_share_ewm5",
    "player_carry_share_season_to_date",
    "player_carry_share_career_prior",
    "player_red_zone_carries_lag1",
    "player_red_zone_carries_roll3_mean",
    "player_red_zone_carries_roll5_mean",
    "player_red_zone_carries_ewm5",
    "player_red_zone_carries_season_to_date",
    "player_goal_line_carries_lag1",
    "player_goal_line_carries_roll3_mean",
    "player_goal_line_carries_roll5_mean",
    "player_goal_line_carries_ewm5",
    "player_goal_line_carries_season_to_date",
]

TARGET_HISTORY = [
    "player_targets_lag1",
    "player_targets_roll3_mean",
    "player_targets_roll5_mean",
    "player_targets_roll8_mean",
    "player_targets_ewm3",
    "player_targets_ewm5",
    "player_targets_season_to_date",
    "player_targets_career_prior",
    "player_target_share_lag1",
    "player_target_share_roll3_mean",
    "player_target_share_roll5_mean",
    "player_target_share_roll8_mean",
    "player_target_share_ewm3",
    "player_target_share_ewm5",
    "player_target_share_season_to_date",
    "player_target_share_career_prior",
    "player_air_yards_share_lag1",
    "player_air_yards_share_roll3_mean",
    "player_air_yards_share_roll5_mean",
    "player_air_yards_share_ewm5",
    "player_air_yards_share_season_to_date",
    "player_red_zone_targets_lag1",
    "player_red_zone_targets_roll3_mean",
    "player_red_zone_targets_roll5_mean",
    "player_red_zone_targets_ewm5",
    "player_red_zone_targets_season_to_date",
    "player_red_zone_target_share_lag1",
    "player_red_zone_target_share_roll3_mean",
    "player_red_zone_target_share_roll5_mean",
    "player_red_zone_target_share_ewm5",
    "player_red_zone_target_share_season_to_date",
]

DEFENSE_HISTORY = [
    "player_defensive_def_snap_pct_lag1",
    "player_defensive_def_snap_pct_roll3",
    "player_defensive_def_participation_lag1",
    "player_defensive_def_participation_roll3",
    "player_defensive_tackles_lag1",
    "player_defensive_tackles_roll3",
    "player_defensive_tackles_roll5",
    "player_defensive_tackle_rate_roll3",
    "player_defensive_tackle_rate_roll5",
    "player_defensive_sacks_lag1",
    "player_defensive_sacks_roll3",
    "player_defensive_sacks_roll5",
    "player_defensive_sack_rate_roll5",
    "player_defensive_qb_hits_roll3",
    "player_defensive_qb_hits_roll5",
    "player_defensive_opponent_plays_roll3",
    "player_defensive_opponent_dropbacks_roll3",
    "player_defensive_opponent_rush_rate_roll3",
    "player_defensive_opponent_pass_rate_roll3",
    "player_defensive_team_def_sack_rate_roll3",
]


COMPONENTS: dict[str, dict[str, Any]] = {
    "qb_pass_attempts": {
        "scope": "player",
        "label_source": "player_opportunity",
        "label_column": "pass_attempts",
        "eligible_rule": "passing_yards",
        "prediction_transform": "floor_zero",
        "features": unique(
            ROLE_OFFENSE
            + QB_HISTORY
            + [
                "team_pass_attempts_lag1",
                "team_pass_attempts_roll3_mean",
                "team_pass_attempts_roll5_mean",
                "team_pass_attempts_ewm5",
                "team_pass_attempts_season_to_date",
                "team_dropbacks_lag1",
                "team_dropbacks_roll3_mean",
                "team_dropbacks_roll5_mean",
                "team_dropbacks_ewm5",
                "team_dropbacks_season_to_date",
                "team_pass_rate_roll3_mean",
                "team_pass_rate_roll5_mean",
                "team_pass_rate_ewm5",
                "opponent_opponent_dropbacks_roll3_mean",
                "opponent_opponent_pass_attempts_roll3_mean",
                "opponent_sacks_roll3_mean",
                "opponent_qb_hits_roll3_mean",
                "matchup_expected_team_plays",
                "matchup_expected_team_dropbacks",
                "matchup_pass_rate_vs_opponent",
                "matchup_off_epa_vs_def_epa",
            ]
            + ENVIRONMENT
        ),
    },
    "team_pass_attempts": {
        "scope": "team",
        "label_source": "team_opportunity",
        "label_column": "pass_attempts",
        "prediction_transform": "floor_zero",
        "features": unique(
            TEAM_OFFENSE_CONTEXT
            + OPPONENT_DEFENSE_CONTEXT
            + [
                "matchup_expected_team_plays",
                "matchup_expected_team_dropbacks",
                "matchup_pass_rate_vs_opponent",
                "matchup_off_epa_vs_def_epa",
            ]
            + ENVIRONMENT
        ),
    },
    "team_rush_attempts": {
        "scope": "team",
        "label_source": "team_opportunity",
        "label_column": "rush_attempts",
        "prediction_transform": "floor_zero",
        "features": unique(
            TEAM_OFFENSE_CONTEXT
            + OPPONENT_DEFENSE_CONTEXT
            + [
                "matchup_expected_team_plays",
                "matchup_expected_team_rush_attempts",
                "matchup_rush_rate_vs_opponent",
                "matchup_off_epa_vs_def_epa",
            ]
            + ENVIRONMENT
        ),
    },
    "player_carry_share": {
        "scope": "player",
        "label_source": "player_opportunity",
        "label_column": "carry_share",
        "eligible_rule": "rushing_yards",
        "prediction_transform": "clip_0_1",
        "reconcile_during_current_week_allocation": True,
        "features": unique(
            ROLE_OFFENSE
            + CARRY_HISTORY
            + [
                "team_rush_attempts_lag1",
                "team_rush_attempts_roll3_mean",
                "team_rush_attempts_roll5_mean",
                "team_rush_attempts_ewm5",
                "team_rush_attempts_season_to_date",
                "team_rush_rate_roll3_mean",
                "team_rush_rate_roll5_mean",
                "team_goal_line_rush_attempts_roll3_mean",
                "team_red_zone_rush_attempts_roll3_mean",
                "matchup_expected_team_rush_attempts",
                "matchup_rush_rate_vs_opponent",
            ]
            + ENVIRONMENT
        ),
    },
    "player_target_share": {
        "scope": "player",
        "label_source": "player_opportunity",
        "label_column": "target_share",
        "eligible_rule": "receiving_yards",
        "prediction_transform": "clip_0_1",
        "reconcile_during_current_week_allocation": True,
        "features": unique(
            ROLE_OFFENSE
            + TARGET_HISTORY
            + [
                "team_dropbacks_lag1",
                "team_dropbacks_roll3_mean",
                "team_dropbacks_roll5_mean",
                "team_pass_attempts_roll3_mean",
                "team_pass_attempts_roll5_mean",
                "team_pass_rate_roll3_mean",
                "team_red_zone_pass_attempts_roll3_mean",
                "matchup_expected_team_dropbacks",
                "matchup_player_target_share_x_opp_targets",
                "matchup_pass_rate_vs_opponent",
            ]
            + ENVIRONMENT
        ),
    },
    "player_red_zone_target_share": {
        "scope": "player",
        "label_source": "player_opportunity",
        "label_column": "red_zone_target_share",
        "eligible_rule": "receiving_tds",
        "prediction_transform": "clip_0_1",
        "reconcile_during_current_week_allocation": True,
        "features": unique(
            ROLE_OFFENSE
            + TARGET_HISTORY
            + [
                "team_red_zone_drives_roll3_mean",
                "team_red_zone_drives_roll5_mean",
                "team_red_zone_pass_attempts_roll3_mean",
                "team_red_zone_pass_attempts_roll5_mean",
                "team_pass_attempts_roll3_mean",
                "team_pass_rate_roll3_mean",
                "opponent_red_zone_pass_attempts_allowed_roll3_mean",
                "opponent_red_zone_td_rate_allowed_roll3_mean",
                "matchup_expected_team_dropbacks",
                "matchup_pass_rate_vs_opponent",
            ]
            + ENVIRONMENT
        ),
    },
    "player_goal_line_carry_share": {
        "scope": "player",
        "label_source": "player_opportunity",
        "label_column": "goal_line_carry_share",
        "eligible_rule": "rushing_tds",
        "prediction_transform": "clip_0_1",
        "reconcile_during_current_week_allocation": True,
        "features": unique(
            ROLE_OFFENSE
            + CARRY_HISTORY
            + [
                "team_goal_line_rush_attempts_roll3_mean",
                "team_goal_line_rush_attempts_roll5_mean",
                "team_red_zone_rush_attempts_roll3_mean",
                "team_red_zone_rush_attempts_roll5_mean",
                "team_rush_attempts_roll3_mean",
                "team_rush_rate_roll3_mean",
                "opponent_goal_line_rush_attempts_allowed_roll3_mean",
                "opponent_red_zone_rush_attempts_allowed_roll3_mean",
                "opponent_red_zone_td_rate_allowed_roll3_mean",
                "matchup_expected_team_rush_attempts",
                "matchup_rush_rate_vs_opponent",
            ]
            + ENVIRONMENT
        ),
    },
    "field_goal_attempts": {
        "scope": "team",
        "label_source": "team_opportunity",
        "label_column": "field_goal_attempts",
        "prediction_transform": "floor_zero",
        "features": unique(
            [
                "home_flag",
                "team_drives_roll3_mean",
                "team_drives_roll5_mean",
                "team_drives_ewm5",
                "team_drives_season_to_date",
                "team_field_goal_attempts_lag1",
                "team_field_goal_attempts_roll3_mean",
                "team_field_goal_attempts_roll5_mean",
                "team_field_goal_attempts_ewm5",
                "team_field_goal_attempts_season_to_date",
                "team_points_per_drive_roll3_mean",
                "team_points_per_drive_roll5_mean",
                "team_points_per_drive_ewm5",
                "team_red_zone_drives_roll3_mean",
                "team_red_zone_drives_roll5_mean",
                "team_red_zone_td_rate_roll3_mean",
                "team_red_zone_td_rate_roll5_mean",
                "team_off_epa_per_play_roll3_mean",
                "team_off_success_rate_roll3_mean",
                "opponent_points_per_drive_allowed_roll3_mean",
                "opponent_points_per_drive_allowed_roll5_mean",
                "opponent_red_zone_td_rate_allowed_roll3_mean",
                "opponent_red_zone_td_rate_allowed_roll5_mean",
                "matchup_expected_team_plays",
                "matchup_off_epa_vs_def_epa",
            ]
            + ENVIRONMENT
        ),
    },
    "extra_point_attempts": {
        "scope": "team",
        "label_source": "team_opportunity",
        "label_column": "extra_point_attempts",
        "prediction_transform": "floor_zero",
        "features": unique(
            [
                "home_flag",
                "team_drives_roll3_mean",
                "team_drives_roll5_mean",
                "team_drives_ewm5",
                "team_drives_season_to_date",
                "team_extra_point_attempts_lag1",
                "team_extra_point_attempts_roll3_mean",
                "team_extra_point_attempts_roll5_mean",
                "team_extra_point_attempts_ewm5",
                "team_extra_point_attempts_season_to_date",
                "team_points_per_drive_roll3_mean",
                "team_points_per_drive_roll5_mean",
                "team_points_per_drive_ewm5",
                "team_red_zone_drives_roll3_mean",
                "team_red_zone_drives_roll5_mean",
                "team_red_zone_td_rate_roll3_mean",
                "team_red_zone_td_rate_roll5_mean",
                "team_off_epa_per_play_roll3_mean",
                "team_off_success_rate_roll3_mean",
                "opponent_points_per_drive_allowed_roll3_mean",
                "opponent_points_per_drive_allowed_roll5_mean",
                "opponent_red_zone_td_rate_allowed_roll3_mean",
                "opponent_red_zone_td_rate_allowed_roll5_mean",
                "matchup_expected_team_plays",
                "matchup_off_epa_vs_def_epa",
            ]
            + ENVIRONMENT
        ),
    },
    "opponent_offensive_plays": {
        "scope": "team_opponent",
        "label_source": "opponent_team_opportunity",
        "label_column": "offensive_plays",
        "prediction_transform": "floor_zero",
        "features": unique(
            [
                "home_flag",
                "matchup_expected_opponent_plays",
                "matchup_expected_opponent_dropbacks",
                "player_defensive_opponent_plays_roll3",
                "player_defensive_opponent_dropbacks_roll3",
                "player_defensive_opponent_rush_rate_roll3",
                "player_defensive_opponent_pass_rate_roll3",
                "player_defensive_team_def_sack_rate_roll3",
            ]
            + ENVIRONMENT
        ),
    },
    "opponent_dropbacks": {
        "scope": "team_opponent",
        "label_source": "opponent_team_opportunity",
        "label_column": "dropbacks",
        "prediction_transform": "floor_zero",
        "features": unique(
            [
                "home_flag",
                "matchup_expected_opponent_plays",
                "matchup_expected_opponent_dropbacks",
                "player_defensive_opponent_plays_roll3",
                "player_defensive_opponent_dropbacks_roll3",
                "player_defensive_opponent_rush_rate_roll3",
                "player_defensive_opponent_pass_rate_roll3",
                "player_defensive_team_def_sack_rate_roll3",
            ]
            + ENVIRONMENT
        ),
    },
    "player_defensive_participation": {
        "scope": "player",
        "label_source": "player_opportunity",
        "label_column": "defense_participation",
        "eligible_rule": "tackles",
        "prediction_transform": "clip_0_1",
        "features": unique(
            ROLE_DEFENSE
            + DEFENSE_HISTORY
            + [
                "matchup_expected_opponent_plays",
                "matchup_expected_opponent_dropbacks",
            ]
            + ENVIRONMENT
        ),
    },
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON does not exist: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required YAML does not exist: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def canonical_team(value: Any) -> str:
    team = common.normalize_team(value)
    return HISTORICAL_FRANCHISE_ALIASES.get(team, team)


def numeric_frame(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    # Build the complete numeric block before constructing the DataFrame.
    # Repeated output[column] assignment fragments pandas' internal block
    # manager for wide feature matrices and emits PerformanceWarning.
    data = {
        column: common.safe_numeric(frame[column]).astype("float64")
        for column in columns
    }
    return pd.DataFrame(data, index=frame.index)


def transform_label(
    series: pd.Series,
    component: str,
) -> pd.Series:
    value = common.safe_numeric(series).astype("float64")

    if component in BOUNDED_COMPONENTS:
        return value.clip(lower=0.0, upper=1.0)

    if component in VOLUME_COMPONENTS:
        return value.clip(lower=0.0)

    raise KeyError(f"Unknown component transform: {component}")


def transform_prediction(
    values: np.ndarray,
    component: str,
) -> np.ndarray:
    pred = np.asarray(values, dtype="float64")

    if component in BOUNDED_COMPONENTS:
        return np.clip(pred, 0.0, 1.0)

    if component in VOLUME_COMPONENTS:
        return np.maximum(pred, 0.0)

    raise KeyError(f"Unknown component transform: {component}")


def stable_json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def write_json_atomic(
    path: Path,
    value: dict[str, Any],
) -> None:
    root = common.prop_root().resolve()
    destination = path.resolve()

    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to write outside Prop Engine root: {destination}"
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)

    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temp_path = Path(handle.name)

    try:
        with handle:
            handle.write(stable_json_bytes(value))
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_model_atomic(
    model: lgb.Booster,
    path: Path,
    num_iteration: int,
) -> None:
    root = common.prop_root().resolve()
    destination = path.resolve()

    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to write outside Prop Engine root: {destination}"
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)

    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    handle.close()

    try:
        model.save_model(
            str(temp_path),
            num_iteration=num_iteration,
        )
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    np.asarray(actual, dtype=float)
                    - np.asarray(pred, dtype=float)
                )
            )
        )
    )


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(
                np.asarray(actual, dtype=float)
                - np.asarray(pred, dtype=float)
            )
        )
    )


def r2(actual: np.ndarray, pred: np.ndarray) -> float | None:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)

    denominator = float(np.sum(np.square(y - y.mean())))
    if denominator <= 0.0:
        return None

    value = 1.0 - float(
        np.sum(np.square(y - p))
        / denominator
    )

    return value if math.isfinite(value) else None


def assert_feature_contract(
    config: dict[str, Any],
    canonical_manifest: dict[str, Any],
) -> None:
    if list(COMPONENTS.keys()) != COMPONENT_ORDER:
        raise ValueError("Component order does not match Issue 22 contract.")

    canonical_features = set(canonical_manifest["feature_columns"])
    canonical_numeric = set(canonical_manifest["numeric_features"])
    target_columns = set(canonical_manifest.get("target_columns", []))

    for component in COMPONENT_ORDER:
        features = COMPONENTS[component]["features"]

        if not features:
            raise ValueError(f"{component}: empty explicit feature list.")

        if len(features) != len(set(features)):
            raise ValueError(f"{component}: duplicate feature name.")

        missing = sorted(set(features) - canonical_features)
        if missing:
            raise ValueError(
                f"{component}: feature(s) absent from canonical Issue 17 "
                f"manifest: {missing}"
            )

        nonnumeric = sorted(set(features) - canonical_numeric)
        if nonnumeric:
            raise ValueError(
                f"{component}: this initial trainer accepts explicit numeric "
                f"features only; nonnumeric={nonnumeric}"
            )

        leaked_target = sorted(set(features) & target_columns)
        if leaked_target:
            raise ValueError(
                f"{component}: target leakage: {leaked_target}"
            )

        bad_prefix = [
            column
            for column in features
            if column.startswith("target_")
            or column.startswith("audit_")
        ]
        if bad_prefix:
            raise ValueError(
                f"{component}: prohibited feature prefix: {bad_prefix}"
            )

        if "played_game_flag" in features:
            raise ValueError(
                f"{component}: played_game_flag is forbidden."
            )

        common.reject_forbidden_feature_columns(features, config)


def verify_backtest_policy(folds: pd.DataFrame) -> None:
    common.require_columns(
        folds,
        [
            "fold_id",
            "train_end_season",
            "validation_start_season",
            "validation_end_season",
            "test_flag",
        ],
        "backtest folds",
    )

    dev = folds[
        folds["test_flag"].eq(0)
        & folds["validation_start_season"].eq(
            DEVELOPMENT_VALIDATION_SEASON
        )
    ]

    if len(dev) != 1:
        raise ValueError(
            "Expected exactly one 2024 development validation fold."
        )

    if int(dev.iloc[0]["train_end_season"]) != MODEL_SELECTION_TRAIN_END:
        raise ValueError(
            "Issue 22 requires model selection to train through 2023."
        )

    test = folds[folds["test_flag"].eq(1)]

    if len(test) != 1:
        raise ValueError("Expected exactly one untouched test fold.")

    if (
        int(test.iloc[0]["validation_start_season"])
        != UNTOUCHED_TEST_SEASON
    ):
        raise ValueError("Untouched test season must be 2025.")

    if int(test.iloc[0]["train_end_season"]) != FINAL_TRAIN_END:
        raise ValueError(
            "Untouched test contract must train through 2024."
        )


def check_team_feature_invariance(
    features: pd.DataFrame,
    team_feature_columns: list[str],
) -> None:
    """
    Team-scope features must not vary across player rows of the same team-game.
    This prevents accidental use of player-specific columns in a team model.
    """
    if not team_feature_columns:
        return

    subset = features[
        TEAM_GAME_GRAIN + team_feature_columns
    ].copy()

    for column in team_feature_columns:
        counts = (
            subset.groupby(
                TEAM_GAME_GRAIN,
                dropna=False,
                sort=False,
            )[column]
            .nunique(dropna=False)
        )

        bad = counts.gt(1)
        if bad.any():
            sample = bad[bad].head(10).index.tolist()
            raise ValueError(
                f"Team-scope feature varies within team-game: {column}; "
                f"sample={sample}"
            )


def team_rows_from_features(
    features: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    use = unique(
        [
            "season",
            "week",
            "game_id",
            "team",
            "opponent",
            *feature_columns,
        ]
    )

    frame = (
        features[use]
        .sort_values(
            TEAM_GAME_GRAIN + ["opponent"],
            kind="mergesort",
            na_position="last",
        )
        .drop_duplicates(
            TEAM_GAME_GRAIN,
            keep="first",
        )
        .reset_index(drop=True)
    )

    frame["_team_key"] = frame["team"].map(canonical_team)
    frame["_opponent_key"] = frame["opponent"].map(canonical_team)

    if frame["_team_key"].eq("").any():
        raise ValueError("Blank canonical team key in team model frame.")

    if frame["_opponent_key"].eq("").any():
        raise ValueError("Blank canonical opponent key in team model frame.")

    common.ensure_unique(
        frame,
        TEAM_GAME_GRAIN,
        "team model feature grain",
    )

    return frame


def build_training_frames(
    config: dict[str, Any],
    features: pd.DataFrame,
    eligibility: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    player_path = config["paths"]["player_opportunity"]
    team_path = config["paths"]["team_opportunity"]

    player_labels = common.read_parquet_required(
        player_path,
        [
            "season",
            "week",
            "game_id",
            "player_id",
            "pass_attempts",
            "carry_share",
            "target_share",
            "red_zone_target_share",
            "goal_line_carry_share",
            "defense_participation",
        ],
    )[
        [
            "season",
            "week",
            "game_id",
            "player_id",
            "pass_attempts",
            "carry_share",
            "target_share",
            "red_zone_target_share",
            "goal_line_carry_share",
            "defense_participation",
        ]
    ].copy()

    common.ensure_unique(
        player_labels,
        GRAIN,
        "player opportunity labels",
    )

    team_labels = common.read_parquet_required(
        team_path,
        [
            "season",
            "week",
            "team",
            "offensive_plays",
            "dropbacks",
            "pass_attempts",
            "rush_attempts",
            "field_goal_attempts",
            "extra_point_attempts",
        ],
    )[
        [
            "season",
            "week",
            "team",
            "offensive_plays",
            "dropbacks",
            "pass_attempts",
            "rush_attempts",
            "field_goal_attempts",
            "extra_point_attempts",
        ]
    ].copy()

    team_labels["_team_key"] = team_labels["team"].map(canonical_team)

    common.ensure_unique(
        team_labels,
        ["season", "week", "_team_key"],
        "team opportunity labels",
    )

    all_component_features = unique(
        [
            column
            for component in COMPONENT_ORDER
            for column in COMPONENTS[component]["features"]
        ]
    )

    # Restrict raw inputs to the final training cutoff before label assembly.
    # 2025 is not included in any training frame.
    base = features.loc[
        pd.to_numeric(
            features["season"],
            errors="raise",
        ).le(FINAL_TRAIN_END)
    ].copy()

    player_base = base.merge(
        player_labels,
        on=GRAIN,
        how="left",
        validate="one_to_one",
        suffixes=("", "_label"),
    )

    team_component_features = unique(
        [
            column
            for component in COMPONENT_ORDER
            if COMPONENTS[component]["scope"] in {
                "team",
                "team_opponent",
            }
            for column in COMPONENTS[component]["features"]
        ]
    )

    check_team_feature_invariance(
        base,
        team_component_features,
    )

    team_base = team_rows_from_features(
        base,
        team_component_features,
    )

    team_same = team_labels.rename(
        columns={
            "team": "_label_team",
            "_team_key": "_label_team_key",
        }
    )

    team_base = team_base.merge(
        team_same[
            [
                "season",
                "week",
                "_label_team_key",
                "offensive_plays",
                "dropbacks",
                "pass_attempts",
                "rush_attempts",
                "field_goal_attempts",
                "extra_point_attempts",
            ]
        ],
        left_on=["season", "week", "_team_key"],
        right_on=["season", "week", "_label_team_key"],
        how="left",
        validate="many_to_one",
    )

    opponent_labels = team_labels.rename(
        columns={
            "team": "_opponent_label_team",
            "_team_key": "_opponent_label_key",
            "offensive_plays": "_opponent_offensive_plays_label",
            "dropbacks": "_opponent_dropbacks_label",
        }
    )

    team_base = team_base.merge(
        opponent_labels[
            [
                "season",
                "week",
                "_opponent_label_key",
                "_opponent_offensive_plays_label",
                "_opponent_dropbacks_label",
            ]
        ],
        left_on=["season", "week", "_opponent_key"],
        right_on=["season", "week", "_opponent_label_key"],
        how="left",
        validate="many_to_one",
    )

    frames: dict[str, pd.DataFrame] = {}

    position = (
        player_base["position"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    for component in COMPONENT_ORDER:
        spec = COMPONENTS[component]
        scope = spec["scope"]
        label_column = spec["label_column"]

        if scope == "player":
            frame = player_base.copy()

            if label_column not in frame.columns:
                raise ValueError(
                    f"{component}: missing player label {label_column}"
                )

            eligible_rule = spec["eligible_rule"]
            positions = eligibility[eligible_rule]["eligible_positions"]
            eligible = {
                str(value).strip().upper()
                for value in positions
            }

            frame = frame.loc[position.isin(eligible)].copy()
            frame["_label"] = transform_label(
                frame[label_column],
                component,
            )

        elif scope == "team":
            frame = team_base.copy()
            if label_column not in frame.columns:
                raise ValueError(
                    f"{component}: missing team label {label_column}"
                )
            frame["_label"] = transform_label(
                frame[label_column],
                component,
            )

        elif scope == "team_opponent":
            frame = team_base.copy()

            source_column = {
                "opponent_offensive_plays":
                    "_opponent_offensive_plays_label",
                "opponent_dropbacks":
                    "_opponent_dropbacks_label",
            }[component]

            frame["_label"] = transform_label(
                frame[source_column],
                component,
            )

        else:
            raise KeyError(f"Unknown scope for {component}: {scope}")

        frame = frame.loc[frame["_label"].notna()].copy()

        if frame.empty:
            raise ValueError(
                f"{component}: no non-null training labels through 2024."
            )

        frames[component] = frame

    return frames


def train_component(
    component: str,
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    spec = COMPONENTS[component]
    features = spec["features"]

    if frame["season"].max() > FINAL_TRAIN_END:
        raise ValueError(
            f"{component}: frame contains data after {FINAL_TRAIN_END}."
        )

    selection_train = frame[
        frame["season"].le(MODEL_SELECTION_TRAIN_END)
    ].copy()

    validation = frame[
        frame["season"].eq(DEVELOPMENT_VALIDATION_SEASON)
    ].copy()

    final_train = frame[
        frame["season"].le(FINAL_TRAIN_END)
    ].copy()

    if selection_train.empty:
        raise ValueError(
            f"{component}: empty model-selection training set."
        )

    if validation.empty:
        raise ValueError(
            f"{component}: empty 2024 validation set."
        )

    if final_train.empty:
        raise ValueError(
            f"{component}: empty final training set."
        )

    X_select = numeric_frame(selection_train, features)
    y_select = selection_train["_label"].astype("float64")

    X_valid = numeric_frame(validation, features)
    y_valid = validation["_label"].astype("float64")

    X_final = numeric_frame(final_train, features)
    y_final = final_train["_label"].astype("float64")

    if X_select.isna().all(axis=1).all():
        raise ValueError(
            f"{component}: every model-selection row has all features missing."
        )

    if X_valid.isna().all(axis=1).all():
        raise ValueError(
            f"{component}: every 2024 validation row has all features missing."
        )

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 40,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "max_bin": 255,
        "verbosity": -1,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "data_random_seed": SEED,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": 1,
    }

    select_set = lgb.Dataset(
        X_select,
        label=y_select,
        feature_name=features,
        free_raw_data=False,
    )

    valid_set = lgb.Dataset(
        X_valid,
        label=y_valid,
        feature_name=features,
        reference=select_set,
        free_raw_data=False,
    )

    selected_model = lgb.train(
        params,
        select_set,
        num_boost_round=2000,
        valid_sets=[valid_set],
        valid_names=["validation_2024"],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=75,
                first_metric_only=True,
                verbose=False,
            ),
            lgb.log_evaluation(period=0),
        ],
    )

    best_iteration = int(
        selected_model.best_iteration
        if selected_model.best_iteration
        else 2000
    )

    if best_iteration < 1:
        raise ValueError(
            f"{component}: invalid best_iteration={best_iteration}"
        )

    valid_raw = selected_model.predict(
        X_valid,
        num_iteration=best_iteration,
    )

    valid_pred = transform_prediction(
        valid_raw,
        component,
    )

    valid_actual = y_valid.to_numpy(dtype="float64")

    validation_metrics = {
        "rmse": rmse(valid_actual, valid_pred),
        "mae": mae(valid_actual, valid_pred),
        "r2": r2(valid_actual, valid_pred),
    }

    final_set = lgb.Dataset(
        X_final,
        label=y_final,
        feature_name=features,
        free_raw_data=False,
    )

    final_model = lgb.train(
        params,
        final_set,
        num_boost_round=best_iteration,
        callbacks=[
            lgb.log_evaluation(period=0),
        ],
    )

    model_root = (
        common.prop_root()
        / "models"
        / "components"
        / component
    )

    model_path = model_root / "model.txt"
    manifest_path = model_root / "feature_manifest.json"
    metadata_path = model_root / "metadata.json"

    model_root.mkdir(parents=True, exist_ok=True)

    save_model_atomic(
        final_model,
        model_path,
        num_iteration=best_iteration,
    )

    feature_manifest = {
        "component": component,
        "model_type": "lightgbm_regression",
        "objective": "regression",
        "scope": spec["scope"],
        "label_source": spec["label_source"],
        "label_column": spec["label_column"],
        "numeric_features": features,
        "categorical_features": [],
        "feature_count": len(features),
        "automatic_all_numeric_selection": False,
        "prediction_transform": spec["prediction_transform"],
        "clip_min": 0.0,
        "clip_max": (
            1.0
            if component in BOUNDED_COMPONENTS
            else None
        ),
        "reconcile_during_current_week_allocation": bool(
            spec.get(
                "reconcile_during_current_week_allocation",
                False,
            )
        ),
        "forbidden_features": config["forbidden_features"],
    }

    write_json_atomic(
        manifest_path,
        feature_manifest,
    )

    importance_gain = final_model.feature_importance(
        importance_type="gain",
        iteration=best_iteration,
    )

    importance_split = final_model.feature_importance(
        importance_type="split",
        iteration=best_iteration,
    )

    importance = [
        {
            "feature": feature,
            "gain": float(gain),
            "split": int(split),
        }
        for feature, gain, split in zip(
            features,
            importance_gain,
            importance_split,
        )
    ]

    importance.sort(
        key=lambda row: (
            -row["gain"],
            -row["split"],
            row["feature"],
        )
    )

    metadata = {
        "component": component,
        "status": "trained",
        "model_type": "lightgbm_regression",
        "lightgbm_version": lgb.__version__,
        "seed": SEED,
        "training_policy": {
            "random_split_used": False,
            "model_selection_train_end_season":
                MODEL_SELECTION_TRAIN_END,
            "development_validation_season":
                DEVELOPMENT_VALIDATION_SEASON,
            "final_train_end_season": FINAL_TRAIN_END,
            "untouched_test_season": UNTOUCHED_TEST_SEASON,
            "untouched_test_used_for_selection": False,
            "untouched_test_used_for_metrics": False,
            "untouched_test_used_for_fit": False,
        },
        "rows": {
            "model_selection_train": int(len(selection_train)),
            "validation_2024": int(len(validation)),
            "final_train_through_2024": int(len(final_train)),
        },
        "label": {
            "source": spec["label_source"],
            "column": spec["label_column"],
            "minimum": float(y_final.min()),
            "maximum": float(y_final.max()),
            "mean": float(y_final.mean()),
            "first_training_season": int(final_train["season"].min()),
            "last_training_season": int(final_train["season"].max()),
        },
        "prediction": {
            "transform": spec["prediction_transform"],
            "clip_min": 0.0,
            "clip_max": (
                1.0
                if component in BOUNDED_COMPONENTS
                else None
            ),
            "reconcile_during_current_week_allocation": bool(
                spec.get(
                    "reconcile_during_current_week_allocation",
                    False,
                )
            ),
        },
        "best_iteration_selected_on_2024": best_iteration,
        "validation_2024_metrics": validation_metrics,
        "params": params,
        "feature_count": len(features),
        "feature_importance": importance,
        "feature_manifest_sha256": sha256_file(manifest_path),
        "model_sha256": sha256_file(model_path),
    }

    write_json_atomic(
        metadata_path,
        metadata,
    )

    return {
        "component": component,
        "model": str(
            model_path.relative_to(common.repo_root())
        ).replace("\\", "/"),
        "feature_manifest": str(
            manifest_path.relative_to(common.repo_root())
        ).replace("\\", "/"),
        "metadata": str(
            metadata_path.relative_to(common.repo_root())
        ).replace("\\", "/"),
        "features": len(features),
        "model_selection_train_rows": int(len(selection_train)),
        "validation_2024_rows": int(len(validation)),
        "final_train_rows": int(len(final_train)),
        "best_iteration": best_iteration,
        "validation_rmse": validation_metrics["rmse"],
        "validation_mae": validation_metrics["mae"],
    }


def main() -> int:
    # ISSUE28_MARKET_EXCLUSION_PREFLIGHT
    _issue28_audit = common.prop_root() / "scripts" / "validate" / "audit_market_exclusion.py"
    _issue28_result = __import__("subprocess").run(
        [__import__("sys").executable, str(_issue28_audit), "--preflight"],
        check=False,
    )
    if _issue28_result.returncode != 0:
        raise RuntimeError("Issue 28 market-exclusion preflight failed.")

    config = common.load_config()
    root = common.repo_root()

    eligibility = load_yaml(
        root / ELIGIBILITY_PATH
    )

    canonical_manifest = load_json(
        root / FEATURE_MANIFEST_PATH
    )

    folds = common.read_parquet_required(
        FOLDS_PATH
    )

    verify_backtest_policy(folds)
    assert_feature_contract(
        config,
        canonical_manifest,
    )

    all_features = unique(
        [
            column
            for component in COMPONENT_ORDER
            for column in COMPONENTS[component]["features"]
        ]
    )

    required_columns = unique(
        [
            "season",
            "week",
            "game_id",
            "player_id",
            "team",
            "opponent",
            "position",
            *all_features,
        ]
    )

    feature_path = (
        root / config["paths"]["historical_features"]
    )

    if not feature_path.is_file():
        raise FileNotFoundError(
            f"Historical feature table does not exist: {feature_path}"
        )

    features = pd.read_parquet(
        feature_path,
        columns=required_columns,
    )

    common.ensure_unique(
        features,
        GRAIN,
        "historical canonical features",
    )

    features["season"] = pd.to_numeric(
        features["season"],
        errors="raise",
    ).astype(int)

    features["week"] = pd.to_numeric(
        features["week"],
        errors="raise",
    ).astype(int)

    frames = build_training_frames(
        config,
        features,
        eligibility,
    )

    results = []

    for component in COMPONENT_ORDER:
        result = train_component(
            component,
            frames[component],
            config,
        )
        results.append(result)

        print(
            json.dumps(
                {
                    "component": component,
                    "status": "trained",
                    "features": result["features"],
                    "model_selection_train_rows":
                        result["model_selection_train_rows"],
                    "validation_2024_rows":
                        result["validation_2024_rows"],
                    "final_train_rows":
                        result["final_train_rows"],
                    "best_iteration":
                        result["best_iteration"],
                    "validation_rmse":
                        result["validation_rmse"],
                    "validation_mae":
                        result["validation_mae"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    payload = {
        "status": "passed",
        "components": len(results),
        "component_names": COMPONENT_ORDER,
        "model_type": "lightgbm_regression",
        "random_split_used": False,
        "model_selection_train_end_season":
            MODEL_SELECTION_TRAIN_END,
        "development_validation_season":
            DEVELOPMENT_VALIDATION_SEASON,
        "final_train_end_season": FINAL_TRAIN_END,
        "untouched_test_season": UNTOUCHED_TEST_SEASON,
        "untouched_test_used": False,
        "share_prediction_clip": [0.0, 1.0],
        "volume_prediction_floor": 0.0,
        "current_week_share_reconciliation_required": True,
        "results": results,
    }

    common.log_run(
        "train_opportunity_models.py",
        payload,
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


if __name__ == "__main__":
    raise SystemExit(main())
