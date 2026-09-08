#!/usr/bin/env python3
"""
Train/evaluate deterministic NFL Prop Engine baseline models.

READS
-----
- config/prop_engine.yaml
- config/target_eligibility.yaml
- data/historical/features/player_game_features.parquet
- evaluation/backtest_folds.parquet

WRITES
------
- evaluation/baseline_oof_predictions.parquet

BASELINE CONTRACT
-----------------
passing_yards =
    projected_attempts_baseline * rolling_yards_per_attempt

passing_tds =
    projected_attempts_baseline * rolling_passing_td_rate

rushing_yards =
    projected_carries_baseline * rolling_yards_per_carry

rushing_tds =
    projected_goal_line_carries * shrunk_goal_line_td_rate

receiving_yards =
    projected_team_pass_attempts
    * projected_target_share
    * rolling_yards_per_target

receiving_tds =
    projected_red_zone_targets * shrunk_receiving_td_rate

kicking_points =
    3 * expected_fg_makes + expected_pat_makes

tackles =
    projected_opponent_plays
    * projected_defensive_participation
    * rolling_tackle_rate

sacks =
    projected_opponent_dropbacks
    * projected_defensive_participation
    * shrunk_sack_rate

POLICY
------
- No random split.
- OOF rows come only from each chronological fold's validation/test window.
- 2025 is scored only by the fixed baseline formulas using training data through
  2024. No formula, fallback order, shrinkage rule, or hyperparameter is chosen
  from 2025 performance.
- Projection formulas never use target_* columns. Targets are read only after
  projections are constructed, for the output column `actual`.
- Missing source data remains missing; it is never converted to zero.
- Yardage efficiency is not clipped nonnegative because signed yardage is valid.
- Count/probability/participation components are bounded where mathematically
  required.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common



_CONFIG_CONTRACT = common.load_config()
_TRAINING_CONTRACT = _CONFIG_CONTRACT["training"]
MODEL_SELECTION_TRAIN_END = int(_TRAINING_CONTRACT["model_selection_train_end_season"])
DEVELOPMENT_VALIDATION_SEASON = int(_TRAINING_CONTRACT["development_validation_season"])
FINAL_TRAIN_END = int(_TRAINING_CONTRACT["final_train_end_season"])
UNTOUCHED_TEST_SEASON = int(_TRAINING_CONTRACT["untouched_test_season"])
OUTPUT_PATH = (
    "docs/win/football/nfl/prop_engine/evaluation/"
    "baseline_oof_predictions.parquet"
)

FOLDS_PATH = (
    "docs/win/football/nfl/prop_engine/evaluation/"
    "backtest_folds.parquet"
)

ELIGIBILITY_PATH = (
    "docs/win/football/nfl/prop_engine/config/"
    "target_eligibility.yaml"
)

OUTPUT_COLUMNS = [
    "fold_id",
    "season",
    "week",
    "game_id",
    "player_id",
    "target",
    "actual",
    "baseline_projection",
]

TARGETS = list(_CONFIG_CONTRACT["targets"].keys())

TARGET_COLUMN = {
    target: f"target_{target}"
    for target in TARGETS
}

# All columns below are pregame feature columns from Issue 17.
BASELINE_FEATURE_COLUMNS = [
    # Passing volume / efficiency.
    "player_pass_attempts_lag1",
    "player_pass_attempts_roll3_mean",
    "player_pass_attempts_roll5_mean",
    "player_pass_attempts_season_to_date",
    "player_pass_attempts_career_prior",
    "player_yards_per_attempt_lag1",
    "player_yards_per_attempt_roll3_mean",
    "player_yards_per_attempt_roll5_mean",
    "player_yards_per_attempt_season_to_date",
    "player_yards_per_attempt_career_prior",
    "player_passing_td_rate_lag1",
    "player_passing_td_rate_roll3_mean",
    "player_passing_td_rate_roll5_mean",
    "player_passing_td_rate_season_to_date",
    "player_passing_td_rate_career_prior",

    # Rushing.
    "player_carries_lag1",
    "player_carries_roll3_mean",
    "player_carries_roll5_mean",
    "player_carries_season_to_date",
    "player_carries_career_prior",
    "player_yards_per_carry_lag1",
    "player_yards_per_carry_roll3_mean",
    "player_yards_per_carry_roll5_mean",
    "player_yards_per_carry_season_to_date",
    "player_yards_per_carry_career_prior",
    "player_goal_line_carries_lag1",
    "player_goal_line_carries_roll3_mean",
    "player_goal_line_carries_roll5_mean",
    "player_goal_line_carries_season_to_date",
    "player_goal_line_carries_career_prior",
    "player_rushing_tds_roll5_mean",

    # Receiving.
    "team_pass_attempts_lag1",
    "team_pass_attempts_roll3_mean",
    "team_pass_attempts_roll5_mean",
    "team_pass_attempts_season_to_date",
    "player_target_share_lag1",
    "player_target_share_roll3_mean",
    "player_target_share_roll5_mean",
    "player_target_share_season_to_date",
    "player_target_share_career_prior",
    "player_yards_per_target_lag1",
    "player_yards_per_target_roll3_mean",
    "player_yards_per_target_roll5_mean",
    "player_yards_per_target_season_to_date",
    "player_yards_per_target_career_prior",
    "player_red_zone_targets_lag1",
    "player_red_zone_targets_roll3_mean",
    "player_red_zone_targets_roll5_mean",
    "player_red_zone_targets_season_to_date",
    "player_red_zone_targets_career_prior",
    "player_receiving_tds_roll5_mean",

    # Kicking.
    "player_kicking_fg_attempts_lag1",
    "player_kicking_fg_attempts_roll3",
    "player_kicking_fg_attempts_roll5",
    "player_kicking_fg_make_pct_career_prior",
    "player_kicking_fg_make_pct_season_prior",
    "player_kicking_pat_attempts_roll3",
    "player_kicking_pat_make_pct_career_prior",

    # Defense.
    "matchup_expected_opponent_plays",
    "matchup_expected_opponent_dropbacks",
    "player_defensive_def_snap_pct_lag1",
    "player_defensive_def_snap_pct_roll3",
    "player_defensive_def_participation_lag1",
    "player_defensive_def_participation_roll3",
    "player_defensive_tackles_roll3",
    "player_defensive_sacks_roll5",
    "player_defensive_opponent_plays_roll3",
    "player_defensive_opponent_dropbacks_roll3",
]


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required YAML does not exist: {path}")

    with path.open("r", encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle)

    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")

    return value


def numeric(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype("float64")
    )


def coalesce_numeric(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> pd.Series:
    """
    First non-null numeric value in the declared order.

    The order is part of the deterministic baseline definition. No data-driven
    feature selection occurs.
    """
    columns = list(columns)

    if not columns:
        raise ValueError("coalesce_numeric requires at least one column.")

    result = pd.Series(
        np.nan,
        index=frame.index,
        dtype="float64",
    )

    for column in columns:
        candidate = numeric(frame[column])
        result = result.where(result.notna(), candidate)

    return result


def clip_probability(series: pd.Series) -> pd.Series:
    return numeric(series).clip(lower=0.0, upper=1.0)


def clip_nonnegative(series: pd.Series) -> pd.Series:
    return numeric(series).clip(lower=0.0)


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    numerator = numeric(numerator)
    denominator = numeric(denominator)

    result = pd.Series(
        np.nan,
        index=numerator.index,
        dtype="float64",
    )

    valid = (
        numerator.notna()
        & denominator.notna()
        & denominator.gt(0.0)
    )

    result.loc[valid] = (
        numerator.loc[valid]
        / denominator.loc[valid]
    )

    return result.replace([np.inf, -np.inf], np.nan)


def season_week_mask(
    frame: pd.DataFrame,
    start_season: int,
    start_week: int,
    end_season: int,
    end_week: int,
) -> pd.Series:
    season = pd.to_numeric(frame["season"], errors="raise").astype(int)
    week = pd.to_numeric(frame["week"], errors="raise").astype(int)

    after_start = (
        (season > start_season)
        | ((season == start_season) & (week >= start_week))
    )

    before_end = (
        (season < end_season)
        | ((season == end_season) & (week <= end_week))
    )

    return after_start & before_end


def position_mask(
    frame: pd.DataFrame,
    eligible_positions: list[str],
) -> pd.Series:
    eligible = {
        str(value).strip().upper()
        for value in eligible_positions
    }

    position = (
        frame["position"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return position.isin(eligible)


def empirical_rate_prior(
    raw_rate: pd.Series,
    exposure: pd.Series,
) -> tuple[float, float]:
    """
    Training-only empirical-Bayes prior.

    Prior rate:
        exposure-weighted mean of pregame player rates in the fold's
        training window.

    Prior exposure:
        median positive historical exposure among those training rows.

    Both values are derived only from strictly pregame features already
    present on training rows. No target/actual column is used.
    """
    rate = clip_probability(raw_rate)
    exposure = clip_nonnegative(exposure)

    valid = (
        rate.notna()
        & exposure.notna()
        & exposure.gt(0.0)
    )

    if not valid.any():
        return float("nan"), float("nan")

    exp = exposure.loc[valid]
    r = rate.loc[valid]

    total_exposure = float(exp.sum())

    if not math.isfinite(total_exposure) or total_exposure <= 0.0:
        return float("nan"), float("nan")

    prior_rate = float((r * exp).sum() / total_exposure)
    prior_exposure = float(exp.median())

    if not math.isfinite(prior_rate):
        prior_rate = float("nan")

    if (
        not math.isfinite(prior_exposure)
        or prior_exposure <= 0.0
    ):
        prior_exposure = float("nan")

    return prior_rate, prior_exposure


def shrink_rate(
    raw_rate: pd.Series,
    exposure: pd.Series,
    prior_rate: float,
    prior_exposure: float,
) -> pd.Series:
    """
    Shrink a pregame player rate toward the training-only prior.

    If the training prior is unavailable because the source has not started,
    retain an available player raw rate. If the player rate is unavailable but
    the training prior exists, use the prior. If neither exists, preserve NaN.
    """
    raw = clip_probability(raw_rate)
    exp = clip_nonnegative(exposure)

    result = pd.Series(
        np.nan,
        index=raw.index,
        dtype="float64",
    )

    prior_ok = (
        math.isfinite(prior_rate)
        and math.isfinite(prior_exposure)
        and prior_exposure > 0.0
    )

    player_ok = (
        raw.notna()
        & exp.notna()
        & exp.gt(0.0)
    )

    if prior_ok:
        result.loc[player_ok] = (
            (
                raw.loc[player_ok] * exp.loc[player_ok]
                + prior_rate * prior_exposure
            )
            / (
                exp.loc[player_ok]
                + prior_exposure
            )
        )

        result.loc[~player_ok] = prior_rate
    else:
        result.loc[player_ok] = raw.loc[player_ok]

    return clip_probability(result)


def base_components(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Construct deterministic pregame components that do not depend on fold
    outcomes or target columns.
    """
    components: dict[str, pd.Series] = {}

    # Passing.
    components["projected_attempts_baseline"] = clip_nonnegative(
        coalesce_numeric(
            frame,
            [
                "player_pass_attempts_roll3_mean",
                "player_pass_attempts_roll5_mean",
                "player_pass_attempts_season_to_date",
                "player_pass_attempts_career_prior",
                "player_pass_attempts_lag1",
            ],
        )
    )

    # Preserve signed efficiency. Do not clip yardage rates to nonnegative.
    components["rolling_yards_per_attempt"] = coalesce_numeric(
        frame,
        [
            "player_yards_per_attempt_roll5_mean",
            "player_yards_per_attempt_roll3_mean",
            "player_yards_per_attempt_season_to_date",
            "player_yards_per_attempt_career_prior",
            "player_yards_per_attempt_lag1",
        ],
    )

    components["rolling_passing_td_rate"] = clip_probability(
        coalesce_numeric(
            frame,
            [
                "player_passing_td_rate_roll5_mean",
                "player_passing_td_rate_roll3_mean",
                "player_passing_td_rate_season_to_date",
                "player_passing_td_rate_career_prior",
                "player_passing_td_rate_lag1",
            ],
        )
    )

    # Rushing.
    components["projected_carries_baseline"] = clip_nonnegative(
        coalesce_numeric(
            frame,
            [
                "player_carries_roll3_mean",
                "player_carries_roll5_mean",
                "player_carries_season_to_date",
                "player_carries_career_prior",
                "player_carries_lag1",
            ],
        )
    )

    components["rolling_yards_per_carry"] = coalesce_numeric(
        frame,
        [
            "player_yards_per_carry_roll5_mean",
            "player_yards_per_carry_roll3_mean",
            "player_yards_per_carry_season_to_date",
            "player_yards_per_carry_career_prior",
            "player_yards_per_carry_lag1",
        ],
    )

    components["projected_goal_line_carries"] = clip_nonnegative(
        coalesce_numeric(
            frame,
            [
                "player_goal_line_carries_roll3_mean",
                "player_goal_line_carries_roll5_mean",
                "player_goal_line_carries_season_to_date",
                "player_goal_line_carries_career_prior",
                "player_goal_line_carries_lag1",
            ],
        )
    )

    # Five-game historical exposure used only for TD-rate shrinkage.
    goal_line_exposure = (
        clip_nonnegative(frame["player_goal_line_carries_roll5_mean"])
        * 5.0
    )
    goal_line_success = (
        clip_nonnegative(frame["player_rushing_tds_roll5_mean"])
        * 5.0
    )

    components["goal_line_rate_exposure"] = goal_line_exposure
    components["goal_line_td_rate_raw"] = clip_probability(
        safe_divide(
            goal_line_success,
            goal_line_exposure,
        )
    )

    # Receiving.
    components["projected_team_pass_attempts"] = clip_nonnegative(
        coalesce_numeric(
            frame,
            [
                "team_pass_attempts_roll3_mean",
                "team_pass_attempts_roll5_mean",
                "team_pass_attempts_season_to_date",
                "team_pass_attempts_lag1",
            ],
        )
    )

    components["projected_target_share"] = clip_probability(
        coalesce_numeric(
            frame,
            [
                "player_target_share_roll3_mean",
                "player_target_share_roll5_mean",
                "player_target_share_season_to_date",
                "player_target_share_career_prior",
                "player_target_share_lag1",
            ],
        )
    )

    components["rolling_yards_per_target"] = coalesce_numeric(
        frame,
        [
            "player_yards_per_target_roll5_mean",
            "player_yards_per_target_roll3_mean",
            "player_yards_per_target_season_to_date",
            "player_yards_per_target_career_prior",
            "player_yards_per_target_lag1",
        ],
    )

    components["projected_red_zone_targets"] = clip_nonnegative(
        coalesce_numeric(
            frame,
            [
                "player_red_zone_targets_roll3_mean",
                "player_red_zone_targets_roll5_mean",
                "player_red_zone_targets_season_to_date",
                "player_red_zone_targets_career_prior",
                "player_red_zone_targets_lag1",
            ],
        )
    )

    receiving_td_exposure = (
        clip_nonnegative(frame["player_red_zone_targets_roll5_mean"])
        * 5.0
    )
    receiving_td_success = (
        clip_nonnegative(frame["player_receiving_tds_roll5_mean"])
        * 5.0
    )

    components["receiving_td_rate_exposure"] = receiving_td_exposure
    components["receiving_td_rate_raw"] = clip_probability(
        safe_divide(
            receiving_td_success,
            receiving_td_exposure,
        )
    )

    # Kicking.
    components["projected_fg_attempts"] = clip_nonnegative(
        coalesce_numeric(
            frame,
            [
                "player_kicking_fg_attempts_roll3",
                "player_kicking_fg_attempts_roll5",
                "player_kicking_fg_attempts_lag1",
            ],
        )
    )

    components["expected_fg_make_pct"] = clip_probability(
        coalesce_numeric(
            frame,
            [
                "player_kicking_fg_make_pct_season_prior",
                "player_kicking_fg_make_pct_career_prior",
            ],
        )
    )

    components["projected_pat_attempts"] = clip_nonnegative(
        numeric(frame["player_kicking_pat_attempts_roll3"])
    )

    components["expected_pat_make_pct"] = clip_probability(
        numeric(frame["player_kicking_pat_make_pct_career_prior"])
    )

    # Defense.
    historical_def_participation = clip_probability(
        coalesce_numeric(
            frame,
            [
                "player_defensive_def_participation_roll3",
                "player_defensive_def_participation_lag1",
                "player_defensive_def_snap_pct_roll3",
                "player_defensive_def_snap_pct_lag1",
            ],
        )
    )

    components[
        "projected_defensive_participation"
    ] = historical_def_participation

    components["projected_opponent_plays"] = clip_nonnegative(
        coalesce_numeric(
            frame,
            [
                "matchup_expected_opponent_plays",
                "player_defensive_opponent_plays_roll3",
            ],
        )
    )

    components["projected_opponent_dropbacks"] = clip_nonnegative(
        coalesce_numeric(
            frame,
            [
                "matchup_expected_opponent_dropbacks",
                "player_defensive_opponent_dropbacks_roll3",
            ],
        )
    )

    tackle_exposure = (
        clip_nonnegative(frame["player_defensive_opponent_plays_roll3"])
        * historical_def_participation
    )
    tackle_success = clip_nonnegative(
        frame["player_defensive_tackles_roll3"]
    )

    components["tackle_rate_exposure"] = tackle_exposure
    components["rolling_tackle_rate"] = clip_probability(
        safe_divide(
            tackle_success,
            tackle_exposure,
        )
    )

    # Approximate five-game pass-rush exposure using the pregame rolling
    # opponent-dropback context and pregame defensive participation.
    sack_exposure = (
        clip_nonnegative(frame["player_defensive_opponent_dropbacks_roll3"])
        * historical_def_participation
        * 5.0
    )
    sack_success = (
        clip_nonnegative(frame["player_defensive_sacks_roll5"])
        * 5.0
    )

    components["sack_rate_exposure"] = sack_exposure
    components["sack_rate_raw"] = clip_probability(
        safe_divide(
            sack_success,
            sack_exposure,
        )
    )

    return components


def build_projection(
    target: str,
    components: dict[str, pd.Series],
    *,
    goal_line_prior: tuple[float, float],
    receiving_td_prior: tuple[float, float],
    sack_prior: tuple[float, float],
) -> pd.Series:
    if target == "passing_yards":
        projection = (
            components["projected_attempts_baseline"]
            * components["rolling_yards_per_attempt"]
        )

    elif target == "passing_tds":
        projection = (
            components["projected_attempts_baseline"]
            * components["rolling_passing_td_rate"]
        )
        projection = clip_nonnegative(projection)

    elif target == "rushing_yards":
        projection = (
            components["projected_carries_baseline"]
            * components["rolling_yards_per_carry"]
        )

    elif target == "rushing_tds":
        shrunk = shrink_rate(
            components["goal_line_td_rate_raw"],
            components["goal_line_rate_exposure"],
            goal_line_prior[0],
            goal_line_prior[1],
        )
        projection = (
            components["projected_goal_line_carries"]
            * shrunk
        )
        projection = clip_nonnegative(projection)

    elif target == "receiving_yards":
        projection = (
            components["projected_team_pass_attempts"]
            * components["projected_target_share"]
            * components["rolling_yards_per_target"]
        )

    elif target == "receiving_tds":
        shrunk = shrink_rate(
            components["receiving_td_rate_raw"],
            components["receiving_td_rate_exposure"],
            receiving_td_prior[0],
            receiving_td_prior[1],
        )
        projection = (
            components["projected_red_zone_targets"]
            * shrunk
        )
        projection = clip_nonnegative(projection)

    elif target == "kicking_points":
        expected_fg_makes = (
            components["projected_fg_attempts"]
            * components["expected_fg_make_pct"]
        )
        expected_pat_makes = (
            components["projected_pat_attempts"]
            * components["expected_pat_make_pct"]
        )
        projection = (
            3.0 * expected_fg_makes
            + expected_pat_makes
        )
        projection = clip_nonnegative(projection)

    elif target == "tackles":
        projection = (
            components["projected_opponent_plays"]
            * components["projected_defensive_participation"]
            * components["rolling_tackle_rate"]
        )
        projection = clip_nonnegative(projection)

    elif target == "sacks":
        shrunk = shrink_rate(
            components["sack_rate_raw"],
            components["sack_rate_exposure"],
            sack_prior[0],
            sack_prior[1],
        )
        projection = (
            components["projected_opponent_dropbacks"]
            * components["projected_defensive_participation"]
            * shrunk
        )
        projection = clip_nonnegative(projection)

    else:
        raise KeyError(f"Unsupported baseline target: {target}")

    return numeric(projection)


def validate_contract(
    config: dict,
    eligibility: dict,
    features: pd.DataFrame,
    folds: pd.DataFrame,
) -> None:
    if list(eligibility.keys()) != TARGETS:
        raise ValueError(
            "target_eligibility.yaml target order/content does not match "
            f"required targets. actual={list(eligibility.keys())}"
        )

    for target in TARGETS:
        spec = eligibility[target]

        if not isinstance(spec, dict):
            raise ValueError(
                f"Eligibility entry must be a mapping: {target}"
            )

        positions = spec.get("eligible_positions")

        if not isinstance(positions, list) or not positions:
            raise ValueError(
                f"Missing eligible_positions for target {target}"
            )

    required_feature_columns = [
        "season",
        "week",
        "game_id",
        "kickoff_timestamp",
        "player_id",
        "position",
        *BASELINE_FEATURE_COLUMNS,
        *[TARGET_COLUMN[target] for target in TARGETS],
    ]

    common.require_columns(
        features,
        required_feature_columns,
        "historical feature table",
    )

    common.ensure_unique(
        features,
        ["season", "week", "game_id", "player_id"],
        "historical feature table",
    )

    common.reject_forbidden_feature_columns(
        BASELINE_FEATURE_COLUMNS,
        config,
    )

    common.require_columns(
        folds,
        [
            "fold_id",
            "train_start_season",
            "train_start_week",
            "train_end_season",
            "train_end_week",
            "validation_start_season",
            "validation_start_week",
            "validation_end_season",
            "validation_end_week",
            "test_flag",
        ],
        "backtest folds",
    )

    common.ensure_unique(
        folds,
        ["fold_id"],
        "backtest folds",
    )

    if len(folds.loc[folds["test_flag"].eq(1)]) != 1:
        raise ValueError(
            "Expected exactly one untouched test fold."
        )

    test = folds.loc[folds["test_flag"].eq(1)].iloc[0]

    if int(test["validation_start_season"]) != UNTOUCHED_TEST_SEASON:
        raise ValueError(
            "Issue 21 requires the untouched test season to be 2025."
        )

    if int(test["train_end_season"]) != FINAL_TRAIN_END:
        raise ValueError(
            "Issue 21 requires the untouched test to train through 2024."
        )

    development_fold = folds.loc[
        folds["validation_start_season"].eq(DEVELOPMENT_VALIDATION_SEASON)
        & folds["test_flag"].eq(0)
    ]

    if len(development_fold) != 1:
        raise ValueError(
            "Expected exactly one 2024 development validation fold."
        )

    if int(development_fold.iloc[0]["train_end_season"]) != MODEL_SELECTION_TRAIN_END:
        raise ValueError(
            "Issue 21 development policy requires train through 2023 "
            "and validate 2024."
        )


def build_oof_predictions(
    features: pd.DataFrame,
    folds: pd.DataFrame,
    eligibility: dict,
) -> pd.DataFrame:
    all_records: list[pd.DataFrame] = []

    # Precompute target-independent components once. They use only pregame
    # features and contain no target columns.
    components = base_components(features)

    kickoff = pd.to_datetime(
        features["kickoff_timestamp"],
        errors="raise",
        utc=True,
    )

    for fold in folds.itertuples(index=False):
        fold_id = str(fold.fold_id)

        train_mask = season_week_mask(
            features,
            int(fold.train_start_season),
            int(fold.train_start_week),
            int(fold.train_end_season),
            int(fold.train_end_week),
        )

        validation_mask = season_week_mask(
            features,
            int(fold.validation_start_season),
            int(fold.validation_start_week),
            int(fold.validation_end_season),
            int(fold.validation_end_week),
        )

        if not train_mask.any():
            raise ValueError(
                f"{fold_id}: training window is empty."
            )

        if not validation_mask.any():
            raise ValueError(
                f"{fold_id}: validation/test window is empty."
            )

        max_train_kickoff = kickoff.loc[train_mask].max()
        min_validation_kickoff = kickoff.loc[validation_mask].min()

        if not max_train_kickoff < min_validation_kickoff:
            raise ValueError(
                f"{fold_id}: training kickoff leakage detected. "
                f"max_train={max_train_kickoff}, "
                f"min_validation={min_validation_kickoff}"
            )

        # Fold-specific shrinkage priors are estimated exclusively from the
        # fold's training rows and exclusively from pregame feature values.
        goal_line_prior = empirical_rate_prior(
            components["goal_line_td_rate_raw"].loc[train_mask],
            components["goal_line_rate_exposure"].loc[train_mask],
        )

        receiving_td_prior = empirical_rate_prior(
            components["receiving_td_rate_raw"].loc[train_mask],
            components["receiving_td_rate_exposure"].loc[train_mask],
        )

        sack_prior = empirical_rate_prior(
            components["sack_rate_raw"].loc[train_mask],
            components["sack_rate_exposure"].loc[train_mask],
        )

        for target in TARGETS:
            eligible_mask = (
                validation_mask
                & position_mask(
                    features,
                    eligibility[target]["eligible_positions"],
                )
            )

            actual = numeric(features[TARGET_COLUMN[target]])

            # OOF evaluation rows require a realized target. A realized zero is
            # retained; only true missing targets are excluded.
            row_mask = (
                eligible_mask
                & actual.notna()
            )

            if not row_mask.any():
                continue

            projection = build_projection(
                target,
                components,
                goal_line_prior=goal_line_prior,
                receiving_td_prior=receiving_td_prior,
                sack_prior=sack_prior,
            )

            subset = features.loc[
                row_mask,
                [
                    "season",
                    "week",
                    "game_id",
                    "player_id",
                ],
            ].copy()

            subset.insert(0, "fold_id", fold_id)
            subset["target"] = target
            subset["actual"] = actual.loc[row_mask].astype("float64")
            subset["baseline_projection"] = (
                projection.loc[row_mask].astype("float64")
            )

            subset = subset[OUTPUT_COLUMNS]
            all_records.append(subset)

    if not all_records:
        raise ValueError("No baseline OOF predictions were produced.")

    output = pd.concat(
        all_records,
        ignore_index=True,
    )

    output["season"] = pd.to_numeric(
        output["season"],
        errors="raise",
    ).astype(int)

    output["week"] = pd.to_numeric(
        output["week"],
        errors="raise",
    ).astype(int)

    output["actual"] = numeric(output["actual"])
    output["baseline_projection"] = numeric(
        output["baseline_projection"]
    )

    output = output.sort_values(
        [
            "season",
            "week",
            "game_id",
            "player_id",
            "target",
            "fold_id",
        ],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)

    return output


def validate_output(
    output: pd.DataFrame,
    folds: pd.DataFrame,
) -> None:
    if list(output.columns) != OUTPUT_COLUMNS:
        raise ValueError(
            "baseline_oof_predictions output column order mismatch."
        )

    common.ensure_unique(
        output,
        [
            "fold_id",
            "season",
            "week",
            "game_id",
            "player_id",
            "target",
        ],
        "baseline OOF predictions",
    )

    if not set(output["target"].unique()) <= set(TARGETS):
        raise ValueError("Unexpected target in baseline OOF output.")

    if output["actual"].isna().any():
        raise ValueError(
            "OOF output contains missing actual values."
        )

    numeric_projection = numeric(output["baseline_projection"])

    if np.isinf(
        numeric_projection.dropna().to_numpy(dtype=float)
    ).any():
        raise ValueError(
            "OOF output contains infinite baseline projections."
        )

    nonnegative_targets = {
        "passing_tds",
        "rushing_tds",
        "receiving_tds",
        "kicking_points",
        "tackles",
        "sacks",
    }

    invalid_negative = (
        output["target"].isin(nonnegative_targets)
        & numeric_projection.lt(0.0)
    )

    if invalid_negative.any():
        sample = output.loc[
            invalid_negative,
            OUTPUT_COLUMNS,
        ].head(20)

        raise ValueError(
            "Negative nonnegative-target baseline projection(s): "
            f"{sample.to_dict(orient='records')}"
        )

    expected_fold_ids = set(folds["fold_id"].astype(str))
    actual_fold_ids = set(output["fold_id"].astype(str))

    if not actual_fold_ids <= expected_fold_ids:
        raise ValueError(
            "OOF output contains unknown fold IDs."
        )

    # A season is validated/tested by exactly one annual fold, so no player
    # target observation may appear in more than one fold.
    common.ensure_unique(
        output,
        [
            "season",
            "week",
            "game_id",
            "player_id",
            "target",
        ],
        "cross-fold OOF observation",
    )

    # 2025 may appear only under the untouched test fold.
    rows_test = output.loc[output["season"].eq(UNTOUCHED_TEST_SEASON)]

    if not rows_test.empty:
        allowed_test_ids = set(
            folds.loc[
                folds["test_flag"].eq(1),
                "fold_id",
            ].astype(str)
        )

        if not set(rows_test["fold_id"].astype(str)) <= allowed_test_ids:
            raise ValueError(
                "2025 predictions appeared in a development fold."
            )


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
    prop_root = common.prop_root()

    eligibility = load_yaml(
        common.repo_root() / ELIGIBILITY_PATH
    )

    features_path = (
        common.repo_root()
        / config["paths"]["historical_features"]
    )

    if not features_path.is_file():
        raise FileNotFoundError(
            f"Historical feature table does not exist: {features_path}"
        )

    # Read only the required columns to keep the baseline trainer lightweight.
    required_columns = [
        "season",
        "week",
        "game_id",
        "kickoff_timestamp",
        "player_id",
        "position",
        *BASELINE_FEATURE_COLUMNS,
        *[TARGET_COLUMN[target] for target in TARGETS],
    ]

    # Preserve declaration order while removing duplicates.
    required_columns = list(dict.fromkeys(required_columns))

    features = pd.read_parquet(
        features_path,
        columns=required_columns,
    )

    folds = common.read_parquet_required(
        FOLDS_PATH,
    )

    validate_contract(
        config,
        eligibility,
        features,
        folds,
    )

    output = build_oof_predictions(
        features,
        folds,
        eligibility,
    )

    validate_output(
        output,
        folds,
    )

    common.write_parquet_atomic(
        output,
        OUTPUT_PATH,
    )

    coverage = (
        output.assign(
            projected=output["baseline_projection"].notna()
        )
        .groupby("target", sort=True)["projected"]
        .agg(["sum", "count"])
    )

    coverage_payload = {
        target: {
            "projected_rows": int(row["sum"]),
            "eligible_actual_rows": int(row["count"]),
            "coverage": (
                float(row["sum"] / row["count"])
                if row["count"]
                else None
            ),
        }
        for target, row in coverage.iterrows()
    }

    payload = {
        "status": "passed",
        "output": OUTPUT_PATH,
        "rows": int(len(output)),
        "folds": int(output["fold_id"].nunique()),
        "targets": int(output["target"].nunique()),
        "random_split_used": False,
        "target_columns_used_in_projection": False,
        "untouched_test_season": UNTOUCHED_TEST_SEASON,
        "test_tuning_used": False,
        "coverage": coverage_payload,
    }

    common.log_run(
        "train_baselines.py",
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
