#!/usr/bin/env python3
"""
Train deterministic LightGBM efficiency models for the NFL Prop Engine.

REQUIRED MODELS
---------------
passing_yards_per_attempt
passing_td_rate
rushing_yards_per_carry
rushing_td_per_goal_line_carry
receiving_yards_per_target
receiving_td_per_red_zone_target
field_goal_conversion
extra_point_conversion
tackle_rate_per_defensive_play
sack_rate_per_defensive_play

MODEL ARTIFACTS
---------------
For each model:
    models/efficiency/{model_name}/model.txt
    models/efficiency/{model_name}/feature_manifest.json
    models/efficiency/{model_name}/metadata.json

POLICY
------
- LightGBM regression.
- Model selection trains through 2023 and validates on 2024.
- Persisted final models train through 2024.
- 2025 is untouched: no fitting, tuning, metrics, priors, or feature generation
  from 2025 are used here.
- Every row-level feature used for model training is available strictly before
  the target game's kickoff.
- Small-sample efficiency history is shrunk player -> position -> league.
- Rookies / zero-history players use the strictly prior position prior when
  available, then the league prior.
- TD efficiency and sack rate use materially stronger prior exposure than
  corresponding yardage / ordinary rate models.
- Player shrinkage history is keyed by player_id, NOT team, so trades preserve
  player efficiency history.
- Opportunity-share features are prohibited from all efficiency models; old
  team carry/target share is therefore never imported into efficiency.
- Exact PBP touchdown flags are used for:
    * rushing TD per goal-line carry (inside the 5)
    * receiving TD per red-zone target (20 or closer)
- Yardage efficiencies remain signed. Probability/rate predictions are clipped
  to [0, 1].
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
        "Issue 23 requires LightGBM. Install with: "
        "python -m pip install lightgbm"
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
ELIGIBILITY_PATH = (
    "docs/win/football/nfl/prop_engine/config/target_eligibility.yaml"
)
FOLDS_PATH = (
    "docs/win/football/nfl/prop_engine/evaluation/backtest_folds.parquet"
)

GRAIN = ["season", "week", "game_id", "player_id"]

MODELS = [
    "passing_yards_per_attempt",
    "passing_td_rate",
    "rushing_yards_per_carry",
    "rushing_td_per_goal_line_carry",
    "receiving_yards_per_target",
    "receiving_td_per_red_zone_target",
    "field_goal_conversion",
    "extra_point_conversion",
    "tackle_rate_per_defensive_play",
    "sack_rate_per_defensive_play",
]

RATE_MODELS = {
    "passing_td_rate",
    "rushing_td_per_goal_line_carry",
    "receiving_td_per_red_zone_target",
    "field_goal_conversion",
    "extra_point_conversion",
    "tackle_rate_per_defensive_play",
    "sack_rate_per_defensive_play",
}

YARDAGE_MODELS = {
    "passing_yards_per_attempt",
    "rushing_yards_per_carry",
    "receiving_yards_per_target",
}

TD_MODELS = {
    "passing_td_rate",
    "rushing_td_per_goal_line_carry",
    "receiving_td_per_red_zone_target",
}

# Exposure units are attempts/carries/targets/defensive plays. These are fixed
# deterministic regularization constants, not tuned from 2025.
SHRINKAGE_EXPOSURE = {
    "passing_yards_per_attempt": 50.0,
    "passing_td_rate": 200.0,
    "rushing_yards_per_carry": 50.0,
    "rushing_td_per_goal_line_carry": 100.0,
    "receiving_yards_per_target": 50.0,
    "receiving_td_per_red_zone_target": 100.0,
    "field_goal_conversion": 25.0,
    "extra_point_conversion": 40.0,
    "tackle_rate_per_defensive_play": 150.0,
    "sack_rate_per_defensive_play": 500.0,
}

DERIVED_FEATURES = [
    "eff_player_prior_rate",
    "eff_position_prior_rate",
    "eff_league_prior_rate",
    "eff_shrunk_rate",
    "eff_player_prior_exposure",
    "eff_prior_weight",
    "eff_rookie_position_prior_flag",
]

COMMON_HISTORY_FEATURES = [
    "history_no_nfl_history_flag",
    "history_history_games",
]

FEATURES: dict[str, list[str]] = {
    "passing_yards_per_attempt": [
        "player_yards_per_attempt_lag1",
        "player_yards_per_attempt_roll3_mean",
        "player_yards_per_attempt_roll5_mean",
        "player_yards_per_attempt_ewm5",
        "player_yards_per_attempt_season_to_date",
        "player_yards_per_attempt_career_prior",
        "player_pass_attempts_career_prior",
        *COMMON_HISTORY_FEATURES,
        *DERIVED_FEATURES,
    ],
    "passing_td_rate": [
        "player_passing_td_rate_lag1",
        "player_passing_td_rate_roll3_mean",
        "player_passing_td_rate_roll5_mean",
        "player_passing_td_rate_ewm5",
        "player_passing_td_rate_season_to_date",
        "player_passing_td_rate_career_prior",
        "player_pass_attempts_career_prior",
        *COMMON_HISTORY_FEATURES,
        *DERIVED_FEATURES,
    ],
    "rushing_yards_per_carry": [
        "player_yards_per_carry_lag1",
        "player_yards_per_carry_roll3_mean",
        "player_yards_per_carry_roll5_mean",
        "player_yards_per_carry_ewm5",
        "player_yards_per_carry_season_to_date",
        "player_yards_per_carry_career_prior",
        "player_carries_career_prior",
        *COMMON_HISTORY_FEATURES,
        *DERIVED_FEATURES,
    ],
    "rushing_td_per_goal_line_carry": [
        "player_goal_line_carries_lag1",
        "player_goal_line_carries_roll3_mean",
        "player_goal_line_carries_roll5_mean",
        "player_goal_line_carries_ewm5",
        "player_goal_line_carries_season_to_date",
        "player_goal_line_carries_career_prior",
        *COMMON_HISTORY_FEATURES,
        *DERIVED_FEATURES,
    ],
    "receiving_yards_per_target": [
        "player_yards_per_target_lag1",
        "player_yards_per_target_roll3_mean",
        "player_yards_per_target_roll5_mean",
        "player_yards_per_target_ewm5",
        "player_yards_per_target_season_to_date",
        "player_yards_per_target_career_prior",
        "player_targets_career_prior",
        *COMMON_HISTORY_FEATURES,
        *DERIVED_FEATURES,
    ],
    "receiving_td_per_red_zone_target": [
        "player_red_zone_targets_lag1",
        "player_red_zone_targets_roll3_mean",
        "player_red_zone_targets_roll5_mean",
        "player_red_zone_targets_ewm5",
        "player_red_zone_targets_season_to_date",
        "player_red_zone_targets_career_prior",
        *COMMON_HISTORY_FEATURES,
        *DERIVED_FEATURES,
    ],
    "field_goal_conversion": [
        "player_field_goal_attempts_lag1",
        "player_field_goal_attempts_roll3_mean",
        "player_field_goal_attempts_roll5_mean",
        "player_field_goal_attempts_season_to_date",
        "player_field_goal_attempts_career_prior",
        "player_field_goals_made_lag1",
        "player_field_goals_made_roll3_mean",
        "player_field_goals_made_roll5_mean",
        "player_field_goals_made_season_to_date",
        "player_field_goals_made_career_prior",
        *COMMON_HISTORY_FEATURES,
        *DERIVED_FEATURES,
    ],
    "extra_point_conversion": [
        "player_extra_point_attempts_lag1",
        "player_extra_point_attempts_roll3_mean",
        "player_extra_point_attempts_roll5_mean",
        "player_extra_point_attempts_season_to_date",
        "player_extra_point_attempts_career_prior",
        "player_extra_points_made_lag1",
        "player_extra_points_made_roll3_mean",
        "player_extra_points_made_roll5_mean",
        "player_extra_points_made_season_to_date",
        "player_extra_points_made_career_prior",
        *COMMON_HISTORY_FEATURES,
        *DERIVED_FEATURES,
    ],
    "tackle_rate_per_defensive_play": [
        "player_tackle_rate_per_def_play_lag1",
        "player_tackle_rate_per_def_play_roll3_mean",
        "player_tackle_rate_per_def_play_roll5_mean",
        "player_tackle_rate_per_def_play_ewm5",
        "player_tackle_rate_per_def_play_season_to_date",
        "player_tackle_rate_per_def_play_career_prior",
        "player_defense_participation_lag1",
        "player_defense_participation_roll3_mean",
        "player_defense_participation_career_prior",
        *COMMON_HISTORY_FEATURES,
        *DERIVED_FEATURES,
    ],
    "sack_rate_per_defensive_play": [
        "player_sack_rate_per_def_play_lag1",
        "player_sack_rate_per_def_play_roll3_mean",
        "player_sack_rate_per_def_play_roll5_mean",
        "player_sack_rate_per_def_play_ewm5",
        "player_sack_rate_per_def_play_season_to_date",
        "player_sack_rate_per_def_play_career_prior",
        "player_defense_participation_lag1",
        "player_defense_participation_roll3_mean",
        "player_defense_participation_career_prior",
        *COMMON_HISTORY_FEATURES,
        *DERIVED_FEATURES,
    ],
}

ELIGIBILITY_RULE = {
    "passing_yards_per_attempt": "passing_yards",
    "passing_td_rate": "passing_tds",
    "rushing_yards_per_carry": "rushing_yards",
    "rushing_td_per_goal_line_carry": "rushing_tds",
    "receiving_yards_per_target": "receiving_yards",
    "receiving_td_per_red_zone_target": "receiving_tds",
    "field_goal_conversion": "kicking_points",
    "extra_point_conversion": "kicking_points",
    "tackle_rate_per_defensive_play": "tackles",
    "sack_rate_per_defensive_play": "sacks",
}

# Opportunity-share features are not permitted in efficiency models.
PROHIBITED_SHARE_TOKENS = (
    "carry_share",
    "target_share",
    "air_yards_share",
    "red_zone_target_share",
    "red_zone_carry_share",
    "goal_line_carry_share",
)


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


def numeric(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype("float64")
    )


def safe_rate(
    numerator: pd.Series,
    exposure: pd.Series,
) -> pd.Series:
    num = numeric(numerator)
    exp = numeric(exposure)

    result = pd.Series(
        np.nan,
        index=num.index,
        dtype="float64",
    )

    valid = (
        num.notna()
        & exp.notna()
        & exp.gt(0.0)
    )

    result.loc[valid] = (
        num.loc[valid]
        / exp.loc[valid]
    )

    return result.replace([np.inf, -np.inf], np.nan)


def normalize_position_group(
    position: pd.Series,
    position_group: pd.Series,
) -> pd.Series:
    group = (
        position_group
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    pos = (
        position
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    defensive_map = {
        "DL": "DL",
        "DE": "DL",
        "DT": "DL",
        "NT": "DL",
        "LDT": "DL",
        "EDGE": "DL",
        "LB": "LB",
        "ILB": "LB",
        "OLB": "LB",
        "MLB": "LB",
        "DB": "DB",
        "CB": "DB",
        "S": "DB",
        "SAF": "DB",
        "FS": "DB",
        "SS": "DB",
        "NB": "DB",
    }

    fallback = pos.map(defensive_map).fillna(pos)
    group = group.where(group.ne(""), fallback)

    return group


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


def save_model_atomic(
    model: lgb.Booster,
    path: Path,
    num_iteration: int,
) -> None:
    destination = path.resolve()
    root = common.prop_root().resolve()

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_backtest_policy(folds: pd.DataFrame) -> None:
    common.require_columns(
        folds,
        [
            "train_end_season",
            "validation_start_season",
            "test_flag",
        ],
        "backtest folds",
    )

    dev = folds.loc[
        folds["test_flag"].eq(0)
        & folds["validation_start_season"].eq(
            DEVELOPMENT_VALIDATION_SEASON
        )
    ]

    if len(dev) != 1:
        raise ValueError(
            "Expected exactly one 2024 development fold."
        )

    if int(dev.iloc[0]["train_end_season"]) != MODEL_SELECTION_TRAIN_END:
        raise ValueError(
            "Issue 23 model selection must train through 2023."
        )

    test = folds.loc[folds["test_flag"].eq(1)]

    if len(test) != 1:
        raise ValueError("Expected exactly one untouched test fold.")

    if int(test.iloc[0]["validation_start_season"]) != UNTOUCHED_TEST_SEASON:
        raise ValueError("Untouched test season must be 2025.")

    if int(test.iloc[0]["train_end_season"]) != FINAL_TRAIN_END:
        raise ValueError(
            "Untouched-test training cutoff must be 2024."
        )


def validate_feature_contract(
    config: dict[str, Any],
    canonical_manifest: dict[str, Any],
) -> None:
    canonical_features = set(canonical_manifest["feature_columns"])
    canonical_numeric = set(canonical_manifest["numeric_features"])

    if set(FEATURES) != set(MODELS):
        raise ValueError("Efficiency feature map does not cover all models.")

    for model_name in MODELS:
        features = FEATURES[model_name]

        if len(features) != len(set(features)):
            raise ValueError(
                f"{model_name}: duplicate feature name."
            )

        canonical_only = [
            feature
            for feature in features
            if feature not in DERIVED_FEATURES
        ]

        missing = sorted(
            set(canonical_only)
            - canonical_features
        )
        if missing:
            raise ValueError(
                f"{model_name}: missing canonical feature(s): {missing}"
            )

        nonnumeric = sorted(
            set(canonical_only)
            - canonical_numeric
        )
        if nonnumeric:
            raise ValueError(
                f"{model_name}: nonnumeric canonical feature(s): {nonnumeric}"
            )

        prohibited = [
            feature
            for feature in features
            if feature.startswith("target_")
            or feature.startswith("audit_")
            or feature == "played_game_flag"
        ]
        if prohibited:
            raise ValueError(
                f"{model_name}: prohibited feature(s): {prohibited}"
            )

        share_features = [
            feature
            for feature in features
            if any(
                token in feature
                for token in PROHIBITED_SHARE_TOKENS
            )
        ]
        if share_features:
            raise ValueError(
                f"{model_name}: opportunity-share feature(s) are "
                f"forbidden in efficiency models: {share_features}"
            )

        common.reject_forbidden_feature_columns(
            features,
            config,
        )

    if not (
        SHRINKAGE_EXPOSURE["passing_td_rate"]
        > SHRINKAGE_EXPOSURE["passing_yards_per_attempt"]
    ):
        raise ValueError("Passing TD shrinkage is not stronger than YPA.")

    if not (
        SHRINKAGE_EXPOSURE["rushing_td_per_goal_line_carry"]
        > SHRINKAGE_EXPOSURE["rushing_yards_per_carry"]
    ):
        raise ValueError(
            "Rushing TD shrinkage must be stronger than rushing-yardage shrinkage."
        )

    if not (
        SHRINKAGE_EXPOSURE["receiving_td_per_red_zone_target"]
        > SHRINKAGE_EXPOSURE["receiving_yards_per_target"]
    ):
        raise ValueError(
            "Receiving TD shrinkage must be stronger than receiving-yardage shrinkage."
        )

    if not (
        SHRINKAGE_EXPOSURE["sack_rate_per_defensive_play"]
        > SHRINKAGE_EXPOSURE["tackle_rate_per_defensive_play"]
    ):
        raise ValueError(
            "Sack shrinkage must be stronger than tackle-rate shrinkage."
        )


def build_exact_conditional_td_labels(
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Rebuild exact same-week conditional TD labels from local PBP.

    Goal line is inside the 5, matching build_player_opportunity.py.
    Red zone is yardline_100 <= 20.
    """
    rows: list[pd.DataFrame] = []
    pattern = config["paths"]["pbp_pattern"]
    root = common.repo_root()

    for season in range(
        int(config["seasons"]["rich_feature_start"]),
        FINAL_TRAIN_END + 1,
    ):
        path = root / pattern.format(season=season)

        if not path.is_file():
            raise FileNotFoundError(
                f"Required rich-feature PBP is missing: {path}"
            )

        usecols = [
            "season_type",
            "week",
            "game_id",
            "yardline_100",
            "rush_attempt",
            "pass_attempt",
            "rusher_player_id",
            "receiver_player_id",
            "rush_touchdown",
            "pass_touchdown",
        ]

        pbp = pd.read_csv(
            path,
            usecols=usecols,
            low_memory=False,
        )

        pbp = pbp.loc[
            pbp["season_type"]
            .astype(str)
            .str.upper()
            .eq("REG")
        ].copy()

        pbp["season"] = season
        pbp["week"] = pd.to_numeric(
            pbp["week"],
            errors="raise",
        ).astype(int)

        pbp["game_id"] = (
            pbp["game_id"]
            .astype(str)
            .str.strip()
        )

        pbp["yardline_100"] = numeric(
            pbp["yardline_100"]
        )
        pbp["rush_attempt"] = numeric(
            pbp["rush_attempt"]
        ).fillna(0.0)
        pbp["pass_attempt"] = numeric(
            pbp["pass_attempt"]
        ).fillna(0.0)
        pbp["rush_touchdown"] = numeric(
            pbp["rush_touchdown"]
        ).fillna(0.0)
        pbp["pass_touchdown"] = numeric(
            pbp["pass_touchdown"]
        ).fillna(0.0)

        pbp["rusher_player_id"] = (
            pbp["rusher_player_id"]
            .map(common.normalize_player_id)
        )
        pbp["receiver_player_id"] = (
            pbp["receiver_player_id"]
            .map(common.normalize_player_id)
        )

        goal_line = (
            pbp["yardline_100"].notna()
            & pbp["yardline_100"].le(5.0)
            & pbp["rush_attempt"].eq(1.0)
            & pbp["rusher_player_id"].ne("")
        )

        red_zone_target = (
            pbp["yardline_100"].notna()
            & pbp["yardline_100"].le(20.0)
            & pbp["pass_attempt"].eq(1.0)
            & pbp["receiver_player_id"].ne("")
        )

        gl = pbp.loc[
            goal_line,
            [
                "season",
                "week",
                "game_id",
                "rusher_player_id",
                "rush_touchdown",
            ],
        ].rename(
            columns={
                "rusher_player_id": "player_id",
            }
        )

        gl["_goal_line_carries_pbp"] = 1.0
        gl["_goal_line_rush_tds"] = (
            gl["rush_touchdown"].eq(1.0).astype(float)
        )

        gl = (
            gl.groupby(
                GRAIN,
                as_index=False,
                dropna=False,
            )
            .agg(
                _goal_line_carries_pbp=(
                    "_goal_line_carries_pbp",
                    "sum",
                ),
                _goal_line_rush_tds=(
                    "_goal_line_rush_tds",
                    "sum",
                ),
            )
        )

        rz = pbp.loc[
            red_zone_target,
            [
                "season",
                "week",
                "game_id",
                "receiver_player_id",
                "pass_touchdown",
            ],
        ].rename(
            columns={
                "receiver_player_id": "player_id",
            }
        )

        rz["_red_zone_targets_pbp"] = 1.0
        rz["_red_zone_receiving_tds"] = (
            rz["pass_touchdown"].eq(1.0).astype(float)
        )

        rz = (
            rz.groupby(
                GRAIN,
                as_index=False,
                dropna=False,
            )
            .agg(
                _red_zone_targets_pbp=(
                    "_red_zone_targets_pbp",
                    "sum",
                ),
                _red_zone_receiving_tds=(
                    "_red_zone_receiving_tds",
                    "sum",
                ),
            )
        )

        merged = gl.merge(
            rz,
            on=GRAIN,
            how="outer",
            validate="one_to_one",
        )

        rows.append(merged)

    if not rows:
        raise ValueError("No conditional TD PBP rows were built.")

    output = pd.concat(
        rows,
        ignore_index=True,
    )

    for column in [
        "_goal_line_carries_pbp",
        "_goal_line_rush_tds",
        "_red_zone_targets_pbp",
        "_red_zone_receiving_tds",
    ]:
        output[column] = numeric(
            output[column]
        ).fillna(0.0)

    common.ensure_unique(
        output,
        GRAIN,
        "conditional TD PBP labels",
    )

    return output


def prepare_label_base(
    config: dict[str, Any],
    features: pd.DataFrame,
) -> pd.DataFrame:
    player_path = config["paths"]["player_opportunity"]

    player = common.read_parquet_required(
        player_path,
        [
            "season",
            "week",
            "game_id",
            "player_id",
            "team",
            "position",
            "position_group",
            "pass_attempts",
            "passing_yards",
            "passing_tds",
            "carries",
            "rushing_yards",
            "rushing_tds",
            "goal_line_carries",
            "targets",
            "receiving_yards",
            "receiving_tds",
            "red_zone_targets",
            "field_goal_attempts",
            "field_goals_made",
            "extra_point_attempts",
            "extra_points_made",
            "tackle_rate_per_def_play",
            "sack_rate_per_def_play",
            "defense_snap_pct",
            "defense_participation",
        ],
    ).copy()

    player = player.loc[
        pd.to_numeric(
            player["season"],
            errors="raise",
        ).le(FINAL_TRAIN_END)
    ].copy()

    common.ensure_unique(
        player,
        GRAIN,
        "player opportunity efficiency labels",
    )

    feature_keys = features[
        [
            *GRAIN,
            "kickoff_timestamp",
            "position",
            "position_group",
        ]
    ].copy()

    labels = feature_keys.merge(
        player,
        on=GRAIN,
        how="left",
        validate="one_to_one",
        suffixes=("", "_opp"),
    )

    labels["position"] = (
        labels["position"]
        .where(
            labels["position"].notna()
            & labels["position"].astype(str).str.strip().ne(""),
            labels["position_opp"],
        )
    )

    labels["position_group"] = (
        labels["position_group"]
        .where(
            labels["position_group"].notna()
            & labels["position_group"].astype(str).str.strip().ne(""),
            labels["position_group_opp"],
        )
    )

    labels["_prior_position_group"] = (
        normalize_position_group(
            labels["position"],
            labels["position_group"],
        )
    )

    labels["kickoff_timestamp"] = pd.to_datetime(
        labels["kickoff_timestamp"],
        errors="raise",
        utc=True,
    )

    exact_td = build_exact_conditional_td_labels(
        config
    )

    labels = labels.merge(
        exact_td,
        on=GRAIN,
        how="left",
        validate="one_to_one",
    )

    for column in [
        "_goal_line_carries_pbp",
        "_goal_line_rush_tds",
        "_red_zone_targets_pbp",
        "_red_zone_receiving_tds",
    ]:
        labels[column] = numeric(labels[column]).fillna(0.0)

    rich = labels["season"].ge(
        int(config["seasons"]["rich_feature_start"])
    )

    gl_opp = numeric(labels["goal_line_carries"])
    rz_opp = numeric(labels["red_zone_targets"])

    gl_mismatch = (
        rich
        & gl_opp.notna()
        & ~np.isclose(
            gl_opp.to_numpy(dtype=float),
            labels["_goal_line_carries_pbp"].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
    )

    if gl_mismatch.any():
        sample = labels.loc[
            gl_mismatch,
            [
                *GRAIN,
                "goal_line_carries",
                "_goal_line_carries_pbp",
            ],
        ].head(20)
        raise ValueError(
            "PBP goal-line carry reconstruction does not match "
            f"player opportunity. Sample={sample.to_dict(orient='records')}"
        )

    rz_mismatch = (
        rich
        & rz_opp.notna()
        & ~np.isclose(
            rz_opp.to_numpy(dtype=float),
            labels["_red_zone_targets_pbp"].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
    )

    if rz_mismatch.any():
        sample = labels.loc[
            rz_mismatch,
            [
                *GRAIN,
                "red_zone_targets",
                "_red_zone_targets_pbp",
            ],
        ].head(20)
        raise ValueError(
            "PBP red-zone target reconstruction does not match "
            f"player opportunity. Sample={sample.to_dict(orient='records')}"
        )

    return labels


def build_component_label(
    labels: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    frame = labels.copy()

    if model_name == "passing_yards_per_attempt":
        numerator = numeric(frame["passing_yards"])
        exposure = numeric(frame["pass_attempts"])

    elif model_name == "passing_td_rate":
        numerator = numeric(frame["passing_tds"])
        exposure = numeric(frame["pass_attempts"])

    elif model_name == "rushing_yards_per_carry":
        numerator = numeric(frame["rushing_yards"])
        exposure = numeric(frame["carries"])

    elif model_name == "rushing_td_per_goal_line_carry":
        numerator = numeric(
            frame["_goal_line_rush_tds"]
        )
        exposure = numeric(
            frame["_goal_line_carries_pbp"]
        )

    elif model_name == "receiving_yards_per_target":
        numerator = numeric(frame["receiving_yards"])
        exposure = numeric(frame["targets"])

    elif model_name == "receiving_td_per_red_zone_target":
        numerator = numeric(
            frame["_red_zone_receiving_tds"]
        )
        exposure = numeric(
            frame["_red_zone_targets_pbp"]
        )

    elif model_name == "field_goal_conversion":
        numerator = numeric(frame["field_goals_made"])
        exposure = numeric(frame["field_goal_attempts"])

    elif model_name == "extra_point_conversion":
        numerator = numeric(frame["extra_points_made"])
        exposure = numeric(frame["extra_point_attempts"])

    elif model_name == "tackle_rate_per_defensive_play":
        rate = numeric(frame["tackle_rate_per_def_play"])
        exposure = numeric(
            frame["defense_snap_pct"]
        )

        fallback = numeric(
            frame["defense_participation"]
        )
        exposure = exposure.where(
            exposure.notna(),
            fallback,
        )

        # Percent/share is the best same-week available player-level exposure
        # proxy retained in the opportunity table. Scale it to 100 "defensive
        # play share units" solely for empirical-Bayes weighting. The realized
        # label itself remains the exact canonical tackle_rate_per_def_play.
        exposure = exposure * 100.0
        numerator = rate * exposure

    elif model_name == "sack_rate_per_defensive_play":
        rate = numeric(frame["sack_rate_per_def_play"])
        exposure = numeric(
            frame["defense_snap_pct"]
        )

        fallback = numeric(
            frame["defense_participation"]
        )
        exposure = exposure.where(
            exposure.notna(),
            fallback,
        )

        exposure = exposure * 100.0
        numerator = rate * exposure

    else:
        raise KeyError(model_name)

    rate = safe_rate(
        numerator,
        exposure,
    )

    if model_name in {
        "tackle_rate_per_defensive_play",
        "sack_rate_per_defensive_play",
    }:
        source_rate = numeric(
            frame[
                {
                    "tackle_rate_per_defensive_play":
                        "tackle_rate_per_def_play",
                    "sack_rate_per_defensive_play":
                        "sack_rate_per_def_play",
                }[model_name]
            ]
        )

        rate = source_rate.where(
            source_rate.notna(),
            rate,
        )

    if model_name in RATE_MODELS:
        rate = rate.clip(lower=0.0, upper=1.0)

    frame["_numerator"] = numerator
    frame["_exposure"] = exposure
    frame["_label"] = rate

    frame = frame.loc[
        frame["_label"].notna()
        & frame["_exposure"].notna()
        & frame["_exposure"].gt(0.0)
    ].copy()

    return frame


def add_strict_prior_features(
    frame: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """
    Add player -> position -> league pregame empirical-Bayes features.

    Every cumulative value excludes all rows at the target kickoff.
    Player history is keyed only by player_id, so team changes do not reset it.
    """
    work = frame.copy()

    work = work.sort_values(
        [
            "kickoff_timestamp",
            "season",
            "week",
            "game_id",
            "player_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    # Player prior. A player cannot play two NFL games at the exact same kickoff;
    # nevertheless use kickoff-group aggregates so strict-before semantics are
    # explicit rather than relying on row order.
    player_time = (
        work.groupby(
            ["player_id", "kickoff_timestamp"],
            as_index=False,
            sort=True,
            dropna=False,
        )
        .agg(
            _n=("_numerator", "sum"),
            _e=("_exposure", "sum"),
        )
        .sort_values(
            ["player_id", "kickoff_timestamp"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    player_time["_player_num_prior"] = (
        player_time.groupby(
            "player_id",
            sort=False,
        )["_n"]
        .cumsum()
        - player_time["_n"]
    )

    player_time["_player_exp_prior"] = (
        player_time.groupby(
            "player_id",
            sort=False,
        )["_e"]
        .cumsum()
        - player_time["_e"]
    )

    work = work.merge(
        player_time[
            [
                "player_id",
                "kickoff_timestamp",
                "_player_num_prior",
                "_player_exp_prior",
            ]
        ],
        on=["player_id", "kickoff_timestamp"],
        how="left",
        validate="many_to_one",
    )

    # Position prior, excluding every row at the same kickoff.
    pos_time = (
        work.groupby(
            [
                "_prior_position_group",
                "kickoff_timestamp",
            ],
            as_index=False,
            sort=True,
            dropna=False,
        )
        .agg(
            _n=("_numerator", "sum"),
            _e=("_exposure", "sum"),
        )
        .sort_values(
            [
                "_prior_position_group",
                "kickoff_timestamp",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    pos_time["_position_num_prior"] = (
        pos_time.groupby(
            "_prior_position_group",
            sort=False,
        )["_n"]
        .cumsum()
        - pos_time["_n"]
    )

    pos_time["_position_exp_prior"] = (
        pos_time.groupby(
            "_prior_position_group",
            sort=False,
        )["_e"]
        .cumsum()
        - pos_time["_e"]
    )

    work = work.merge(
        pos_time[
            [
                "_prior_position_group",
                "kickoff_timestamp",
                "_position_num_prior",
                "_position_exp_prior",
            ]
        ],
        on=[
            "_prior_position_group",
            "kickoff_timestamp",
        ],
        how="left",
        validate="many_to_one",
    )

    # League prior.
    league_time = (
        work.groupby(
            "kickoff_timestamp",
            as_index=False,
            sort=True,
            dropna=False,
        )
        .agg(
            _n=("_numerator", "sum"),
            _e=("_exposure", "sum"),
        )
        .sort_values(
            "kickoff_timestamp",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    league_time["_league_num_prior"] = (
        league_time["_n"].cumsum()
        - league_time["_n"]
    )
    league_time["_league_exp_prior"] = (
        league_time["_e"].cumsum()
        - league_time["_e"]
    )

    work = work.merge(
        league_time[
            [
                "kickoff_timestamp",
                "_league_num_prior",
                "_league_exp_prior",
            ]
        ],
        on="kickoff_timestamp",
        how="left",
        validate="many_to_one",
    )

    work["eff_player_prior_rate"] = safe_rate(
        work["_player_num_prior"],
        work["_player_exp_prior"],
    )

    work["eff_position_prior_rate"] = safe_rate(
        work["_position_num_prior"],
        work["_position_exp_prior"],
    )

    work["eff_league_prior_rate"] = safe_rate(
        work["_league_num_prior"],
        work["_league_exp_prior"],
    )

    if model_name in RATE_MODELS:
        for column in [
            "eff_player_prior_rate",
            "eff_position_prior_rate",
            "eff_league_prior_rate",
        ]:
            work[column] = numeric(
                work[column]
            ).clip(lower=0.0, upper=1.0)

    prior_rate = (
        work["eff_position_prior_rate"]
        .where(
            work["eff_position_prior_rate"].notna(),
            work["eff_league_prior_rate"],
        )
    )

    prior_weight = float(
        SHRINKAGE_EXPOSURE[model_name]
    )

    player_exp = numeric(
        work["_player_exp_prior"]
    ).fillna(0.0)

    player_num = numeric(
        work["_player_num_prior"]
    ).fillna(0.0)

    shrunk = pd.Series(
        np.nan,
        index=work.index,
        dtype="float64",
    )

    prior_ok = prior_rate.notna()
    player_ok = player_exp.gt(0.0)

    both = prior_ok & player_ok
    shrunk.loc[both] = (
        player_num.loc[both]
        + prior_weight * prior_rate.loc[both]
    ) / (
        player_exp.loc[both]
        + prior_weight
    )

    rookie = prior_ok & ~player_ok
    shrunk.loc[rookie] = prior_rate.loc[rookie]

    no_prior_player = ~prior_ok & player_ok
    shrunk.loc[no_prior_player] = (
        work.loc[
            no_prior_player,
            "eff_player_prior_rate",
        ]
    )

    if model_name in RATE_MODELS:
        shrunk = shrunk.clip(
            lower=0.0,
            upper=1.0,
        )

    work["eff_shrunk_rate"] = shrunk
    work["eff_player_prior_exposure"] = player_exp
    work["eff_prior_weight"] = prior_weight

    work["eff_rookie_position_prior_flag"] = (
        player_exp.eq(0.0)
        & work["eff_position_prior_rate"].notna()
    ).astype(float)

    return work


def apply_eligibility(
    frame: pd.DataFrame,
    model_name: str,
    eligibility: dict[str, Any],
) -> pd.DataFrame:
    rule = ELIGIBILITY_RULE[model_name]
    positions = {
        str(value).strip().upper()
        for value in eligibility[rule]["eligible_positions"]
    }

    pos = (
        frame["position"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return frame.loc[
        pos.isin(positions)
    ].copy()


def feature_matrix(
    frame: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    columns = FEATURES[model_name]

    return pd.DataFrame(
        {
            column: numeric(frame[column])
            for column in columns
        },
        index=frame.index,
    )


def transform_prediction(
    values: np.ndarray,
    model_name: str,
) -> np.ndarray:
    pred = np.asarray(
        values,
        dtype="float64",
    )

    if model_name in RATE_MODELS:
        return np.clip(
            pred,
            0.0,
            1.0,
        )

    return pred


def rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    np.asarray(actual, dtype=float)
                    - np.asarray(predicted, dtype=float)
                )
            )
        )
    )


def mae(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    return float(
        np.mean(
            np.abs(
                np.asarray(actual, dtype=float)
                - np.asarray(predicted, dtype=float)
            )
        )
    )


def r2(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float | None:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)

    denominator = float(
        np.sum(
            np.square(
                y - y.mean()
            )
        )
    )

    if denominator <= 0.0:
        return None

    value = 1.0 - float(
        np.sum(np.square(y - p))
        / denominator
    )

    return (
        value
        if math.isfinite(value)
        else None
    )


def final_prior_snapshot(
    frame: pd.DataFrame,
    model_name: str,
) -> dict[str, Any]:
    source = frame.loc[
        frame["season"].le(FINAL_TRAIN_END)
    ].copy()

    league_num = float(
        numeric(source["_numerator"]).sum()
    )
    league_exp = float(
        numeric(source["_exposure"]).sum()
    )

    league_rate = (
        league_num / league_exp
        if league_exp > 0.0
        else None
    )

    positions: dict[str, Any] = {}

    grouped = source.groupby(
        "_prior_position_group",
        dropna=False,
        sort=True,
    )

    for position_group, group in grouped:
        numerator = float(
            numeric(group["_numerator"]).sum()
        )
        exposure = float(
            numeric(group["_exposure"]).sum()
        )

        positions[str(position_group)] = {
            "numerator": numerator,
            "exposure": exposure,
            "rate": (
                numerator / exposure
                if exposure > 0.0
                else None
            ),
        }

    if model_name in RATE_MODELS:
        if league_rate is not None:
            league_rate = float(
                np.clip(
                    league_rate,
                    0.0,
                    1.0,
                )
            )

        for payload in positions.values():
            if payload["rate"] is not None:
                payload["rate"] = float(
                    np.clip(
                        payload["rate"],
                        0.0,
                        1.0,
                    )
                )

    return {
        "through_season": FINAL_TRAIN_END,
        "league": {
            "numerator": league_num,
            "exposure": league_exp,
            "rate": league_rate,
        },
        "positions": positions,
    }


def train_model(
    model_name: str,
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    selection_train = frame.loc[
        frame["season"].le(
            MODEL_SELECTION_TRAIN_END
        )
    ].copy()

    validation = frame.loc[
        frame["season"].eq(
            DEVELOPMENT_VALIDATION_SEASON
        )
    ].copy()

    final_train = frame.loc[
        frame["season"].le(
            FINAL_TRAIN_END
        )
    ].copy()

    if selection_train.empty:
        raise ValueError(
            f"{model_name}: empty selection training set."
        )

    if validation.empty:
        raise ValueError(
            f"{model_name}: empty 2024 validation set."
        )

    if final_train.empty:
        raise ValueError(
            f"{model_name}: empty final training set."
        )

    X_train = feature_matrix(
        selection_train,
        model_name,
    )
    y_train = numeric(
        selection_train["_label"]
    )

    X_valid = feature_matrix(
        validation,
        model_name,
    )
    y_valid = numeric(
        validation["_label"]
    )

    X_final = feature_matrix(
        final_train,
        model_name,
    )
    y_final = numeric(
        final_train["_label"]
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

    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=FEATURES[model_name],
        free_raw_data=False,
    )

    valid_set = lgb.Dataset(
        X_valid,
        label=y_valid,
        feature_name=FEATURES[model_name],
        reference=train_set,
        free_raw_data=False,
    )

    selected = lgb.train(
        params,
        train_set,
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
        selected.best_iteration
        if selected.best_iteration
        else 2000
    )

    valid_pred = transform_prediction(
        selected.predict(
            X_valid,
            num_iteration=best_iteration,
        ),
        model_name,
    )

    valid_actual = y_valid.to_numpy(
        dtype="float64"
    )

    metrics = {
        "rmse": rmse(
            valid_actual,
            valid_pred,
        ),
        "mae": mae(
            valid_actual,
            valid_pred,
        ),
        "r2": r2(
            valid_actual,
            valid_pred,
        ),
    }

    final_set = lgb.Dataset(
        X_final,
        label=y_final,
        feature_name=FEATURES[model_name],
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
        / "efficiency"
        / model_name
    )

    model_path = model_root / "model.txt"
    manifest_path = model_root / "feature_manifest.json"
    metadata_path = model_root / "metadata.json"

    model_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_model_atomic(
        final_model,
        model_path,
        best_iteration,
    )

    canonical_features = [
        feature
        for feature in FEATURES[model_name]
        if feature not in DERIVED_FEATURES
    ]

    manifest = {
        "model": model_name,
        "model_type": "lightgbm_regression",
        "objective": "regression",
        "numeric_features": FEATURES[model_name],
        "categorical_features": [],
        "canonical_features": canonical_features,
        "derived_features": DERIVED_FEATURES,
        "feature_count": len(FEATURES[model_name]),
        "automatic_all_numeric_selection": False,
        "prediction_transform": (
            "clip_0_1"
            if model_name in RATE_MODELS
            else "identity_signed"
        ),
        "shrinkage": {
            "hierarchy": [
                "player",
                "position",
                "league",
            ],
            "prior_exposure": float(
                SHRINKAGE_EXPOSURE[model_name]
            ),
            "strictly_prior_kickoff": True,
            "rookie_fallback": (
                "position_prior_then_league_prior"
            ),
            "player_history_key": "player_id",
            "team_resets_player_efficiency": False,
        },
        "trade_policy": {
            "preserve_player_efficiency_history": True,
            "old_team_opportunity_share_features_allowed": False,
        },
        "forbidden_features": config["forbidden_features"],
    }

    write_json_atomic(
        manifest_path,
        manifest,
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
            FEATURES[model_name],
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
        "model": model_name,
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
            "final_train_end_season":
                FINAL_TRAIN_END,
            "untouched_test_season":
                UNTOUCHED_TEST_SEASON,
            "untouched_test_used_for_selection": False,
            "untouched_test_used_for_metrics": False,
            "untouched_test_used_for_fit": False,
            "untouched_test_used_for_priors": False,
        },
        "rows": {
            "model_selection_train": int(
                len(selection_train)
            ),
            "validation_2024": int(
                len(validation)
            ),
            "final_train_through_2024": int(
                len(final_train)
            ),
        },
        "label": {
            "minimum": float(
                y_final.min()
            ),
            "maximum": float(
                y_final.max()
            ),
            "mean": float(
                y_final.mean()
            ),
            "first_training_season": int(
                final_train["season"].min()
            ),
            "last_training_season": int(
                final_train["season"].max()
            ),
        },
        "shrinkage": {
            "hierarchy": "player_to_position_to_league",
            "prior_exposure": float(
                SHRINKAGE_EXPOSURE[model_name]
            ),
            "strong_shrinkage": bool(
                model_name in TD_MODELS
                or model_name
                == "sack_rate_per_defensive_play"
            ),
            "strictly_prior_kickoff": True,
            "rookies_use_position_prior": True,
            "player_history_key": "player_id",
            "team_change_resets_efficiency": False,
            "final_priors_through_2024":
                final_prior_snapshot(
                    final_train,
                    model_name,
                ),
        },
        "trade_policy": {
            "preserve_player_efficiency": True,
            "old_team_opportunity_share": False,
            "share_features_in_model": False,
        },
        "conditional_td_label_policy": {
            "exact_pbp_touchdown_flags": bool(
                model_name in {
                    "rushing_td_per_goal_line_carry",
                    "receiving_td_per_red_zone_target",
                }
            ),
            "goal_line_definition": (
                "yardline_100 <= 5"
                if model_name
                == "rushing_td_per_goal_line_carry"
                else None
            ),
            "red_zone_definition": (
                "yardline_100 <= 20"
                if model_name
                == "receiving_td_per_red_zone_target"
                else None
            ),
        },
        "best_iteration_selected_on_2024":
            best_iteration,
        "validation_2024_metrics": metrics,
        "params": params,
        "feature_count": len(
            FEATURES[model_name]
        ),
        "feature_importance": importance,
        "model_sha256": sha256_file(
            model_path
        ),
        "feature_manifest_sha256":
            sha256_file(manifest_path),
    }

    write_json_atomic(
        metadata_path,
        metadata,
    )

    return {
        "model": model_name,
        "features": len(
            FEATURES[model_name]
        ),
        "model_selection_train_rows": int(
            len(selection_train)
        ),
        "validation_2024_rows": int(
            len(validation)
        ),
        "final_train_rows": int(
            len(final_train)
        ),
        "best_iteration": best_iteration,
        "validation_rmse": metrics["rmse"],
        "validation_mae": metrics["mae"],
        "model_path": str(
            model_path.relative_to(
                common.repo_root()
            )
        ).replace("\\", "/"),
        "feature_manifest_path": str(
            manifest_path.relative_to(
                common.repo_root()
            )
        ).replace("\\", "/"),
        "metadata_path": str(
            metadata_path.relative_to(
                common.repo_root()
            )
        ).replace("\\", "/"),
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

    verify_backtest_policy(
        folds
    )

    validate_feature_contract(
        config,
        canonical_manifest,
    )

    canonical_features = sorted(
        {
            feature
            for model_name in MODELS
            for feature in FEATURES[model_name]
            if feature not in DERIVED_FEATURES
        }
    )

    required_columns = [
        *GRAIN,
        "kickoff_timestamp",
        "position",
        "position_group",
        *canonical_features,
    ]

    feature_path = (
        root
        / config["paths"]["historical_features"]
    )

    features = pd.read_parquet(
        feature_path,
        columns=list(
            dict.fromkeys(
                required_columns
            )
        ),
    )

    features["season"] = pd.to_numeric(
        features["season"],
        errors="raise",
    ).astype(int)

    features["week"] = pd.to_numeric(
        features["week"],
        errors="raise",
    ).astype(int)

    features = features.loc[
        features["season"].le(
            FINAL_TRAIN_END
        )
    ].copy()

    common.ensure_unique(
        features,
        GRAIN,
        "canonical historical features",
    )

    label_base = prepare_label_base(
        config,
        features,
    )

    results: list[dict[str, Any]] = []

    for model_name in MODELS:
        frame = build_component_label(
            label_base,
            model_name,
        )

        frame = apply_eligibility(
            frame,
            model_name,
            eligibility,
        )

        frame = add_strict_prior_features(
            frame,
            model_name,
        )

        feature_join = features[
            [
                *GRAIN,
                *[
                    feature
                    for feature
                    in FEATURES[model_name]
                    if feature not in DERIVED_FEATURES
                ],
            ]
        ].copy()

        frame = frame.merge(
            feature_join,
            on=GRAIN,
            how="left",
            validate="one_to_one",
        )

        if frame["season"].max() > FINAL_TRAIN_END:
            raise ValueError(
                f"{model_name}: data after 2024 entered training."
            )

        result = train_model(
            model_name,
            frame,
            config,
        )

        results.append(result)

        print(
            json.dumps(
                {
                    "model": model_name,
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
                    "shrinkage_exposure":
                        SHRINKAGE_EXPOSURE[model_name],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    payload = {
        "status": "passed",
        "models": len(results),
        "model_names": MODELS,
        "model_type": "lightgbm_regression",
        "random_split_used": False,
        "model_selection_train_end_season":
            MODEL_SELECTION_TRAIN_END,
        "development_validation_season":
            DEVELOPMENT_VALIDATION_SEASON,
        "final_train_end_season":
            FINAL_TRAIN_END,
        "untouched_test_season":
            UNTOUCHED_TEST_SEASON,
        "untouched_test_used": False,
        "shrinkage_hierarchy":
            "player_to_position_to_league",
        "rookies_use_position_priors": True,
        "stronger_shrinkage_for_tds_and_sacks": True,
        "trades_preserve_player_efficiency": True,
        "old_team_opportunity_share_features_used": False,
        "exact_pbp_conditional_td_labels": True,
        "results": results,
    }

    common.log_run(
        "train_efficiency_models.py",
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
