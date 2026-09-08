#!/usr/bin/env python3
"""
Select the NFL Prop Engine architecture for each modeled target.

CANDIDATES
----------
baseline
direct
component
direct_component_blend

SELECTION POLICY
----------------
- The final development fold is the only architecture-selection window.
- Blend weights are chosen only from that chronological validation window.
- The untouched test fold is scored only after architecture and blend weights
  are frozen. Test metrics are reporting-only and never affect selection.
- Validation direct/component models are rebuilt using only rows through the
  development fold's training cutoff. Persisted final models are used only for
  the untouched test fold because those artifacts were fit through the test
  fold's configured training cutoff.
- Component shares are reconciled within team-game before target assembly.
- No sportsbook or market data is read.

REQUIRED OUTPUTS
----------------
evaluation/model_selection.csv
models/{target}/selected_model.json

AUDIT OUTPUT
------------
evaluation/model_selection_predictions.parquet
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import common
import train_direct_models as direct
import train_efficiency_models as efficiency
import train_opportunity_models as opportunity


FOLDS_PATH = (
    "docs/win/football/nfl/prop_engine/evaluation/backtest_folds.parquet"
)
BASELINE_PATH = (
    "docs/win/football/nfl/prop_engine/evaluation/baseline_oof_predictions.parquet"
)
ELIGIBILITY_PATH = (
    "docs/win/football/nfl/prop_engine/config/target_eligibility.yaml"
)
FEATURE_CONFIG_ROOT = (
    "docs/win/football/nfl/prop_engine/config/features"
)
SELECTION_OUTPUT = (
    "docs/win/football/nfl/prop_engine/evaluation/model_selection.csv"
)
AUDIT_OUTPUT = (
    "docs/win/football/nfl/prop_engine/evaluation/"
    "model_selection_predictions.parquet"
)
ACCEPTANCE_THRESHOLDS_PATH = (
    "docs/win/football/nfl/prop_engine/config/acceptance_thresholds.yaml"
)

# RUSHING_YARDS_ROBUST_GATE_SELECTION
# For rushing_yards only, choose the direct/component blend weight on the
# chronological 2024 validation fold by maximizing the weakest normalized
# locked-gate margin after the existing quantile point calibration candidates.
# No test-season row participates in this selection.
RUSHING_YARDS_ROBUST_GATE_SELECTION = True
POINT_PREDICTION_BLEND_CANDIDATES = (0.0, 0.25, 0.50, 0.75, 1.0)

GRAIN = ["season", "week", "game_id", "player_id"]
TEAM_GRAIN = ["season", "week", "game_id", "team"]
CANDIDATES = [
    "baseline",
    "direct",
    "component",
    "direct_component_blend",
]
OUTPUT_COLUMNS = [
    "target",
    "candidate",
    "validation_mae",
    "validation_rmse",
    "validation_median_ae",
    "validation_poisson_deviance",
    "validation_brier_1plus",
    "validation_logloss_1plus",
    "test_mae",
    "test_rmse",
    "test_poisson_deviance",
    "selected_flag",
    "selection_reason",
]
AUDIT_COLUMNS = [
    "split",
    "fold_id",
    "season",
    "week",
    "game_id",
    "player_id",
    "target",
    "actual",
    "baseline_projection",
    "direct_projection",
    "component_projection",
    "blend_projection",
    "blend_direct_weight",
    "blend_component_weight",
    "direct_variant",
]

# No separately trained team red-zone / goal-line volume model exists in Issue
# 22. The component architecture therefore uses these already-pregame team-form
# fields as deterministic volume proxies, in declared fallback order.
RED_ZONE_PASS_VOLUME_FEATURES = [
    "team_red_zone_pass_attempts_roll3_mean",
    "team_red_zone_pass_attempts_roll5_mean",
    "team_red_zone_pass_attempts_ewm5",
    "team_red_zone_pass_attempts_season_to_date",
]
GOAL_LINE_RUSH_VOLUME_FEATURES = [
    "team_goal_line_rush_attempts_roll3_mean",
    "team_goal_line_rush_attempts_roll5_mean",
    "team_goal_line_rush_attempts_ewm5",
    "team_goal_line_rush_attempts_season_to_date",
]

COMPONENT_DEPENDENCIES = {
    "passing_yards": [
        "qb_pass_attempts",
        "passing_yards_per_attempt",
    ],
    "passing_tds": [
        "qb_pass_attempts",
        "passing_td_rate",
    ],
    "rushing_yards": [
        "team_rush_attempts",
        "player_carry_share",
        "rushing_yards_per_carry",
    ],
    "rushing_tds": [
        "player_goal_line_carry_share",
        "rushing_td_per_goal_line_carry",
        *GOAL_LINE_RUSH_VOLUME_FEATURES,
    ],
    "receiving_yards": [
        "team_pass_attempts",
        "player_target_share",
        "receiving_yards_per_target",
    ],
    "receiving_tds": [
        "player_red_zone_target_share",
        "receiving_td_per_red_zone_target",
        *RED_ZONE_PASS_VOLUME_FEATURES,
    ],
    "kicking_points": [
        "field_goal_attempts",
        "field_goal_conversion",
        "extra_point_attempts",
        "extra_point_conversion",
    ],
    "tackles": [
        "opponent_offensive_plays",
        "player_defensive_participation",
        "tackle_rate_per_defensive_play",
    ],
    "sacks": [
        "opponent_offensive_plays",
        "player_defensive_participation",
        "sack_rate_per_defensive_play",
    ],
}


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required YAML missing: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON missing: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def numeric(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype("float64")
    )



def numeric_matrix(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Build a numeric feature matrix in one allocation.

    This is intentionally equivalent to train_opportunity_models.numeric_frame
    but avoids pandas DataFrame fragmentation warnings from repeated column
    insertion when Issue 25 scores the large opportunity feature sets.
    """
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing numeric matrix column(s): {missing}")

    data = {
        column: common.safe_numeric(frame[column]).astype("float64")
        for column in columns
    }
    return pd.DataFrame(data, index=frame.index, columns=columns)


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def coalesce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for column in columns:
        if column not in frame.columns:
            continue
        candidate = numeric(frame[column])
        result = result.where(result.notna(), candidate)
    return result


def target_is_signed(config: dict[str, Any], target: str) -> bool:
    return str(config["targets"][target].get("type", "")) == "continuous_signed"


def target_is_count(config: dict[str, Any], target: str) -> bool:
    return str(config["targets"][target].get("type", "")) == "count_nonnegative"


def transform_target_prediction(
    values: np.ndarray | pd.Series,
    config: dict[str, Any],
    target: str,
) -> np.ndarray:
    output = np.asarray(values, dtype="float64")
    if not target_is_signed(config, target):
        output = np.maximum(output, 0.0)
    return output


def metric_values(
    actual: np.ndarray | pd.Series,
    prediction: np.ndarray | pd.Series,
    *,
    count_target: bool,
) -> dict[str, float | None]:
    y = np.asarray(actual, dtype="float64")
    p = np.asarray(prediction, dtype="float64")

    valid = np.isfinite(y) & np.isfinite(p)
    if not valid.all():
        raise ValueError(
            f"Metric input contains nonfinite rows: {int((~valid).sum())}"
        )
    if len(y) == 0:
        raise ValueError("Metric input is empty.")

    error = np.abs(y - p)
    result: dict[str, float | None] = {
        "mae": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(np.square(y - p)))),
        "median_ae": float(np.median(error)),
        "poisson_deviance": None,
        "brier_1plus": None,
        "logloss_1plus": None,
    }

    if count_target:
        if np.any(y < 0.0):
            raise ValueError("Negative actual encountered for count target.")
        lam = np.maximum(p, 1e-12)
        terms = np.where(
            y > 0.0,
            y * np.log(y / lam) - (y - lam),
            lam,
        )
        result["poisson_deviance"] = float(2.0 * np.mean(terms))

        probability = 1.0 - np.exp(-np.maximum(p, 0.0))
        probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
        event = (y >= 1.0).astype("float64")
        result["brier_1plus"] = float(
            np.mean(np.square(probability - event))
        )
        result["logloss_1plus"] = float(
            -np.mean(
                event * np.log(probability)
                + (1.0 - event) * np.log(1.0 - probability)
            )
        )

    return result


def resolve_folds(folds: pd.DataFrame) -> dict[str, Any]:
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
    common.ensure_unique(folds, ["fold_id"], "backtest folds")

    test = folds.loc[folds["test_flag"].eq(1)].copy()
    if len(test) != 1:
        raise ValueError("Expected exactly one untouched test fold.")
    test_row = test.iloc[0]

    final_train_end = int(test_row["train_end_season"])
    test_start = int(test_row["validation_start_season"])
    test_end = int(test_row["validation_end_season"])
    if test_start != test_end:
        raise ValueError("Issue 25 expects a single-season untouched test fold.")

    development = folds.loc[
        folds["test_flag"].eq(0)
        & folds["validation_start_season"].eq(final_train_end)
        & folds["validation_end_season"].eq(final_train_end)
    ].copy()
    if len(development) != 1:
        raise ValueError(
            "Expected exactly one final development fold immediately before test."
        )
    dev_row = development.iloc[0]

    return {
        "validation_fold_id": str(dev_row["fold_id"]),
        "validation_season": final_train_end,
        "selection_train_end_season": int(dev_row["train_end_season"]),
        "test_fold_id": str(test_row["fold_id"]),
        "test_season": test_start,
        "final_train_end_season": final_train_end,
    }


def assert_trainer_cutoffs(policy: dict[str, Any]) -> None:
    expected = (
        policy["selection_train_end_season"],
        policy["validation_season"],
        policy["final_train_end_season"],
        policy["test_season"],
    )
    for module, label in [
        (direct, "direct"),
        (opportunity, "opportunity"),
        (efficiency, "efficiency"),
    ]:
        actual = (
            int(module.MODEL_SELECTION_TRAIN_END),
            int(module.DEVELOPMENT_VALIDATION_SEASON),
            int(module.FINAL_TRAIN_END),
            int(module.UNTOUCHED_TEST_SEASON),
        )
        if actual != expected:
            raise ValueError(
                f"{label} trainer cutoff contract {actual} differs from folds {expected}."
            )


def load_baseline_windows(
    baseline: pd.DataFrame,
    policy: dict[str, Any],
    targets: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common.require_columns(
        baseline,
        [
            "fold_id",
            *GRAIN,
            "target",
            "actual",
            "baseline_projection",
        ],
        "baseline OOF predictions",
    )
    validation = baseline.loc[
        baseline["fold_id"].astype(str).eq(policy["validation_fold_id"])
    ].copy()
    test = baseline.loc[
        baseline["fold_id"].astype(str).eq(policy["test_fold_id"])
    ].copy()

    for label, frame, season in [
        ("validation", validation, policy["validation_season"]),
        ("test", test, policy["test_season"]),
    ]:
        if frame.empty:
            raise ValueError(f"Baseline {label} window is empty.")
        if set(pd.to_numeric(frame["season"]).astype(int)) != {season}:
            raise ValueError(f"Baseline {label} season differs from fold policy.")
        if set(frame["target"].astype(str)) != set(targets):
            raise ValueError(f"Baseline {label} target coverage is incomplete.")
        common.ensure_unique(
            frame,
            [*GRAIN, "target"],
            f"baseline {label} target grain",
        )
        actual_values = numeric(frame["actual"])
        invalid_actual = frame["actual"].notna() & actual_values.isna()
        if invalid_actual.any() or actual_values.isna().any():
            raise ValueError(
                f"Baseline {label} contains missing or nonnumeric actual."
            )
        frame["actual"] = actual_values

        baseline_values = numeric(frame["baseline_projection"])
        invalid_baseline = (
            frame["baseline_projection"].notna()
            & baseline_values.isna()
        )
        if invalid_baseline.any():
            sample = (
                frame.loc[invalid_baseline, [*GRAIN, "target", "baseline_projection"]]
                .head(10)
                .to_dict(orient="records")
            )
            raise ValueError(
                f"Baseline {label} contains nonnumeric baseline_projection; "
                f"sample={sample}"
            )
        frame["baseline_projection"] = baseline_values

        # A missing baseline projection means its deterministic formula could
        # not be evaluated from strictly pregame history. Do not fabricate a
        # zero. Architecture comparison uses the common evaluable cohort, so
        # these rows are excluded before any validation metric or blend weight
        # is selected.
        frame = frame.loc[frame["baseline_projection"].notna()].copy()
        if frame.empty:
            raise ValueError(
                f"Baseline {label} has no evaluable rows after excluding "
                "missing baseline projections."
            )
        if set(frame["target"].astype(str)) != set(targets):
            raise ValueError(
                f"Baseline {label} loses target coverage after excluding "
                "missing baseline projections."
            )

        if label == "validation":
            validation = frame
        else:
            test = frame

    return validation, test


def direct_feature_requirements(
    root: Path,
    targets: list[str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    manifests: dict[str, dict[str, Any]] = {}
    columns: list[str] = []
    for target in targets:
        manifest = load_json(root / FEATURE_CONFIG_ROOT / f"{target}.json")
        manifests[target] = manifest
        columns.extend(manifest["numeric_features"])
        columns.extend(manifest["categorical_features"])
        columns.append(f"target_{target}")
    return manifests, unique(columns)


def train_fixed_booster(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    feature_names: list[str],
    categorical_features: list[str],
    params: dict[str, Any],
    rounds: int,
) -> lgb.Booster:
    if rounds < 1:
        raise ValueError(f"Invalid boosting round count: {rounds}")
    dataset = lgb.Dataset(
        X,
        label=numeric(y),
        feature_name=feature_names,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )
    return lgb.train(
        dict(params),
        dataset,
        num_boost_round=int(rounds),
        callbacks=[lgb.log_evaluation(period=0)],
    )


def prepare_direct_frames(
    features: pd.DataFrame,
    target: str,
    manifest: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positions = {
        str(value).strip().upper()
        for value in manifest["eligible_positions"]
    }
    position = (
        features["position"].fillna("").astype(str).str.strip().str.upper()
    )
    actual = numeric(features[f"target_{target}"])
    eligible = position.isin(positions) & actual.notna()

    train = features.loc[
        eligible
        & pd.to_numeric(features["season"]).le(
            policy["selection_train_end_season"]
        )
    ].copy()
    validation = features.loc[
        eligible
        & pd.to_numeric(features["season"]).eq(policy["validation_season"])
    ].copy()
    final_test = features.loc[
        eligible
        & pd.to_numeric(features["season"]).eq(policy["test_season"])
    ].copy()

    if train.empty or validation.empty or final_test.empty:
        raise ValueError(
            f"{target}: empty direct train/validation/test frame."
        )
    return train, validation, final_test


def score_direct_variants(
    config: dict[str, Any],
    root: Path,
    direct_features: pd.DataFrame,
    target: str,
    manifest: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train, validation, test = prepare_direct_frames(
        direct_features, target, manifest, policy
    )
    numeric_features = list(manifest["numeric_features"])
    categorical_features = list(manifest["categorical_features"])
    feature_names = [*numeric_features, *categorical_features]

    common.reject_forbidden_feature_columns(feature_names, config)

    metadata = load_json(root / "docs/win/football/nfl/prop_engine/models" / target / "metadata.json")
    persisted_manifest = load_json(
        root
        / "docs/win/football/nfl/prop_engine/models"
        / target
        / "feature_manifest.json"
    )
    if persisted_manifest.get("market_features_used") is not False:
        raise ValueError(f"{target}: persisted direct manifest market flag is not false.")
    common.reject_forbidden_feature_columns(
        list(persisted_manifest["selected_features"]), config
    )

    levels_selection = direct.categorical_levels(train, categorical_features)
    X_train = direct.model_matrix(
        train, numeric_features, categorical_features, levels_selection
    )
    X_valid = direct.model_matrix(
        validation, numeric_features, categorical_features, levels_selection
    )
    y_train = numeric(train[f"target_{target}"])

    variants: list[dict[str, Any]] = [
        {
            "name": "primary",
            "objective": str(metadata["primary"]["objective"]),
            "rounds": int(metadata["primary"]["best_iteration_selected_on_2024"]),
            "model_file": root
            / "docs/win/football/nfl/prop_engine/models"
            / target
            / "direct_model.txt",
        }
    ]
    challenger = metadata.get("tackles_regression_challenger", {})
    if target == "tackles" and challenger.get("present") is True:
        variants.append(
            {
                "name": "regression_challenger",
                "objective": str(challenger["objective"]),
                "rounds": int(challenger["best_iteration_selected_on_2024"]),
                "model_file": root / str(challenger["path"]),
            }
        )

    validation_actual = numeric(validation[f"target_{target}"]).to_numpy()
    variant_predictions: dict[str, np.ndarray] = {}
    variant_metrics: dict[str, dict[str, float | None]] = {}

    for variant in variants:
        params = direct.params_for(variant["objective"])
        model = train_fixed_booster(
            X_train,
            y_train,
            feature_names=feature_names,
            categorical_features=categorical_features,
            params=params,
            rounds=variant["rounds"],
        )
        pred = model.predict(X_valid, num_iteration=variant["rounds"])
        pred = direct.transform_prediction(pred, variant["objective"])
        pred = transform_target_prediction(pred, config, target)
        variant_predictions[variant["name"]] = pred
        variant_metrics[variant["name"]] = metric_values(
            validation_actual,
            pred,
            count_target=target_is_count(config, target),
        )

    chosen = min(
        variants,
        key=lambda v: (
            float(variant_metrics[v["name"]]["mae"]),
            float(variant_metrics[v["name"]]["rmse"]),
            v["name"],
        ),
    )
    chosen_name = str(chosen["name"])

    valid_output = validation[GRAIN].copy()
    valid_output["direct_projection"] = variant_predictions[chosen_name]

    final_levels = persisted_manifest[
        "categorical_levels_final_through_2024"
    ]
    X_test = direct.model_matrix(
        test, numeric_features, categorical_features, final_levels
    )
    persisted = lgb.Booster(model_file=str(chosen["model_file"]))
    if persisted.feature_name() != feature_names:
        raise ValueError(f"{target}: persisted direct feature order mismatch.")
    test_pred = persisted.predict(X_test)
    test_pred = direct.transform_prediction(test_pred, chosen["objective"])
    test_pred = transform_target_prediction(test_pred, config, target)

    test_output = test[GRAIN].copy()
    test_output["direct_projection"] = test_pred

    choice = {
        "variant": chosen_name,
        "objective": chosen["objective"],
        "model_file": str(Path(chosen["model_file"]).relative_to(root)).replace("\\", "/"),
        "selected_on_validation_only": True,
        "validation_variant_metrics": {
            name: {
                key: finite_float(value)
                for key, value in metrics.items()
            }
            for name, metrics in variant_metrics.items()
        },
    }
    return valid_output, test_output, choice


def component_inference_rows(
    features: pd.DataFrame,
    component: str,
    eligibility: dict[str, Any],
) -> pd.DataFrame:
    spec = opportunity.COMPONENTS[component]
    scope = spec["scope"]
    component_features = list(spec["features"])

    if scope == "player":
        frame = features.copy()
        rule = str(spec["eligible_rule"])
        positions = {
            str(value).strip().upper()
            for value in eligibility[rule]["eligible_positions"]
        }
        position = (
            frame["position"].fillna("").astype(str).str.strip().str.upper()
        )
        return frame.loc[position.isin(positions)].copy()

    opportunity.check_team_feature_invariance(features, component_features)
    return opportunity.team_rows_from_features(features, component_features)


def score_opportunity_models(
    config: dict[str, Any],
    root: Path,
    opp_features: pd.DataFrame,
    eligibility: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    training_features = opp_features.loc[
        pd.to_numeric(opp_features["season"]).le(policy["final_train_end_season"])
    ].copy()
    frames = opportunity.build_training_frames(
        config, training_features, eligibility
    )

    validation_features = opp_features.loc[
        pd.to_numeric(opp_features["season"]).eq(policy["validation_season"])
    ].copy()
    test_features = opp_features.loc[
        pd.to_numeric(opp_features["season"]).eq(policy["test_season"])
    ].copy()

    validation_predictions: dict[str, pd.DataFrame] = {}
    test_predictions: dict[str, pd.DataFrame] = {}

    for component in opportunity.COMPONENT_ORDER:
        spec = opportunity.COMPONENTS[component]
        feature_names = list(spec["features"])
        common.reject_forbidden_feature_columns(feature_names, config)
        metadata = load_json(
            root
            / "docs/win/football/nfl/prop_engine/models/components"
            / component
            / "metadata.json"
        )
        train = frames[component].loc[
            pd.to_numeric(frames[component]["season"]).le(
                policy["selection_train_end_season"]
            )
        ].copy()
        X_train = numeric_matrix(train, feature_names)
        y_train = numeric(train["_label"])
        model = train_fixed_booster(
            X_train,
            y_train,
            feature_names=feature_names,
            categorical_features=[],
            params=dict(metadata["params"]),
            rounds=int(metadata["best_iteration_selected_on_2024"]),
        )

        valid_rows = component_inference_rows(
            validation_features, component, eligibility
        )
        X_valid = numeric_matrix(valid_rows, feature_names)
        valid_pred = opportunity.transform_prediction(
            model.predict(X_valid), component
        )

        key = GRAIN if spec["scope"] == "player" else TEAM_GRAIN
        valid_output = valid_rows[key].copy()
        valid_output[component] = valid_pred
        common.ensure_unique(valid_output, key, f"validation {component} predictions")
        validation_predictions[component] = valid_output

        test_rows = component_inference_rows(test_features, component, eligibility)
        X_test = numeric_matrix(test_rows, feature_names)
        persisted = lgb.Booster(
            model_file=str(
                root
                / "docs/win/football/nfl/prop_engine/models/components"
                / component
                / "model.txt"
            )
        )
        if persisted.feature_name() != feature_names:
            raise ValueError(f"{component}: persisted opportunity feature order mismatch.")
        test_pred = opportunity.transform_prediction(
            persisted.predict(X_test), component
        )
        test_output = test_rows[key].copy()
        test_output[component] = test_pred
        common.ensure_unique(test_output, key, f"test {component} predictions")
        test_predictions[component] = test_output

    return validation_predictions, test_predictions


def raw_efficiency_histories(
    config: dict[str, Any],
    eff_features: pd.DataFrame,
    eligibility: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    history_features = eff_features.loc[
        pd.to_numeric(eff_features["season"]).le(policy["final_train_end_season"])
    ].copy()
    label_base = efficiency.prepare_label_base(config, history_features)
    histories: dict[str, pd.DataFrame] = {}
    for model_name in efficiency.MODELS:
        frame = efficiency.build_component_label(label_base, model_name)
        frame = efficiency.apply_eligibility(
            frame, model_name, eligibility
        )
        histories[model_name] = frame
    return history_features, histories


def efficiency_inference_frame(
    eff_features: pd.DataFrame,
    raw_history: pd.DataFrame,
    model_name: str,
    year: int,
    eligibility: dict[str, Any],
) -> pd.DataFrame:
    rule = efficiency.ELIGIBILITY_RULE[model_name]
    positions = {
        str(value).strip().upper()
        for value in eligibility[rule]["eligible_positions"]
    }
    target_rows = eff_features.loc[
        pd.to_numeric(eff_features["season"]).eq(year)
    ].copy()
    position = (
        target_rows["position"].fillna("").astype(str).str.strip().str.upper()
    )
    target_rows = target_rows.loc[position.isin(positions)].copy()
    if target_rows.empty:
        raise ValueError(f"{model_name}: empty inference rows for {year}.")

    prior_columns = [
        *GRAIN,
        "kickoff_timestamp",
        "position",
        "position_group",
        "_prior_position_group",
        "_numerator",
        "_exposure",
        "_label",
    ]
    history = raw_history[prior_columns].copy()
    history["_inference_marker"] = 0

    placeholder = target_rows[
        [*GRAIN, "kickoff_timestamp", "position", "position_group"]
    ].copy()
    placeholder["_prior_position_group"] = efficiency.normalize_position_group(
        placeholder["position"], placeholder["position_group"]
    )
    placeholder["_numerator"] = np.nan
    placeholder["_exposure"] = np.nan
    placeholder["_label"] = np.nan
    placeholder["_inference_marker"] = 1

    combined = pd.concat([history, placeholder], ignore_index=True, sort=False)
    enriched = efficiency.add_strict_prior_features(combined, model_name)
    inference_rows = enriched.loc[enriched["_inference_marker"].eq(1)].copy()

    canonical_features = [
        feature
        for feature in efficiency.FEATURES[model_name]
        if feature not in efficiency.DERIVED_FEATURES
    ]
    feature_join = target_rows[[*GRAIN, *canonical_features]].copy()
    inference_rows = inference_rows.merge(
        feature_join,
        on=GRAIN,
        how="left",
        validate="one_to_one",
        suffixes=("", "_canonical"),
    )
    common.ensure_unique(
        inference_rows, GRAIN, f"{model_name} inference {year}"
    )
    return inference_rows


def score_efficiency_models(
    config: dict[str, Any],
    root: Path,
    eff_features: pd.DataFrame,
    eligibility: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    history_features, raw_histories = raw_efficiency_histories(
        config, eff_features, eligibility, policy
    )
    validation_predictions: dict[str, pd.DataFrame] = {}
    test_predictions: dict[str, pd.DataFrame] = {}

    for model_name in efficiency.MODELS:
        raw_history = raw_histories[model_name]
        enriched_training = efficiency.add_strict_prior_features(
            raw_history, model_name
        )
        canonical_features = [
            feature
            for feature in efficiency.FEATURES[model_name]
            if feature not in efficiency.DERIVED_FEATURES
        ]
        feature_join = history_features[[*GRAIN, *canonical_features]].copy()
        enriched_training = enriched_training.merge(
            feature_join,
            on=GRAIN,
            how="left",
            validate="one_to_one",
        )
        train = enriched_training.loc[
            pd.to_numeric(enriched_training["season"]).le(
                policy["selection_train_end_season"]
            )
        ].copy()

        feature_names = list(efficiency.FEATURES[model_name])
        common.reject_forbidden_feature_columns(feature_names, config)
        metadata = load_json(
            root
            / "docs/win/football/nfl/prop_engine/models/efficiency"
            / model_name
            / "metadata.json"
        )
        X_train = efficiency.feature_matrix(train, model_name)
        y_train = numeric(train["_label"])
        model = train_fixed_booster(
            X_train,
            y_train,
            feature_names=feature_names,
            categorical_features=[],
            params=dict(metadata["params"]),
            rounds=int(metadata["best_iteration_selected_on_2024"]),
        )

        valid_rows = efficiency_inference_frame(
            eff_features,
            raw_history,
            model_name,
            policy["validation_season"],
            eligibility,
        )
        valid_pred = efficiency.transform_prediction(
            model.predict(efficiency.feature_matrix(valid_rows, model_name)),
            model_name,
        )
        valid_output = valid_rows[GRAIN].copy()
        valid_output[model_name] = valid_pred
        validation_predictions[model_name] = valid_output

        test_rows = efficiency_inference_frame(
            eff_features,
            raw_history,
            model_name,
            policy["test_season"],
            eligibility,
        )
        persisted = lgb.Booster(
            model_file=str(
                root
                / "docs/win/football/nfl/prop_engine/models/efficiency"
                / model_name
                / "model.txt"
            )
        )
        if persisted.feature_name() != feature_names:
            raise ValueError(f"{model_name}: persisted efficiency feature order mismatch.")
        test_pred = efficiency.transform_prediction(
            persisted.predict(efficiency.feature_matrix(test_rows, model_name)),
            model_name,
        )
        test_output = test_rows[GRAIN].copy()
        test_output[model_name] = test_pred
        test_predictions[model_name] = test_output

    return validation_predictions, test_predictions


def attach_team_and_reconcile_share(
    prediction: pd.DataFrame,
    context: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    frame = prediction.merge(
        context[[*GRAIN, "team"]],
        on=GRAIN,
        how="left",
        validate="one_to_one",
    )
    if frame["team"].isna().any():
        raise ValueError(f"{column}: missing team during share reconciliation.")
    raw = numeric(frame[column]).clip(lower=0.0, upper=1.0)
    totals = raw.groupby(
        [frame[k] for k in TEAM_GRAIN],
        sort=False,
    ).transform("sum")
    adjusted = raw.where(~totals.gt(0.0), raw / totals)
    frame[column] = adjusted.clip(lower=0.0, upper=1.0)
    return frame[[*GRAIN, column]]


def build_component_target_prediction(
    config: dict[str, Any],
    target: str,
    base_rows: pd.DataFrame,
    context: pd.DataFrame,
    opportunity_predictions: dict[str, pd.DataFrame],
    efficiency_predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    frame = base_rows[GRAIN].merge(
        context[
            unique(
                [
                    *GRAIN,
                    "team",
                    *RED_ZONE_PASS_VOLUME_FEATURES,
                    *GOAL_LINE_RUSH_VOLUME_FEATURES,
                ]
            )
        ],
        on=GRAIN,
        how="left",
        validate="one_to_one",
    )
    if frame["team"].isna().any():
        raise ValueError(f"{target}: missing team context.")

    share_components = {
        "player_carry_share",
        "player_target_share",
        "player_red_zone_target_share",
        "player_goal_line_carry_share",
    }

    dependencies = COMPONENT_DEPENDENCIES[target]
    for dependency in dependencies:
        if dependency in opportunity_predictions:
            source = opportunity_predictions[dependency]
            spec = opportunity.COMPONENTS[dependency]
            if dependency in share_components:
                source = attach_team_and_reconcile_share(
                    source, context, dependency
                )
                frame = frame.merge(
                    source,
                    on=GRAIN,
                    how="left",
                    validate="one_to_one",
                )
            elif spec["scope"] == "player":
                frame = frame.merge(
                    source,
                    on=GRAIN,
                    how="left",
                    validate="one_to_one",
                )
            else:
                frame = frame.merge(
                    source,
                    on=TEAM_GRAIN,
                    how="left",
                    validate="many_to_one",
                )
        elif dependency in efficiency_predictions:
            frame = frame.merge(
                efficiency_predictions[dependency],
                on=GRAIN,
                how="left",
                validate="one_to_one",
            )

    if target == "passing_yards":
        projection = frame["qb_pass_attempts"] * frame["passing_yards_per_attempt"]
    elif target == "passing_tds":
        projection = frame["qb_pass_attempts"] * frame["passing_td_rate"]
    elif target == "rushing_yards":
        projection = (
            frame["team_rush_attempts"]
            * frame["player_carry_share"]
            * frame["rushing_yards_per_carry"]
        )
    elif target == "rushing_tds":
        goal_line_volume = coalesce_numeric(frame, GOAL_LINE_RUSH_VOLUME_FEATURES).clip(lower=0.0)
        projection = (
            goal_line_volume
            * frame["player_goal_line_carry_share"]
            * frame["rushing_td_per_goal_line_carry"]
        )
    elif target == "receiving_yards":
        projection = (
            frame["team_pass_attempts"]
            * frame["player_target_share"]
            * frame["receiving_yards_per_target"]
        )
    elif target == "receiving_tds":
        red_zone_volume = coalesce_numeric(frame, RED_ZONE_PASS_VOLUME_FEATURES).clip(lower=0.0)
        projection = (
            red_zone_volume
            * frame["player_red_zone_target_share"]
            * frame["receiving_td_per_red_zone_target"]
        )
    elif target == "kicking_points":
        projection = (
            3.0 * frame["field_goal_attempts"] * frame["field_goal_conversion"]
            + frame["extra_point_attempts"] * frame["extra_point_conversion"]
        )
    elif target == "tackles":
        projection = (
            frame["opponent_offensive_plays"]
            * frame["player_defensive_participation"]
            * frame["tackle_rate_per_defensive_play"]
        )
    elif target == "sacks":
        projection = (
            frame["opponent_offensive_plays"]
            * frame["player_defensive_participation"]
            * frame["sack_rate_per_defensive_play"]
        )
    else:
        raise KeyError(f"No component formula for target {target}")

    projection = transform_target_prediction(projection, config, target)
    if not np.isfinite(projection).all():
        bad = int((~np.isfinite(projection)).sum())
        raise ValueError(f"{target}: component projection has {bad} nonfinite rows.")

    output = frame[GRAIN].copy()
    output["component_projection"] = projection
    return output


def align_target_predictions(
    base_rows: pd.DataFrame,
    direct_prediction: pd.DataFrame,
    component_prediction: pd.DataFrame,
) -> pd.DataFrame:
    output = base_rows[
        [*GRAIN, "actual", "baseline_projection"]
    ].merge(
        direct_prediction,
        on=GRAIN,
        how="left",
        validate="one_to_one",
    ).merge(
        component_prediction,
        on=GRAIN,
        how="left",
        validate="one_to_one",
    )
    for column in [
        "actual",
        "baseline_projection",
        "direct_projection",
        "component_projection",
    ]:
        output[column] = numeric(output[column])
        if output[column].isna().any():
            sample = output.loc[output[column].isna(), GRAIN].head(10)
            raise ValueError(
                f"Missing {column} after candidate alignment. "
                f"Sample={sample.to_dict(orient='records')}"
            )
    return output


def select_blend_weight(
    validation: pd.DataFrame,
    *,
    target: str,
    acceptance: dict[str, Any],
) -> float:
    y = validation["actual"].to_numpy(dtype="float64")
    d = validation["direct_projection"].to_numpy(dtype="float64")
    c = validation["component_projection"].to_numpy(dtype="float64")

    if target != "rushing_yards":
        best_weight = 0.0
        best_key: tuple[float, float, float, float, float] | None = None
        for weight in np.linspace(0.0, 1.0, 101):
            pred = weight * d + (1.0 - weight) * c
            error = np.abs(y - pred)
            key = (
                float(np.mean(error)),
                float(np.sqrt(np.mean(np.square(y - pred)))),
                float(np.median(error)),
                abs(float(weight) - 0.5),
                float(weight),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_weight = float(weight)
        return best_weight

    baseline = validation["baseline_projection"].to_numpy(dtype="float64")
    baseline_mae = float(np.mean(np.abs(baseline - y)))
    max_bias = float(acceptance["maximum_allowed_bias"])
    max_mae = float(acceptance["maximum_validation_mae"])
    min_improvement = float(
        acceptance["minimum_improvement_vs_baseline_pct"]
    )
    if baseline_mae <= 0.0 or max_bias <= 0.0 or max_mae <= 0.0:
        raise ValueError("rushing_yards: invalid locked acceptance gate.")

    passing: list[tuple[float, float, float, float, float, float]] = []
    for weight in np.linspace(0.0, 1.0, 101):
        raw = weight * d + (1.0 - weight) * c
        q50 = float(np.quantile(y - raw, 0.50))
        calibrated_base = raw + q50

        for alpha in POINT_PREDICTION_BLEND_CANDIDATES:
            pred = raw + float(alpha) * (calibrated_base - raw)
            candidate_mae = float(np.mean(np.abs(pred - y)))
            candidate_bias = float(np.mean(pred - y))
            abs_bias = abs(candidate_bias)
            improvement = (
                (baseline_mae - candidate_mae) / baseline_mae * 100.0
            )

            if (
                candidate_mae > max_mae + 1e-12
                or abs_bias > max_bias + 1e-12
                or improvement + 1e-12 < min_improvement
            ):
                continue

            robust_margin = min(
                (max_bias - abs_bias) / max_bias,
                (max_mae - candidate_mae) / max_mae,
                (improvement - min_improvement)
                / max(abs(min_improvement), 1.0),
            )
            passing.append(
                (
                    -robust_margin,
                    candidate_mae,
                    abs_bias,
                    float(alpha),
                    abs(float(weight) - 0.5),
                    float(weight),
                )
            )

    if not passing:
        raise ValueError(
            "rushing_yards: no 2024 blend/calibration candidate passes "
            "all locked acceptance gates."
        )

    passing.sort()
    return float(passing[0][5])


def candidate_metrics_for_frame(
    config: dict[str, Any],
    target: str,
    frame: pd.DataFrame,
) -> dict[str, dict[str, float | None]]:
    mapping = {
        "baseline": "baseline_projection",
        "direct": "direct_projection",
        "component": "component_projection",
        "direct_component_blend": "blend_projection",
    }
    return {
        candidate: metric_values(
            frame["actual"],
            frame[column],
            count_target=target_is_count(config, target),
        )
        for candidate, column in mapping.items()
    }


def selection_key(metrics: dict[str, float | None], candidate: str) -> tuple[float, float, float, int]:
    return (
        float(metrics["mae"]),
        float(metrics["rmse"]),
        float(metrics["median_ae"]),
        CANDIDATES.index(candidate),
    )


def build_selected_json(
    *,
    target: str,
    selected: str,
    validation_metrics: dict[str, dict[str, float | None]],
    test_metrics: dict[str, dict[str, float | None]],
    direct_choice: dict[str, Any],
    blend_weight: float,
    policy: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    selected_validation = validation_metrics[selected]
    selected_test = test_metrics[selected]
    return {
        "target": target,
        "selected_architecture": selected,
        "selected_candidate": selected,
        "selection_metric": "validation_mae",
        "selection_tiebreakers": [
            "validation_rmse",
            "validation_median_ae",
            "candidate_order",
        ],
        "selection_reason": reason,
        "model_selection_train_end_season": policy["selection_train_end_season"],
        "validation_season": policy["validation_season"],
        "final_train_end_season": policy["final_train_end_season"],
        "test_season": policy["test_season"],
        "test_reporting_only": True,
        "test_used_for_selection": False,
        "selection_frozen_before_test_reporting": True,
        "comparison_row_policy": (
            "common_evaluable_rows; missing baseline projections excluded; "
            "no missing projection is imputed to zero"
        ),
        "blend_weights": {
            "direct": float(blend_weight),
            "component": float(1.0 - blend_weight),
            "selected_from_validation_only": True,
        },
        "direct_variant": direct_choice,
        "component_dependencies": COMPONENT_DEPENDENCIES[target],
        "component_share_reconciliation": {
            "player_carry_share": True,
            "player_target_share": True,
            "player_red_zone_target_share": True,
            "player_goal_line_carry_share": True,
            "method": "normalize_positive_predicted_shares_within_team_game",
        },
        "red_zone_volume_proxy_order": RED_ZONE_PASS_VOLUME_FEATURES,
        "goal_line_volume_proxy_order": GOAL_LINE_RUSH_VOLUME_FEATURES,
        "selected_validation_metrics": {
            key: finite_float(value)
            for key, value in selected_validation.items()
        },
        "selected_test_metrics_reporting_only": {
            key: finite_float(value)
            for key, value in selected_test.items()
        },
        "market_features_used": False,
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
    acceptance_thresholds = load_yaml(root / ACCEPTANCE_THRESHOLDS_PATH)
    targets = list(config["targets"].keys())

    if set(targets) != set(COMPONENT_DEPENDENCIES):
        raise ValueError(
            "Config targets do not match the implemented component target formulas."
        )

    eligibility = load_yaml(root / ELIGIBILITY_PATH)
    if set(eligibility) != set(targets):
        raise ValueError("Target eligibility coverage differs from config targets.")

    folds = common.read_parquet_required(FOLDS_PATH)
    policy = resolve_folds(folds)
    assert_trainer_cutoffs(policy)

    # _CONFIG_ENFORCED_SELECTION_POLICY
    training = config["training"]
    configured_policy = {
        "selection_train_end_season": int(
            training["model_selection_train_end_season"]
        ),
        "validation_season": int(
            training["development_validation_season"]
        ),
        "final_train_end_season": int(
            training["final_train_end_season"]
        ),
        "test_season": int(
            training["untouched_test_season"]
        ),
    }
    observed_policy = {
        key: int(policy[key])
        for key in configured_policy
    }
    if observed_policy != configured_policy:
        raise ValueError(
            "Backtest-fold policy differs from config.training. "
            f"observed={observed_policy}, configured={configured_policy}"
        )

    baseline = common.read_parquet_required(BASELINE_PATH)
    baseline_validation, baseline_test = load_baseline_windows(
        baseline, policy, targets
    )
    print("CHECK 01: contracts and baseline validation/test windows")

    manifests, direct_required = direct_feature_requirements(root, targets)
    direct_columns = unique(
        [
            *GRAIN,
            "position",
            *direct_required,
        ]
    )
    historical_path = root / config["paths"]["historical_features"]
    direct_features = pd.read_parquet(historical_path, columns=direct_columns)
    direct_features["season"] = pd.to_numeric(
        direct_features["season"], errors="raise"
    ).astype(int)
    direct_features["week"] = pd.to_numeric(
        direct_features["week"], errors="raise"
    ).astype(int)
    common.ensure_unique(direct_features, GRAIN, "direct historical features")

    opportunity_feature_columns = unique(
        [
            column
            for component in opportunity.COMPONENT_ORDER
            for column in opportunity.COMPONENTS[component]["features"]
        ]
    )
    opp_columns = unique(
        [
            *GRAIN,
            "team",
            "opponent",
            "position",
            *opportunity_feature_columns,
            *RED_ZONE_PASS_VOLUME_FEATURES,
            *GOAL_LINE_RUSH_VOLUME_FEATURES,
        ]
    )
    opp_features = pd.read_parquet(historical_path, columns=opp_columns)
    opp_features["season"] = pd.to_numeric(
        opp_features["season"], errors="raise"
    ).astype(int)
    opp_features["week"] = pd.to_numeric(
        opp_features["week"], errors="raise"
    ).astype(int)
    common.ensure_unique(opp_features, GRAIN, "opportunity historical features")

    efficiency_canonical = unique(
        [
            feature
            for model_name in efficiency.MODELS
            for feature in efficiency.FEATURES[model_name]
            if feature not in efficiency.DERIVED_FEATURES
        ]
    )
    eff_columns = unique(
        [
            *GRAIN,
            "kickoff_timestamp",
            "position",
            "position_group",
            *efficiency_canonical,
        ]
    )
    eff_features = pd.read_parquet(historical_path, columns=eff_columns)
    eff_features["season"] = pd.to_numeric(
        eff_features["season"], errors="raise"
    ).astype(int)
    eff_features["week"] = pd.to_numeric(
        eff_features["week"], errors="raise"
    ).astype(int)
    eff_features["kickoff_timestamp"] = pd.to_datetime(
        eff_features["kickoff_timestamp"], errors="raise", utc=True
    )
    common.ensure_unique(eff_features, GRAIN, "efficiency historical features")

    print("CHECK 02: score 12 opportunity components")
    opportunity_validation, opportunity_test = score_opportunity_models(
        config, root, opp_features, eligibility, policy
    )
    print("CHECK 03: score 10 efficiency components")
    efficiency_validation, efficiency_test = score_efficiency_models(
        config, root, eff_features, eligibility, policy
    )

    print("CHECK 04: score direct candidates")
    direct_validation: dict[str, pd.DataFrame] = {}
    direct_test: dict[str, pd.DataFrame] = {}
    direct_choices: dict[str, dict[str, Any]] = {}
    for target in targets:
        valid, test, choice = score_direct_variants(
            config,
            root,
            direct_features,
            target,
            manifests[target],
            policy,
        )
        direct_validation[target] = valid
        direct_test[target] = test
        direct_choices[target] = choice
        print(f"  direct {target}: {choice['variant']}")

    context = opp_features[
        unique(
            [
                *GRAIN,
                "team",
                *RED_ZONE_PASS_VOLUME_FEATURES,
                *GOAL_LINE_RUSH_VOLUME_FEATURES,
            ]
        )
    ].copy()
    common.ensure_unique(context, GRAIN, "component assembly context")

    # Phase 1: validation-only architecture and blend selection.
    print("CHECK 05: select architecture and blend weights on 2024 only")
    validation_frames: dict[str, pd.DataFrame] = {}
    validation_metrics_by_target: dict[
        str, dict[str, dict[str, float | None]]
    ] = {}
    blend_weights: dict[str, float] = {}
    selected_candidates: dict[str, str] = {}

    for target in targets:
        base_rows = baseline_validation.loc[
            baseline_validation["target"].astype(str).eq(target)
        ].copy()
        component_pred = build_component_target_prediction(
            config,
            target,
            base_rows,
            context,
            opportunity_validation,
            efficiency_validation,
        )
        frame = align_target_predictions(
            base_rows,
            direct_validation[target],
            component_pred,
        )
        if target not in acceptance_thresholds:
            raise ValueError(f"Missing acceptance thresholds for {target}.")
        weight = select_blend_weight(
            frame,
            target=target,
            acceptance=acceptance_thresholds[target],
        )
        frame["blend_projection"] = (
            weight * frame["direct_projection"]
            + (1.0 - weight) * frame["component_projection"]
        )
        metrics = candidate_metrics_for_frame(config, target, frame)
        selected = min(
            CANDIDATES,
            key=lambda candidate: selection_key(metrics[candidate], candidate),
        )
        validation_frames[target] = frame
        validation_metrics_by_target[target] = metrics
        blend_weights[target] = weight
        selected_candidates[target] = selected
        print(
            f"  {target}: selected={selected}, "
            f"blend_direct_weight={weight:.2f}, "
            f"validation_mae={metrics[selected]['mae']:.6f}"
        )

    # Architecture and blend decisions are frozen before test scoring begins.
    print("CHECK 06: report frozen selections on untouched 2025")
    test_frames: dict[str, pd.DataFrame] = {}
    test_metrics_by_target: dict[
        str, dict[str, dict[str, float | None]]
    ] = {}

    for target in targets:
        base_rows = baseline_test.loc[
            baseline_test["target"].astype(str).eq(target)
        ].copy()
        component_pred = build_component_target_prediction(
            config,
            target,
            base_rows,
            context,
            opportunity_test,
            efficiency_test,
        )
        frame = align_target_predictions(
            base_rows,
            direct_test[target],
            component_pred,
        )
        weight = blend_weights[target]
        frame["blend_projection"] = (
            weight * frame["direct_projection"]
            + (1.0 - weight) * frame["component_projection"]
        )
        test_frames[target] = frame
        test_metrics_by_target[target] = candidate_metrics_for_frame(
            config, target, frame
        )

    rows: list[dict[str, Any]] = []
    audit_rows: list[pd.DataFrame] = []

    for target in targets:
        selected = selected_candidates[target]
        selected_metrics = validation_metrics_by_target[target][selected]
        reason = (
            f"selected on chronological validation only: {selected} had the "
            f"lowest validation MAE={selected_metrics['mae']:.12g}; "
            "RMSE, median AE, then fixed candidate order are tie-breakers; "
            f"{policy['test_season']} metrics are reporting-only"
        )

        for candidate in CANDIDATES:
            valid_metrics = validation_metrics_by_target[target][candidate]
            test_metrics = test_metrics_by_target[target][candidate]
            if candidate == selected:
                row_reason = reason
            elif candidate == "direct_component_blend":
                row_reason = (
                    f"not selected; validation-only blend weight direct="
                    f"{blend_weights[target]:.2f}, component="
                    f"{1.0 - blend_weights[target]:.2f}; selected={selected}"
                )
            else:
                row_reason = (
                    f"not selected on validation; selected={selected} with "
                    f"validation_mae={selected_metrics['mae']:.12g}"
                )

            rows.append(
                {
                    "target": target,
                    "candidate": candidate,
                    "validation_mae": valid_metrics["mae"],
                    "validation_rmse": valid_metrics["rmse"],
                    "validation_median_ae": valid_metrics["median_ae"],
                    "validation_poisson_deviance": valid_metrics[
                        "poisson_deviance"
                    ],
                    "validation_brier_1plus": valid_metrics["brier_1plus"],
                    "validation_logloss_1plus": valid_metrics[
                        "logloss_1plus"
                    ],
                    "test_mae": test_metrics["mae"],
                    "test_rmse": test_metrics["rmse"],
                    "test_poisson_deviance": test_metrics[
                        "poisson_deviance"
                    ],
                    "selected_flag": int(candidate == selected),
                    "selection_reason": row_reason,
                }
            )

        selected_json = build_selected_json(
            target=target,
            selected=selected,
            validation_metrics=validation_metrics_by_target[target],
            test_metrics=test_metrics_by_target[target],
            direct_choice=direct_choices[target],
            blend_weight=blend_weights[target],
            policy=policy,
            reason=reason,
        )
        direct.write_json_atomic(
            common.prop_root() / "models" / target / "selected_model.json",
            selected_json,
        )

        for split, fold_id, frame in [
            ("validation", policy["validation_fold_id"], validation_frames[target]),
            ("test", policy["test_fold_id"], test_frames[target]),
        ]:
            audit = frame.copy()
            audit["split"] = split
            audit["fold_id"] = fold_id
            audit["target"] = target
            audit["blend_direct_weight"] = blend_weights[target]
            audit["blend_component_weight"] = 1.0 - blend_weights[target]
            audit["direct_variant"] = direct_choices[target]["variant"]
            audit_rows.append(audit[AUDIT_COLUMNS])

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    common.write_csv_atomic(result, SELECTION_OUTPUT)

    audit_output = pd.concat(audit_rows, ignore_index=True)[AUDIT_COLUMNS]
    common.write_parquet_atomic(audit_output, AUDIT_OUTPUT)

    payload = {
        "status": "passed",
        "targets": targets,
        "candidates": CANDIDATES,
        "validation_fold_id": policy["validation_fold_id"],
        "validation_season": policy["validation_season"],
        "selection_train_end_season": policy["selection_train_end_season"],
        "test_fold_id": policy["test_fold_id"],
        "test_season": policy["test_season"],
        "test_reporting_only": True,
        "test_used_for_selection": False,
        "blend_weights_selected_from_validation_only": True,
        "comparison_row_policy": (
            "common_evaluable_rows; missing baseline projections excluded; "
            "no missing projection is imputed to zero"
        ),
        "selected_architectures": selected_candidates,
        "blend_direct_weights": blend_weights,
        "market_features_used": False,
        "output": SELECTION_OUTPUT,
        "audit_output": AUDIT_OUTPUT,
    }
    common.log_run("select_model_architecture.py", payload)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
