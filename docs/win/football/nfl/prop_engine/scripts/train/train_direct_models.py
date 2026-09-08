#!/usr/bin/env python3
"""
Train deterministic direct target models for NFL Prop Engine Issue 24.

PRIMARY MODELS
--------------
passing_yards: LightGBM regression
passing_tds: LightGBM Poisson
rushing_yards: LightGBM regression
rushing_tds: LightGBM Poisson
receiving_yards: LightGBM regression
receiving_tds: LightGBM Poisson
kicking_points: LightGBM regression
tackles: LightGBM Poisson + regression challenger
sacks: LightGBM Poisson

REQUIRED OUTPUTS PER TARGET
---------------------------
models/{target}/direct_model.txt
models/{target}/feature_manifest.json
models/{target}/metadata.json

ADDITIONAL TACKLES OUTPUT
-------------------------
models/tackles/challenger_model.txt

POLICY
------
- Target-specific Issue 19 feature manifests are authoritative.
- Missing selected or required feature => hard failure.
- No target_*, audit_*, played_game_flag, market, odds, DRAT, EPRED, or
  other configured forbidden feature may enter a model.
- Feature schema hash is computed from the exact ordered numeric/categorical
  feature schema and must match metadata + persisted model manifest.
- Weather and travel features are consumed only from the canonical Issue 17
  feature table, whose environment family is joined by game_id. This trainer
  performs no alternate weather/travel join.
- Model selection: train through 2023, validate on 2024.
- Persisted final models: fit through 2024.
- 2025 is untouched and cannot affect fitting, early stopping, feature schema,
  categorical vocabularies, metrics, or challenger evaluation.
- Tackles Poisson remains the required primary direct model. Regression is
  persisted as a challenger for Issue 25 architecture selection.
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
        "Issue 24 requires LightGBM. Install with: "
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

FEATURE_CONFIG_ROOT = (
    "docs/win/football/nfl/prop_engine/config/features"
)
CANONICAL_MANIFEST_PATH = (
    "docs/win/football/nfl/prop_engine/data/historical/features/"
    "feature_manifest.json"
)
ELIGIBILITY_PATH = (
    "docs/win/football/nfl/prop_engine/config/target_eligibility.yaml"
)
FOLDS_PATH = (
    "docs/win/football/nfl/prop_engine/evaluation/backtest_folds.parquet"
)

TARGETS = list(_CONFIG_CONTRACT["targets"].keys())

PRIMARY_OBJECTIVE = {
    "passing_yards": "regression",
    "passing_tds": "poisson",
    "rushing_yards": "regression",
    "rushing_tds": "poisson",
    "receiving_yards": "regression",
    "receiving_tds": "poisson",
    "kicking_points": "regression",
    "tackles": "poisson",
    "sacks": "poisson",
}

PRIMARY_MODEL_FAMILY = {
    target: (
        "lightgbm_poisson"
        if objective == "poisson"
        else "lightgbm_regression"
    )
    for target, objective in PRIMARY_OBJECTIVE.items()
}

REQUIRED_SOURCE_MANIFEST_KEYS = [
    "target",
    "eligible_positions",
    "numeric_features",
    "categorical_features",
    "required_features",
    "optional_features",
    "forbidden_features",
]

ENVIRONMENT_JOIN_CONTRACT = {
    "weather_join_key": "game_id",
    "travel_join_key": "game_id",
    "source": "canonical_issue17_feature_table",
    "secondary_join_performed": False,
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON missing: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required YAML missing: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    destination = path.resolve()
    prop = common.prop_root().resolve()
    try:
        destination.relative_to(prop)
    except ValueError as exc:
        raise ValueError(
            f"Refusing write outside Prop Engine: {destination}"
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)

    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
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
    booster: lgb.Booster,
    path: Path,
    num_iteration: int,
) -> None:
    destination = path.resolve()
    prop = common.prop_root().resolve()
    try:
        destination.relative_to(prop)
    except ValueError as exc:
        raise ValueError(
            f"Refusing write outside Prop Engine: {destination}"
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)

    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(handle.name)
    handle.close()

    try:
        booster.save_model(
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


def schema_payload(
    numeric_features: list[str],
    categorical_features: list[str],
) -> list[dict[str, str]]:
    return [
        *[
            {"name": feature, "type": "numeric"}
            for feature in numeric_features
        ],
        *[
            {"name": feature, "type": "categorical"}
            for feature in categorical_features
        ],
    ]


def feature_hash(
    numeric_features: list[str],
    categorical_features: list[str],
) -> str:
    payload = json.dumps(
        schema_payload(
            numeric_features,
            categorical_features,
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def numeric(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype("float64")
    )


def clean_category(series: pd.Series) -> pd.Series:
    result = (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )
    result = result.mask(
        result.str.casefold().isin(
            {"", "nan", "none", "null", "<na>", "nat"}
        ),
        "",
    )
    return result


def categorical_levels(
    frame: pd.DataFrame,
    categorical_features: list[str],
) -> dict[str, list[str]]:
    levels: dict[str, list[str]] = {}
    for feature in categorical_features:
        values = clean_category(frame[feature])
        observed = sorted(
            value
            for value in values.unique().tolist()
            if value != ""
        )
        levels[feature] = observed
    return levels


def model_matrix(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    levels: dict[str, list[str]],
) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}

    for feature in numeric_features:
        data[feature] = numeric(frame[feature])

    for feature in categorical_features:
        values = clean_category(frame[feature])
        category = pd.Categorical(
            values.where(values.ne(""), None),
            categories=levels[feature],
            ordered=False,
        )
        # LightGBM categorical convention: negative value means missing/unseen.
        data[feature] = pd.Series(
            category.codes.astype("int32"),
            index=frame.index,
        )

    return pd.DataFrame(
        data,
        index=frame.index,
        columns=[
            *numeric_features,
            *categorical_features,
        ],
    )


def validate_folds(folds: pd.DataFrame) -> None:
    common.require_columns(
        folds,
        [
            "train_end_season",
            "validation_start_season",
            "test_flag",
        ],
        "backtest folds",
    )

    development = folds.loc[
        folds["test_flag"].eq(0)
        & folds["validation_start_season"].eq(
            DEVELOPMENT_VALIDATION_SEASON
        )
    ]

    if len(development) != 1:
        raise ValueError(
            "Expected exactly one 2024 development validation fold."
        )

    if int(
        development.iloc[0]["train_end_season"]
    ) != MODEL_SELECTION_TRAIN_END:
        raise ValueError(
            "Issue 24 model selection must train through 2023."
        )

    test = folds.loc[
        folds["test_flag"].eq(1)
    ]

    if len(test) != 1:
        raise ValueError(
            "Expected exactly one untouched test fold."
        )

    if int(
        test.iloc[0]["validation_start_season"]
    ) != UNTOUCHED_TEST_SEASON:
        raise ValueError(
            "Issue 24 untouched test season must be 2025."
        )

    if int(
        test.iloc[0]["train_end_season"]
    ) != FINAL_TRAIN_END:
        raise ValueError(
            "Issue 24 final training cutoff must be 2024."
        )


def validate_source_manifest(
    target: str,
    source: dict[str, Any],
    eligibility: dict[str, Any],
    canonical_manifest: dict[str, Any],
    config: dict[str, Any],
    available_columns: set[str],
) -> None:
    missing_keys = [
        key
        for key in REQUIRED_SOURCE_MANIFEST_KEYS
        if key not in source
    ]
    if missing_keys:
        raise ValueError(
            f"{target}: source feature manifest missing keys: {missing_keys}"
        )

    if source["target"] != target:
        raise ValueError(
            f"{target}: source manifest target={source['target']!r}"
        )

    if target not in eligibility:
        raise ValueError(
            f"{target}: missing target eligibility rule."
        )

    expected_positions = list(
        eligibility[target]["eligible_positions"]
    )
    if list(source["eligible_positions"]) != expected_positions:
        raise ValueError(
            f"{target}: eligible_positions differ between Issue 18 and "
            f"Issue 19 manifest."
        )

    numeric_features = list(source["numeric_features"])
    categorical_features = list(source["categorical_features"])
    selected = [
        *numeric_features,
        *categorical_features,
    ]

    if not selected:
        raise ValueError(
            f"{target}: selected feature list is empty."
        )

    if len(selected) != len(set(selected)):
        raise ValueError(
            f"{target}: duplicate selected feature."
        )

    required = list(source["required_features"])
    optional = list(source["optional_features"])

    missing_required_from_selection = sorted(
        set(required) - set(selected)
    )
    if missing_required_from_selection:
        raise ValueError(
            f"{target}: required feature not selected: "
            f"{missing_required_from_selection}"
        )

    unclassified = sorted(
        set(selected)
        - set(required)
        - set(optional)
    )
    if unclassified:
        raise ValueError(
            f"{target}: selected features not classified as required or "
            f"optional: {unclassified}"
        )

    missing_selected = sorted(
        set(selected) - available_columns
    )
    if missing_selected:
        raise ValueError(
            f"{target}: selected model feature missing from canonical table: "
            f"{missing_selected}"
        )

    missing_required = sorted(
        set(required) - available_columns
    )
    if missing_required:
        raise ValueError(
            f"{target}: missing REQUIRED model feature: {missing_required}"
        )

    canonical_features = set(
        canonical_manifest["feature_columns"]
    )
    canonical_numeric = set(
        canonical_manifest["numeric_features"]
    )
    canonical_categorical = set(
        canonical_manifest["categorical_features"]
    )

    absent_manifest = sorted(
        set(selected) - canonical_features
    )
    if absent_manifest:
        raise ValueError(
            f"{target}: feature absent from canonical Issue 17 manifest: "
            f"{absent_manifest}"
        )

    wrong_numeric = sorted(
        set(numeric_features) - canonical_numeric
    )
    if wrong_numeric:
        raise ValueError(
            f"{target}: numeric features not canonical numeric: "
            f"{wrong_numeric}"
        )

    wrong_categorical = sorted(
        set(categorical_features) - canonical_categorical
    )
    if wrong_categorical:
        raise ValueError(
            f"{target}: categorical features not canonical categorical: "
            f"{wrong_categorical}"
        )

    prohibited = [
        feature
        for feature in selected
        if feature.startswith("target_")
        or feature.startswith("audit_")
        or feature == "played_game_flag"
    ]
    if prohibited:
        raise ValueError(
            f"{target}: prohibited target/audit/outcome features: {prohibited}"
        )

    common.reject_forbidden_feature_columns(
        selected,
        config,
    )

    configured_forbidden = {
        str(value).casefold()
        for value in config["forbidden_features"]
    }
    source_forbidden = {
        str(value).casefold()
        for value in source["forbidden_features"]
    }

    if not configured_forbidden.issubset(
        source_forbidden
    ):
        missing_forbidden = sorted(
            configured_forbidden - source_forbidden
        )
        raise ValueError(
            f"{target}: Issue 19 manifest omits configured forbidden "
            f"tokens: {missing_forbidden}"
        )


def params_for(objective: str) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objective": objective,
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

    if objective == "poisson":
        params["poisson_max_delta_step"] = 0.7

    return params


def transform_prediction(
    prediction: np.ndarray,
    objective: str,
) -> np.ndarray:
    values = np.asarray(
        prediction,
        dtype="float64",
    )
    if objective == "poisson":
        return np.maximum(values, 0.0)
    return values


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    np.asarray(y, dtype=float)
                    - np.asarray(p, dtype=float)
                )
            )
        )
    )


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(
                np.asarray(y, dtype=float)
                - np.asarray(p, dtype=float)
            )
        )
    )


def r2(y: np.ndarray, p: np.ndarray) -> float | None:
    actual = np.asarray(y, dtype=float)
    pred = np.asarray(p, dtype=float)
    denominator = float(
        np.sum(
            np.square(
                actual - actual.mean()
            )
        )
    )
    if denominator <= 0.0:
        return None
    value = 1.0 - float(
        np.sum(np.square(actual - pred))
        / denominator
    )
    return value if math.isfinite(value) else None


def train_candidate(
    target: str,
    objective: str,
    selection_train: pd.DataFrame,
    validation: pd.DataFrame,
    final_train: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    output_path: Path,
) -> dict[str, Any]:
    selection_levels = categorical_levels(
        selection_train,
        categorical_features,
    )

    final_levels = categorical_levels(
        final_train,
        categorical_features,
    )

    X_train = model_matrix(
        selection_train,
        numeric_features,
        categorical_features,
        selection_levels,
    )
    X_valid = model_matrix(
        validation,
        numeric_features,
        categorical_features,
        selection_levels,
    )
    X_final = model_matrix(
        final_train,
        numeric_features,
        categorical_features,
        final_levels,
    )

    y_train = numeric(
        selection_train[f"target_{target}"]
    )
    y_valid = numeric(
        validation[f"target_{target}"]
    )
    y_final = numeric(
        final_train[f"target_{target}"]
    )

    if objective == "poisson":
        if y_train.lt(0.0).any() or y_valid.lt(0.0).any():
            raise ValueError(
                f"{target}: negative label encountered for Poisson."
            )

    feature_names = [
        *numeric_features,
        *categorical_features,
    ]

    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=feature_names,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    valid_set = lgb.Dataset(
        X_valid,
        label=y_valid,
        feature_name=feature_names,
        categorical_feature=categorical_features,
        reference=train_set,
        free_raw_data=False,
    )

    params = params_for(
        objective
    )

    selected = lgb.train(
        params,
        train_set,
        num_boost_round=2500,
        valid_sets=[valid_set],
        valid_names=["validation_2024"],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100,
                first_metric_only=True,
                verbose=False,
            ),
            lgb.log_evaluation(period=0),
        ],
    )

    best_iteration = int(
        selected.best_iteration
        if selected.best_iteration
        else 2500
    )

    validation_prediction = transform_prediction(
        selected.predict(
            X_valid,
            num_iteration=best_iteration,
        ),
        objective,
    )

    y_valid_array = y_valid.to_numpy(
        dtype="float64"
    )

    metrics = {
        "rmse": rmse(
            y_valid_array,
            validation_prediction,
        ),
        "mae": mae(
            y_valid_array,
            validation_prediction,
        ),
        "r2": r2(
            y_valid_array,
            validation_prediction,
        ),
    }

    final_set = lgb.Dataset(
        X_final,
        label=y_final,
        feature_name=feature_names,
        categorical_feature=categorical_features,
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

    save_model_atomic(
        final_model,
        output_path,
        best_iteration,
    )

    loaded = lgb.Booster(
        model_file=str(output_path)
    )

    if loaded.feature_name() != feature_names:
        raise ValueError(
            f"{target}: persisted LightGBM feature order differs from "
            f"selected manifest."
        )

    return {
        "objective": objective,
        "model_family": (
            "lightgbm_poisson"
            if objective == "poisson"
            else "lightgbm_regression"
        ),
        "best_iteration": best_iteration,
        "validation_2024_metrics": metrics,
        "categorical_levels_final_through_2024": final_levels,
        "model_sha256": sha256_file(output_path),
        "params": params,
    }


def build_target_frame(
    historical: pd.DataFrame,
    target: str,
    source_manifest: dict[str, Any],
) -> pd.DataFrame:
    eligible_positions = {
        str(value).strip().upper()
        for value in source_manifest["eligible_positions"]
    }

    position = (
        historical["position"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    actual = numeric(
        historical[f"target_{target}"]
    )

    frame = historical.loc[
        historical["season"].le(
            FINAL_TRAIN_END
        )
        & position.isin(
            eligible_positions
        )
        & actual.notna()
    ].copy()

    if frame.empty:
        raise ValueError(
            f"{target}: no eligible non-null target rows through 2024."
        )

    if frame["season"].max() > FINAL_TRAIN_END:
        raise ValueError(
            f"{target}: data after 2024 entered training frame."
        )

    return frame


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

    canonical_manifest = load_json(
        root / CANONICAL_MANIFEST_PATH
    )
    eligibility = load_yaml(
        root / ELIGIBILITY_PATH
    )
    folds = common.read_parquet_required(
        FOLDS_PATH
    )
    validate_folds(folds)

    source_manifests: dict[str, dict[str, Any]] = {}

    all_selected_features: list[str] = []
    all_target_columns: list[str] = []
    categorical_union: set[str] = set()

    for target in TARGETS:
        manifest_path = (
            root
            / FEATURE_CONFIG_ROOT
            / f"{target}.json"
        )
        source = load_json(
            manifest_path
        )
        source_manifests[target] = source

        all_selected_features.extend(
            source["numeric_features"]
        )
        all_selected_features.extend(
            source["categorical_features"]
        )
        categorical_union.update(
            source["categorical_features"]
        )
        all_target_columns.append(
            f"target_{target}"
        )

    selected_union = list(
        dict.fromkeys(
            all_selected_features
        )
    )

    required_columns = list(
        dict.fromkeys(
            [
                "season",
                "week",
                "game_id",
                "kickoff_timestamp",
                "player_id",
                "position",
                *selected_union,
                *all_target_columns,
            ]
        )
    )

    historical_path = (
        root
        / config["paths"]["historical_features"]
    )

    if not historical_path.is_file():
        raise FileNotFoundError(
            f"Historical feature table missing: {historical_path}"
        )

    historical = pd.read_parquet(
        historical_path,
        columns=required_columns,
    )

    historical["season"] = pd.to_numeric(
        historical["season"],
        errors="raise",
    ).astype(int)
    historical["week"] = pd.to_numeric(
        historical["week"],
        errors="raise",
    ).astype(int)
    historical["kickoff_timestamp"] = pd.to_datetime(
        historical["kickoff_timestamp"],
        errors="raise",
        utc=True,
    )

    available_columns = set(
        historical.columns
    )

    for target in TARGETS:
        validate_source_manifest(
            target,
            source_manifests[target],
            eligibility,
            canonical_manifest,
            config,
            available_columns,
        )

    results: list[dict[str, Any]] = []

    for target in TARGETS:
        source = source_manifests[target]

        numeric_features = list(
            source["numeric_features"]
        )
        categorical_features = list(
            source["categorical_features"]
        )

        schema_hash = feature_hash(
            numeric_features,
            categorical_features,
        )

        frame = build_target_frame(
            historical,
            target,
            source,
        )

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
                f"{target}: empty selection training rows."
            )
        if validation.empty:
            raise ValueError(
                f"{target}: empty 2024 validation rows."
            )
        if final_train.empty:
            raise ValueError(
                f"{target}: empty final training rows."
            )

        target_dir = (
            common.prop_root()
            / "models"
            / target
        )
        target_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        direct_model_path = (
            target_dir
            / "direct_model.txt"
        )

        primary = train_candidate(
            target=target,
            objective=PRIMARY_OBJECTIVE[target],
            selection_train=selection_train,
            validation=validation,
            final_train=final_train,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            output_path=direct_model_path,
        )

        challenger: dict[str, Any] | None = None

        if target == "tackles":
            challenger_path = (
                target_dir
                / "challenger_model.txt"
            )
            challenger = train_candidate(
                target=target,
                objective="regression",
                selection_train=selection_train,
                validation=validation,
                final_train=final_train,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                output_path=challenger_path,
            )
            challenger["path"] = str(
                challenger_path.relative_to(root)
            ).replace("\\", "/")

        source_manifest_path = (
            root
            / FEATURE_CONFIG_ROOT
            / f"{target}.json"
        )

        output_feature_manifest = {
            "target": target,
            "source_manifest": str(
                source_manifest_path.relative_to(root)
            ).replace("\\", "/"),
            "source_manifest_sha256": sha256_file(
                source_manifest_path
            ),
            "eligible_positions": list(
                source["eligible_positions"]
            ),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "selected_features": [
                *numeric_features,
                *categorical_features,
            ],
            "required_features": list(
                source["required_features"]
            ),
            "optional_features": list(
                source["optional_features"]
            ),
            "forbidden_features": list(
                source["forbidden_features"]
            ),
            "feature_count": (
                len(numeric_features)
                + len(categorical_features)
            ),
            "feature_hash": schema_hash,
            "schema_hash_algorithm": "sha256",
            "schema_hash_payload": (
                "ordered_feature_name_and_type"
            ),
            "categorical_levels_final_through_2024":
                primary[
                    "categorical_levels_final_through_2024"
                ],
            "environment_join_contract":
                ENVIRONMENT_JOIN_CONTRACT,
            "market_features_used": False,
        }

        feature_manifest_path = (
            target_dir
            / "feature_manifest.json"
        )
        write_json_atomic(
            feature_manifest_path,
            output_feature_manifest,
        )

        # Reload persisted manifest and independently recompute its schema hash.
        persisted_manifest = load_json(
            feature_manifest_path
        )

        persisted_hash = feature_hash(
            list(
                persisted_manifest["numeric_features"]
            ),
            list(
                persisted_manifest["categorical_features"]
            ),
        )

        if persisted_hash != schema_hash:
            raise ValueError(
                f"{target}: persisted feature schema hash mismatch."
            )

        if (
            persisted_manifest["feature_hash"]
            != schema_hash
        ):
            raise ValueError(
                f"{target}: feature_manifest feature_hash mismatch."
            )

        training_start = (
            final_train["kickoff_timestamp"]
            .min()
            .isoformat()
        )
        training_end = (
            final_train["kickoff_timestamp"]
            .max()
            .isoformat()
        )

        metadata = {
            # Required keys:
            "target": target,
            "model_family":
                PRIMARY_MODEL_FAMILY[target],
            "objective":
                PRIMARY_OBJECTIVE[target],
            "training_start":
                training_start,
            "training_end":
                training_end,
            "feature_count":
                len(numeric_features)
                + len(categorical_features),
            "feature_hash":
                schema_hash,
            "training_rows":
                int(len(final_train)),
            "random_seed":
                SEED,
            "market_features_used":
                False,

            # Additional audit metadata:
            "status": "trained",
            "lightgbm_version":
                lgb.__version__,
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
                "untouched_test_used_for_selection":
                    False,
                "untouched_test_used_for_metrics":
                    False,
                "untouched_test_used_for_fit":
                    False,
                "untouched_test_used_for_feature_schema":
                    False,
                "untouched_test_used_for_categories":
                    False,
            },
            "rows": {
                "model_selection_train":
                    int(len(selection_train)),
                "validation_2024":
                    int(len(validation)),
                "final_train_through_2024":
                    int(len(final_train)),
            },
            "primary": {
                "model_family":
                    primary["model_family"],
                "objective":
                    primary["objective"],
                "best_iteration_selected_on_2024":
                    primary["best_iteration"],
                "validation_2024_metrics":
                    primary[
                        "validation_2024_metrics"
                    ],
                "model_sha256":
                    primary["model_sha256"],
                "params":
                    primary["params"],
            },
            "tackles_regression_challenger":
                (
                    {
                        "present": True,
                        "model_family":
                            challenger["model_family"],
                        "objective":
                            challenger["objective"],
                        "best_iteration_selected_on_2024":
                            challenger["best_iteration"],
                        "validation_2024_metrics":
                            challenger[
                                "validation_2024_metrics"
                            ],
                        "model_sha256":
                            challenger["model_sha256"],
                        "path":
                            challenger["path"],
                        "selection_deferred_to_issue_25":
                            True,
                    }
                    if challenger is not None
                    else {
                        "present": False,
                    }
                ),
            "feature_manifest": {
                "path": str(
                    feature_manifest_path.relative_to(
                        root
                    )
                ).replace("\\", "/"),
                "sha256": sha256_file(
                    feature_manifest_path
                ),
                "feature_hash": schema_hash,
            },
            "environment_join_contract":
                ENVIRONMENT_JOIN_CONTRACT,
            "target_columns_used_as_features":
                False,
            "forbidden_columns_used":
                False,
            "missing_required_feature_policy":
                "fail",
        }

        metadata_path = (
            target_dir
            / "metadata.json"
        )
        write_json_atomic(
            metadata_path,
            metadata,
        )

        # Final hard contract checks.
        persisted_metadata = load_json(
            metadata_path
        )

        required_metadata_keys = {
            "target",
            "model_family",
            "objective",
            "training_start",
            "training_end",
            "feature_count",
            "feature_hash",
            "training_rows",
            "random_seed",
            "market_features_used",
        }

        missing_metadata = sorted(
            required_metadata_keys
            - set(persisted_metadata)
        )
        if missing_metadata:
            raise ValueError(
                f"{target}: required metadata keys missing: "
                f"{missing_metadata}"
            )

        if persisted_metadata["market_features_used"] is not False:
            raise ValueError(
                f"{target}: market_features_used must equal false."
            )

        if (
            persisted_metadata["feature_hash"]
            != persisted_manifest["feature_hash"]
        ):
            raise ValueError(
                f"{target}: metadata/manifest feature_hash mismatch."
            )

        result = {
            "target": target,
            "model_family":
                PRIMARY_MODEL_FAMILY[target],
            "objective":
                PRIMARY_OBJECTIVE[target],
            "features":
                metadata["feature_count"],
            "training_rows":
                int(len(final_train)),
            "validation_2024_rows":
                int(len(validation)),
            "best_iteration":
                primary["best_iteration"],
            "validation_rmse":
                primary[
                    "validation_2024_metrics"
                ]["rmse"],
            "validation_mae":
                primary[
                    "validation_2024_metrics"
                ]["mae"],
            "feature_hash":
                schema_hash,
            "tackles_challenger":
                (
                    {
                        "objective":
                            challenger["objective"],
                        "best_iteration":
                            challenger["best_iteration"],
                        "validation_rmse":
                            challenger[
                                "validation_2024_metrics"
                            ]["rmse"],
                        "validation_mae":
                            challenger[
                                "validation_2024_metrics"
                            ]["mae"],
                    }
                    if challenger is not None
                    else None
                ),
        }

        results.append(result)

        print(
            json.dumps(
                {
                    "target": target,
                    "status": "trained",
                    "model_family":
                        result["model_family"],
                    "objective":
                        result["objective"],
                    "features":
                        result["features"],
                    "training_rows":
                        result["training_rows"],
                    "validation_2024_rows":
                        result["validation_2024_rows"],
                    "best_iteration":
                        result["best_iteration"],
                    "validation_rmse":
                        result["validation_rmse"],
                    "validation_mae":
                        result["validation_mae"],
                    "feature_hash":
                        result["feature_hash"],
                    "tackles_challenger":
                        result["tackles_challenger"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    payload = {
        "status": "passed",
        "targets": len(results),
        "target_names": TARGETS,
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
        "market_features_used": False,
        "target_columns_used_as_features": False,
        "forbidden_columns_used": False,
        "feature_schema_hash_verified": True,
        "missing_required_feature_policy": "fail",
        "environment_join_contract":
            ENVIRONMENT_JOIN_CONTRACT,
        "tackles_regression_challenger_trained":
            True,
        "tackles_challenger_selection_deferred_to_issue_25":
            True,
        "results": results,
    }

    common.log_run(
        "train_direct_models.py",
        payload,
    )

    print(
        json.dumps(
            {
                "script":
                    Path(__file__).name,
                "payload":
                    payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
