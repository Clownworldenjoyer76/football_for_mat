#!/usr/bin/env python3
"""
Independent acceptance validator for NFL Prop Engine Issue 24.

Checks:
- all 9 required target directories
- direct_model.txt, feature_manifest.json, metadata.json for every target
- tackles challenger_model.txt exists and loads
- exact required primary model families/objectives
- required metadata keys and market_features_used == false
- target-specific Issue 19 manifest parity
- selected/required features exist in canonical Issue 17 feature schema
- no target_*, audit_*, played_game_flag, or configured forbidden features
- saved LightGBM feature order exactly matches persisted selected manifest
- feature schema SHA256 independently recomputes and matches manifest + metadata
- source Issue 19 manifest SHA256 matches
- canonical weather/travel join contract is game_id
- model selection through 2023, validation 2024, final fit through 2024
- 2025 untouched for selection, metrics, fit, schema, and categorical levels
- finite validation metrics
- deterministic LightGBM parameters
- tackles Poisson primary + regression challenger contract
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import lightgbm as lgb
import yaml


ROOT = Path(r"C:\Users\Mat\Documents\GitHub\football_for_mat")
PROP = ROOT / "docs/win/football/nfl/prop_engine"

MODEL_ROOT = PROP / "models"
TRAINER = PROP / "scripts/train/train_direct_models.py"
FEATURE_CONFIG_ROOT = PROP / "config/features"
CANONICAL_MANIFEST_PATH = (
    PROP / "data/historical/features/feature_manifest.json"
)
ELIGIBILITY_PATH = PROP / "config/target_eligibility.yaml"
CONFIG_PATH = PROP / "config/prop_engine.yaml"

TARGETS = [
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
    "kicking_points",
    "tackles",
    "sacks",
]

EXPECTED_OBJECTIVE = {
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

EXPECTED_FAMILY = {
    target: (
        "lightgbm_poisson"
        if objective == "poisson"
        else "lightgbm_regression"
    )
    for target, objective in EXPECTED_OBJECTIVE.items()
}

REQUIRED_METADATA_KEYS = {
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

REQUIRED_SOURCE_MANIFEST_KEYS = {
    "target",
    "eligible_positions",
    "numeric_features",
    "categorical_features",
    "required_features",
    "optional_features",
    "forbidden_features",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def schema_payload(numeric_features, categorical_features):
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


def feature_hash(numeric_features, categorical_features):
    payload = json.dumps(
        schema_payload(numeric_features, categorical_features),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


print("CHECK 01: required trainer and target directories")
assert TRAINER.is_file(), f"Missing trainer: {TRAINER}"

for target in TARGETS:
    target_dir = MODEL_ROOT / target
    assert target_dir.is_dir(), f"{target}: missing target model directory"

print("CHECK 02: load canonical contracts")
canonical = load_json(CANONICAL_MANIFEST_PATH)

with CONFIG_PATH.open("r", encoding="utf-8-sig") as f:
    config = yaml.safe_load(f)

with ELIGIBILITY_PATH.open("r", encoding="utf-8-sig") as f:
    eligibility = yaml.safe_load(f)

canonical_features = set(canonical["feature_columns"])
canonical_numeric = set(canonical["numeric_features"])
canonical_categorical = set(canonical["categorical_features"])
forbidden = list(config["forbidden_features"])

summary = {}

print("CHECK 03: validate required artifacts, metadata, source-manifest parity")

for target in TARGETS:
    target_dir = MODEL_ROOT / target

    model_path = target_dir / "direct_model.txt"
    output_manifest_path = target_dir / "feature_manifest.json"
    metadata_path = target_dir / "metadata.json"
    source_manifest_path = FEATURE_CONFIG_ROOT / f"{target}.json"

    assert model_path.is_file(), f"{target}: missing direct_model.txt"
    assert output_manifest_path.is_file(), f"{target}: missing feature_manifest.json"
    assert metadata_path.is_file(), f"{target}: missing metadata.json"
    assert source_manifest_path.is_file(), f"{target}: missing Issue 19 source manifest"

    output_manifest = load_json(output_manifest_path)
    metadata = load_json(metadata_path)
    source_manifest = load_json(source_manifest_path)

    missing_source_keys = REQUIRED_SOURCE_MANIFEST_KEYS - set(source_manifest)
    assert not missing_source_keys, (
        f"{target}: source manifest missing keys {sorted(missing_source_keys)}"
    )

    missing_metadata = REQUIRED_METADATA_KEYS - set(metadata)
    assert not missing_metadata, (
        f"{target}: metadata missing required keys {sorted(missing_metadata)}"
    )

    assert metadata["target"] == target
    assert metadata["model_family"] == EXPECTED_FAMILY[target]
    assert metadata["objective"] == EXPECTED_OBJECTIVE[target]
    assert metadata["market_features_used"] is False
    assert metadata["random_seed"] == 24024
    assert metadata["training_rows"] > 0
    assert metadata["feature_count"] > 0

    assert output_manifest["target"] == target
    assert output_manifest["market_features_used"] is False

    assert source_manifest["target"] == target
    assert source_manifest["eligible_positions"] == eligibility[target]["eligible_positions"]

    # Persisted model manifest must exactly preserve Issue 19 selected schemas.
    assert output_manifest["numeric_features"] == source_manifest["numeric_features"], (
        f"{target}: numeric feature schema differs from Issue 19"
    )
    assert output_manifest["categorical_features"] == source_manifest["categorical_features"], (
        f"{target}: categorical feature schema differs from Issue 19"
    )
    assert output_manifest["required_features"] == source_manifest["required_features"], (
        f"{target}: required feature list differs from Issue 19"
    )
    assert output_manifest["optional_features"] == source_manifest["optional_features"], (
        f"{target}: optional feature list differs from Issue 19"
    )
    assert output_manifest["forbidden_features"] == source_manifest["forbidden_features"], (
        f"{target}: forbidden feature list differs from Issue 19"
    )

    assert output_manifest["source_manifest_sha256"] == sha256(source_manifest_path), (
        f"{target}: source manifest SHA256 mismatch"
    )

    numeric_features = list(output_manifest["numeric_features"])
    categorical_features = list(output_manifest["categorical_features"])
    selected = numeric_features + categorical_features

    assert output_manifest["selected_features"] == selected
    assert len(selected) == len(set(selected))
    assert output_manifest["feature_count"] == len(selected)
    assert metadata["feature_count"] == len(selected)

    missing_required_from_selected = sorted(
        set(output_manifest["required_features"]) - set(selected)
    )
    assert not missing_required_from_selected, (
        f"{target}: required feature not selected: {missing_required_from_selected}"
    )

    absent_canonical = sorted(set(selected) - canonical_features)
    assert not absent_canonical, (
        f"{target}: selected feature absent from canonical Issue 17 manifest: "
        f"{absent_canonical}"
    )

    wrong_numeric = sorted(set(numeric_features) - canonical_numeric)
    assert not wrong_numeric, (
        f"{target}: numeric feature not canonical numeric: {wrong_numeric}"
    )

    wrong_categorical = sorted(set(categorical_features) - canonical_categorical)
    assert not wrong_categorical, (
        f"{target}: categorical feature not canonical categorical: "
        f"{wrong_categorical}"
    )

    prohibited = [
        feature for feature in selected
        if feature.startswith("target_")
        or feature.startswith("audit_")
        or feature == "played_game_flag"
    ]
    assert not prohibited, f"{target}: target/audit/outcome leakage {prohibited}"

    for feature in selected:
        lowered = feature.casefold()
        hits = [
            token for token in forbidden
            if str(token).casefold() in lowered
        ]
        assert not hits, (
            f"{target}: configured forbidden feature {feature}: {hits}"
        )

    assert metadata["target_columns_used_as_features"] is False
    assert metadata["forbidden_columns_used"] is False
    assert metadata["missing_required_feature_policy"] == "fail"

    print(f"  loading {target}/direct_model.txt")
    booster = lgb.Booster(model_file=str(model_path))

    assert booster.feature_name() == selected, (
        f"{target}: saved model feature order != persisted manifest"
    )
    assert booster.num_feature() == len(selected)
    assert booster.num_trees() >= 1

    independently_computed_hash = feature_hash(
        numeric_features,
        categorical_features,
    )

    assert output_manifest["feature_hash"] == independently_computed_hash, (
        f"{target}: output manifest feature hash mismatch"
    )
    assert metadata["feature_hash"] == independently_computed_hash, (
        f"{target}: metadata feature hash mismatch"
    )
    assert (
        metadata["feature_manifest"]["feature_hash"]
        == independently_computed_hash
    )

    assert (
        metadata["feature_manifest"]["sha256"]
        == sha256(output_manifest_path)
    )

    primary = metadata["primary"]

    assert primary["model_family"] == EXPECTED_FAMILY[target]
    assert primary["objective"] == EXPECTED_OBJECTIVE[target]
    assert primary["model_sha256"] == sha256(model_path)

    best_iteration = primary["best_iteration_selected_on_2024"]
    assert isinstance(best_iteration, int) and best_iteration >= 1

    metrics = primary["validation_2024_metrics"]
    for metric_name in ["rmse", "mae"]:
        value = float(metrics[metric_name])
        assert math.isfinite(value) and value >= 0.0, (
            f"{target}: invalid {metric_name}={value}"
        )

    if metrics["r2"] is not None:
        assert math.isfinite(float(metrics["r2"]))

    params = primary["params"]
    assert params["objective"] == EXPECTED_OBJECTIVE[target]
    assert params["boosting_type"] == "gbdt"
    assert params["deterministic"] is True
    assert params["num_threads"] == 1
    assert params["seed"] == 24024

    policy = metadata["training_policy"]
    assert policy["random_split_used"] is False
    assert policy["model_selection_train_end_season"] == 2023
    assert policy["development_validation_season"] == 2024
    assert policy["final_train_end_season"] == 2024
    assert policy["untouched_test_season"] == 2025
    assert policy["untouched_test_used_for_selection"] is False
    assert policy["untouched_test_used_for_metrics"] is False
    assert policy["untouched_test_used_for_fit"] is False
    assert policy["untouched_test_used_for_feature_schema"] is False
    assert policy["untouched_test_used_for_categories"] is False

    rows = metadata["rows"]
    assert rows["model_selection_train"] > 0
    assert rows["validation_2024"] > 0
    assert rows["final_train_through_2024"] == metadata["training_rows"]

    # Required weather/travel join contract.
    join_contract = metadata["environment_join_contract"]
    assert join_contract["weather_join_key"] == "game_id"
    assert join_contract["travel_join_key"] == "game_id"
    assert join_contract["source"] == "canonical_issue17_feature_table"
    assert join_contract["secondary_join_performed"] is False

    assert output_manifest["environment_join_contract"] == join_contract

    summary[target] = {
        "family": metadata["model_family"],
        "objective": metadata["objective"],
        "features": metadata["feature_count"],
        "training_rows": metadata["training_rows"],
        "validation_rows": rows["validation_2024"],
        "best_iteration": best_iteration,
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
        "feature_hash": independently_computed_hash,
        "trees": booster.num_trees(),
    }


print("CHECK 04: tackles Poisson primary plus regression challenger")

tackles_dir = MODEL_ROOT / "tackles"
challenger_path = tackles_dir / "challenger_model.txt"
assert challenger_path.is_file(), "tackles: missing challenger_model.txt"

tackles_metadata = load_json(tackles_dir / "metadata.json")
challenger_meta = tackles_metadata["tackles_regression_challenger"]

assert challenger_meta["present"] is True
assert challenger_meta["model_family"] == "lightgbm_regression"
assert challenger_meta["objective"] == "regression"
assert challenger_meta["selection_deferred_to_issue_25"] is True
assert challenger_meta["model_sha256"] == sha256(challenger_path)

challenger = lgb.Booster(model_file=str(challenger_path))
tackles_manifest = load_json(tackles_dir / "feature_manifest.json")
tackles_selected = (
    tackles_manifest["numeric_features"]
    + tackles_manifest["categorical_features"]
)

assert challenger.feature_name() == tackles_selected
assert challenger.num_feature() == len(tackles_selected)
assert challenger.num_trees() >= 1

for metric_name in ["rmse", "mae"]:
    value = float(
        challenger_meta["validation_2024_metrics"][metric_name]
    )
    assert math.isfinite(value) and value >= 0.0

print(
    "  tackles primary poisson: "
    f"rmse={summary['tackles']['rmse']:.6f}, "
    f"mae={summary['tackles']['mae']:.6f}"
)
print(
    "  tackles regression challenger: "
    f"rmse={challenger_meta['validation_2024_metrics']['rmse']:.6f}, "
    f"mae={challenger_meta['validation_2024_metrics']['mae']:.6f}"
)


print("CHECK 05: static source policy markers")

source = TRAINER.read_text(encoding="utf-8")

required_markers = [
    "MODEL_SELECTION_TRAIN_END = 2023",
    "DEVELOPMENT_VALIDATION_SEASON = 2024",
    "FINAL_TRAIN_END = 2024",
    "UNTOUCHED_TEST_SEASON = 2025",
    '"passing_yards": "regression"',
    '"passing_tds": "poisson"',
    '"rushing_yards": "regression"',
    '"rushing_tds": "poisson"',
    '"receiving_yards": "regression"',
    '"receiving_tds": "poisson"',
    '"kicking_points": "regression"',
    '"tackles": "poisson"',
    '"sacks": "poisson"',
    "challenger_model.txt",
    '"weather_join_key": "game_id"',
    '"travel_join_key": "game_id"',
    '"market_features_used":',
    '"fail"',
    "feature_hash(",
    "target_columns_used_as_features",
]

for marker in required_markers:
    assert marker in source, f"Trainer source missing marker: {marker}"


print("CHECK 06: summarize all independently validated direct models")

for target in TARGETS:
    s = summary[target]
    print(
        f"{target}: "
        f"family={s['family']}, "
        f"objective={s['objective']}, "
        f"features={s['features']}, "
        f"training_rows={s['training_rows']}, "
        f"validation_2024_rows={s['validation_rows']}, "
        f"trees={s['trees']}, "
        f"best_iteration={s['best_iteration']}, "
        f"rmse={s['rmse']:.6f}, "
        f"mae={s['mae']:.6f}, "
        f"feature_hash={s['feature_hash']}"
    )

print(f"targets={len(summary)}")
print("market_features_used=false")
print("feature_schema_hash_verified=true")
print("missing_required_feature_policy=fail")
print("weather_join_key=game_id")
print("travel_join_key=game_id")
print("selection=train_through_2023_validate_2024")
print("final_fit=train_through_2024")
print("untouched_test=2025")
print("ISSUE 24 ACCEPTANCE: PASS")
