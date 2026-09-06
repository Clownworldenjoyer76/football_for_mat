#!/usr/bin/env python3
"""
Independent acceptance validator for NFL Prop Engine Issue 22.

Checks:
- all 12 required component directories
- model.txt, feature_manifest.json, metadata.json for every component
- LightGBM model files load successfully
- model feature names exactly match explicit manifests
- no automatic all-numeric selection
- no target/audit/played_game_flag/forbidden market features
- metadata/model/manifest SHA256 integrity
- LightGBM regression contract
- fixed 2023 -> 2024 model-selection policy
- final fit through 2024 only
- untouched 2025 policy
- share/participation clipping to [0,1]
- volume zero-floor policy
- current-week share reconciliation flag
- finite validation metrics
- deterministic feature counts / iterations
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

MODEL_ROOT = PROP / "models/components"
CANONICAL_MANIFEST = (
    PROP / "data/historical/features/feature_manifest.json"
)
CONFIG_PATH = PROP / "config/prop_engine.yaml"
TRAINER_PATH = (
    PROP / "scripts/train/train_opportunity_models.py"
)

COMPONENTS = [
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

SHARE_COMPONENTS = {
    "player_carry_share",
    "player_target_share",
    "player_red_zone_target_share",
    "player_goal_line_carry_share",
}

BOUNDED_COMPONENTS = (
    SHARE_COMPONENTS
    | {"player_defensive_participation"}
)

VOLUME_COMPONENTS = {
    "qb_pass_attempts",
    "team_pass_attempts",
    "team_rush_attempts",
    "field_goal_attempts",
    "extra_point_attempts",
    "opponent_offensive_plays",
    "opponent_dropbacks",
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


print("CHECK 01: required trainer and component directories")
assert TRAINER_PATH.is_file(), f"Missing trainer: {TRAINER_PATH}"
assert MODEL_ROOT.is_dir(), f"Missing component model root: {MODEL_ROOT}"

actual_dirs = sorted(
    p.name for p in MODEL_ROOT.iterdir()
    if p.is_dir() and p.name in COMPONENTS
)
assert actual_dirs == sorted(COMPONENTS), (
    f"Required component directories mismatch: {actual_dirs}"
)

print("CHECK 02: load shared contracts")
canonical = load_json(CANONICAL_MANIFEST)

with CONFIG_PATH.open("r", encoding="utf-8-sig") as f:
    config = yaml.safe_load(f)

canonical_features = set(canonical["feature_columns"])
canonical_numeric = set(canonical["numeric_features"])
target_columns = set(canonical.get("target_columns", []))
forbidden = list(config["forbidden_features"])

print("CHECK 03: validate all required artifacts and LightGBM models")

summary = {}

for component in COMPONENTS:
    root = MODEL_ROOT / component
    model_path = root / "model.txt"
    manifest_path = root / "feature_manifest.json"
    metadata_path = root / "metadata.json"

    assert model_path.is_file(), f"{component}: missing model.txt"
    assert manifest_path.is_file(), f"{component}: missing feature_manifest.json"
    assert metadata_path.is_file(), f"{component}: missing metadata.json"

    manifest = load_json(manifest_path)
    metadata = load_json(metadata_path)

    assert manifest["component"] == component
    assert metadata["component"] == component

    assert manifest["model_type"] == "lightgbm_regression"
    assert metadata["model_type"] == "lightgbm_regression"
    assert manifest["objective"] == "regression"

    features = manifest["numeric_features"]
    cats = manifest["categorical_features"]

    assert isinstance(features, list) and features, (
        f"{component}: numeric feature list is empty"
    )
    assert cats == [], (
        f"{component}: initial Issue 22 trainer should have no categorical features"
    )
    assert len(features) == len(set(features)), (
        f"{component}: duplicate feature name"
    )
    assert manifest["feature_count"] == len(features)
    assert metadata["feature_count"] == len(features)

    assert manifest["automatic_all_numeric_selection"] is False, (
        f"{component}: automatic all-numeric selection is not allowed"
    )

    unknown = sorted(set(features) - canonical_features)
    assert not unknown, (
        f"{component}: features absent from canonical Issue 17 manifest: {unknown}"
    )

    nonnumeric = sorted(set(features) - canonical_numeric)
    assert not nonnumeric, (
        f"{component}: nonnumeric feature in numeric manifest: {nonnumeric}"
    )

    leaked = sorted(set(features) & target_columns)
    assert not leaked, (
        f"{component}: target leakage: {leaked}"
    )

    bad_prefix = [
        x for x in features
        if x.startswith("target_") or x.startswith("audit_")
    ]
    assert not bad_prefix, (
        f"{component}: prohibited target/audit feature: {bad_prefix}"
    )

    assert "played_game_flag" not in features, (
        f"{component}: played_game_flag leakage"
    )

    for feature in features:
        normalized = feature.casefold()
        hits = [
            token for token in forbidden
            if str(token).casefold() in normalized
        ]
        assert not hits, (
            f"{component}: forbidden feature {feature}: {hits}"
        )

    assert manifest["forbidden_features"] == forbidden

    print(f"  loading {component}/model.txt")
    booster = lgb.Booster(model_file=str(model_path))

    model_features = booster.feature_name()
    assert model_features == features, (
        f"{component}: model feature order does not match manifest"
    )

    assert booster.num_feature() == len(features)
    assert booster.num_trees() >= 1

    assert metadata["model_sha256"] == sha256(model_path), (
        f"{component}: model SHA256 mismatch"
    )
    assert metadata["feature_manifest_sha256"] == sha256(manifest_path), (
        f"{component}: feature-manifest SHA256 mismatch"
    )

    policy = metadata["training_policy"]

    assert policy["random_split_used"] is False
    assert policy["model_selection_train_end_season"] == 2023
    assert policy["development_validation_season"] == 2024
    assert policy["final_train_end_season"] == 2024
    assert policy["untouched_test_season"] == 2025
    assert policy["untouched_test_used_for_selection"] is False
    assert policy["untouched_test_used_for_metrics"] is False
    assert policy["untouched_test_used_for_fit"] is False

    rows = metadata["rows"]
    assert rows["model_selection_train"] > 0
    assert rows["validation_2024"] > 0
    assert rows["final_train_through_2024"] > 0
    assert (
        rows["final_train_through_2024"]
        >= rows["model_selection_train"]
    )

    label = metadata["label"]

    assert label["last_training_season"] == 2024, (
        f"{component}: final model did not end in 2024"
    )
    assert label["first_training_season"] <= 2024

    if component in BOUNDED_COMPONENTS:
        assert label["minimum"] >= -1e-12, (
            f"{component}: bounded label below zero"
        )
        assert label["maximum"] <= 1.0 + 1e-12, (
            f"{component}: bounded label above one"
        )

    if component in VOLUME_COMPONENTS:
        assert label["minimum"] >= -1e-12, (
            f"{component}: volume label below zero"
        )

    pred = metadata["prediction"]

    if component in BOUNDED_COMPONENTS:
        assert pred["transform"] == "clip_0_1"
        assert pred["clip_min"] == 0.0
        assert pred["clip_max"] == 1.0
        assert manifest["prediction_transform"] == "clip_0_1"
        assert manifest["clip_min"] == 0.0
        assert manifest["clip_max"] == 1.0

    if component in VOLUME_COMPONENTS:
        assert pred["transform"] == "floor_zero"
        assert pred["clip_min"] == 0.0
        assert pred["clip_max"] is None
        assert manifest["prediction_transform"] == "floor_zero"
        assert manifest["clip_min"] == 0.0
        assert manifest["clip_max"] is None

    if component in SHARE_COMPONENTS:
        assert (
            pred["reconcile_during_current_week_allocation"]
            is True
        ), f"{component}: share reconciliation flag missing"
        assert (
            manifest["reconcile_during_current_week_allocation"]
            is True
        ), f"{component}: manifest share reconciliation flag missing"
    else:
        assert (
            pred["reconcile_during_current_week_allocation"]
            is False
        )

    best_iteration = metadata["best_iteration_selected_on_2024"]
    assert isinstance(best_iteration, int) and best_iteration >= 1

    metrics = metadata["validation_2024_metrics"]
    for metric_name in ["rmse", "mae"]:
        value = float(metrics[metric_name])
        assert math.isfinite(value) and value >= 0.0, (
            f"{component}: invalid {metric_name}={value}"
        )

    r2 = metrics["r2"]
    if r2 is not None:
        assert math.isfinite(float(r2))

    params = metadata["params"]
    assert params["objective"] == "regression"
    assert params["boosting_type"] == "gbdt"
    assert params["deterministic"] is True
    assert params["num_threads"] == 1
    assert params["seed"] == 22022

    summary[component] = {
        "features": len(features),
        "trees": booster.num_trees(),
        "best_iteration": best_iteration,
        "validation_rmse": float(metrics["rmse"]),
        "validation_mae": float(metrics["mae"]),
    }

print("CHECK 04: static 2025 exclusion and fixed training cutoff")
source = TRAINER_PATH.read_text(encoding="utf-8")

required_source_markers = [
    "MODEL_SELECTION_TRAIN_END = 2023",
    "DEVELOPMENT_VALIDATION_SEASON = 2024",
    "FINAL_TRAIN_END = 2024",
    "UNTOUCHED_TEST_SEASON = 2025",
    '.le(FINAL_TRAIN_END)',
    "untouched_test_used_for_selection",
    "untouched_test_used_for_metrics",
    "untouched_test_used_for_fit",
    "reconcile_during_current_week_allocation",
    "automatic_all_numeric_selection",
]

for marker in required_source_markers:
    assert marker in source, (
        f"Trainer source missing policy marker: {marker}"
    )

print("CHECK 05: all 12 components independently validated")
for component in COMPONENTS:
    s = summary[component]
    print(
        f"{component}: "
        f"features={s['features']}, "
        f"trees={s['trees']}, "
        f"best_iteration={s['best_iteration']}, "
        f"rmse={s['validation_rmse']:.6f}, "
        f"mae={s['validation_mae']:.6f}"
    )

print(f"components={len(summary)}")
print("model_type=lightgbm_regression")
print("selection=train_through_2023_validate_2024")
print("final_fit=train_through_2024")
print("untouched_test=2025")
print("ISSUE 22 ACCEPTANCE: PASS")
