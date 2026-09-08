#!/usr/bin/env python3
"""
Independent acceptance validator for NFL Prop Engine Issue 23.

Validates:
- all 10 required efficiency model directories/artifacts
- LightGBM model loadability and exact model/manifest feature order
- explicit numeric feature selection only
- no target/audit/played_game_flag leakage
- no forbidden market features
- no old-team opportunity-share features
- player -> position -> league shrinkage contract
- rookies use position prior then league fallback
- player history keyed by player_id, not team
- trades preserve efficiency history
- stronger shrinkage for ALL TD efficiencies vs corresponding yardage models
- stronger shrinkage for sacks vs tackle rate
- exact-PBP policy for goal-line rushing TD and red-zone receiving TD
- goal line definition <= 5 and red zone definition <= 20
- 2023 -> 2024 model-selection policy
- final fit through 2024
- 2025 untouched, including priors
- rate models clip predictions to [0,1]
- yardage models preserve signed outputs
- artifact SHA256 integrity
- deterministic LightGBM settings
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

MODEL_ROOT = PROP / "models/efficiency"
TRAINER = PROP / "scripts/train/train_efficiency_models.py"
CANONICAL_MANIFEST = (
    PROP / "data/historical/features/feature_manifest.json"
)
CONFIG = PROP / "config/prop_engine.yaml"

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

PROHIBITED_SHARE_TOKENS = (
    "carry_share",
    "target_share",
    "air_yards_share",
    "red_zone_target_share",
    "red_zone_carry_share",
    "goal_line_carry_share",
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


print("CHECK 01: required trainer and model directories")
assert TRAINER.is_file(), f"Missing trainer: {TRAINER}"
assert MODEL_ROOT.is_dir(), f"Missing model root: {MODEL_ROOT}"

actual = sorted(
    p.name for p in MODEL_ROOT.iterdir()
    if p.is_dir() and p.name in MODELS
)
assert actual == sorted(MODELS), (
    f"Required efficiency model directories mismatch: {actual}"
)

print("CHECK 02: shared manifests/config")
canonical = load_json(CANONICAL_MANIFEST)

with CONFIG.open("r", encoding="utf-8-sig") as f:
    config = yaml.safe_load(f)

canonical_features = set(canonical["feature_columns"])
canonical_numeric = set(canonical["numeric_features"])
forbidden = list(config["forbidden_features"])
training = config["training"]
EXPECTED_SEED = int(training["random_seed"])
EXPECTED_SELECTION_END = int(training["model_selection_train_end_season"])
EXPECTED_VALIDATION_SEASON = int(training["development_validation_season"])
EXPECTED_FINAL_TRAIN_END = int(training["final_train_end_season"])
EXPECTED_UNTOUCHED_TEST_SEASON = int(training["untouched_test_season"])

summary = {}
shrinkage = {}

print("CHECK 03: validate artifacts, features, and LightGBM models")

for model_name in MODELS:
    root = MODEL_ROOT / model_name
    model_path = root / "model.txt"
    manifest_path = root / "feature_manifest.json"
    metadata_path = root / "metadata.json"

    assert model_path.is_file(), f"{model_name}: missing model.txt"
    assert manifest_path.is_file(), f"{model_name}: missing feature_manifest.json"
    assert metadata_path.is_file(), f"{model_name}: missing metadata.json"

    manifest = load_json(manifest_path)
    metadata = load_json(metadata_path)

    assert manifest["model"] == model_name
    assert metadata["model"] == model_name

    assert manifest["model_type"] == "lightgbm_regression"
    assert metadata["model_type"] == "lightgbm_regression"
    assert manifest["objective"] == "regression"

    features = manifest["numeric_features"]
    canonical_used = manifest["canonical_features"]
    derived = manifest["derived_features"]

    assert isinstance(features, list) and features
    assert len(features) == len(set(features))
    assert manifest["feature_count"] == len(features)
    assert metadata["feature_count"] == len(features)
    assert manifest["categorical_features"] == []
    assert manifest["automatic_all_numeric_selection"] is False

    assert set(features) == set(canonical_used) | set(derived)
    assert not (set(canonical_used) & set(derived))

    missing = sorted(set(canonical_used) - canonical_features)
    assert not missing, (
        f"{model_name}: canonical features not present in Issue 17 manifest: {missing}"
    )

    nonnumeric = sorted(set(canonical_used) - canonical_numeric)
    assert not nonnumeric, (
        f"{model_name}: nonnumeric feature in canonical numeric set: {nonnumeric}"
    )

    bad = [
        x for x in features
        if x.startswith("target_")
        or x.startswith("audit_")
        or x == "played_game_flag"
    ]
    assert not bad, f"{model_name}: prohibited feature(s): {bad}"

    shares = [
        x for x in features
        if any(token in x for token in PROHIBITED_SHARE_TOKENS)
    ]
    assert not shares, (
        f"{model_name}: old-team opportunity-share feature(s) found: {shares}"
    )

    for feature in features:
        lowered = feature.casefold()
        hits = [
            token for token in forbidden
            if str(token).casefold() in lowered
        ]
        assert not hits, (
            f"{model_name}: forbidden market feature {feature}: {hits}"
        )

    assert manifest["forbidden_features"] == forbidden

    print(f"  loading {model_name}/model.txt")
    booster = lgb.Booster(model_file=str(model_path))

    assert booster.feature_name() == features, (
        f"{model_name}: model feature order differs from manifest"
    )
    assert booster.num_feature() == len(features)
    assert booster.num_trees() >= 1

    assert metadata["model_sha256"] == sha256(model_path)
    assert (
        metadata["feature_manifest_sha256"]
        == sha256(manifest_path)
    )

    policy = metadata["training_policy"]
    assert policy["random_split_used"] is False
    assert policy["model_selection_train_end_season"] == EXPECTED_SELECTION_END
    assert policy["development_validation_season"] == EXPECTED_VALIDATION_SEASON
    assert policy["final_train_end_season"] == EXPECTED_FINAL_TRAIN_END
    assert policy["untouched_test_season"] == EXPECTED_UNTOUCHED_TEST_SEASON
    assert policy["untouched_test_used_for_selection"] is False
    assert policy["untouched_test_used_for_metrics"] is False
    assert policy["untouched_test_used_for_fit"] is False
    assert policy["untouched_test_used_for_priors"] is False

    rows = metadata["rows"]
    assert rows["model_selection_train"] > 0
    assert rows["validation_2024"] > 0
    assert rows["final_train_through_2024"] > 0
    assert (
        rows["final_train_through_2024"]
        >= rows["model_selection_train"]
    )

    label = metadata["label"]
    assert label["last_training_season"] == EXPECTED_FINAL_TRAIN_END

    if model_name in RATE_MODELS:
        assert label["minimum"] >= -1e-12
        assert label["maximum"] <= 1.0 + 1e-12
        assert manifest["prediction_transform"] == "clip_0_1"

    if model_name in YARDAGE_MODELS:
        assert manifest["prediction_transform"] == "identity_signed"

    shrink = manifest["shrinkage"]
    assert shrink["hierarchy"] == ["player", "position", "league"]
    assert shrink["strictly_prior_kickoff"] is True
    assert (
        shrink["rookie_fallback"]
        == "position_prior_then_league_prior"
    )
    assert shrink["player_history_key"] == "player_id"
    assert shrink["team_resets_player_efficiency"] is False

    shrinkage[model_name] = float(shrink["prior_exposure"])

    trade = manifest["trade_policy"]
    assert trade["preserve_player_efficiency_history"] is True
    assert trade["old_team_opportunity_share_features_allowed"] is False

    metadata_shrink = metadata["shrinkage"]
    assert metadata_shrink["hierarchy"] == "player_to_position_to_league"
    assert metadata_shrink["strictly_prior_kickoff"] is True
    assert metadata_shrink["rookies_use_position_prior"] is True
    assert metadata_shrink["player_history_key"] == "player_id"
    assert metadata_shrink["team_change_resets_efficiency"] is False
    assert (
        float(metadata_shrink["prior_exposure"])
        == shrinkage[model_name]
    )

    final_priors = metadata_shrink["final_priors_through_2024"]
    assert final_priors["through_season"] == 2024
    assert final_priors["league"]["exposure"] > 0.0
    assert len(final_priors["positions"]) >= 1

    metadata_trade = metadata["trade_policy"]
    assert metadata_trade["preserve_player_efficiency"] is True
    assert metadata_trade["old_team_opportunity_share"] is False
    assert metadata_trade["share_features_in_model"] is False

    conditional = metadata["conditional_td_label_policy"]

    if model_name == "rushing_td_per_goal_line_carry":
        assert conditional["exact_pbp_touchdown_flags"] is True
        assert conditional["goal_line_definition"] == "yardline_100 <= 5"

    elif model_name == "receiving_td_per_red_zone_target":
        assert conditional["exact_pbp_touchdown_flags"] is True
        assert conditional["red_zone_definition"] == "yardline_100 <= 20"

    else:
        assert conditional["exact_pbp_touchdown_flags"] is False

    best = metadata["best_iteration_selected_on_2024"]
    assert isinstance(best, int) and best >= 1

    metrics = metadata["validation_2024_metrics"]
    for metric in ["rmse", "mae"]:
        value = float(metrics[metric])
        assert math.isfinite(value) and value >= 0.0

    if metrics["r2"] is not None:
        assert math.isfinite(float(metrics["r2"]))

    params = metadata["params"]
    assert params["objective"] == "regression"
    assert params["boosting_type"] == "gbdt"
    assert params["deterministic"] is True
    assert params["num_threads"] == 1
    assert params["seed"] == EXPECTED_SEED

    summary[model_name] = {
        "features": len(features),
        "trees": booster.num_trees(),
        "best_iteration": best,
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
    }

print("CHECK 04: stronger TD and sack shrinkage")

assert (
    shrinkage["passing_td_rate"]
    > shrinkage["passing_yards_per_attempt"]
), "Passing TD shrinkage is not stronger than passing-yardage shrinkage"

assert (
    shrinkage["rushing_td_per_goal_line_carry"]
    > shrinkage["rushing_yards_per_carry"]
), "Rushing TD shrinkage is not stronger than rushing-yardage shrinkage"

assert (
    shrinkage["receiving_td_per_red_zone_target"]
    > shrinkage["receiving_yards_per_target"]
), "Receiving TD shrinkage is not stronger than receiving-yardage shrinkage"

assert (
    shrinkage["sack_rate_per_defensive_play"]
    > shrinkage["tackle_rate_per_defensive_play"]
), "Sack shrinkage is not stronger than tackle-rate shrinkage"

print(
    "  passing: "
    f"{shrinkage['passing_yards_per_attempt']} -> "
    f"{shrinkage['passing_td_rate']}"
)
print(
    "  rushing: "
    f"{shrinkage['rushing_yards_per_carry']} -> "
    f"{shrinkage['rushing_td_per_goal_line_carry']}"
)
print(
    "  receiving: "
    f"{shrinkage['receiving_yards_per_target']} -> "
    f"{shrinkage['receiving_td_per_red_zone_target']}"
)
print(
    "  defense: "
    f"{shrinkage['tackle_rate_per_defensive_play']} -> "
    f"{shrinkage['sack_rate_per_defensive_play']}"
)

print("CHECK 05: static trainer policy / trade semantics")

source = TRAINER.read_text(encoding="utf-8")

markers = [
    '_CONFIG_CONTRACT = common.load_config()',
    '_TRAINING_CONTRACT = _CONFIG_CONTRACT["training"]',
    'MODEL_SELECTION_TRAIN_END = int(_TRAINING_CONTRACT["model_selection_train_end_season"])',
    'DEVELOPMENT_VALIDATION_SEASON = int(_TRAINING_CONTRACT["development_validation_season"])',
    'FINAL_TRAIN_END = int(_TRAINING_CONTRACT["final_train_end_season"])',
    'UNTOUCHED_TEST_SEASON = int(_TRAINING_CONTRACT["untouched_test_season"])',
    '"rushing_td_per_goal_line_carry": 100.0',
    '"receiving_td_per_red_zone_target": 100.0',
    '"passing_td_rate": 200.0',
    '"sack_rate_per_defensive_play": 500.0',
    '["player_id", "kickoff_timestamp"]',
    '"_prior_position_group"',
    '"kickoff_timestamp"',
    "position_prior_then_league_prior",
    "team_resets_player_efficiency",
    "old_team_opportunity_share_features_allowed",
    "yardline_100",
    "rush_touchdown",
    "pass_touchdown",
]

for marker in markers:
    assert marker in source, f"Trainer missing policy marker: {marker}"

# The prior-history calculation must not group player history by team.
assert '["player_id", "team"' not in source
assert '["team", "player_id"' not in source

print("CHECK 06: all 10 models independently validated")

for model_name in MODELS:
    s = summary[model_name]
    print(
        f"{model_name}: "
        f"features={s['features']}, "
        f"trees={s['trees']}, "
        f"best_iteration={s['best_iteration']}, "
        f"rmse={s['rmse']:.6f}, "
        f"mae={s['mae']:.6f}, "
        f"shrinkage={shrinkage[model_name]:.1f}"
    )

print(f"models={len(summary)}")
print("model_type=lightgbm_regression")
print("shrinkage=player_to_position_to_league")
print("rookies=position_prior_then_league")
print("trades=preserve_player_efficiency")
print("old_team_opportunity_share=forbidden")
print("selection=train_through_2023_validate_2024")
print("final_fit=train_through_2024")
print("untouched_test=2025")
print("ISSUE 23 ACCEPTANCE: PASS")
