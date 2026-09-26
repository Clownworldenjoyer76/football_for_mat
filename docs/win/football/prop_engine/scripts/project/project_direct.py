#!/usr/bin/env python3
"""
Run current-week direct target projections for NFL Prop Engine Issue 35.

READS
-----
- data/current/features/{season}_week_{week}_features.parquet
- data/current/{season}_week_{week}_roles.parquet
- data/current/{season}_week_{week}_universe.parquet
- data/current/{season}_week_{week}_allocated_opportunity.parquet  (sequence gate)
- config/target_eligibility.yaml
- models/{target}/direct_model.txt
- models/{target}/feature_manifest.json
- models/{target}/metadata.json

WRITES
------
- data/current/{season}_week_{week}_direct_projections.parquet
- logs/direct_projections_{season}_week_{week}.json

POLICY
------
- Score only current target/player combinations satisfying the canonical target
  eligibility position + current-role requirement.
- Ineligible target/player combinations remain null.
- All scored direct outputs are floored at zero, including regression models.
- Predictions are never rounded internally.
- Persisted feature schema hash, model hash, categorical levels, and LightGBM
  feature order are verified before scoring.
- No target columns or forbidden market-derived inputs are consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    import lightgbm as lgb
except ModuleNotFoundError as exc:
    raise SystemExit("Issue 35 requires LightGBM in the active environment.") from exc

# noinspection DuplicatedCode
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common


_CONFIG_CONTRACT = common.load_config()
GRAIN = ["season", "week", "game_id", "player_id"]
TARGETS = list(_CONFIG_CONTRACT["targets"].keys())
OUTPUT_MAP = {target: f"direct_{target}" for target in TARGETS}
OUTPUT_COLUMNS = [*GRAIN, *[OUTPUT_MAP[t] for t in TARGETS]]

EXPECTED_CURRENT_REQUIREMENT = {
    "passing_yards": "identified_qb_role",
    "passing_tds": "identified_qb_role",
    "rushing_yards": "recent_usage_or_plausible_depth_or_participation_role",
    "rushing_tds": "recent_usage_or_plausible_depth_or_participation_role",
    "receiving_yards": "recent_usage_or_plausible_depth_or_participation_role",
    "receiving_tds": "recent_usage_or_plausible_depth_or_participation_role",
    "kicking_points": "identified_primary_kicker_role",
    "tackles": "recent_defensive_participation_or_current_starter_or_promotion_role",
    "sacks": "recent_defensive_participation_or_current_starter_or_promotion_role",
}

ROLE_REQUIRED = [
    *GRAIN,
    "team",
    "position",
    "depth_rank",
    "starter_flag",
    "primary_qb_flag",
    "primary_kicker_flag",
    "primary_role_flag",
    "committee_role_flag",
]
UNIVERSE_REQUIRED = [
    *GRAIN,
    "team",
    "position",
    "injury_game_status",
    "eligibility_status",
]
FEATURE_CONTEXT_REQUIRED = [
    *GRAIN,
    "team",
    "opponent",
    "position",
    "role_depth_starter_flag_pregame",
    "role_prior_offense_snap_pct",
    "role_snap_pct_roll3",
    "role_snap_pct_roll5",
    "role_prior_offense_participation",
    "role_participation_roll3",
    "role_participation_roll5",
    "role_prior_defense_participation",
    "role_starter_promotion_flag",
    "role_defensive_starter_flag",
]

RUSH_USAGE_COLUMNS = [
    "player_carries_lag1",
    "player_carries_roll3_mean",
    "player_carries_roll5_mean",
    "player_carry_share_lag1",
    "player_carry_share_roll3_mean",
]
RECEIVE_USAGE_COLUMNS = [
    "player_targets_lag1",
    "player_targets_roll3_mean",
    "player_targets_roll5_mean",
    "player_target_share_lag1",
    "player_target_share_roll3_mean",
]
OFFENSE_PARTICIPATION_COLUMNS = [
    "role_prior_offense_snap_pct",
    "role_snap_pct_roll3",
    "role_snap_pct_roll5",
    "role_prior_offense_participation",
    "role_participation_roll3",
    "role_participation_roll5",
]
DEF_PARTICIPATION_COLUMNS = [
    "role_prior_defense_participation",
    "player_defensive_def_participation_lag1",
    "player_defensive_def_participation_roll3",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run current-week direct target projections.")
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--week", type=int, required=True)
    return p.parse_args()



def run_market_preflight() -> dict[str, Any]:
    return common.run_market_exclusion_audit(
        missing_message="Issue 28 market validator missing: {path}",
        failure_prefix=(
            "Market-exclusion preflight failed before "
            "direct projection. "
        ),
    )


def flag(series: pd.Series) -> pd.Series:
    return common.safe_numeric_float64(series).fillna(0.0).gt(0.0)


def any_positive(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing current eligibility signal(s): {missing}")
    result = pd.Series(False, index=frame.index, dtype=bool)
    for c in columns:
        result |= common.safe_numeric_float64(frame[c]).fillna(0.0).gt(0.0)
    return result


def model_matrix(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    levels: dict[str, list[str]],
) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    for feature in numeric_features:
        data[feature] = common.safe_numeric_float64(frame[feature])
    for feature in categorical_features:
        if feature not in levels or not isinstance(levels[feature], list):
            raise ValueError(f"Missing persisted categorical levels for {feature}")
        values = common.clean_category_series(frame[feature])
        category = pd.Categorical(
            values.where(values.ne(""), None),
            categories=[str(v) for v in levels[feature]],
            ordered=False,
        )
        data[feature] = pd.Series(category.codes.astype("int32"), index=frame.index)
    return pd.DataFrame(
        data,
        index=frame.index,
        columns=[*numeric_features, *categorical_features],
    )


def feature_hash(numeric_features: list[str], categorical_features: list[str]) -> str:
    schema = [
        *[{"name": f, "type": "numeric"} for f in numeric_features],
        *[{"name": f, "type": "categorical"} for f in categorical_features],
    ]
    payload = json.dumps(
        schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda: h.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_current_context(
    features: pd.DataFrame,
    roles: pd.DataFrame,
    universe_all: pd.DataFrame,
    season: int,
    week: int,
) -> pd.DataFrame:
    common.require_columns(features, FEATURE_CONTEXT_REQUIRED, "Issue 31 current features")
    common.require_columns(roles, ROLE_REQUIRED, "Issue 30 current roles")
    common.require_columns(universe_all, UNIVERSE_REQUIRED, "Issue 29 current universe")
    common.ensure_unique(features, GRAIN, "Issue 31 current features")
    common.ensure_unique(roles, GRAIN, "Issue 30 current roles")
    common.ensure_unique(universe_all, GRAIN, "Issue 29 current universe")

    for frame, label in [(features, "features"), (roles, "roles"), (universe_all, "universe")]:
        common.require_current_week_frame(
            frame,
            season=season,
            week=week,
            mismatch_message=(
                f"Issue 35 {label} season/week mismatch"
            ),
        )

    eligible_universe = universe_all.loc[
        universe_all["eligibility_status"].fillna("").astype(str).str.strip().str.casefold().eq("eligible")
    ].copy()
    if eligible_universe.empty:
        raise ValueError("Issue 35 eligible current universe is empty")
    if eligible_universe["injury_game_status"].fillna("").astype(str).str.strip().str.casefold().eq("out").any():
        raise ValueError("Issue 35 eligible current universe unexpectedly contains an Out player")
    common.ensure_unique(eligible_universe, GRAIN, "Issue 35 eligible universe")

    keyset = lambda df: set(map(tuple, df[GRAIN].to_numpy()))
    expected = keyset(eligible_universe)
    if keyset(features) != expected or keyset(roles) != expected:
        raise ValueError("Issue 35 features/roles grain does not exactly match eligible current universe")

    role_add = roles[
        [
            *GRAIN, "team", "position", "depth_rank", "starter_flag",
            "primary_qb_flag", "primary_kicker_flag", "primary_role_flag",
            "committee_role_flag",
        ]
    ].rename(columns={"team": "_role_team", "position": "_role_position"})
    uni_add = eligible_universe[[*GRAIN, "team", "position"]].rename(
        columns={"team": "_universe_team", "position": "_universe_position"}
    )
    work = features.merge(role_add, on=GRAIN, how="left", validate="one_to_one")
    work = work.merge(uni_add, on=GRAIN, how="left", validate="one_to_one")
    if len(work) != len(features):
        raise ValueError("Issue 35 context join changed row count")

    pos = common.normalize_position_series(work["position"])
    bad = (
        work["team"].astype(str).ne(work["_role_team"].astype(str))
        | work["team"].astype(str).ne(work["_universe_team"].astype(str))
        | pos.ne(common.normalize_position_series(work["_role_position"]))
        | pos.ne(common.normalize_position_series(work["_universe_position"]))
    )
    if bad.any():
        raise ValueError(
            "Issue 35 current feature/role/universe context mismatch; sample="
            f"{work.loc[bad, [*GRAIN, 'team', 'position', '_role_team', '_role_position', '_universe_team', '_universe_position']].head(10).to_dict('records')}"
        )
    return work


def eligibility_mask(
    work: pd.DataFrame,
    target: str,
    eligibility: dict[str, Any],
    manifest: dict[str, Any],
) -> pd.Series:
    if target not in eligibility:
        raise ValueError(f"{target}: missing target eligibility rule")
    rule = eligibility[target]
    if not isinstance(rule, dict):
        raise ValueError(f"{target}: target eligibility rule must be mapping")
    expected_requirement = EXPECTED_CURRENT_REQUIREMENT[target]
    if str(rule.get("current_requirement", "")) != expected_requirement:
        raise ValueError(
            f"{target}: current_requirement changed; expected {expected_requirement!r}, "
            f"found {rule.get('current_requirement')!r}"
        )
    positions = [str(x).strip().upper() for x in rule.get("eligible_positions", [])]
    manifest_positions = [str(x).strip().upper() for x in manifest.get("eligible_positions", [])]
    if positions != manifest_positions or not positions:
        raise ValueError(f"{target}: eligible positions disagree between eligibility config and model manifest")

    pos_ok = common.normalize_position_series(work["position"]).isin(set(positions))

    if expected_requirement == "identified_qb_role":
        role_ok = flag(work["primary_qb_flag"])
    elif expected_requirement == "identified_primary_kicker_role":
        role_ok = flag(work["primary_kicker_flag"])
    elif expected_requirement == "recent_usage_or_plausible_depth_or_participation_role":
        usage_cols = RUSH_USAGE_COLUMNS if target.startswith("rushing_") else RECEIVE_USAGE_COLUMNS
        recent_usage = any_positive(work, usage_cols)
        participation = any_positive(work, OFFENSE_PARTICIPATION_COLUMNS)
        depth = common.safe_numeric_float64(work["depth_rank"])
        plausible_depth = depth.notna() & depth.le(3.0)
        current_role = (
            flag(work["starter_flag"])
            | flag(work["primary_role_flag"])
            | flag(work["committee_role_flag"])
            | flag(work["role_depth_starter_flag_pregame"])
        )
        role_ok = recent_usage | participation | plausible_depth | current_role
    elif expected_requirement == "recent_defensive_participation_or_current_starter_or_promotion_role":
        recent_def = any_positive(work, DEF_PARTICIPATION_COLUMNS)
        current_def = (
            flag(work["role_defensive_starter_flag"])
            | flag(work["role_starter_promotion_flag"])
            | flag(work["starter_flag"])
            | flag(work["primary_role_flag"])
        )
        role_ok = recent_def | current_def
    else:
        raise ValueError(f"{target}: unsupported current eligibility requirement {expected_requirement}")

    return (pos_ok & role_ok).astype(bool)


def validate_and_load_model(
    prop: Path,
    target: str,
    available_columns: set[str],
) -> tuple[lgb.Booster, dict[str, Any], dict[str, Any], list[str], list[str], dict[str, list[str]]]:
    model_dir = prop / "models" / target
    model_path = model_dir / "direct_model.txt"
    manifest_path = model_dir / "feature_manifest.json"
    metadata_path = model_dir / "metadata.json"
    for p in [model_path, manifest_path, metadata_path]:
        if not p.is_file():
            raise FileNotFoundError(f"{target}: required direct-model artifact missing: {p}")

    manifest = common.load_json_mapping(manifest_path)
    metadata = common.load_json_mapping(metadata_path)
    if str(manifest.get("target")) != target or str(metadata.get("target")) != target:
        raise ValueError(f"{target}: target mismatch in direct-model artifacts")
    if str(metadata.get("status")) != "trained":
        raise ValueError(f"{target}: direct model metadata is not trained")
    if bool(manifest.get("market_features_used", True)) or bool(metadata.get("market_features_used", True)):
        raise ValueError(f"{target}: direct-model artifact reports market features used")

    numeric_features = list(manifest.get("numeric_features", []))
    categorical_features = list(manifest.get("categorical_features", []))
    feature_names = [*numeric_features, *categorical_features]
    common.reject_forbidden_feature_columns(feature_names, common.load_config())
    if not feature_names or len(feature_names) != len(set(feature_names)):
        raise ValueError(f"{target}: invalid/duplicate direct feature list")
    if int(manifest.get("feature_count", -1)) != len(feature_names):
        raise ValueError(f"{target}: manifest feature_count mismatch")
    missing = [c for c in feature_names if c not in available_columns]
    if missing:
        raise ValueError(f"{target}: missing required current direct-model feature(s): {missing[:40]}")
    if any(c.startswith("target_") or c.startswith("audit_") or c == "played_game_flag" for c in feature_names):
        raise ValueError(f"{target}: prohibited target/audit/outcome feature in direct manifest")

    computed_hash = feature_hash(numeric_features, categorical_features)
    if computed_hash != str(manifest.get("feature_hash")):
        raise ValueError(f"{target}: direct feature schema hash mismatch")
    if computed_hash != str(metadata.get("feature_hash")):
        raise ValueError(f"{target}: metadata feature schema hash mismatch")

    primary = metadata.get("primary", {})
    if not isinstance(primary, dict):
        raise ValueError(f"{target}: missing primary direct-model metadata")
    stored_model_hash = str(primary.get("model_sha256", ""))
    if not stored_model_hash or sha256_file(model_path) != stored_model_hash:
        raise ValueError(f"{target}: persisted direct model hash mismatch")

    levels = manifest.get("categorical_levels_final_through_2024", {})
    if not isinstance(levels, dict):
        raise ValueError(f"{target}: categorical level contract missing")
    for c in categorical_features:
        if c not in levels or not isinstance(levels[c], list):
            raise ValueError(f"{target}: missing final categorical levels for {c}")

    booster = lgb.Booster(model_file=str(model_path))
    if list(booster.feature_name()) != feature_names:
        raise ValueError(f"{target}: persisted LightGBM feature order mismatch")
    return booster, manifest, metadata, numeric_features, categorical_features, levels


def main() -> int:
    args = parse_args()
    config = common.load_config()
    season, week = common.resolve_projection_season_week(
        args.season,
        args.week,
        config,
    )

    prop = common.prop_root()
    market = run_market_preflight()

    context_paths = common.current_projection_context_paths(
        season,
        week,
    )
    features_path = context_paths["features"]
    roles_path = context_paths["roles"]
    universe_path = context_paths["universe"]
    allocated_path = prop / "data" / "current" / f"{season}_week_{week}_allocated_opportunity.parquet"
    issue34_log_path = prop / "logs" / f"allocated_opportunity_{season}_week_{week}.json"
    eligibility_path = context_paths["eligibility"]
    output_path = prop / "data" / "current" / f"{season}_week_{week}_direct_projections.parquet"
    log_path = prop / "logs" / f"direct_projections_{season}_week_{week}.json"

    common.require_existing_files(
        [
            features_path,
            roles_path,
            universe_path,
            allocated_path,
            issue34_log_path,
            eligibility_path,
        ],
        missing_message=(
            "Issue 35 required input/sequence artifact missing: {path}"
        ),
    )

    features, roles, universe = common.read_current_projection_frames(
        context_paths
    )
    allocated = pd.read_parquet(allocated_path)
    issue34_log = common.load_json_mapping(issue34_log_path)
    eligibility = common.load_yaml_mapping(eligibility_path)

    if str(issue34_log.get("status")) != "passed":
        raise ValueError("Issue 35 requires passed Issue 34 allocation log")
    common.ensure_unique(allocated, GRAIN, "Issue 34 allocated opportunity")
    if set(map(tuple, allocated[GRAIN].to_numpy())) != set(map(tuple, features[GRAIN].to_numpy())):
        raise ValueError("Issue 35 allocation sequence-gate grain differs from current features")

    common.reject_forbidden_feature_columns(features.columns, config)
    if any(str(c).startswith("target_") for c in features.columns):
        raise ValueError("Issue 35 current feature table contains target columns")

    work = build_current_context(features, roles, universe, season, week)
    output = work[GRAIN].copy()
    for target in TARGETS:
        output[OUTPUT_MAP[target]] = np.nan

    model_audit: dict[str, Any] = {}
    eligibility_counts: dict[str, int] = {}
    for target in TARGETS:
        booster, manifest, metadata, numeric_features, categorical_features, levels = validate_and_load_model(
            prop, target, set(work.columns)
        )
        common.reject_forbidden_feature_columns([*numeric_features, *categorical_features], config)
        mask = eligibility_mask(work, target, eligibility, manifest)
        eligible_rows = int(mask.sum())
        if eligible_rows <= 0:
            raise ValueError(f"{target}: no eligible current target/player combinations")

        rows = work.loc[mask].copy()
        model_input = model_matrix(rows, numeric_features, categorical_features, levels)
        pred = np.asarray(booster.predict(model_input), dtype="float64")
        if pred.shape[0] != len(rows) or not np.isfinite(pred).all():
            raise ValueError(f"{target}: invalid direct-model prediction")
        pred = np.maximum(pred, 0.0)
        output.loc[mask, OUTPUT_MAP[target]] = pred

        # Contract: only eligible rows are scored; ineligible rows remain null.
        if output.loc[~mask, OUTPUT_MAP[target]].notna().any():
            raise ValueError(f"{target}: ineligible direct projection row is non-null")
        if output.loc[mask, OUTPUT_MAP[target]].isna().any():
            raise ValueError(f"{target}: eligible direct projection row is null")
        if common.safe_numeric_float64(output.loc[mask, OUTPUT_MAP[target]]).lt(0.0).any():
            raise ValueError(f"{target}: negative direct projection after flooring")

        eligibility_counts[target] = eligible_rows
        model_audit[target] = {
            "eligible_rows_scored": eligible_rows,
            "feature_count": len(numeric_features) + len(categorical_features),
            "numeric_feature_count": len(numeric_features),
            "categorical_feature_count": len(categorical_features),
            "feature_hash": str(manifest.get("feature_hash")),
            "objective": str(metadata.get("objective")),
            "model_family": str(metadata.get("model_family")),
            "floor_zero_applied": True,
            "internal_rounding": False,
            "persisted_feature_order_checked": True,
            "persisted_model_hash_checked": True,
        }

    output = output[OUTPUT_COLUMNS].copy()
    common.ensure_unique(output, GRAIN, "Issue 35 direct projections")
    if len(output) != len(work):
        raise ValueError("Issue 35 output row count changed")
    common.reject_forbidden_feature_columns(output.columns, config)
    common.write_parquet_atomic(output, output_path)

    payload = {
        "script": Path(__file__).name,
        "status": "passed",
        "season": season,
        "week": week,
        "rows": int(len(output)),
        "games": int(work["game_id"].nunique()),
        "teams": int(work["team"].nunique()),
        "columns": len(OUTPUT_COLUMNS),
        "targets_scored": len(TARGETS),
        "eligible_rows_by_target": eligibility_counts,
        "ineligible_combinations_null": True,
        "floor_zero_applied": True,
        "internal_rounding": False,
        "market_exclusion_passed": bool(market["passed"]),
        "market_features_used": False,
        "output": common.repo_relative_path(output_path),
        "log": common.repo_relative_path(log_path),
    }
    log_payload = {
        **payload,
        "inputs": {
            "features": common.repo_relative_path(features_path),
            "roles": common.repo_relative_path(roles_path),
            "universe": common.repo_relative_path(universe_path),
            "allocated_opportunity_sequence_gate": common.repo_relative_path(allocated_path),
            "issue34_log": common.repo_relative_path(issue34_log_path),
            "target_eligibility": common.repo_relative_path(eligibility_path),
        },
        "models": model_audit,
        "policy": {
            "full_player_grain_preserved": True,
            "eligible_target_player_combinations_only": True,
            "ineligible_target_player_combinations_null": True,
            "nonnegative_floor": True,
            "no_internal_rounding": True,
            "missing_required_model_feature_fails": True,
            "persisted_model_feature_order_checked": True,
            "persisted_feature_schema_hash_checked": True,
            "persisted_model_hash_checked": True,
            "market_exclusion_preflight": True,
        },
    }
    common.write_json_default_str_atomic(
        log_path,
        log_payload,
        ensure_ascii=False,
    )
    print(json.dumps({"script": Path(__file__).name, "payload": payload}, sort_keys=True, separators=(",", ":")))
    print("DIRECT PROJECTIONS BUILD: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
