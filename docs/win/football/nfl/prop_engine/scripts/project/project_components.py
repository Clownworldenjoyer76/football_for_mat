#!/usr/bin/env python3
"""Run current-week opportunity and efficiency component projections.

Issue 33 contract
-----------------
READS:
  data/current/features/{season}_week_{week}_features.parquet
  data/current/{season}_week_{week}_roles.parquet
  data/current/{season}_week_1_priors.parquet (Week 1 sequence gate)
  data/historical/features/player_game_features.parquet
  models/components/*/{model.txt,feature_manifest.json}
  models/efficiency/*/{model.txt,feature_manifest.json}
  config/target_eligibility.yaml

WRITES:
  data/current/{season}_week_{week}_component_projections.parquet
  logs/component_projections_{season}_week_{week}.json

POLICY:
  - Score persisted Issue 22 opportunity models and Issue 23 efficiency models.
  - Missing model features or persisted feature-order mismatch hard-fail.
  - Component share predictions remain raw in Issue 33; reconciliation belongs
    to the current-week allocation stage required by Issue 22.
  - Team pass/rush volumes repeat at player grain. QB and kicker player-facing
    volumes are assigned only to Issue 30 primary roles.
  - Efficiency-only eff_* features are reconstructed model-by-model from
    strictly prior realized history; they are never written into Issue 31.
  - Week 1 requires the accepted explicit Issue 32 prior artifact as a sequence
    gate, but does not replace trained model inputs with invented priors.
  - No sportsbook/market-derived input is permitted.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
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
    raise SystemExit("Issue 33 requires LightGBM in the active environment.") from exc

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
TRAIN_DIR = SCRIPTS_ROOT / "train"
for p in (SCRIPTS_ROOT, TRAIN_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import common
import train_opportunity_models as opportunity
import train_efficiency_models as efficiency

GRAIN = ["season", "week", "game_id", "player_id"]
TEAM_GRAIN = ["season", "week", "game_id", "team"]
OUTPUT_COLUMNS = [
    "season", "week", "game_id", "player_id", "team", "opponent", "position",
    "projected_team_pass_attempts", "projected_qb_pass_attempts",
    "projected_team_rush_attempts", "projected_player_carries",
    "projected_target_share", "projected_targets", "projected_yards_per_attempt",
    "projected_yards_per_carry", "projected_yards_per_target",
    "projected_red_zone_targets", "projected_goal_line_carries",
    "projected_fg_attempts", "projected_fg_make_probability",
    "projected_pat_attempts",
]

OPPORTUNITY_NEEDED = [
    "qb_pass_attempts", "team_pass_attempts", "team_rush_attempts",
    "player_carry_share", "player_target_share", "player_red_zone_target_share",
    "player_goal_line_carry_share", "field_goal_attempts", "extra_point_attempts",
]
EFFICIENCY_NEEDED = [
    "passing_yards_per_attempt", "rushing_yards_per_carry",
    "receiving_yards_per_target", "field_goal_conversion",
]
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
ROLE_REQUIRED = [
    "season", "week", "game_id", "player_id", "team", "position",
    "primary_qb_flag", "primary_kicker_flag",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run current-week component projections.")
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--week", type=int, required=True)
    return p.parse_args()


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(common.repo_root().resolve()))


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required YAML missing: {path}")
    with path.open("r", encoding="utf-8-sig") as h:
        value = yaml.safe_load(h)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON missing: {path}")
    with path.open("r", encoding="utf-8-sig") as h:
        value = json.load(h)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", prefix=f".{path.name}.",
        suffix=".tmp", dir=path.parent, delete=False,
    )
    temp = Path(h.name)
    try:
        with h:
            json.dump(payload, h, indent=2, sort_keys=True, ensure_ascii=False, default=str)
            h.write("\n")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def run_market_preflight() -> dict[str, Any]:
    path = SCRIPTS_ROOT / "validate" / "audit_market_exclusion.py"
    if not path.is_file():
        raise FileNotFoundError(f"Issue 28 market validator missing: {path}")
    cp = subprocess.run(
        [sys.executable, str(path)], cwd=common.repo_root(), capture_output=True,
        text=True, check=False,
    )
    if cp.returncode != 0 or "MARKET EXCLUSION AUDIT: PASS" not in cp.stdout:
        raise RuntimeError(
            "Market-exclusion preflight failed before component projection. "
            f"stdout={cp.stdout[-2000:]!r} stderr={cp.stderr[-2000:]!r}"
        )
    return {"passed": True, "validator": repo_relative(path)}


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")


def coalesce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing deterministic volume proxy columns: {missing}")
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    for c in columns:
        value = numeric(frame[c])
        out = out.where(out.notna(), value)
    return out


def validate_booster_manifest(root: Path, family: str, name: str) -> tuple[lgb.Booster, dict[str, Any], list[str]]:
    model_dir = root / "models" / family / name
    manifest = load_json(model_dir / "feature_manifest.json")
    model_path = model_dir / "model.txt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Required model missing: {model_path}")
    numeric_features = list(manifest.get("numeric_features", []))
    categorical = list(manifest.get("categorical_features", []))
    feature_names = numeric_features + categorical
    common.reject_forbidden_feature_columns(feature_names, common.load_config())
    if int(manifest.get("feature_count", len(feature_names))) != len(feature_names):
        raise ValueError(f"{family}/{name}: manifest feature_count mismatch")
    booster = lgb.Booster(model_file=str(model_path))
    if list(booster.feature_name()) != feature_names:
        raise ValueError(f"{family}/{name}: persisted LightGBM feature order mismatch")
    return booster, manifest, feature_names


def component_inference_rows(features: pd.DataFrame, component: str, eligibility: dict[str, Any]) -> pd.DataFrame:
    spec = opportunity.COMPONENTS[component]
    feature_names = list(spec["features"])
    missing = [c for c in feature_names if c not in features.columns]
    if missing:
        raise ValueError(f"{component}: missing current feature(s): {missing[:30]}")
    if spec["scope"] == "team":
        opportunity.check_team_feature_invariance(features, feature_names)
        return opportunity.team_rows_from_features(features, feature_names)
    rule = str(spec["eligible_rule"])
    positions = {str(x).strip().upper() for x in eligibility[rule]["eligible_positions"]}
    pos = features["position"].fillna("").astype(str).str.strip().str.upper()
    return features.loc[pos.isin(positions)].copy()


def score_opportunity(root: Path, features: pd.DataFrame, eligibility: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    predictions: dict[str, pd.DataFrame] = {}
    audits: dict[str, Any] = {}
    for component in OPPORTUNITY_NEEDED:
        booster, manifest, feature_names = validate_booster_manifest(root, "components", component)
        rows = component_inference_rows(features, component, eligibility)
        X = opportunity.numeric_frame(rows, feature_names)
        pred = opportunity.transform_prediction(booster.predict(X), component)
        if not np.isfinite(pred).all():
            raise ValueError(f"{component}: nonfinite persisted-model prediction")
        key = GRAIN if opportunity.COMPONENTS[component]["scope"] == "player" else TEAM_GRAIN
        out = rows[key].copy()
        out[component] = pred
        common.ensure_unique(out, key, f"Issue 33 {component}")
        predictions[component] = out
        audits[component] = {
            "scope": str(manifest.get("scope")), "rows": int(len(out)),
            "feature_count": len(feature_names),
            "transform": str(manifest.get("prediction_transform")),
        }
    return predictions, audits


def efficiency_columns() -> list[str]:
    cols = [*GRAIN, "kickoff_timestamp", "position", "position_group"]
    for model_name in EFFICIENCY_NEEDED:
        for f in efficiency.FEATURES[model_name]:
            if f not in efficiency.DERIVED_FEATURES and f not in cols:
                cols.append(f)
    return cols


def prepare_efficiency_history(config: dict[str, Any], hist: pd.DataFrame, eligibility: dict[str, Any]) -> dict[str, pd.DataFrame]:
    label_base = efficiency.prepare_label_base(config, hist)
    result: dict[str, pd.DataFrame] = {}
    for name in EFFICIENCY_NEEDED:
        raw = efficiency.build_component_label(label_base, name)
        raw = efficiency.apply_eligibility(raw, name, eligibility)
        result[name] = raw
    return result


def efficiency_inference_frame(current: pd.DataFrame, raw_history: pd.DataFrame, model_name: str, eligibility: dict[str, Any]) -> pd.DataFrame:
    rule = efficiency.ELIGIBILITY_RULE[model_name]
    positions = {str(x).strip().upper() for x in eligibility[rule]["eligible_positions"]}
    pos = current["position"].fillna("").astype(str).str.strip().str.upper()
    target = current.loc[pos.isin(positions)].copy()
    if target.empty:
        raise ValueError(f"{model_name}: no current eligible rows")

    prior_columns = [
        *GRAIN, "kickoff_timestamp", "position", "position_group",
        "_prior_position_group", "_numerator", "_exposure", "_label",
    ]
    missing = [c for c in prior_columns if c not in raw_history.columns]
    if missing:
        raise ValueError(f"{model_name}: raw prior history missing {missing}")
    history = raw_history[prior_columns].copy()
    history["_inference_marker"] = 0
    placeholder = target[[*GRAIN, "kickoff_timestamp", "position", "position_group"]].copy()
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

    canonical = [f for f in efficiency.FEATURES[model_name] if f not in efficiency.DERIVED_FEATURES]
    missing = [c for c in canonical if c not in target.columns]
    if missing:
        raise ValueError(f"{model_name}: missing current canonical efficiency features: {missing}")
    inference_rows = inference_rows.merge(
        target[[*GRAIN, *canonical]], on=GRAIN, how="left", validate="one_to_one",
        suffixes=("", "_canonical"),
    )
    common.ensure_unique(inference_rows, GRAIN, f"Issue 33 {model_name} inference")
    return inference_rows


def score_efficiency(root: Path, current: pd.DataFrame, history: pd.DataFrame, eligibility: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    raw_histories = prepare_efficiency_history(config, history, eligibility)
    predictions: dict[str, pd.DataFrame] = {}
    audits: dict[str, Any] = {}
    for name in EFFICIENCY_NEEDED:
        booster, manifest, feature_names = validate_booster_manifest(root, "efficiency", name)
        rows = efficiency_inference_frame(current, raw_histories[name], name, eligibility)
        expected = list(efficiency.FEATURES[name])
        if feature_names != expected:
            raise ValueError(f"{name}: manifest differs from trainer efficiency feature order")
        X = efficiency.feature_matrix(rows, name)
        pred = efficiency.transform_prediction(booster.predict(X), name)
        if not np.isfinite(pred).all():
            raise ValueError(f"{name}: nonfinite persisted-model prediction")
        out = rows[GRAIN].copy()
        out[name] = pred
        common.ensure_unique(out, GRAIN, f"Issue 33 {name}")
        predictions[name] = out
        audits[name] = {
            "rows": int(len(out)), "feature_count": len(feature_names),
            "derived_features": list(manifest.get("derived_features", [])),
            "strict_prior_history_through": int(pd.to_numeric(history["season"]).max()),
        }
    return predictions, audits


def merge_player_prediction(base: pd.DataFrame, pred: pd.DataFrame, source_col: str, output_col: str) -> pd.DataFrame:
    return base.merge(
        pred.rename(columns={source_col: output_col}), on=GRAIN,
        how="left", validate="one_to_one",
    )


def merge_team_prediction(base: pd.DataFrame, pred: pd.DataFrame, source_col: str, output_col: str) -> pd.DataFrame:
    return base.merge(
        pred.rename(columns={source_col: output_col}), on=TEAM_GRAIN,
        how="left", validate="many_to_one",
    )


def main() -> int:
    args = parse_args()
    config = common.load_config()
    season = int(args.season) if args.season is not None else int(config["seasons"]["current"])
    week = int(args.week)
    if not 1 <= week <= 25:
        raise ValueError(f"Invalid week: {week}")
    market = run_market_preflight()
    repo = common.repo_root()
    prop = common.prop_root()

    features_path = prop / "data" / "current" / "features" / f"{season}_week_{week}_features.parquet"
    roles_path = prop / "data" / "current" / f"{season}_week_{week}_roles.parquet"
    output_path = prop / "data" / "current" / f"{season}_week_{week}_component_projections.parquet"
    log_path = prop / "logs" / f"component_projections_{season}_week_{week}.json"
    eligibility_path = prop / "config" / "target_eligibility.yaml"
    historical_path = repo / config["paths"]["historical_features"]
    required_paths = [features_path, roles_path, eligibility_path, historical_path]
    if week == 1:
        required_paths.append(prop / "data" / "current" / f"{season}_week_1_priors.parquet")
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Issue 33 required input missing: {path}")

    features = pd.read_parquet(features_path)
    roles = pd.read_parquet(roles_path)
    common.require_columns(features, [*GRAIN, "team", "opponent", "position", "position_group", "kickoff_timestamp", *RED_ZONE_PASS_VOLUME_FEATURES, *GOAL_LINE_RUSH_VOLUME_FEATURES], "Issue 33 current features")
    common.require_columns(roles, ROLE_REQUIRED, "Issue 33 current roles")
    common.ensure_unique(features, GRAIN, "Issue 33 current features")
    common.ensure_unique(roles, GRAIN, "Issue 33 current roles")
    common.reject_forbidden_feature_columns(features.columns, config)
    if any(str(c).startswith("target_") for c in features.columns):
        raise ValueError("Issue 33 current features unexpectedly contain target columns")

    features = features.copy()
    features["season"] = pd.to_numeric(features["season"], errors="raise").astype(int)
    features["week"] = pd.to_numeric(features["week"], errors="raise").astype(int)
    features["kickoff_timestamp"] = pd.to_datetime(features["kickoff_timestamp"], errors="raise", utc=True)
    if set(features["season"]) != {season} or set(features["week"]) != {week}:
        raise ValueError("Issue 33 current feature season/week mismatch")

    eligibility = load_yaml(eligibility_path)
    root = prop
    opp_pred, opp_audit = score_opportunity(root, features, eligibility)

    hist_cols = efficiency_columns()
    history = pd.read_parquet(historical_path, columns=hist_cols)
    history["season"] = pd.to_numeric(history["season"], errors="raise").astype(int)
    history["week"] = pd.to_numeric(history["week"], errors="raise").astype(int)
    history["kickoff_timestamp"] = pd.to_datetime(history["kickoff_timestamp"], errors="raise", utc=True)
    history = history.loc[history["season"].lt(season)].copy()
    if history.empty:
        raise ValueError("Issue 33 has no strict-prior historical efficiency context")
    common.ensure_unique(history, GRAIN, "Issue 33 historical efficiency context")
    eff_pred, eff_audit = score_efficiency(root, features, history, eligibility, config)

    base = features[[*GRAIN, "team", "opponent", "position", *RED_ZONE_PASS_VOLUME_FEATURES, *GOAL_LINE_RUSH_VOLUME_FEATURES]].copy()
    for comp, out_col in [
        ("team_pass_attempts", "projected_team_pass_attempts"),
        ("team_rush_attempts", "projected_team_rush_attempts"),
        ("field_goal_attempts", "_team_fg_attempts"),
        ("extra_point_attempts", "_team_pat_attempts"),
    ]:
        base = merge_team_prediction(base, opp_pred[comp], comp, out_col)
    for comp, out_col in [
        ("qb_pass_attempts", "_raw_qb_pass_attempts"),
        ("player_carry_share", "_raw_carry_share"),
        ("player_target_share", "projected_target_share"),
        ("player_red_zone_target_share", "_raw_red_zone_target_share"),
        ("player_goal_line_carry_share", "_raw_goal_line_carry_share"),
    ]:
        base = merge_player_prediction(base, opp_pred[comp], comp, out_col)
    for name, out_col in [
        ("passing_yards_per_attempt", "projected_yards_per_attempt"),
        ("rushing_yards_per_carry", "projected_yards_per_carry"),
        ("receiving_yards_per_target", "projected_yards_per_target"),
        ("field_goal_conversion", "_raw_fg_make_probability"),
    ]:
        base = merge_player_prediction(base, eff_pred[name], name, out_col)

    role_cols = roles[[*GRAIN, "primary_qb_flag", "primary_kicker_flag"]].copy()
    base = base.merge(role_cols, on=GRAIN, how="left", validate="one_to_one")
    if base[["primary_qb_flag", "primary_kicker_flag"]].isna().any().any():
        raise ValueError("Issue 33 role merge missing primary-role flags")
    qb_primary = numeric(base["primary_qb_flag"]).fillna(0).gt(0)
    kicker_primary = numeric(base["primary_kicker_flag"]).fillna(0).gt(0)
    if int(qb_primary.sum()) != int(base["team"].nunique()):
        raise ValueError("Issue 33 requires exactly one primary QB per team")
    if int(kicker_primary.sum()) != int(base["team"].nunique()):
        raise ValueError("Issue 33 requires exactly one primary kicker per team")

    base["projected_qb_pass_attempts"] = 0.0
    base.loc[qb_primary, "projected_qb_pass_attempts"] = numeric(base.loc[qb_primary, "_raw_qb_pass_attempts"]).to_numpy()
    if base.loc[qb_primary, "projected_qb_pass_attempts"].isna().any():
        raise ValueError("Primary QB is missing qb_pass_attempts component prediction")

    base["_raw_carry_share"] = numeric(base["_raw_carry_share"]).fillna(0.0).clip(0.0, 1.0)
    base["projected_target_share"] = numeric(base["projected_target_share"]).fillna(0.0).clip(0.0, 1.0)
    base["_raw_red_zone_target_share"] = numeric(base["_raw_red_zone_target_share"]).fillna(0.0).clip(0.0, 1.0)
    base["_raw_goal_line_carry_share"] = numeric(base["_raw_goal_line_carry_share"]).fillna(0.0).clip(0.0, 1.0)
    base["projected_player_carries"] = numeric(base["projected_team_rush_attempts"]).clip(lower=0.0) * base["_raw_carry_share"]
    base["projected_targets"] = numeric(base["projected_team_pass_attempts"]).clip(lower=0.0) * base["projected_target_share"]
    rz_volume = coalesce_numeric(base, RED_ZONE_PASS_VOLUME_FEATURES).clip(lower=0.0)
    gl_volume = coalesce_numeric(base, GOAL_LINE_RUSH_VOLUME_FEATURES).clip(lower=0.0)
    base["projected_red_zone_targets"] = rz_volume * base["_raw_red_zone_target_share"]
    base["projected_goal_line_carries"] = gl_volume * base["_raw_goal_line_carry_share"]

    base["projected_fg_attempts"] = 0.0
    base.loc[kicker_primary, "projected_fg_attempts"] = numeric(base.loc[kicker_primary, "_team_fg_attempts"]).to_numpy()
    base["projected_pat_attempts"] = 0.0
    base.loc[kicker_primary, "projected_pat_attempts"] = numeric(base.loc[kicker_primary, "_team_pat_attempts"]).to_numpy()
    base["projected_fg_make_probability"] = np.nan
    base.loc[kicker_primary, "projected_fg_make_probability"] = numeric(base.loc[kicker_primary, "_raw_fg_make_probability"]).to_numpy()
    if base.loc[kicker_primary, ["projected_fg_attempts", "projected_pat_attempts", "projected_fg_make_probability"]].isna().any().any():
        raise ValueError("Primary kicker is missing field-goal/PAT component or efficiency prediction")
    base["projected_fg_make_probability"] = numeric(base["projected_fg_make_probability"]).clip(0.0, 1.0)

    nonnegative = [
        "projected_team_pass_attempts", "projected_qb_pass_attempts",
        "projected_team_rush_attempts", "projected_player_carries",
        "projected_target_share", "projected_targets", "projected_red_zone_targets",
        "projected_goal_line_carries", "projected_fg_attempts", "projected_pat_attempts",
    ]
    for c in nonnegative:
        base[c] = numeric(base[c])
        if base[c].isna().any() or base[c].lt(0.0).any():
            raise ValueError(f"Issue 33 invalid nonnegative projection column: {c}")
    # Signed yardage efficiencies may legitimately be negative. They must be
    # finite where the corresponding model is position-eligible; irrelevant
    # player rows remain null.
    if base["projected_target_share"].gt(1.0).any():
        raise ValueError("Issue 33 target-share prediction exceeds 1")

    output = base[OUTPUT_COLUMNS].copy()
    common.ensure_unique(output, GRAIN, "Issue 33 component projections")
    if len(output) != len(features):
        raise ValueError("Issue 33 output row count differs from current features")
    common.reject_forbidden_feature_columns(output.columns, config)
    common.write_parquet_atomic(output, output_path)

    payload = {
        "script": Path(__file__).name, "status": "passed", "season": season,
        "week": week, "rows": int(len(output)), "games": int(output["game_id"].nunique()),
        "teams": int(output["team"].nunique()), "columns": len(OUTPUT_COLUMNS),
        "opportunity_models_scored": len(OPPORTUNITY_NEEDED),
        "efficiency_models_scored": len(EFFICIENCY_NEEDED),
        "strict_prior_efficiency_history_end_season": int(history["season"].max()),
        "shares_reconciled": False,
        "share_reconciliation_stage": "current_week_allocation",
        "primary_qbs": int(qb_primary.sum()), "primary_kickers": int(kicker_primary.sum()),
        "market_exclusion_passed": bool(market["passed"]), "market_features_used": False,
        "output": repo_relative(output_path), "log": repo_relative(log_path),
    }
    log_payload = {
        **payload,
        "inputs": {
            "features": repo_relative(features_path), "roles": repo_relative(roles_path),
            "historical_features": repo_relative(historical_path),
            "eligibility": repo_relative(eligibility_path),
            **({"week1_priors": repo_relative(prop / "data" / "current" / f"{season}_week_1_priors.parquet")} if week == 1 else {}),
        },
        "opportunity_models": opp_audit,
        "efficiency_models": eff_audit,
        "policy": {
            "raw_share_components_only": True,
            "no_share_reconciliation_before_allocation": True,
            "primary_qb_assignment": True,
            "primary_kicker_assignment": True,
            "efficiency_derived_features_model_local": True,
            "strict_prior_efficiency_history": True,
            "missing_required_model_feature_fails": True,
            "persisted_model_feature_order_checked": True,
            "market_exclusion_preflight": True,
        },
    }
    write_json_atomic(log_payload, log_path)
    print(json.dumps({"script": Path(__file__).name, "payload": payload}, sort_keys=True, separators=(",", ":")))
    print("COMPONENT PROJECTIONS BUILD: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
