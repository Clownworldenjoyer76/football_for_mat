#!/usr/bin/env python3
"""Independent acceptance validator for NFL Prop Engine Issue 35."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    import lightgbm as lgb
except ModuleNotFoundError as exc:
    raise SystemExit("Issue 35 validation requires LightGBM.") from exc

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import common

GRAIN = ["season", "week", "game_id", "player_id"]
TARGETS = [
    "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
    "receiving_yards", "receiving_tds", "kicking_points", "tackles", "sacks",
]
OUT = {t: f"direct_{t}" for t in TARGETS}
EXPECTED_COLUMNS = [*GRAIN, *[OUT[t] for t in TARGETS]]
REQ = {
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
RUSH_USAGE = [
    "player_carries_lag1", "player_carries_roll3_mean", "player_carries_roll5_mean",
    "player_carry_share_lag1", "player_carry_share_roll3_mean",
]
REC_USAGE = [
    "player_targets_lag1", "player_targets_roll3_mean", "player_targets_roll5_mean",
    "player_target_share_lag1", "player_target_share_roll3_mean",
]
OFF_PART = [
    "role_prior_offense_snap_pct", "role_snap_pct_roll3", "role_snap_pct_roll5",
    "role_prior_offense_participation", "role_participation_roll3", "role_participation_roll5",
]
DEF_PART = [
    "role_prior_defense_participation", "player_defensive_def_participation_lag1",
    "player_defensive_def_participation_roll3",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f"Missing JSON: {path}")
    with path.open("r", encoding="utf-8-sig") as h:
        value = json.load(h)
    if not isinstance(value, dict):
        raise AssertionError(f"Expected JSON object: {path}")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f"Missing YAML: {path}")
    with path.open("r", encoding="utf-8-sig") as h:
        value = yaml.safe_load(h)
    if not isinstance(value, dict):
        raise AssertionError(f"Expected YAML mapping: {path}")
    return value


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")


def flg(series: pd.Series) -> pd.Series:
    return num(series).fillna(0.0).gt(0.0)


def pos(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def any_pos(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise AssertionError(f"Missing eligibility signal(s): {missing}")
    result = pd.Series(False, index=frame.index, dtype=bool)
    for c in cols:
        result |= num(frame[c]).fillna(0.0).gt(0.0)
    return result


def clean_category(series: pd.Series) -> pd.Series:
    value = series.fillna("").astype(str).str.strip()
    return value.mask(value.str.casefold().isin({"", "nan", "none", "null", "<na>", "nat"}), "")


def independent_matrix(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    levels: dict[str, list[str]],
) -> pd.DataFrame:
    pieces: dict[str, pd.Series] = {}
    for c in numeric_features:
        pieces[c] = num(frame[c])
    for c in categorical_features:
        values = clean_category(frame[c])
        cat = pd.Categorical(
            values.where(values.ne(""), None), categories=[str(x) for x in levels[c]], ordered=False
        )
        pieces[c] = pd.Series(cat.codes.astype("int32"), index=frame.index)
    return pd.DataFrame(pieces, index=frame.index, columns=[*numeric_features, *categorical_features])


def independent_hash(numeric_features: list[str], categorical_features: list[str]) -> str:
    schema = (
        [{"name": f, "type": "numeric"} for f in numeric_features]
        + [{"name": f, "type": "categorical"} for f in categorical_features]
    )
    raw = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda: h.read(1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def independent_eligibility(work: pd.DataFrame, target: str, cfg: dict[str, Any], manifest: dict[str, Any]) -> pd.Series:
    rule = cfg[target]
    if str(rule.get("current_requirement")) != REQ[target]:
        raise AssertionError(f"{target}: current eligibility requirement mismatch")
    positions = [str(x).strip().upper() for x in rule.get("eligible_positions", [])]
    model_positions = [str(x).strip().upper() for x in manifest.get("eligible_positions", [])]
    if positions != model_positions:
        raise AssertionError(f"{target}: eligible positions mismatch between YAML and manifest")
    position_ok = pos(work["position"]).isin(set(positions))

    requirement = REQ[target]
    if requirement == "identified_qb_role":
        current_ok = flg(work["primary_qb_flag"])
    elif requirement == "identified_primary_kicker_role":
        current_ok = flg(work["primary_kicker_flag"])
    elif requirement == "recent_usage_or_plausible_depth_or_participation_role":
        usage = any_pos(work, RUSH_USAGE if target.startswith("rushing_") else REC_USAGE)
        participation = any_pos(work, OFF_PART)
        rank = num(work["depth_rank"])
        depth = rank.notna() & rank.le(3.0)
        role = (
            flg(work["starter_flag"]) | flg(work["primary_role_flag"])
            | flg(work["committee_role_flag"]) | flg(work["role_depth_starter_flag_pregame"])
        )
        current_ok = usage | participation | depth | role
    else:
        recent = any_pos(work, DEF_PART)
        role = (
            flg(work["role_defensive_starter_flag"]) | flg(work["role_starter_promotion_flag"])
            | flg(work["starter_flag"]) | flg(work["primary_role_flag"])
        )
        current_ok = recent | role
    return (position_ok & current_ok).astype(bool)


def main() -> int:
    args = parse_args()
    season, week = int(args.season), int(args.week)
    prop = common.prop_root()
    builder = prop / "scripts" / "project" / "project_direct.py"
    output_path = prop / "data" / "current" / f"{season}_week_{week}_direct_projections.parquet"
    log_path = prop / "logs" / f"direct_projections_{season}_week_{week}.json"
    features_path = prop / "data" / "current" / "features" / f"{season}_week_{week}_features.parquet"
    roles_path = prop / "data" / "current" / f"{season}_week_{week}_roles.parquet"
    universe_path = prop / "data" / "current" / f"{season}_week_{week}_universe.parquet"
    allocated_path = prop / "data" / "current" / f"{season}_week_{week}_allocated_opportunity.parquet"
    issue34_log_path = prop / "logs" / f"allocated_opportunity_{season}_week_{week}.json"
    elig_path = prop / "config" / "target_eligibility.yaml"

    print("CHECK 01: required builder/output/log, Issue 34 sequence gate, and exact headers")
    for p in [builder, output_path, log_path, features_path, roles_path, universe_path, allocated_path, issue34_log_path, elig_path]:
        if not p.is_file():
            raise AssertionError(f"Missing Issue 35 artifact/input: {p}")
    issue34 = load_json(issue34_log_path)
    if issue34.get("status") != "passed":
        raise AssertionError("Issue 34 sequence log is not passed")
    out = pd.read_parquet(output_path)
    if list(out.columns) != EXPECTED_COLUMNS:
        raise AssertionError(f"Issue 35 exact headers/order mismatch: {list(out.columns)}")
    common.ensure_unique(out, GRAIN, "Issue 35 output")

    print("CHECK 02: exact eligible current player grain and independently reconstruct target eligibility")
    features = pd.read_parquet(features_path)
    roles = pd.read_parquet(roles_path)
    universe = pd.read_parquet(universe_path)
    allocated = pd.read_parquet(allocated_path)
    elig = load_yaml(elig_path)
    for df, label in [(features, "features"), (roles, "roles"), (universe, "universe"), (allocated, "allocated")]:
        common.ensure_unique(df, GRAIN, label)
    eligible_universe = universe.loc[
        universe["eligibility_status"].fillna("").astype(str).str.strip().str.casefold().eq("eligible")
    ].copy()
    if eligible_universe["injury_game_status"].fillna("").astype(str).str.strip().str.casefold().eq("out").any():
        raise AssertionError("Eligible Issue 29 universe contains an Out player")
    keys = lambda df: set(map(tuple, df[GRAIN].to_numpy()))
    expected = keys(eligible_universe)
    if keys(features) != expected or keys(roles) != expected or keys(allocated) != expected or keys(out) != expected:
        raise AssertionError("Issue 35 output/current artifacts do not share exact eligible player grain")
    if set(pd.to_numeric(out["season"], errors="raise").astype(int)) != {season} or set(pd.to_numeric(out["week"], errors="raise").astype(int)) != {week}:
        raise AssertionError("Issue 35 output season/week mismatch")

    work = features.merge(
        roles[[*GRAIN, "depth_rank", "starter_flag", "primary_qb_flag", "primary_kicker_flag", "primary_role_flag", "committee_role_flag"]],
        on=GRAIN, how="left", validate="one_to_one",
    )
    if len(work) != len(features):
        raise AssertionError("Issue 35 independent role merge changed row count")

    print("CHECK 03: independently verify direct manifests, hashes, categorical encoding, and persisted model order")
    config = common.load_config()
    common.reject_forbidden_feature_columns(features.columns, config)
    if any(str(c).startswith("target_") for c in features.columns):
        raise AssertionError("Issue 35 current features contain target columns")

    eligibility_counts: dict[str, int] = {}
    max_error = 0.0
    for target in TARGETS:
        model_dir = prop / "models" / target
        model_path = model_dir / "direct_model.txt"
        manifest_path = model_dir / "feature_manifest.json"
        metadata_path = model_dir / "metadata.json"
        for p in [model_path, manifest_path, metadata_path]:
            if not p.is_file():
                raise AssertionError(f"{target}: missing direct artifact {p}")
        manifest = load_json(manifest_path)
        metadata = load_json(metadata_path)
        if manifest.get("target") != target or metadata.get("target") != target:
            raise AssertionError(f"{target}: artifact target mismatch")
        numeric_features = list(manifest.get("numeric_features", []))
        categorical_features = list(manifest.get("categorical_features", []))
        feature_names = [*numeric_features, *categorical_features]
        if int(manifest.get("feature_count", -1)) != len(feature_names):
            raise AssertionError(f"{target}: feature_count mismatch")
        missing = [c for c in feature_names if c not in work.columns]
        if missing:
            raise AssertionError(f"{target}: missing current direct features: {missing[:30]}")
        schema_hash = independent_hash(numeric_features, categorical_features)
        if schema_hash != manifest.get("feature_hash") or schema_hash != metadata.get("feature_hash"):
            raise AssertionError(f"{target}: feature schema hash mismatch")
        primary = metadata.get("primary", {})
        if sha256_file(model_path) != str(primary.get("model_sha256", "")):
            raise AssertionError(f"{target}: direct model file hash mismatch")
        levels = manifest.get("categorical_levels_final_through_2024", {})
        for c in categorical_features:
            if c not in levels or not isinstance(levels[c], list):
                raise AssertionError(f"{target}: missing categorical levels for {c}")
        booster = lgb.Booster(model_file=str(model_path))
        if list(booster.feature_name()) != feature_names:
            raise AssertionError(f"{target}: persisted LightGBM feature order mismatch")
        common.reject_forbidden_feature_columns(feature_names, config)

        mask = independent_eligibility(work, target, elig, manifest)
        eligibility_counts[target] = int(mask.sum())
        if not mask.any():
            raise AssertionError(f"{target}: no independently eligible rows")
        col = OUT[target]
        if out.loc[~mask, col].notna().any():
            raise AssertionError(f"{target}: ineligible target/player combination is non-null")
        if out.loc[mask, col].isna().any():
            raise AssertionError(f"{target}: eligible target/player combination is null")
        if num(out.loc[mask, col]).lt(0.0).any():
            raise AssertionError(f"{target}: direct output is negative")

        X = independent_matrix(work.loc[mask], numeric_features, categorical_features, levels)
        expected_pred = np.maximum(np.asarray(booster.predict(X), dtype="float64"), 0.0)
        actual = num(out.loc[mask, col]).to_numpy(dtype="float64")
        if not np.isfinite(actual).all():
            raise AssertionError(f"{target}: nonfinite eligible output")
        err = float(np.max(np.abs(expected_pred - actual))) if len(actual) else 0.0
        max_error = max(max_error, err)
        if not np.allclose(expected_pred, actual, atol=1e-12, rtol=1e-12):
            raise AssertionError(f"{target}: direct prediction mismatch; max_abs_error={err}")

    print("CHECK 04: ineligible combinations null, nonnegative floor, and no internal rounding")
    # Rescoring at 1e-12 tolerance above detects any material internal rounding.
    for c in [OUT[t] for t in TARGETS]:
        values = num(out[c].dropna())
        if values.lt(0.0).any():
            raise AssertionError(f"Negative direct output: {c}")

    print("CHECK 05: market exclusion and final log policy")
    audit = prop / "scripts" / "validate" / "audit_market_exclusion.py"
    cp = subprocess.run([sys.executable, str(audit)], cwd=common.repo_root(), capture_output=True, text=True, check=False)
    if cp.returncode != 0 or "MARKET EXCLUSION AUDIT: PASS" not in cp.stdout:
        raise AssertionError("Issue 35 market-exclusion audit failed")
    log = load_json(log_path)
    required_log = {
        "status": "passed",
        "season": season,
        "week": week,
        "rows": len(out),
        "columns": len(EXPECTED_COLUMNS),
        "targets_scored": len(TARGETS),
        "ineligible_combinations_null": True,
        "floor_zero_applied": True,
        "internal_rounding": False,
        "market_features_used": False,
    }
    for k, v in required_log.items():
        if log.get(k) != v:
            raise AssertionError(f"Issue 35 log mismatch for {k}: expected={v!r} actual={log.get(k)!r}")
    if log.get("eligible_rows_by_target") != eligibility_counts:
        raise AssertionError("Issue 35 logged eligibility counts differ from independent reconstruction")

    print(f"season={season}")
    print(f"week={week}")
    print(f"games={out['game_id'].nunique()}")
    print(f"rows={len(out)}")
    print(f"columns={len(out.columns)}")
    print(f"targets_rescored={len(TARGETS)}")
    print(f"max_direct_prediction_abs_error={max_error:.3g}")
    for target in TARGETS:
        print(f"eligible_{target}={eligibility_counts[target]}")
    print("ineligible_combinations_null=true")
    print("floor_zero_applied=true")
    print("internal_rounding=false")
    print("market_features_used=false")
    print("ISSUE 35 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
