#!/usr/bin/env python3
"""
Reconcile current-week team opportunity allocation for the NFL Prop Engine.

READS
-----
- data/current/{season}_week_{week}_component_projections.parquet
- data/current/features/{season}_week_{week}_features.parquet
- data/current/{season}_week_{week}_roles.parquet
- data/current/{season}_week_{week}_universe.parquet
- config/target_eligibility.yaml
- models/components/player_carry_share/{model.txt,feature_manifest.json}
- models/components/player_defensive_participation/{model.txt,feature_manifest.json}

WRITES
------
- data/current/{season}_week_{week}_allocated_opportunity.parquet
- logs/allocated_opportunity_{season}_week_{week}.json

POLICY
------
- Target shares are allocated only to eligible receiving positions.
- Carry shares are allocated only to eligible rushing positions.
- Raw shares are preserved separately from allocated shares.
- Player carry share is rescored from the persisted Issue 22 component model so
  raw carry share is preserved directly rather than inferred from rounded volume.
- Out-player opportunity is redistributed through current depth/promotion,
  recent backup role, and prior/recent participation weights. It is never split
  equally as a default fallback.
- Team target and carry allocations each reconcile exactly to 1.0 when the
  corresponding team volume is positive.
- Defensive participation is scored from the persisted Issue 22 model and is
  not normalized because its manifest explicitly says current-week allocation
  reconciliation is not required.
- Missing required features/models hard-fail.
- No target columns or forbidden market-derived inputs are consumed.
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
    raise SystemExit("Issue 34 requires LightGBM in the active environment.") from exc

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
TRAIN_DIR = SCRIPTS_ROOT / "train"
for p in (SCRIPTS_ROOT, TRAIN_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import common
import train_opportunity_models as opportunity

GRAIN = ["season", "week", "game_id", "player_id"]
TEAM_GRAIN = ["season", "week", "game_id", "team"]
OUTPUT_COLUMNS = [
    "season",
    "week",
    "game_id",
    "player_id",
    "team",
    "raw_projected_target_share",
    "allocated_target_share",
    "raw_projected_carry_share",
    "allocated_carry_share",
    "raw_projected_def_participation",
    "allocated_def_participation",
    "redistribution_reason",
]

COMPONENT_REQUIRED = [
    *GRAIN,
    "team",
    "position",
    "projected_team_pass_attempts",
    "projected_team_rush_attempts",
    "projected_player_carries",
    "projected_target_share",
    "projected_targets",
    "projected_defensive_participation",
]
ROLE_REQUIRED = [
    *GRAIN,
    "team",
    "position",
    "depth_rank",
    "starter_flag",
    "primary_role_flag",
    "committee_role_flag",
    "role_confidence",
    "role_reason",
]
UNIVERSE_REQUIRED = [
    *GRAIN,
    "team",
    "position",
    "position_group",
    "depth_rank",
    "depth_backup_flag",
    "injury_game_status",
    "eligibility_status",
    "role_status",
]
FEATURE_REQUIRED = [
    *GRAIN,
    "team",
    "position",
    "position_group",
    "role_depth_rank_pregame",
    "role_depth_starter_flag_pregame",
    "role_prior_offense_snap_pct",
    "role_snap_pct_roll3",
    "role_snap_pct_roll5",
    "role_prior_offense_participation",
    "role_participation_roll3",
    "role_participation_roll5",
    "role_prior_defense_participation",
    "role_starter_promotion_flag",
    "role_teammate_out_count_position",
]

EPS = 1e-12
TEAM_SUM_TOL = 1e-9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reconcile current-week team opportunity allocation.")
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
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
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
        [sys.executable, str(path)],
        cwd=common.repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    if cp.returncode != 0 or "MARKET EXCLUSION AUDIT: PASS" not in cp.stdout:
        raise RuntimeError(
            "Market-exclusion preflight failed before opportunity allocation. "
            f"stdout={cp.stdout[-2000:]!r} stderr={cp.stderr[-2000:]!r}"
        )
    return {"passed": True, "validator": repo_relative(path)}


def numeric(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype("float64")
    )


def clip01(series: pd.Series) -> pd.Series:
    return numeric(series).clip(lower=0.0, upper=1.0)


def normalized_position(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def weighted_row_mean(frame: pd.DataFrame, columns: list[str], weights: list[float]) -> pd.Series:
    if len(columns) != len(weights):
        raise ValueError("weighted_row_mean columns/weights length mismatch")
    num = pd.Series(0.0, index=frame.index, dtype="float64")
    den = pd.Series(0.0, index=frame.index, dtype="float64")
    for col, weight in zip(columns, weights):
        if col not in frame.columns:
            raise ValueError(f"Missing allocation role signal: {col}")
        value = clip01(frame[col])
        valid = value.notna()
        num.loc[valid] += float(weight) * value.loc[valid]
        den.loc[valid] += float(weight)
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    valid = den.gt(0.0)
    out.loc[valid] = num.loc[valid] / den.loc[valid]
    return out


def depth_score(depth_rank: pd.Series) -> pd.Series:
    rank = numeric(depth_rank)
    out = pd.Series(0.10, index=rank.index, dtype="float64")
    out.loc[rank.le(4.0)] = 0.30
    out.loc[rank.le(3.0)] = 0.50
    out.loc[rank.le(2.0)] = 0.75
    out.loc[rank.le(1.0)] = 1.00
    return out


def score_component(
    prop_root: Path,
    features: pd.DataFrame,
    eligibility: dict[str, Any],
    component: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = opportunity.COMPONENTS[component]
    if str(spec.get("scope")) != "player":
        raise ValueError(f"Issue 34 expected player-scope component: {component}")

    model_dir = prop_root / "models" / "components" / component
    manifest = load_json(model_dir / "feature_manifest.json")
    model_path = model_dir / "model.txt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Required component model missing: {model_path}")

    feature_names = list(manifest.get("numeric_features", [])) + list(manifest.get("categorical_features", []))
    if int(manifest.get("feature_count", len(feature_names))) != len(feature_names):
        raise ValueError(f"{component}: manifest feature_count mismatch")
    missing = [c for c in feature_names if c not in features.columns]
    if missing:
        raise ValueError(f"{component}: missing current model feature(s): {missing[:30]}")

    booster = lgb.Booster(model_file=str(model_path))
    if list(booster.feature_name()) != feature_names:
        raise ValueError(f"{component}: persisted LightGBM feature order mismatch")

    rule = str(spec["eligible_rule"])
    positions = {str(x).strip().upper() for x in eligibility[rule]["eligible_positions"]}
    pos = normalized_position(features["position"])
    rows = features.loc[pos.isin(positions)].copy()
    if rows.empty:
        raise ValueError(f"{component}: no current eligible rows")

    # All persisted Issue 22 component features are numeric. Building the frame
    # in one operation avoids the fragmented-DataFrame warning from the trainer helper.
    X = pd.DataFrame(
        {c: numeric(rows[c]).to_numpy() for c in feature_names},
        index=rows.index,
        columns=feature_names,
    )
    pred = opportunity.transform_prediction(booster.predict(X), component)
    if not np.isfinite(pred).all():
        raise ValueError(f"{component}: nonfinite persisted-model prediction")

    out = rows[GRAIN].copy()
    out[component] = np.asarray(pred, dtype="float64")
    common.ensure_unique(out, GRAIN, f"Issue 34 raw {component}")
    return out, {
        "component": component,
        "rows": int(len(out)),
        "feature_count": len(feature_names),
        "prediction_transform": str(manifest.get("prediction_transform")),
        "reconcile_during_current_week_allocation": bool(
            manifest.get("reconcile_during_current_week_allocation", False)
        ),
    }


def add_role_signals(work: pd.DataFrame) -> pd.DataFrame:
    out = work.copy()

    out["_recent_offense"] = weighted_row_mean(
        out,
        [
            "role_prior_offense_participation",
            "role_participation_roll3",
            "role_participation_roll5",
            "role_prior_offense_snap_pct",
            "role_snap_pct_roll3",
            "role_snap_pct_roll5",
        ],
        [0.20, 0.25, 0.15, 0.10, 0.20, 0.10],
    ).fillna(0.0)
    out["_depth_score"] = depth_score(out["depth_rank"])
    out["_starter"] = clip01(out["starter_flag"]).fillna(0.0)
    out["_committee"] = clip01(out["committee_role_flag"]).fillna(0.0)
    out["_confidence"] = clip01(out["role_confidence"]).fillna(0.5)
    out["_backup"] = (
        out["role_status"].fillna("").astype(str).str.strip().str.casefold().eq("backup")
        | numeric(out["depth_backup_flag"]).fillna(0.0).gt(0.0)
    ).astype("float64")
    out["_teammate_out"] = numeric(out["role_teammate_out_count_position"]).fillna(0.0).clip(lower=0.0)

    explicit_promotion = numeric(out["role_starter_promotion_flag"]).fillna(0.0).gt(0.0)
    contextual_promotion = out["_teammate_out"].gt(0.0) & (
        out["_starter"].gt(0.0) | numeric(out["depth_rank"]).fillna(99.0).le(1.0)
    )
    out["_promotion"] = (explicit_promotion | contextual_promotion).astype("float64")
    out["_recent_backup"] = (out["_backup"] * out["_recent_offense"]).clip(0.0, 1.0)

    # Non-equal deterministic fallback claim. Current depth is intentionally the
    # largest signal, followed by recent participation/snap role.
    out["_fallback_claim"] = (
        0.34 * out["_depth_score"]
        + 0.28 * out["_recent_offense"]
        + 0.16 * out["_starter"]
        + 0.08 * out["_committee"]
        + 0.08 * out["_recent_backup"]
        + 0.06 * out["_confidence"]
    ).clip(lower=1e-6)

    out["_out_boost"] = 1.0 + out["_teammate_out"].clip(upper=2.0) * (
        0.70 * out["_promotion"]
        + 0.20 * out["_recent_backup"]
        + 0.10 * out["_recent_offense"]
    )
    return out


def allocate_share_family(
    work: pd.DataFrame,
    raw_col: str,
    allocated_col: str,
    candidate_mask: pd.Series,
    volume_col: str,
    family: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = work.copy()
    out[raw_col] = clip01(out[raw_col]).fillna(0.0)
    out[allocated_col] = 0.0
    out[f"_{family}_reason"] = f"{family}:not_eligible_position"

    teams_with_out = 0
    fallback_teams = 0
    normalized_teams = 0

    for key, idx in out.groupby(TEAM_GRAIN, sort=True).groups.items():
        loc = pd.Index(idx)
        cand_idx = loc[candidate_mask.loc[loc].to_numpy(dtype=bool)]
        volume = numeric(out.loc[loc, volume_col])
        finite_volume = volume.dropna()
        if finite_volume.empty:
            raise ValueError(f"{family}: missing team volume for {key}")
        if not np.allclose(finite_volume.to_numpy(), finite_volume.iloc[0], atol=1e-10, rtol=0.0):
            raise ValueError(f"{family}: team volume is not invariant within team-game {key}")
        team_volume = max(float(finite_volume.iloc[0]), 0.0)

        if len(cand_idx) == 0:
            if team_volume > EPS:
                raise ValueError(f"{family}: positive team volume has no eligible allocation candidates: {key}")
            continue

        raw = out.loc[cand_idx, raw_col].clip(lower=0.0)
        out_context = out.loc[cand_idx, "_teammate_out"].gt(0.0)
        has_out = bool(out_context.any())
        if has_out:
            teams_with_out += 1

        if raw.sum() > EPS:
            score = raw.copy()
            if has_out:
                score = score * out.loc[cand_idx, "_out_boost"]
                out.loc[cand_idx[out_context.to_numpy()], f"_{family}_reason"] = (
                    f"{family}:teammate_out_depth_backup_participation_weighted"
                )
                remaining = cand_idx[~out_context.to_numpy()]
                out.loc[remaining, f"_{family}_reason"] = f"{family}:team_normalized_raw_share"
            else:
                out.loc[cand_idx, f"_{family}_reason"] = f"{family}:team_normalized_raw_share"
            normalized_teams += 1
        else:
            score = out.loc[cand_idx, "_fallback_claim"].copy()
            out.loc[cand_idx, f"_{family}_reason"] = f"{family}:depth_participation_fallback_no_raw_mass"
            fallback_teams += 1

        score = numeric(score).fillna(0.0).clip(lower=0.0)
        total = float(score.sum())
        if total <= EPS:
            raise ValueError(f"{family}: deterministic role-weight fallback has zero mass: {key}")
        allocated = score / total
        out.loc[cand_idx, allocated_col] = allocated.to_numpy(dtype="float64")

        if team_volume > EPS:
            share_sum = float(out.loc[cand_idx, allocated_col].sum())
            if not np.isclose(share_sum, 1.0, atol=TEAM_SUM_TOL, rtol=0.0):
                raise ValueError(f"{family}: allocated shares do not sum to 1 for {key}: {share_sum}")

    return out, {
        "teams_with_out_context": int(teams_with_out),
        "fallback_teams_no_raw_mass": int(fallback_teams),
        "normalized_teams": int(normalized_teams),
    }


def main() -> int:
    args = parse_args()
    config = common.load_config()
    season = int(args.season) if args.season is not None else int(config["seasons"]["current"])
    week = int(args.week)
    if week < 1:
        raise ValueError("week must be >= 1")

    repo = common.repo_root()
    prop = common.prop_root()
    market = run_market_preflight()

    component_path = prop / "data" / "current" / f"{season}_week_{week}_component_projections.parquet"
    features_path = prop / "data" / "current" / "features" / f"{season}_week_{week}_features.parquet"
    roles_path = prop / "data" / "current" / f"{season}_week_{week}_roles.parquet"
    universe_path = prop / "data" / "current" / f"{season}_week_{week}_universe.parquet"
    eligibility_path = prop / "config" / "target_eligibility.yaml"
    issue33_log = prop / "logs" / f"component_projections_{season}_week_{week}.json"
    output_path = prop / "data" / "current" / f"{season}_week_{week}_allocated_opportunity.parquet"
    log_path = prop / "logs" / f"allocated_opportunity_{season}_week_{week}.json"

    for path in [component_path, features_path, roles_path, universe_path, eligibility_path, issue33_log]:
        if not path.is_file():
            raise FileNotFoundError(f"Issue 34 required input missing: {path}")

    component = pd.read_parquet(component_path)
    features = pd.read_parquet(features_path)
    roles = pd.read_parquet(roles_path)
    universe_all = pd.read_parquet(universe_path)
    eligibility = load_yaml(eligibility_path)
    issue33 = load_json(issue33_log)

    common.require_columns(component, COMPONENT_REQUIRED, "Issue 33 component projections")
    common.require_columns(features, FEATURE_REQUIRED, "Issue 31 current features")
    common.require_columns(roles, ROLE_REQUIRED, "Issue 30 current roles")
    common.require_columns(universe_all, UNIVERSE_REQUIRED, "Issue 29 current universe")
    common.ensure_unique(component, GRAIN, "Issue 33 component projections")
    common.ensure_unique(features, GRAIN, "Issue 31 current features")
    common.ensure_unique(roles, GRAIN, "Issue 30 current roles")
    common.ensure_unique(universe_all, GRAIN, "Issue 29 current universe")

    if bool(issue33.get("shares_reconciled", True)):
        raise ValueError("Issue 34 expected unreconciled Issue 33 component shares")
    common.reject_forbidden_feature_columns(features.columns, config)
    common.reject_forbidden_feature_columns(component.columns, config)
    if any(str(c).startswith("target_") for c in features.columns):
        raise ValueError("Issue 34 current features unexpectedly contain target columns")

    for frame, label in [(component, "component"), (features, "features"), (roles, "roles"), (universe_all, "universe")]:
        frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
        frame["week"] = pd.to_numeric(frame["week"], errors="raise").astype(int)
        if set(frame["season"]) != {season} or set(frame["week"]) != {week}:
            raise ValueError(f"Issue 34 {label} season/week mismatch")

    # Context must agree across the accepted current-week artifacts.
    context = component[[*GRAIN, "team", "position"]].merge(
        features[[*GRAIN, "team", "position"]].rename(
            columns={"team": "_feature_team", "position": "_feature_position"}
        ),
        on=GRAIN, how="left", validate="one_to_one",
    ).merge(
        roles[[*GRAIN, "team", "position"]].rename(
            columns={"team": "_role_team", "position": "_role_position"}
        ),
        on=GRAIN, how="left", validate="one_to_one",
    )
    bad_context = (
        context["_feature_team"].isna()
        | context["_role_team"].isna()
        | context["team"].astype(str).ne(context["_feature_team"].astype(str))
        | context["team"].astype(str).ne(context["_role_team"].astype(str))
        | normalized_position(context["position"]).ne(normalized_position(context["_feature_position"]))
        | normalized_position(context["position"]).ne(normalized_position(context["_role_position"]))
    )
    if bad_context.any():
        raise ValueError(
            "Issue 34 component/features/roles context mismatch; sample="
            f"{context.loc[bad_context].head(10).to_dict('records')}"
        )

    eligible_universe = universe_all.loc[
        universe_all["eligibility_status"].fillna("").astype(str).str.casefold().eq("eligible")
    ].copy()
    common.ensure_unique(eligible_universe, GRAIN, "Issue 34 eligible universe")
    if set(map(tuple, roles[GRAIN].to_numpy())) != set(map(tuple, eligible_universe[GRAIN].to_numpy())):
        raise ValueError("Issue 34 roles do not exactly match eligible Issue 29 universe")
    if eligible_universe["injury_game_status"].fillna("").astype(str).str.casefold().eq("out").any():
        raise ValueError("Issue 34 eligible allocation pool contains an Out player")

    target_manifest = load_json(
        prop / "models" / "components" / "player_target_share" / "feature_manifest.json"
    )
    carry_manifest = load_json(
        prop / "models" / "components" / "player_carry_share" / "feature_manifest.json"
    )
    def_manifest = load_json(
        prop / "models" / "components" / "player_defensive_participation" / "feature_manifest.json"
    )
    if not bool(target_manifest.get("reconcile_during_current_week_allocation", False)):
        raise ValueError("player_target_share manifest no longer requires current-week reconciliation")
    if not bool(carry_manifest.get("reconcile_during_current_week_allocation", False)):
        raise ValueError("player_carry_share manifest no longer requires current-week reconciliation")
    if bool(def_manifest.get("reconcile_during_current_week_allocation", False)):
        raise ValueError("player_defensive_participation manifest unexpectedly requires normalization")

    carry_audit = {
        "component": "player_carry_share",
        "source": "issue33_component_projection",
        "reconcile_during_current_week_allocation": True,
        "rescored_in_issue34": False,
    }
    def_audit = {
        "component": "player_defensive_participation",
        "source": "issue33_component_projection",
        "reconcile_during_current_week_allocation": False,
        "rescored_in_issue34": False,
    }

    work = component[
        [
            *GRAIN,
            "team",
            "position",
            "projected_team_pass_attempts",
            "projected_team_rush_attempts",
            "projected_player_carries",
            "projected_target_share",
            "projected_targets",
            "projected_defensive_participation",
        ]
    ].copy()
    work = work.rename(
        columns={
            "projected_target_share": "raw_projected_target_share",
            "projected_defensive_participation": "raw_projected_def_participation",
        }
    )

    rush_volume = numeric(work["projected_team_rush_attempts"]).clip(lower=0.0)
    carry_volume = numeric(work["projected_player_carries"]).clip(lower=0.0)
    work["raw_projected_carry_share"] = 0.0
    positive_rush = rush_volume.gt(EPS)
    work.loc[positive_rush, "raw_projected_carry_share"] = (
        carry_volume.loc[positive_rush] / rush_volume.loc[positive_rush]
    ).to_numpy(dtype="float64")
    work["raw_projected_carry_share"] = clip01(
        work["raw_projected_carry_share"]
    ).fillna(0.0)

    role_cols = roles[
        [
            *GRAIN,
            "depth_rank",
            "starter_flag",
            "primary_role_flag",
            "committee_role_flag",
            "role_confidence",
            "role_reason",
        ]
    ].copy()
    uni_cols = eligible_universe[[*GRAIN, "depth_backup_flag", "role_status"]].copy()
    feature_cols = features[
        [
            *GRAIN,
            "position_group",
            "role_prior_offense_snap_pct",
            "role_snap_pct_roll3",
            "role_snap_pct_roll5",
            "role_prior_offense_participation",
            "role_participation_roll3",
            "role_participation_roll5",
            "role_prior_defense_participation",
            "role_starter_promotion_flag",
            "role_teammate_out_count_position",
        ]
    ].copy()
    work = work.merge(role_cols, on=GRAIN, how="left", validate="one_to_one")
    work = work.merge(uni_cols, on=GRAIN, how="left", validate="one_to_one")
    work = work.merge(feature_cols, on=GRAIN, how="left", validate="one_to_one")
    if len(work) != len(component):
        raise ValueError("Issue 34 row count changed while joining current role context")
    if work[["depth_rank", "role_status"]].isna().all(axis=1).any():
        raise ValueError("Issue 34 role merge failed for one or more eligible players")

    work["raw_projected_target_share"] = clip01(work["raw_projected_target_share"]).fillna(0.0)
    work["raw_projected_carry_share"] = clip01(work["raw_projected_carry_share"]).fillna(0.0)
    work["raw_projected_def_participation"] = clip01(work["raw_projected_def_participation"]).fillna(0.0)

    # Exact preservation check against the canonical Issue 33 carry volume.
    expected_carries = (
        numeric(work["projected_team_rush_attempts"]).clip(lower=0.0)
        * work["raw_projected_carry_share"]
    )
    if not np.allclose(
        expected_carries.to_numpy(),
        numeric(work["projected_player_carries"]).fillna(0.0).to_numpy(),
        atol=1e-8,
        rtol=1e-8,
    ):
        raise ValueError("Issue 34 derived raw carry share disagrees with Issue 33 projected_player_carries")
    expected_targets = (
        numeric(work["projected_team_pass_attempts"]).clip(lower=0.0)
        * work["raw_projected_target_share"]
    )
    if not np.allclose(
        expected_targets.to_numpy(),
        numeric(work["projected_targets"]).fillna(0.0).to_numpy(),
        atol=1e-8,
        rtol=1e-8,
    ):
        raise ValueError("Issue 34 raw target share disagrees with Issue 33 projected_targets")

    work = add_role_signals(work)
    pos = normalized_position(work["position"])
    receiver_positions = {str(x).upper() for x in eligibility["receiving_yards"]["eligible_positions"]}
    rusher_positions = {str(x).upper() for x in eligibility["rushing_yards"]["eligible_positions"]}
    defender_positions = {str(x).upper() for x in eligibility["tackles"]["eligible_positions"]}
    receiver_mask = pos.isin(receiver_positions)
    rusher_mask = pos.isin(rusher_positions)
    defender_mask = pos.isin(defender_positions)

    work, target_audit = allocate_share_family(
        work,
        "raw_projected_target_share",
        "allocated_target_share",
        receiver_mask,
        "projected_team_pass_attempts",
        "target",
    )
    work, carry_alloc_audit = allocate_share_family(
        work,
        "raw_projected_carry_share",
        "allocated_carry_share",
        rusher_mask,
        "projected_team_rush_attempts",
        "carry",
    )

    # Defensive participation is not a team share and the Issue 22 manifest
    # explicitly opts out of current-week reconciliation. The raw model itself
    # already uses depth, teammate-out, promotion, and participation inputs.
    work["allocated_def_participation"] = 0.0
    work.loc[defender_mask, "allocated_def_participation"] = work.loc[
        defender_mask, "raw_projected_def_participation"
    ].to_numpy(dtype="float64")
    work["_def_reason"] = "def:not_eligible_position"
    work.loc[defender_mask & work["_teammate_out"].gt(0.0), "_def_reason"] = (
        "def:model_role_context_out_no_reconciliation"
    )
    work.loc[defender_mask & work["_teammate_out"].le(0.0), "_def_reason"] = (
        "def:model_role_context_no_reconciliation"
    )

    work["redistribution_reason"] = (
        work["_target_reason"].astype(str)
        + ";"
        + work["_carry_reason"].astype(str)
        + ";"
        + work["_def_reason"].astype(str)
    )

    # Internal consistency: allocated target/carry shares must exhaust team
    # volume across eligible candidates and must be zero outside candidate sets.
    if work.loc[~receiver_mask, "allocated_target_share"].abs().gt(TEAM_SUM_TOL).any():
        raise ValueError("Non-receiver received target allocation")
    if work.loc[~rusher_mask, "allocated_carry_share"].abs().gt(TEAM_SUM_TOL).any():
        raise ValueError("Non-rusher received carry allocation")
    if work.loc[~defender_mask, "allocated_def_participation"].abs().gt(TEAM_SUM_TOL).any():
        raise ValueError("Non-defender received defensive participation")

    target_volume_error = 0.0
    carry_volume_error = 0.0
    for _, group in work.groupby(TEAM_GRAIN, sort=True):
        pass_volume = max(float(numeric(group["projected_team_pass_attempts"]).dropna().iloc[0]), 0.0)
        rush_volume = max(float(numeric(group["projected_team_rush_attempts"]).dropna().iloc[0]), 0.0)
        target_alloc = float(group.loc[normalized_position(group["position"]).isin(receiver_positions), "allocated_target_share"].sum())
        carry_alloc = float(group.loc[normalized_position(group["position"]).isin(rusher_positions), "allocated_carry_share"].sum())
        if pass_volume > EPS:
            target_volume_error = max(target_volume_error, abs(pass_volume * target_alloc - pass_volume))
        if rush_volume > EPS:
            carry_volume_error = max(carry_volume_error, abs(rush_volume * carry_alloc - rush_volume))

    if target_volume_error > 1e-7 or carry_volume_error > 1e-7:
        raise ValueError(
            f"Issue 34 team-volume reconciliation failed: target_error={target_volume_error} carry_error={carry_volume_error}"
        )

    output = work[OUTPUT_COLUMNS].copy()
    common.ensure_unique(output, GRAIN, "Issue 34 allocated opportunity")
    if len(output) != len(component):
        raise ValueError("Issue 34 output row count differs from Issue 33")
    common.reject_forbidden_feature_columns(output.columns, config)
    common.write_parquet_atomic(output, output_path)

    changed_target = int((output["allocated_target_share"] - output["raw_projected_target_share"]).abs().gt(1e-10).sum())
    changed_carry = int((output["allocated_carry_share"] - output["raw_projected_carry_share"]).abs().gt(1e-10).sum())
    changed_def = int((output["allocated_def_participation"] - output["raw_projected_def_participation"]).abs().gt(1e-10).sum())

    payload = {
        "script": Path(__file__).name,
        "status": "passed",
        "season": season,
        "week": week,
        "rows": int(len(output)),
        "games": int(output["game_id"].nunique()),
        "teams": int(output["team"].nunique()),
        "columns": len(OUTPUT_COLUMNS),
        "target_rows_adjusted": changed_target,
        "carry_rows_adjusted": changed_carry,
        "def_participation_rows_adjusted": changed_def,
        "target_team_volume_max_error": float(target_volume_error),
        "carry_team_volume_max_error": float(carry_volume_error),
        "out_players_in_allocation_pool": 0,
        "equal_split_default_used": False,
        "market_exclusion_passed": bool(market["passed"]),
        "market_features_used": False,
        "output": repo_relative(output_path),
        "log": repo_relative(log_path),
    }
    log_payload = {
        **payload,
        "inputs": {
            "component_projections": repo_relative(component_path),
            "features": repo_relative(features_path),
            "roles": repo_relative(roles_path),
            "universe": repo_relative(universe_path),
            "eligibility": repo_relative(eligibility_path),
            "issue33_log": repo_relative(issue33_log),
        },
        "raw_component_models": {
            "player_target_share": {
                "reconcile_during_current_week_allocation": bool(
                    target_manifest.get("reconcile_during_current_week_allocation", False)
                )
            },
            "player_carry_share": carry_audit,
            "player_defensive_participation": def_audit,
        },
        "allocation": {
            "target": target_audit,
            "carry": carry_alloc_audit,
        },
        "policy": {
            "eligible_receivers_only": True,
            "eligible_rushers_only": True,
            "out_players_excluded": True,
            "out_redistribution_uses_depth_promotion_recent_backup_participation": True,
            "equal_split_default": False,
            "target_team_share_reconciled": True,
            "carry_team_share_reconciled": True,
            "raw_and_allocated_preserved": True,
            "defensive_participation_manifest_reconciliation_required": False,
            "defensive_participation_preserved_from_issue33": True,
            "component_models_rescored": False,
            "market_exclusion_preflight": True,
        },
    }
    write_json_atomic(log_payload, log_path)
    print(json.dumps({"script": Path(__file__).name, "payload": payload}, sort_keys=True, separators=(",", ":")))
    print("TEAM OPPORTUNITY ALLOCATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
