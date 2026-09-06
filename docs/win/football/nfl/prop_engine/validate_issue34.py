#!/usr/bin/env python3
"""Independent acceptance validator for Issue 34 team opportunity allocation."""

from __future__ import annotations

import argparse
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
    raise SystemExit("Issue 34 validator requires LightGBM.") from exc

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
TRAIN = SCRIPTS / "train"
for p in (SCRIPTS, TRAIN):
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
EPS = 1e-12
TOL = 1e-9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate Issue 34.")
    p.add_argument("--season", type=int, default=None)
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


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")


def clip01(s: pd.Series) -> pd.Series:
    return numeric(s).clip(0.0, 1.0)


def norm_pos(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.upper()


def assert_close(actual: pd.Series, expected: pd.Series, label: str, atol: float = 1e-9) -> None:
    a = numeric(actual).to_numpy(dtype="float64")
    e = numeric(expected).to_numpy(dtype="float64")
    if not np.allclose(a, e, atol=atol, rtol=1e-9, equal_nan=True):
        diff = np.abs(a - e)
        idx = np.where(~np.isclose(a, e, atol=atol, rtol=1e-9, equal_nan=True))[0][:10]
        raise AssertionError(
            f"{label} mismatch; max_abs_diff={np.nanmax(diff)} sample_indices={idx.tolist()}"
        )


def run_market_preflight(repo: Path) -> None:
    path = repo / "docs" / "win" / "football" / "nfl" / "prop_engine" / "scripts" / "validate" / "audit_market_exclusion.py"
    cp = subprocess.run([sys.executable, str(path)], cwd=repo, capture_output=True, text=True, check=False)
    if cp.returncode != 0 or "MARKET EXCLUSION AUDIT: PASS" not in cp.stdout:
        raise AssertionError(
            "Independent market exclusion audit failed: "
            f"stdout={cp.stdout[-1500:]!r} stderr={cp.stderr[-1500:]!r}"
        )


def score_component(
    prop: Path,
    features: pd.DataFrame,
    eligibility: dict[str, Any],
    component: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = opportunity.COMPONENTS[component]
    if str(spec.get("scope")) != "player":
        raise AssertionError(f"Expected player component: {component}")
    model_dir = prop / "models" / "components" / component
    manifest = load_json(model_dir / "feature_manifest.json")
    model_path = model_dir / "model.txt"
    if not model_path.is_file():
        raise AssertionError(f"Missing model: {model_path}")
    feature_names = list(manifest.get("numeric_features", [])) + list(manifest.get("categorical_features", []))
    if int(manifest.get("feature_count", -1)) != len(feature_names):
        raise AssertionError(f"{component}: manifest feature_count mismatch")
    missing = [c for c in feature_names if c not in features.columns]
    if missing:
        raise AssertionError(f"{component}: missing current features: {missing[:30]}")
    booster = lgb.Booster(model_file=str(model_path))
    if list(booster.feature_name()) != feature_names:
        raise AssertionError(f"{component}: persisted model feature order mismatch")
    rule = str(spec["eligible_rule"])
    positions = {str(x).strip().upper() for x in eligibility[rule]["eligible_positions"]}
    rows = features.loc[norm_pos(features["position"]).isin(positions)].copy()
    X = pd.DataFrame(
        {c: numeric(rows[c]).to_numpy() for c in feature_names},
        index=rows.index,
        columns=feature_names,
    )
    pred = opportunity.transform_prediction(booster.predict(X), component)
    if not np.isfinite(pred).all():
        raise AssertionError(f"{component}: nonfinite prediction")
    out = rows[GRAIN].copy()
    out[component] = np.asarray(pred, dtype="float64")
    common.ensure_unique(out, GRAIN, f"Issue 34 validator {component}")
    return out, manifest


def weighted_mean(frame: pd.DataFrame, columns: list[str], weights: list[float]) -> pd.Series:
    n = pd.Series(0.0, index=frame.index, dtype="float64")
    d = pd.Series(0.0, index=frame.index, dtype="float64")
    for c, w in zip(columns, weights):
        v = clip01(frame[c])
        ok = v.notna()
        n.loc[ok] += float(w) * v.loc[ok]
        d.loc[ok] += float(w)
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    ok = d.gt(0.0)
    out.loc[ok] = n.loc[ok] / d.loc[ok]
    return out


def depth_score(rank_series: pd.Series) -> pd.Series:
    rank = numeric(rank_series)
    out = pd.Series(0.10, index=rank.index, dtype="float64")
    out.loc[rank.le(4.0)] = 0.30
    out.loc[rank.le(3.0)] = 0.50
    out.loc[rank.le(2.0)] = 0.75
    out.loc[rank.le(1.0)] = 1.00
    return out


def add_signals(work: pd.DataFrame) -> pd.DataFrame:
    out = work.copy()
    out["_recent"] = weighted_mean(
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
    out["_depth"] = depth_score(out["depth_rank"])
    out["_starter"] = clip01(out["starter_flag"]).fillna(0.0)
    out["_committee"] = clip01(out["committee_role_flag"]).fillna(0.0)
    out["_confidence"] = clip01(out["role_confidence"]).fillna(0.5)
    out["_backup"] = (
        out["role_status"].fillna("").astype(str).str.strip().str.casefold().eq("backup")
        | numeric(out["depth_backup_flag"]).fillna(0.0).gt(0.0)
    ).astype("float64")
    out["_out"] = numeric(out["role_teammate_out_count_position"]).fillna(0.0).clip(lower=0.0)
    explicit = numeric(out["role_starter_promotion_flag"]).fillna(0.0).gt(0.0)
    contextual = out["_out"].gt(0.0) & (
        out["_starter"].gt(0.0) | numeric(out["depth_rank"]).fillna(99.0).le(1.0)
    )
    out["_promotion"] = (explicit | contextual).astype("float64")
    out["_recent_backup"] = (out["_backup"] * out["_recent"]).clip(0.0, 1.0)
    out["_claim"] = (
        0.34 * out["_depth"]
        + 0.28 * out["_recent"]
        + 0.16 * out["_starter"]
        + 0.08 * out["_committee"]
        + 0.08 * out["_recent_backup"]
        + 0.06 * out["_confidence"]
    ).clip(lower=1e-6)
    out["_boost"] = 1.0 + out["_out"].clip(upper=2.0) * (
        0.70 * out["_promotion"]
        + 0.20 * out["_recent_backup"]
        + 0.10 * out["_recent"]
    )
    return out


def reconstruct_family(
    work: pd.DataFrame,
    raw_col: str,
    alloc_col: str,
    candidate_mask: pd.Series,
    volume_col: str,
    family: str,
) -> pd.DataFrame:
    out = work.copy()
    out[alloc_col] = 0.0
    out[f"_{family}_expected_reason"] = f"{family}:not_eligible_position"
    for key, idx in out.groupby(TEAM_GRAIN, sort=True).groups.items():
        loc = pd.Index(idx)
        cand = loc[candidate_mask.loc[loc].to_numpy(dtype=bool)]
        volume = numeric(out.loc[loc, volume_col]).dropna()
        if volume.empty:
            raise AssertionError(f"{family}: missing team volume for {key}")
        team_volume = max(float(volume.iloc[0]), 0.0)
        if len(cand) == 0:
            if team_volume > EPS:
                raise AssertionError(f"{family}: positive volume without candidates: {key}")
            continue
        raw = clip01(out.loc[cand, raw_col]).fillna(0.0)
        out_ctx = out.loc[cand, "_out"].gt(0.0)
        if raw.sum() > EPS:
            score = raw.copy()
            if bool(out_ctx.any()):
                score = score * out.loc[cand, "_boost"]
                out.loc[cand[out_ctx.to_numpy()], f"_{family}_expected_reason"] = (
                    f"{family}:teammate_out_depth_backup_participation_weighted"
                )
                out.loc[cand[~out_ctx.to_numpy()], f"_{family}_expected_reason"] = (
                    f"{family}:team_normalized_raw_share"
                )
            else:
                out.loc[cand, f"_{family}_expected_reason"] = f"{family}:team_normalized_raw_share"
        else:
            score = out.loc[cand, "_claim"]
            out.loc[cand, f"_{family}_expected_reason"] = f"{family}:depth_participation_fallback_no_raw_mass"
        score = numeric(score).fillna(0.0).clip(lower=0.0)
        if float(score.sum()) <= EPS:
            raise AssertionError(f"{family}: zero deterministic fallback mass: {key}")
        out.loc[cand, alloc_col] = (score / score.sum()).to_numpy(dtype="float64")
    return out


def main() -> int:
    args = parse_args()
    config = common.load_config()
    season = int(args.season) if args.season is not None else int(config["seasons"]["current"])
    week = int(args.week)
    repo = common.repo_root()
    prop = common.prop_root()

    builder = prop / "scripts" / "project" / "allocate_team_opportunity.py"
    output_path = prop / "data" / "current" / f"{season}_week_{week}_allocated_opportunity.parquet"
    log_path = prop / "logs" / f"allocated_opportunity_{season}_week_{week}.json"
    component_path = prop / "data" / "current" / f"{season}_week_{week}_component_projections.parquet"
    features_path = prop / "data" / "current" / "features" / f"{season}_week_{week}_features.parquet"
    roles_path = prop / "data" / "current" / f"{season}_week_{week}_roles.parquet"
    universe_path = prop / "data" / "current" / f"{season}_week_{week}_universe.parquet"
    eligibility_path = prop / "config" / "target_eligibility.yaml"
    issue33_log_path = prop / "logs" / f"component_projections_{season}_week_{week}.json"

    print("CHECK 01: required builder/output/log, sequence inputs, and exact headers")
    for p in [builder, output_path, log_path, component_path, features_path, roles_path, universe_path, eligibility_path, issue33_log_path]:
        if not p.is_file():
            raise AssertionError(f"Missing Issue 34 artifact/input: {p}")
    out = pd.read_parquet(output_path)
    if list(out.columns) != OUTPUT_COLUMNS:
        raise AssertionError(f"Issue 34 exact headers mismatch: {list(out.columns)}")
    common.ensure_unique(out, GRAIN, "Issue 34 output")
    if len(out) == 0:
        raise AssertionError("Issue 34 output is empty")

    component = pd.read_parquet(component_path)
    features = pd.read_parquet(features_path)
    roles = pd.read_parquet(roles_path)
    universe = pd.read_parquet(universe_path)
    eligibility = load_yaml(eligibility_path)
    issue33 = load_json(issue33_log_path)
    log = load_json(log_path)
    if bool(issue33.get("shares_reconciled", True)):
        raise AssertionError("Issue 33 input was already reconciled")
    if int(log.get("season", -1)) != season or int(log.get("week", -1)) != week or log.get("status") != "passed":
        raise AssertionError("Issue 34 log season/week/status mismatch")

    for frame in [out, component, features, roles, universe]:
        frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
        frame["week"] = pd.to_numeric(frame["week"], errors="raise").astype(int)

    print("CHECK 02: exact eligible player grain, current context, and Out-player exclusion")
    eligible = universe.loc[
        universe["eligibility_status"].fillna("").astype(str).str.casefold().eq("eligible")
    ].copy()
    common.ensure_unique(eligible, GRAIN, "Issue 34 validator eligible universe")
    keys_out = set(map(tuple, out[GRAIN].to_numpy()))
    keys_component = set(map(tuple, component[GRAIN].to_numpy()))
    keys_roles = set(map(tuple, roles[GRAIN].to_numpy()))
    keys_eligible = set(map(tuple, eligible[GRAIN].to_numpy()))
    if not (keys_out == keys_component == keys_roles == keys_eligible):
        raise AssertionError("Issue 34 player grain does not match Issue 29/30/33 accepted eligible grain")
    if eligible["injury_game_status"].fillna("").astype(str).str.casefold().eq("out").any():
        raise AssertionError("Out player exists in eligible allocation pool")
    context = out[[*GRAIN, "team"]].merge(
        component[[*GRAIN, "team", "position"]].rename(columns={"team": "_component_team"}),
        on=GRAIN,
        how="left",
        validate="one_to_one",
    )
    if context["team"].astype(str).ne(context["_component_team"].astype(str)).any():
        raise AssertionError("Issue 34 team context differs from Issue 33")

    print("CHECK 03: independently rescore raw carry share and defensive participation")
    target_manifest = load_json(
        prop / "models" / "components" / "player_target_share" / "feature_manifest.json"
    )
    carry_pred, carry_manifest = score_component(prop, features, eligibility, "player_carry_share")
    def_pred, def_manifest = score_component(prop, features, eligibility, "player_defensive_participation")
    if not bool(target_manifest.get("reconcile_during_current_week_allocation", False)):
        raise AssertionError("player_target_share manifest should require allocation reconciliation")
    if not bool(carry_manifest.get("reconcile_during_current_week_allocation", False)):
        raise AssertionError("player_carry_share manifest should require allocation reconciliation")
    if bool(def_manifest.get("reconcile_during_current_week_allocation", False)):
        raise AssertionError("player_defensive_participation should not require allocation reconciliation")

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
        ]
    ].copy()
    work = work.merge(out[OUTPUT_COLUMNS], on=[*GRAIN, "team"], how="left", validate="one_to_one")
    work = work.merge(
        carry_pred.rename(columns={"player_carry_share": "_expected_raw_carry"}),
        on=GRAIN, how="left", validate="one_to_one",
    )
    work = work.merge(
        def_pred.rename(columns={"player_defensive_participation": "_expected_raw_def"}),
        on=GRAIN, how="left", validate="one_to_one",
    )
    work["_expected_raw_carry"] = clip01(work["_expected_raw_carry"]).fillna(0.0)
    work["_expected_raw_def"] = clip01(work["_expected_raw_def"]).fillna(0.0)
    assert_close(work["raw_projected_target_share"], clip01(work["projected_target_share"]).fillna(0.0), "raw target share")
    assert_close(work["raw_projected_carry_share"], work["_expected_raw_carry"], "raw carry share")
    assert_close(work["raw_projected_def_participation"], work["_expected_raw_def"], "raw defensive participation")
    assert_close(
        numeric(work["projected_player_carries"]).fillna(0.0),
        numeric(work["projected_team_rush_attempts"]).clip(lower=0.0) * work["raw_projected_carry_share"],
        "Issue 33 carry volume identity",
        atol=1e-8,
    )
    assert_close(
        numeric(work["projected_targets"]).fillna(0.0),
        numeric(work["projected_team_pass_attempts"]).clip(lower=0.0) * work["raw_projected_target_share"],
        "Issue 33 target volume identity",
        atol=1e-8,
    )

    print("CHECK 04: independently reconstruct depth/backup/participation-weighted allocation")
    role_cols = roles[[*GRAIN, "depth_rank", "starter_flag", "committee_role_flag", "role_confidence"]].copy()
    uni_cols = eligible[[*GRAIN, "depth_backup_flag", "role_status"]].copy()
    feat_cols = features[
        [
            *GRAIN,
            "role_prior_offense_snap_pct",
            "role_snap_pct_roll3",
            "role_snap_pct_roll5",
            "role_prior_offense_participation",
            "role_participation_roll3",
            "role_participation_roll5",
            "role_starter_promotion_flag",
            "role_teammate_out_count_position",
        ]
    ].copy()
    work = work.merge(role_cols, on=GRAIN, how="left", validate="one_to_one")
    work = work.merge(uni_cols, on=GRAIN, how="left", validate="one_to_one")
    work = work.merge(feat_cols, on=GRAIN, how="left", validate="one_to_one")
    work = add_signals(work)

    pos = norm_pos(work["position"])
    receiver_positions = {str(x).upper() for x in eligibility["receiving_yards"]["eligible_positions"]}
    rusher_positions = {str(x).upper() for x in eligibility["rushing_yards"]["eligible_positions"]}
    defender_positions = {str(x).upper() for x in eligibility["tackles"]["eligible_positions"]}
    receiver = pos.isin(receiver_positions)
    rusher = pos.isin(rusher_positions)
    defender = pos.isin(defender_positions)

    expected = reconstruct_family(
        work,
        "raw_projected_target_share",
        "_expected_target_alloc",
        receiver,
        "projected_team_pass_attempts",
        "target",
    )
    expected = reconstruct_family(
        expected,
        "raw_projected_carry_share",
        "_expected_carry_alloc",
        rusher,
        "projected_team_rush_attempts",
        "carry",
    )
    expected["_expected_def_alloc"] = 0.0
    expected.loc[defender, "_expected_def_alloc"] = expected.loc[defender, "raw_projected_def_participation"].to_numpy()
    expected["_expected_def_reason"] = "def:not_eligible_position"
    expected.loc[defender & expected["_out"].gt(0.0), "_expected_def_reason"] = "def:model_role_context_out_no_reconciliation"
    expected.loc[defender & expected["_out"].le(0.0), "_expected_def_reason"] = "def:model_role_context_no_reconciliation"
    expected["_expected_reason"] = (
        expected["_target_expected_reason"].astype(str)
        + ";"
        + expected["_carry_expected_reason"].astype(str)
        + ";"
        + expected["_expected_def_reason"].astype(str)
    )

    assert_close(expected["allocated_target_share"], expected["_expected_target_alloc"], "allocated target share")
    assert_close(expected["allocated_carry_share"], expected["_expected_carry_alloc"], "allocated carry share")
    assert_close(expected["allocated_def_participation"], expected["_expected_def_alloc"], "allocated defensive participation")
    if not expected["redistribution_reason"].astype(str).eq(expected["_expected_reason"].astype(str)).all():
        bad = expected.loc[
            ~expected["redistribution_reason"].astype(str).eq(expected["_expected_reason"].astype(str)),
            [*GRAIN, "redistribution_reason", "_expected_reason"],
        ].head(10)
        raise AssertionError(f"redistribution_reason mismatch: {bad.to_dict('records')}")

    print("CHECK 05: team-volume consistency, bounds, raw preservation, and market/log policy")
    if expected.loc[~receiver, "allocated_target_share"].abs().gt(TOL).any():
        raise AssertionError("Non-receiver received target share")
    if expected.loc[~rusher, "allocated_carry_share"].abs().gt(TOL).any():
        raise AssertionError("Non-rusher received carry share")
    if expected.loc[~defender, "allocated_def_participation"].abs().gt(TOL).any():
        raise AssertionError("Non-defender received defensive participation")
    for c in [
        "raw_projected_target_share",
        "allocated_target_share",
        "raw_projected_carry_share",
        "allocated_carry_share",
        "raw_projected_def_participation",
        "allocated_def_participation",
    ]:
        v = numeric(expected[c])
        if v.isna().any() or not v.between(0.0, 1.0).all():
            raise AssertionError(f"{c} is not finite within [0,1]")

    max_target_error = 0.0
    max_carry_error = 0.0
    for key, g in expected.groupby(TEAM_GRAIN, sort=True):
        pass_volume = max(float(numeric(g["projected_team_pass_attempts"]).dropna().iloc[0]), 0.0)
        rush_volume = max(float(numeric(g["projected_team_rush_attempts"]).dropna().iloc[0]), 0.0)
        t_sum = float(g.loc[norm_pos(g["position"]).isin(receiver_positions), "allocated_target_share"].sum())
        c_sum = float(g.loc[norm_pos(g["position"]).isin(rusher_positions), "allocated_carry_share"].sum())
        if pass_volume > EPS:
            max_target_error = max(max_target_error, abs(pass_volume * t_sum - pass_volume))
            if not np.isclose(t_sum, 1.0, atol=TOL, rtol=0.0):
                raise AssertionError(f"Target shares do not sum to 1 for {key}: {t_sum}")
        if rush_volume > EPS:
            max_carry_error = max(max_carry_error, abs(rush_volume * c_sum - rush_volume))
            if not np.isclose(c_sum, 1.0, atol=TOL, rtol=0.0):
                raise AssertionError(f"Carry shares do not sum to 1 for {key}: {c_sum}")

    if not expected.loc[defender, "allocated_def_participation"].equals(expected.loc[defender, "raw_projected_def_participation"]):
        # Exact equality is expected because there is intentionally no defensive reconciliation.
        assert_close(
            expected.loc[defender, "allocated_def_participation"],
            expected.loc[defender, "raw_projected_def_participation"],
            "defensive raw preservation",
            atol=0.0,
        )

    policy = log.get("policy", {})
    required_policy = {
        "eligible_receivers_only": True,
        "eligible_rushers_only": True,
        "out_players_excluded": True,
        "out_redistribution_uses_depth_promotion_recent_backup_participation": True,
        "equal_split_default": False,
        "target_team_share_reconciled": True,
        "carry_team_share_reconciled": True,
        "raw_and_allocated_preserved": True,
        "defensive_participation_manifest_reconciliation_required": False,
        "defensive_participation_preserved_from_model": True,
        "market_exclusion_preflight": True,
    }
    for k, v in required_policy.items():
        if policy.get(k) != v:
            raise AssertionError(f"Issue 34 log policy mismatch for {k}: {policy.get(k)!r}")
    if log.get("market_features_used") is not False or log.get("equal_split_default_used") is not False:
        raise AssertionError("Issue 34 log reports forbidden market use or equal-split default")
    if int(log.get("out_players_in_allocation_pool", -1)) != 0:
        raise AssertionError("Issue 34 log reports Out player in allocation pool")
    common.reject_forbidden_feature_columns(out.columns, config)
    run_market_preflight(repo)

    print(f"season={season}")
    print(f"week={week}")
    print(f"games={out['game_id'].nunique()}")
    print(f"teams={out['team'].nunique()}")
    print(f"rows={len(out)}")
    print(f"columns={len(out.columns)}")
    print(f"target_rows_adjusted={(numeric(out['allocated_target_share']) - numeric(out['raw_projected_target_share'])).abs().gt(1e-10).sum()}")
    print(f"carry_rows_adjusted={(numeric(out['allocated_carry_share']) - numeric(out['raw_projected_carry_share'])).abs().gt(1e-10).sum()}")
    print("def_participation_reconciled=false")
    print(f"target_team_volume_max_error={max_target_error:.12g}")
    print(f"carry_team_volume_max_error={max_carry_error:.12g}")
    print("out_players_in_allocation_pool=0")
    print("equal_split_default_used=false")
    print("market_features_used=false")
    print("ISSUE 34 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
