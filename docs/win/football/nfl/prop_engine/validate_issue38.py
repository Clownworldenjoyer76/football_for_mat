#!/usr/bin/env python3
"""Independent acceptance validator for Issue 38 current-week validation gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import common

TARGETS = {
    "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
    "receiving_yards", "receiving_tds", "kicking_points", "tackles", "sacks",
}
GRAIN = ["season", "week", "game_id", "player_id"]
TEAM_GRAIN = ["season", "week", "game_id", "team"]
TOL = 1e-9
REQUIRED_CHECKS = {
    "required current-week artifacts load",
    "every projected team is scheduled and every game ID exists in schedule",
    "every projected player has canonical GSIS identity",
    "no duplicate player_id + game_id + target",
    "nonnegative projections, ordered intervals, and probabilities in [0,1]",
    "Out and ineligible players excluded from active-only output",
    "no two QBs receive full QB1 volume for one team",
    "target and carry shares are logically bounded and reconciled",
    "one primary kicker receives majority kicking opportunity",
    "defensive projections limited to plausible participants",
    "current feature schema matches every selected production model",
    "market audit passes and weekly outputs report no market features",
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--week", type=int, required=True)
    return p.parse_args()


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"} else text


def canon(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "season" in out:
        out["season"] = pd.to_numeric(out["season"], errors="raise").astype(int)
    if "week" in out:
        out["week"] = pd.to_numeric(out["week"], errors="raise").astype(int)
    if "game_id" in out:
        out["game_id"] = out["game_id"].map(clean)
    if "player_id" in out:
        out["player_id"] = out["player_id"].map(common.normalize_player_id)
    if "team" in out:
        out["team"] = out["team"].map(common.normalize_team)
    if "opponent" in out:
        out["opponent"] = out["opponent"].map(common.normalize_team)
    return out


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as h:
        value = json.load(h)
    if not isinstance(value, dict):
        raise AssertionError(f"Expected JSON object: {path}")
    return value


def verify_model_schema(prop: Path, features: pd.DataFrame) -> tuple[int, int]:
    import lightgbm as lgb

    current = set(features.columns)
    models_checked = 0
    deps: set[str] = set()
    seen: set[str] = set()

    def inspect(label: str, model: Path, manifest_path: Path, required: list[str] | None = None) -> None:
        nonlocal models_checked
        if label in seen:
            return
        seen.add(label)
        manifest = read_json(manifest_path)
        ordered = list(manifest.get("numeric_features", [])) + list(manifest.get("categorical_features", []))
        if int(manifest.get("feature_count", len(ordered))) != len(ordered):
            raise AssertionError(f"{label}: manifest feature_count mismatch")
        booster = lgb.Booster(model_file=str(model))
        if list(booster.feature_name()) != ordered:
            raise AssertionError(f"{label}: persisted model feature order mismatch")
        needed = required if required is not None else ordered
        missing = [x for x in needed if x not in current]
        if missing:
            raise AssertionError(f"{label}: current features missing {missing[:20]}")
        models_checked += 1

    for target in sorted(TARGETS):
        selected = read_json(prop / "models" / target / "selected_model.json")
        arch = clean(selected.get("selected_architecture") or selected.get("selected_candidate"))
        if arch in {"direct", "direct_component_blend"}:
            rel = clean((selected.get("direct_variant") or {}).get("model_file"))
            model = common.repo_root() / rel if rel else prop / "models" / target / "direct_model.txt"
            inspect(f"direct/{target}", model, prop / "models" / target / "feature_manifest.json")
        if arch in {"component", "direct_component_blend"}:
            deps.update(clean(x) for x in selected.get("component_dependencies", []) if clean(x))

    for dep in sorted(deps):
        cdir = prop / "models" / "components" / dep
        edir = prop / "models" / "efficiency" / dep
        if (cdir / "feature_manifest.json").is_file():
            inspect(f"component/{dep}", cdir / "model.txt", cdir / "feature_manifest.json")
        elif (edir / "feature_manifest.json").is_file():
            manifest = read_json(edir / "feature_manifest.json")
            required = list(manifest.get("canonical_features", []))
            if not required:
                derived = set(manifest.get("derived_features", []))
                required = [x for x in list(manifest.get("numeric_features", [])) + list(manifest.get("categorical_features", [])) if x not in derived]
            inspect(f"efficiency/{dep}", edir / "model.txt", edir / "feature_manifest.json", required)
        else:
            # Selected component formulas can include deterministic current
            # feature/proxy dependencies that are not standalone models.
            # Accept them only when the current feature schema actually
            # contains the dependency column.
            if dep not in set(features.columns):
                raise AssertionError(
                    f"Dependency is neither a persisted model nor a current feature: {dep}"
                )
    return models_checked, len(deps)


def main() -> int:
    a = args()
    config = common.load_config()
    season = int(a.season if a.season is not None else config["seasons"]["current"])
    week = int(a.week)
    repo = common.repo_root()
    prop = common.prop_root()

    builder = prop / "scripts" / "validate" / "validate_week.py"
    report_path = prop / "output" / str(season) / f"week_{week}_validation.json"
    universe_path = prop / "data" / "current" / f"{season}_week_{week}_universe.parquet"
    roles_path = prop / "data" / "current" / f"{season}_week_{week}_roles.parquet"
    component_path = prop / "data" / "current" / f"{season}_week_{week}_component_projections.parquet"
    allocation_path = prop / "data" / "current" / f"{season}_week_{week}_allocated_opportunity.parquet"
    feature_path = prop / "data" / "current" / "features" / f"{season}_week_{week}_features.parquet"
    long_path = prop / "output" / str(season) / f"week_{week}_player_projections.csv"
    active_path = prop / "output" / str(season) / f"week_{week}_active_player_projections.csv"
    wide_path = prop / "output" / str(season) / f"week_{week}_player_projections_wide.csv"
    schedule_path = repo / str(config["paths"]["current_schedule"]).format(season=season, week=week)

    print("CHECK 01: required Issue 38 builder/report and passed check registry")
    for p in (builder, report_path, universe_path, roles_path, component_path, allocation_path, feature_path, long_path, active_path, wide_path, schedule_path):
        if not p.is_file():
            raise AssertionError(f"Missing Issue 38 artifact/input: {p}")
    report = read_json(report_path)
    if report.get("status") != "passed":
        raise AssertionError("Issue 38 validation report is not passed")
    checks = report.get("checks")
    if not isinstance(checks, list):
        raise AssertionError("Issue 38 report checks must be a list")
    names = {str(x.get("name")) for x in checks if isinstance(x, dict)}
    if names != REQUIRED_CHECKS:
        raise AssertionError(f"Issue 38 check registry mismatch: missing={sorted(REQUIRED_CHECKS-names)}, extra={sorted(names-REQUIRED_CHECKS)}")
    if not all(x.get("passed") is True for x in checks):
        raise AssertionError("Issue 38 report contains a failed check")
    if int(report.get("checks_total", -1)) != 12 or int(report.get("checks_passed", -1)) != 12 or int(report.get("checks_failed", -1)) != 0:
        raise AssertionError("Issue 38 report check counts are invalid")

    universe = canon(pd.read_parquet(universe_path))
    roles = canon(pd.read_parquet(roles_path))
    component = canon(pd.read_parquet(component_path))
    allocation = canon(pd.read_parquet(allocation_path))
    features = canon(pd.read_parquet(feature_path))
    long = canon(pd.read_csv(long_path, low_memory=False))
    active = canon(pd.read_csv(active_path, low_memory=False))
    wide = canon(pd.read_csv(wide_path, low_memory=False))
    schedule = pd.read_csv(schedule_path, low_memory=False)

    print("CHECK 02: independently verify schedule/game IDs, GSIS identity, and projection grain")
    s = schedule.loc[
        pd.to_numeric(schedule["season"], errors="coerce").eq(season)
        & pd.to_numeric(schedule["week"], errors="coerce").eq(week)
        & schedule["season_type"].astype(str).str.casefold().isin({"reg", "regular", "regular season"})
    ]
    game_ids = {clean(x) for x in s["game_id"]}
    if not game_ids:
        raise AssertionError("No target-week schedule games")
    for label, frame in (("universe", universe), ("long", long), ("active", active), ("wide", wide)):
        bad_games = set(frame["game_id"]) - game_ids
        if bad_games:
            raise AssertionError(f"{label}: nonscheduled game IDs {sorted(bad_games)[:10]}")
        if frame["player_id"].map(common.normalize_player_id).eq("").any():
            raise AssertionError(f"{label}: blank GSIS player ID")
    universe_keys = set(zip(universe["game_id"], universe["player_id"]))
    if set(zip(long["game_id"], long["player_id"])) - universe_keys:
        raise AssertionError("Long output contains non-universe player")
    if set(zip(active["game_id"], active["player_id"])) - universe_keys:
        raise AssertionError("Active output contains non-universe player")
    common.ensure_unique(long, ["game_id", "player_id", "target"], "Issue 38 long")
    common.ensure_unique(active, ["game_id", "player_id", "target"], "Issue 38 active")
    if set(long["target"].astype(str)) != TARGETS or len(long) != len(universe) * 9:
        raise AssertionError("Long audit target coverage/cardinality mismatch")

    print("CHECK 03: independently verify nonnegative projections, intervals, probabilities, and Out exclusion")
    for label, frame in (("long", long), ("active", active)):
        projection = num(frame["projection"])
        if projection.isna().any() or projection.lt(-TOL).any():
            raise AssertionError(f"{label}: invalid/negative projection")
        low = num(frame["low"]); high = num(frame["high"])
        if (low.isna() ^ high.isna()).any():
            raise AssertionError(f"{label}: asymmetric low/high null mask")
        bounded = low.notna()
        if (low.loc[bounded] > projection.loc[bounded] + TOL).any() or (projection.loc[bounded] > high.loc[bounded] + TOL).any():
            raise AssertionError(f"{label}: interval ordering violation")
        p1 = num(frame["probability_1_plus"]); p2 = num(frame["probability_2_plus"])
        for p in (p1, p2):
            finite = p.notna()
            if ((p.loc[finite] < -TOL) | (p.loc[finite] > 1 + TOL)).any():
                raise AssertionError(f"{label}: probability outside [0,1]")
        both = p1.notna() & p2.notna()
        if (p2.loc[both] > p1.loc[both] + TOL).any():
            raise AssertionError(f"{label}: p2 exceeds p1")
    ustatus = universe[["game_id", "player_id", "injury_game_status", "eligibility_status"]]
    astatus = active.merge(ustatus, on=["game_id", "player_id"], how="left", validate="many_to_one", suffixes=("", "_u"))
    if astatus["injury_game_status_u"].astype(str).str.casefold().eq("out").any():
        raise AssertionError("Out player in active-only output")
    if astatus["eligibility_status_u"].astype(str).str.casefold().ne("eligible").any():
        raise AssertionError("Ineligible player in active-only output")

    print("CHECK 04: independently verify QB, share, kicker, and defensive-participation controls")
    joined = component.merge(roles[[*GRAIN, "primary_qb_flag", "primary_kicker_flag"]], on=GRAIN, how="left", validate="one_to_one")
    joined["_kick"] = num(joined["projected_fg_attempts"]).fillna(0) + num(joined["projected_pat_attempts"]).fillna(0)
    for team, group in joined.groupby("team"):
        if int(num(group["primary_qb_flag"]).fillna(0).gt(0).sum()) != 1:
            raise AssertionError(f"{team}: primary QB count != 1")
        qbs = group.loc[group["position"].astype(str).str.upper().eq("QB")]
        team_pass = float(num(group["projected_team_pass_attempts"]).dropna().median())
        qbv = num(qbs["projected_qb_pass_attempts"]).fillna(0)
        full = qbv.ge(0.5 * team_pass - TOL) & qbv.gt(TOL) if team_pass > TOL else qbv.gt(TOL)
        if int(full.sum()) > 1:
            raise AssertionError(f"{team}: multiple full QB1 volumes")
        primary_k = num(group["primary_kicker_flag"]).fillna(0).gt(0)
        if int(primary_k.sum()) != 1:
            raise AssertionError(f"{team}: primary kicker count != 1")
        total = float(group["_kick"].sum())
        if total <= TOL or float(group.loc[primary_k, "_kick"].sum()) / total <= 0.5 + TOL:
            raise AssertionError(f"{team}: primary kicker lacks majority opportunity")

    for c in ("raw_projected_target_share", "allocated_target_share", "raw_projected_carry_share", "allocated_carry_share", "allocated_def_participation"):
        v = num(allocation[c]); finite = v.notna()
        if ((v.loc[finite] < -TOL) | (v.loc[finite] > 1 + TOL)).any():
            raise AssertionError(f"{c}: outside [0,1]")
    sums = allocation.groupby(TEAM_GRAIN).agg(target=("allocated_target_share", "sum"), carry=("allocated_carry_share", "sum")).reset_index()
    vols = component.groupby(TEAM_GRAIN).agg(passv=("projected_team_pass_attempts", "first"), rushv=("projected_team_rush_attempts", "first")).reset_index()
    sums = sums.merge(vols, on=TEAM_GRAIN, validate="one_to_one")
    if (num(sums.loc[num(sums["passv"]).gt(TOL), "target"]) - 1).abs().gt(1e-8).any():
        raise AssertionError("Target shares do not reconcile")
    if (num(sums.loc[num(sums["rushv"]).gt(TOL), "carry"]) - 1).abs().gt(1e-8).any():
        raise AssertionError("Carry shares do not reconcile")

    with (prop / "config" / "target_eligibility.yaml").open("r", encoding="utf-8-sig") as h:
        elig = yaml.safe_load(h)
    ctx = universe[[*GRAIN, "position", "eligibility_status", "injury_game_status"]].merge(allocation[[*GRAIN, "allocated_def_participation"]], on=GRAIN, validate="one_to_one")
    d = active.loc[active["target"].astype(str).isin({"tackles", "sacks"})].merge(ctx, on=GRAIN, how="left", validate="many_to_one", suffixes=("", "_u"))
    for target in ("tackles", "sacks"):
        rows = d.loc[d["target"].astype(str).eq(target)]
        allowed = {str(x).upper() for x in elig[target]["eligible_positions"]}
        if (~rows["position_u"].astype(str).str.upper().isin(allowed)).any():
            raise AssertionError(f"{target}: implausible position")
        if rows["eligibility_status_u"].astype(str).str.casefold().ne("eligible").any() or rows["injury_game_status_u"].astype(str).str.casefold().eq("out").any():
            raise AssertionError(f"{target}: implausible participation status")
        p = num(rows["allocated_def_participation"])
        if p.isna().any() or p.lt(-TOL).any() or p.gt(1 + TOL).any():
            raise AssertionError(f"{target}: invalid defensive participation")

    print("CHECK 05: independently verify selected model feature schemas and market audit")
    models_checked, deps_checked = verify_model_schema(prop, features)
    audit = read_json(prop / "evaluation" / "market_exclusion_audit.json")
    if audit.get("passed") is not True or audit.get("forbidden_source_references") not in ([], None) or audit.get("forbidden_feature_columns") not in ([], None):
        raise AssertionError("Market audit does not pass cleanly")
    if report.get("market_features_used") is not False:
        raise AssertionError("Issue 38 report says market features were used")

    print("CHECK 06: failure path is wired to a nonzero exit code")
    spec = importlib.util.spec_from_file_location("issue38_week_validator", builder)
    if spec is None or spec.loader is None:
        raise AssertionError("Could not import validate_week.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.exit_code(False) == 0 or module.exit_code(True) != 0:
        raise AssertionError("Issue 38 failure/pass exit-code contract is wrong")

    print(f"season={season}")
    print(f"week={week}")
    print(f"players={len(universe)}")
    print(f"games={len(game_ids)}")
    print(f"audit_rows={len(long)}")
    print(f"active_rows={len(active)}")
    print("checks_total=12")
    print(f"models_checked={models_checked}")
    print(f"component_dependencies_checked={deps_checked}")
    print("negative_projection_rows=0")
    print("interval_violations=0")
    print("probability_violations=0")
    print("out_active_rows=0")
    print("multiple_full_qb1_teams=0")
    print("primary_kicker_majority=true")
    print("defensive_plausibility=true")
    print("feature_schema_matches_model=true")
    print("market_features_used=false")
    print("failure_returns_nonzero_exit_code=true")
    print("ISSUE 38 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
