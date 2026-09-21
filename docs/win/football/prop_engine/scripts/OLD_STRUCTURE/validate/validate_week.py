#!/usr/bin/env python3
"""Validate one current-week NFL Prop Engine production run (Issue 38)."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common


_CONFIG_CONTRACT = common.load_config()
TARGETS = list(_CONFIG_CONTRACT["targets"].keys())
# SIX_TARGET_PRODUCTION_REGISTRY_MODE
GRAIN = ["season", "week", "game_id", "player_id"]
PLAYER_GAME = ["game_id", "player_id"]
TEAM_GRAIN = ["season", "week", "game_id", "team"]
TOL = 1e-9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate every current-week Prop Engine run.")
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


def canonical_ids(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "season" in out.columns:
        out["season"] = pd.to_numeric(out["season"], errors="raise").astype(int)
    if "week" in out.columns:
        out["week"] = pd.to_numeric(out["week"], errors="raise").astype(int)
    if "game_id" in out.columns:
        out["game_id"] = out["game_id"].map(clean)
    if "player_id" in out.columns:
        out["player_id"] = out["player_id"].map(common.normalize_player_id)
    if "team" in out.columns:
        out["team"] = out["team"].map(common.normalize_team)
    if "opponent" in out.columns:
        out["opponent"] = out["opponent"].map(common.normalize_team)
    return out


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON missing: {path}")
    with path.open("r", encoding="utf-8-sig") as h:
        value = json.load(h)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    root = common.prop_root().resolve()
    destination = path.resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Issue 38 output must remain under Prop Engine: {destination}") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    h = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=destination.parent,
        prefix=f".{destination.name}.", suffix=".tmp", delete=False,
    )
    temp = Path(h.name)
    try:
        with h:
            json.dump(payload, h, indent=2, sort_keys=True, ensure_ascii=False, default=str, allow_nan=False)
            h.write("\n")
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def production_target_state(prop: Path) -> tuple[list[str], list[str]]:
    registry = read_json(prop / "models" / "production_registry.json")
    if list(registry.keys()) != TARGETS:
        raise AssertionError("Production registry target set/order mismatch")
    approved: list[str] = []
    deferred: list[str] = []
    for target in TARGETS:
        entry = registry[target]
        if not isinstance(entry, dict):
            raise AssertionError(f"{target}: invalid production registry entry")
        if entry.get("production_approved") is True:
            version = entry.get("version")
            if not isinstance(version, str) or not version.strip():
                raise AssertionError(f"{target}: approved registry entry has blank version")
            approved.append(target)
        elif entry.get("production_approved") is False and entry.get("version") is None:
            deferred.append(target)
        else:
            raise AssertionError(f"{target}: invalid production registry state")
    if not approved:
        raise AssertionError("No approved production targets")
    return approved, deferred


def build_team_maps(team_master: pd.DataFrame) -> tuple[dict[str, str], set[str]]:
    common.require_columns(team_master, ["team_abbr"], "team master")
    aliases: dict[str, str] = {}
    abbrs: set[str] = set()
    source_cols = [c for c in ("canonical_team", "team", "alias", "nickname", "shortDisplayName", "team_abbr") if c in team_master.columns]
    for row in team_master.to_dict("records"):
        abbr = common.normalize_team(row.get("team_abbr"))
        if not abbr:
            continue
        abbrs.add(abbr)
        for c in source_cols:
            value = clean(row.get(c))
            if value:
                aliases.setdefault(value.casefold(), abbr)
    return aliases, abbrs


def resolve_team(value: Any, aliases: dict[str, str], abbrs: set[str]) -> str:
    text = clean(value)
    normalized = common.normalize_team(text)
    if normalized in abbrs:
        return normalized
    return aliases.get(text.casefold(), "")


def schedule_contract(schedule: pd.DataFrame, team_master: pd.DataFrame, season: int, week: int) -> tuple[dict[str, tuple[str, str]], set[str]]:
    common.require_columns(schedule, ["season", "season_type", "week", "game_id", "home_team", "away_team"], "current schedule")
    aliases, abbrs = build_team_maps(team_master)
    season_col = pd.to_numeric(schedule["season"], errors="coerce")
    week_col = pd.to_numeric(schedule["week"], errors="coerce")
    regular = schedule["season_type"].astype(str).str.strip().str.casefold().isin({"reg", "regular", "regular season"})
    frame = schedule.loc[season_col.eq(season) & week_col.eq(week) & regular].copy()
    if frame.empty:
        raise AssertionError("No target-week regular-season schedule rows")
    team_map: dict[str, tuple[str, str]] = {}
    game_ids: set[str] = set()
    for row in frame.to_dict("records"):
        gid = clean(row["game_id"])
        home = resolve_team(row["home_team"], aliases, abbrs)
        away = resolve_team(row["away_team"], aliases, abbrs)
        if not gid or not home or not away or home == away:
            raise AssertionError(f"Invalid schedule row: game_id={gid!r}, home={home!r}, away={away!r}")
        if gid in game_ids:
            raise AssertionError(f"Duplicate schedule game_id: {gid}")
        if home in team_map or away in team_map:
            raise AssertionError("A team appears in multiple target-week games")
        game_ids.add(gid)
        team_map[home] = (gid, away)
        team_map[away] = (gid, home)
    return team_map, game_ids


def ensure_week(frame: pd.DataFrame, season: int, week: int, label: str) -> None:
    if "season" in frame.columns and pd.to_numeric(frame["season"], errors="coerce").ne(season).any():
        raise AssertionError(f"{label}: wrong season")
    if "week" in frame.columns and pd.to_numeric(frame["week"], errors="coerce").ne(week).any():
        raise AssertionError(f"{label}: wrong week")


def run_market_audit(prop: Path) -> dict[str, Any]:
    script = prop / "scripts" / "validate" / "audit_market_exclusion.py"
    if not script.is_file():
        raise FileNotFoundError(f"Issue 28 market audit script missing: {script}")
    cp = subprocess.run([sys.executable, str(script)], cwd=common.repo_root(), capture_output=True, text=True, check=False)
    if cp.returncode != 0 or "MARKET EXCLUSION AUDIT: PASS" not in cp.stdout:
        raise AssertionError(
            "Market audit failed. "
            f"stdout={cp.stdout[-1500:]!r} stderr={cp.stderr[-1500:]!r}"
        )
    audit_path = prop / "evaluation" / "market_exclusion_audit.json"
    audit = read_json(audit_path)
    if audit.get("passed") is not True:
        raise AssertionError("Market audit JSON is not passed=true")
    if audit.get("forbidden_source_references") not in ([], None):
        raise AssertionError("Market audit contains forbidden source references")
    if audit.get("forbidden_feature_columns") not in ([], None):
        raise AssertionError("Market audit contains forbidden feature columns")
    return {"files_scanned": int(audit.get("files_scanned", 0)), "audit": str(audit_path.relative_to(common.repo_root()))}


def validate_model_schemas(
    prop: Path,
    current_features: pd.DataFrame,
    production_targets: list[str],
) -> dict[str, Any]:
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as exc:
        raise AssertionError("LightGBM is required to validate persisted model feature order") from exc

    current_columns = set(current_features.columns)
    checked_models: dict[str, dict[str, Any]] = {}
    dependencies: set[str] = set()
    deterministic_feature_dependencies: set[str] = set()
    architectures: dict[str, str] = {}

    def verify_model(label: str, model_path: Path, manifest_path: Path, current_required: list[str] | None = None) -> None:
        if label in checked_models:
            return
        if not model_path.is_file() or not manifest_path.is_file():
            raise AssertionError(f"Missing persisted model contract for {label}: {model_path}, {manifest_path}")
        manifest = read_json(manifest_path)
        numeric_features = list(manifest.get("numeric_features", []))
        categorical_features = list(manifest.get("categorical_features", []))
        feature_names = numeric_features + categorical_features
        if int(manifest.get("feature_count", len(feature_names))) != len(feature_names):
            raise AssertionError(f"{label}: feature_count mismatch")
        if not feature_names:
            raise AssertionError(f"{label}: empty feature manifest")
        booster = lgb.Booster(model_file=str(model_path))
        if list(booster.feature_name()) != feature_names:
            raise AssertionError(f"{label}: LightGBM feature order differs from manifest")
        required = list(current_required if current_required is not None else feature_names)
        missing = [c for c in required if c not in current_columns]
        if missing:
            raise AssertionError(f"{label}: current feature schema missing {missing[:20]}")
        checked_models[label] = {"features": len(feature_names), "current_required": len(required)}

    for target in production_targets:
        selected_path = prop / "models" / target / "selected_model.json"
        selected = read_json(selected_path)
        architecture = clean(selected.get("selected_architecture") or selected.get("selected_candidate"))
        if architecture not in {"direct", "component", "direct_component_blend"}:
            raise AssertionError(f"{target}: unsupported selected architecture {architecture!r}")
        architectures[target] = architecture

        if architecture in {"direct", "direct_component_blend"}:
            direct_info = selected.get("direct_variant") or {}
            model_rel = clean(direct_info.get("model_file"))
            model_path = (common.repo_root() / model_rel).resolve() if model_rel else prop / "models" / target / "direct_model.txt"
            verify_model(
                f"direct/{target}",
                model_path,
                prop / "models" / target / "feature_manifest.json",
            )

        if architecture in {"component", "direct_component_blend"}:
            for dep in selected.get("component_dependencies", []):
                dependency = clean(dep)
                if dependency:
                    dependencies.add(dependency)

    for dep in sorted(dependencies):
        component_dir = prop / "models" / "components" / dep
        efficiency_dir = prop / "models" / "efficiency" / dep
        if (component_dir / "feature_manifest.json").is_file():
            verify_model(
                f"component/{dep}",
                component_dir / "model.txt",
                component_dir / "feature_manifest.json",
            )
            continue
        if (efficiency_dir / "feature_manifest.json").is_file():
            manifest = read_json(efficiency_dir / "feature_manifest.json")
            canonical_features = list(manifest.get("canonical_features", []))
            if not canonical_features:
                # Older/alternate manifests: model features minus explicitly derived fields.
                all_features = list(manifest.get("numeric_features", [])) + list(manifest.get("categorical_features", []))
                derived = set(manifest.get("derived_features", []))
                canonical_features = [f for f in all_features if f not in derived]
            verify_model(
                f"efficiency/{dep}",
                efficiency_dir / "model.txt",
                efficiency_dir / "feature_manifest.json",
                current_required=canonical_features,
            )
            continue
        # Some Issue 25 component formulas persist deterministic pregame
        # feature/proxy dependencies alongside actual model dependencies
        # (for example team_goal_line_rush_attempts_ewm5).  Those are valid
        # only when they exist in the current feature schema; they are not
        # expected to have a model directory.
        if dep in current_columns:
            deterministic_feature_dependencies.add(dep)
            continue
        raise AssertionError(
            f"Selected dependency is neither a persisted model nor a current feature: {dep}"
        )

    return {
        "models_checked": len(checked_models),
        "component_dependencies_checked": len(dependencies),
        "deterministic_feature_dependencies_checked": len(deterministic_feature_dependencies),
        "architectures": architectures,
    }


def exit_code(passed: bool) -> int:
    return 0 if bool(passed) else 1


def main() -> int:
    args = parse_args()
    config = common.load_config()
    season = int(args.season if args.season is not None else config["seasons"]["current"])
    week = int(args.week)
    repo = common.repo_root()
    prop = common.prop_root()
    production_targets, deferred_targets = production_target_state(prop)
    output_path = prop / "output" / str(season) / f"week_{week}_validation.json"
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    paths = {
        "schedule": (repo / str(config["paths"]["current_schedule"]).format(season=season, week=week)).resolve(),
        "team_master": (repo / str(config["paths"]["team_master"])).resolve(),
        "universe": prop / "data" / "current" / f"{season}_week_{week}_universe.parquet",
        "roles": prop / "data" / "current" / f"{season}_week_{week}_roles.parquet",
        "component": prop / "data" / "current" / f"{season}_week_{week}_component_projections.parquet",
        "allocation": prop / "data" / "current" / f"{season}_week_{week}_allocated_opportunity.parquet",
        "features": prop / "data" / "current" / "features" / f"{season}_week_{week}_features.parquet",
        "long": prop / "output" / str(season) / f"week_{week}_player_projections.csv",
        "active": prop / "output" / str(season) / f"week_{week}_active_player_projections.csv",
        "wide": prop / "output" / str(season) / f"week_{week}_player_projections_wide.csv",
    }

    checks: list[dict[str, Any]] = []

    def record(name: str, fn: Callable[[], dict[str, Any] | None]) -> None:
        print(f"CHECK {len(checks)+1:02d}: {name}")
        try:
            details = fn() or {}
            checks.append({"name": name, "passed": True, "details": details})
        except Exception as exc:
            checks.append({"name": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"})

    # Load inputs once so all downstream checks operate on the same snapshot.
    frames: dict[str, pd.DataFrame] = {}
    required_error: Exception | None = None
    try:
        missing = [str(p) for p in paths.values() if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing required current-week artifact(s): {missing}")
        frames["schedule"] = pd.read_csv(paths["schedule"], low_memory=False)
        frames["team_master"] = pd.read_csv(paths["team_master"], low_memory=False)
        frames["universe"] = canonical_ids(pd.read_parquet(paths["universe"]))
        frames["roles"] = canonical_ids(pd.read_parquet(paths["roles"]))
        frames["component"] = canonical_ids(pd.read_parquet(paths["component"]))
        frames["allocation"] = canonical_ids(pd.read_parquet(paths["allocation"]))
        frames["features"] = canonical_ids(pd.read_parquet(paths["features"]))
        frames["long"] = canonical_ids(pd.read_csv(paths["long"], low_memory=False))
        frames["active"] = canonical_ids(pd.read_csv(paths["active"], low_memory=False))
        frames["wide"] = canonical_ids(pd.read_csv(paths["wide"], low_memory=False))
        for label, frame in frames.items():
            if label not in {"schedule", "team_master"}:
                ensure_week(frame, season, week, label)
    except Exception as exc:
        required_error = exc

    def required_inputs() -> dict[str, Any]:
        if required_error is not None:
            raise required_error
        return {"required_artifacts": len(paths), "long_rows": len(frames["long"]), "active_rows": len(frames["active"]), "wide_rows": len(frames["wide"])}

    record("required current-week artifacts load", required_inputs)

    if required_error is None:
        schedule_map, schedule_game_ids = schedule_contract(frames["schedule"], frames["team_master"], season, week)

        def scheduled_teams_games() -> dict[str, Any]:
            universe = frames["universe"]
            universe_teams = set(universe["team"].map(common.normalize_team))
            if universe_teams != set(schedule_map):
                missing = sorted(universe_teams - set(schedule_map))
                extra = sorted(set(schedule_map) - universe_teams)
                raise AssertionError(f"Projected/scheduled team mismatch: nonscheduled={missing}, scheduled_without_universe={extra}")
            for label in ("universe", "roles", "component", "allocation", "features", "long", "active", "wide"):
                frame = frames[label]
                if "game_id" in frame.columns:
                    bad = sorted(set(frame["game_id"].map(clean)) - schedule_game_ids)
                    if bad:
                        raise AssertionError(f"{label}: game IDs not in schedule: {bad[:10]}")
            bad_context = []
            for row in universe[["game_id", "team", "opponent", "player_id"]].to_dict("records"):
                expected = schedule_map.get(common.normalize_team(row["team"]))
                if expected is None or (clean(row["game_id"]), common.normalize_team(row["opponent"])) != expected:
                    bad_context.append(row)
                    if len(bad_context) >= 10:
                        break
            if bad_context:
                raise AssertionError(f"Universe schedule context mismatch: {bad_context}")
            return {"scheduled_teams": len(schedule_map), "scheduled_games": len(schedule_game_ids)}

        record("every projected team is scheduled and every game ID exists in schedule", scheduled_teams_games)

        def projected_player_ids() -> dict[str, Any]:
            universe_keys = set(zip(frames["universe"]["game_id"], frames["universe"]["player_id"]))
            for label in ("long", "active", "wide"):
                frame = frames[label]
                if frame["player_id"].map(common.normalize_player_id).eq("").any():
                    raise AssertionError(f"{label}: blank/noncanonical GSIS player_id")
                unknown = set(zip(frame["game_id"], frame["player_id"])) - universe_keys
                if unknown:
                    raise AssertionError(f"{label}: projected player not in canonical GSIS universe: {list(unknown)[:10]}")
            return {"projected_players": int(frames["wide"]["player_id"].nunique())}

        record("every projected player has canonical GSIS identity", projected_player_ids)

        def duplicate_grain() -> dict[str, Any]:
            long = frames["long"]
            active = frames["active"]
            common.ensure_unique(long, ["game_id", "player_id", "target"], "Issue 38 long output")
            common.ensure_unique(active, ["game_id", "player_id", "target"], "Issue 38 active output")
            if set(long["target"].astype(str)) != set(production_targets):
                raise AssertionError(
                    "Long output target set differs from approved production registry"
                )
            if set(active["target"].astype(str)) - set(production_targets):
                raise AssertionError("Active output contains deferred/unapproved target")
            expected = len(frames["universe"]) * len(production_targets)
            if len(long) != expected:
                raise AssertionError(
                    f"Long audit row count {len(long)} != "
                    f"universe*approved_targets {expected}"
                )
            return {"audit_rows": len(long), "active_rows": len(active), "duplicate_rows": 0}

        record("no duplicate player_id + game_id + target", duplicate_grain)

        def numeric_projection_contract() -> dict[str, Any]:
            checked = 0
            for label in ("long", "active"):
                frame = frames[label]
                common.require_columns(frame, ["projection", "low", "high", "probability_1_plus", "probability_2_plus"], label)
                projection = numeric(frame["projection"])
                if projection.isna().any():
                    raise AssertionError(f"{label}: nonfinite projection")
                if projection.lt(-TOL).any():
                    bad = frame.loc[projection.lt(-TOL), ["game_id", "player_id", "target", "projection"]].head(10).to_dict("records")
                    raise AssertionError(f"{label}: negative projection(s): {bad}")
                low = numeric(frame["low"])
                high = numeric(frame["high"])
                asymmetric = low.isna() ^ high.isna()
                if asymmetric.any():
                    raise AssertionError(f"{label}: low/high null masks differ")
                bounded = low.notna() & high.notna()
                if (low.loc[bounded] > projection.loc[bounded] + TOL).any() or (projection.loc[bounded] > high.loc[bounded] + TOL).any():
                    raise AssertionError(f"{label}: low <= projection <= high violated")
                p1 = numeric(frame["probability_1_plus"])
                p2 = numeric(frame["probability_2_plus"])
                for pname, p in (("probability_1_plus", p1), ("probability_2_plus", p2)):
                    finite = p.notna()
                    if ((p.loc[finite] < -TOL) | (p.loc[finite] > 1.0 + TOL)).any():
                        raise AssertionError(f"{label}: {pname} outside [0,1]")
                both = p1.notna() & p2.notna()
                if (p2.loc[both] > p1.loc[both] + TOL).any():
                    raise AssertionError(f"{label}: probability_2_plus exceeds probability_1_plus")
                checked += len(frame)
            return {"rows_checked": checked, "negative_projections": 0, "interval_violations": 0, "probability_violations": 0}

        record("nonnegative projections, ordered intervals, and probabilities in [0,1]", numeric_projection_contract)

        def out_exclusion() -> dict[str, Any]:
            universe = frames["universe"][["game_id", "player_id", "injury_game_status", "eligibility_status"]].copy()
            active = frames["active"].merge(universe, on=["game_id", "player_id"], how="left", validate="many_to_one", suffixes=("", "_universe"))
            if active["injury_game_status_universe"].isna().any():
                raise AssertionError("Active output has players missing universe injury state")
            out = active["injury_game_status_universe"].astype(str).str.strip().str.casefold().eq("out")
            if out.any():
                raise AssertionError(f"Out player appears in active-only output: {active.loc[out, ['game_id','player_id','target']].head(10).to_dict('records')}")
            ineligible = active["eligibility_status_universe"].astype(str).str.casefold().ne("eligible")
            if ineligible.any():
                raise AssertionError("Ineligible universe player appears in active-only output")
            return {"out_rows_in_active": 0, "ineligible_rows_in_active": 0}

        record("Out and ineligible players excluded from active-only output", out_exclusion)

        def qb_volume() -> dict[str, Any]:
            roles = frames["roles"]
            component = frames["component"]
            common.require_columns(roles, [*GRAIN, "team", "position", "primary_qb_flag"], "roles")
            common.require_columns(component, [*GRAIN, "team", "position", "projected_team_pass_attempts", "projected_qb_pass_attempts"], "component")
            joined = component.merge(roles[[*GRAIN, "primary_qb_flag"]], on=GRAIN, how="left", validate="one_to_one")
            violations = []
            for team, group in joined.groupby("team", sort=False):
                if int(numeric(group["primary_qb_flag"]).fillna(0).gt(0).sum()) != 1:
                    violations.append((team, "primary_qb_count"))
                    continue
                qbs = group.loc[group["position"].astype(str).str.upper().eq("QB")].copy()
                team_pass = float(numeric(group["projected_team_pass_attempts"]).dropna().median())
                if not math.isfinite(team_pass) or team_pass < 0:
                    violations.append((team, "invalid_team_pass_volume"))
                    continue
                qbv = numeric(qbs["projected_qb_pass_attempts"]).fillna(0.0)
                threshold = 0.5 * team_pass
                full = qbv.ge(threshold - TOL) & qbv.gt(TOL) if team_pass > TOL else qbv.gt(TOL)
                if int(full.sum()) > 1:
                    violations.append((team, "multiple_full_qb1_volumes"))
            if violations:
                raise AssertionError(f"QB volume validation failed: {violations[:10]}")
            return {"teams_checked": int(joined["team"].nunique()), "multiple_full_qb1_teams": 0}

        record("no two QBs receive full QB1 volume for one team", qb_volume)

        def share_bounds() -> dict[str, Any]:
            allocation = frames["allocation"]
            component = frames["component"]
            cols = ["raw_projected_target_share", "allocated_target_share", "raw_projected_carry_share", "allocated_carry_share"]
            common.require_columns(allocation, [*GRAIN, "team", *cols, "allocated_def_participation"], "allocation")
            for c in [*cols, "allocated_def_participation"]:
                value = numeric(allocation[c])
                finite = value.notna()
                if ((value.loc[finite] < -TOL) | (value.loc[finite] > 1.0 + TOL)).any():
                    raise AssertionError(f"{c}: share/participation outside [0,1]")
            team_volume = component.groupby(TEAM_GRAIN, sort=False).agg(
                pass_volume=("projected_team_pass_attempts", "first"),
                rush_volume=("projected_team_rush_attempts", "first"),
            ).reset_index()
            sums = allocation.groupby(TEAM_GRAIN, sort=False).agg(
                target_sum=("allocated_target_share", "sum"), carry_sum=("allocated_carry_share", "sum")
            ).reset_index().merge(team_volume, on=TEAM_GRAIN, how="left", validate="one_to_one")
            positive_pass = numeric(sums["pass_volume"]).gt(TOL)
            positive_rush = numeric(sums["rush_volume"]).gt(TOL)
            if (numeric(sums.loc[positive_pass, "target_sum"]) - 1.0).abs().gt(1e-8).any():
                raise AssertionError("Allocated target shares do not reconcile to 1 for positive-volume teams")
            if (numeric(sums.loc[positive_rush, "carry_sum"]) - 1.0).abs().gt(1e-8).any():
                raise AssertionError("Allocated carry shares do not reconcile to 1 for positive-volume teams")
            return {
                "teams_checked": len(sums),
                "max_target_sum_error": float((numeric(sums.loc[positive_pass, "target_sum"]) - 1.0).abs().max()),
                "max_carry_sum_error": float((numeric(sums.loc[positive_rush, "carry_sum"]) - 1.0).abs().max()),
            }

        record("target and carry shares are logically bounded and reconciled", share_bounds)

        def kicker_majority() -> dict[str, Any]:
            roles = frames["roles"]
            component = frames["component"]
            common.require_columns(roles, [*GRAIN, "team", "position", "primary_kicker_flag"], "roles")
            common.require_columns(component, [*GRAIN, "team", "projected_fg_attempts", "projected_pat_attempts"], "component")
            joined = component.merge(roles[[*GRAIN, "primary_kicker_flag"]], on=GRAIN, how="left", validate="one_to_one")
            joined["_kick_opp"] = numeric(joined["projected_fg_attempts"]).fillna(0.0) + numeric(joined["projected_pat_attempts"]).fillna(0.0)
            bad = []
            min_share = 1.0
            for team, group in joined.groupby("team", sort=False):
                primary = numeric(group["primary_kicker_flag"]).fillna(0).gt(0)
                if int(primary.sum()) != 1:
                    bad.append((team, "primary_kicker_count"))
                    continue
                total = float(group["_kick_opp"].sum())
                primary_opp = float(group.loc[primary, "_kick_opp"].sum())
                if total <= TOL:
                    bad.append((team, "zero_team_kicking_opportunity"))
                    continue
                share = primary_opp / total
                min_share = min(min_share, share)
                if share <= 0.5 + TOL:
                    bad.append((team, f"primary_share={share:.6f}"))
            if bad:
                raise AssertionError(f"Primary kicker majority failed: {bad[:10]}")
            return {"teams_checked": int(joined["team"].nunique()), "minimum_primary_kicker_opportunity_share": float(min_share)}

        record("one primary kicker receives majority kicking opportunity", kicker_majority)

        def defensive_plausibility() -> dict[str, Any]:
            eligibility_path = prop / "config" / "target_eligibility.yaml"
            with eligibility_path.open("r", encoding="utf-8-sig") as h:
                eligibility = yaml.safe_load(h)
            universe = frames["universe"]
            allocation = frames["allocation"][[*GRAIN, "allocated_def_participation"]].copy()
            context = universe[[*GRAIN, "position", "eligibility_status", "injury_game_status", "eligibility_reason"]].merge(
                allocation, on=GRAIN, how="left", validate="one_to_one"
            )
            defensive_targets = {
                target for target in ("tackles", "sacks")
                if target in production_targets
            }
            active_def = frames["active"].loc[
                frames["active"]["target"].astype(str).isin(defensive_targets)
            ].merge(
                context, on=GRAIN, how="left", validate="many_to_one", suffixes=("", "_universe")
            )
            if active_def["position_universe"].isna().any():
                raise AssertionError("Defensive active projection missing universe context")
            for target in sorted(defensive_targets):
                allowed = {str(x).strip().upper() for x in eligibility[target]["eligible_positions"]}
                rows = active_def.loc[active_def["target"].astype(str).eq(target)]
                bad_pos = ~rows["position_universe"].astype(str).str.upper().isin(allowed)
                bad_status = rows["eligibility_status_universe"].astype(str).str.casefold().ne("eligible")
                out = rows["injury_game_status_universe"].astype(str).str.casefold().eq("out")
                participation = numeric(rows["allocated_def_participation"])
                bad_part = participation.isna() | participation.lt(-TOL) | participation.gt(1.0 + TOL)
                if (bad_pos | bad_status | out | bad_part).any():
                    sample = rows.loc[bad_pos | bad_status | out | bad_part, ["game_id", "player_id", "target", "position_universe", "eligibility_status_universe", "injury_game_status_universe"]].head(10).to_dict("records")
                    raise AssertionError(f"Implausible defensive projection rows: {sample}")
            return {"defensive_rows_checked": len(active_def), "implausible_rows": 0}

        record("defensive projections limited to plausible participants", defensive_plausibility)

        def schema_contract() -> dict[str, Any]:
            return validate_model_schemas(
                prop, frames["features"], production_targets
            )

        record("current feature schema matches every selected production model", schema_contract)

        def market_contract() -> dict[str, Any]:
            details = run_market_audit(prop)
            for log_name in (f"week_projections_{season}_week_{week}.json", f"wide_output_{season}_week_{week}.json"):
                log = read_json(prop / "logs" / log_name)
                if log.get("market_features_used") is not False:
                    raise AssertionError(f"{log_name}: market_features_used must be false")
                if log.get("market_exclusion_passed") is not True:
                    raise AssertionError(f"{log_name}: market_exclusion_passed must be true")
            details["weekly_logs_checked"] = 2
            return details

        record("market audit passes and weekly outputs report no market features", market_contract)

    overall_passed = bool(checks) and all(bool(c.get("passed")) for c in checks)
    payload = {
        "status": "passed" if overall_passed else "failed",
        "season": season,
        "week": week,
        "generated_at": generated_at,
        "output": str(output_path.resolve().relative_to(repo.resolve())).replace("\\", "/"),
        "checks_total": len(checks),
        "checks_passed": sum(1 for c in checks if c.get("passed") is True),
        "checks_failed": sum(1 for c in checks if c.get("passed") is not True),
        "checks": checks,
        "production_targets": list(production_targets),
        "deferred_targets": list(deferred_targets),
        "failure_returns_nonzero_exit_code": True,
        "market_features_used": False,
    }
    write_json_atomic(output_path, payload)
    print(json.dumps({"script": Path(__file__).name, "payload": payload}, sort_keys=True, default=str))
    print("WEEK VALIDATION: PASS" if overall_passed else "WEEK VALIDATION: FAIL")
    return exit_code(overall_passed)


if __name__ == "__main__":
    raise SystemExit(main())
