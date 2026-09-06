#!/usr/bin/env python3
"""Independent acceptance validator for Prop Engine Issue 31."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import common

TARGETS = [
    "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
    "receiving_yards", "receiving_tds", "kicking_points", "tackles", "sacks",
]
GRAIN = ["season", "week", "game_id", "player_id"]
TEAM_HISTORY_ALIASES = {"SD": "LAC", "OAK": "LV", "STL": "LAR"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    return p.parse_args()


def clean(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    x = str(v).strip()
    return "" if x.casefold() in {"", "nan", "none", "null", "<na>", "nat"} else x


def norm_team(v: Any) -> str:
    x = common.normalize_team(v)
    return TEAM_HISTORY_ALIASES.get(x, x)


def norm_id(v: Any) -> str:
    return common.normalize_player_id(v)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise AssertionError(f"Expected JSON object: {path}")
    return value


def manifest_hash(numeric: list[str], categorical: list[str]) -> str:
    payload = [
        *[{"name": x, "type": "numeric"} for x in numeric],
        *[{"name": x, "type": "categorical"} for x in categorical],
    ]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_feature_names(path: Path) -> list[str] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("feature_names="):
                return line.rstrip("\r\n").split("=", 1)[1].split()
    return None


def selected_manifests(
    repo: Path,
) -> tuple[list[tuple[str, Path, Path, dict[str, Any]]], list[tuple[str, str]]]:
    root = repo / "docs/win/football/nfl/prop_engine/models"
    result: list[tuple[str, Path, Path, dict[str, Any]]] = []
    proxy_dependencies: list[tuple[str, str]] = []
    seen: set[Path] = set()

    def add(owner: str, manifest_path: Path, model_path: Path) -> None:
        if manifest_path.resolve() in seen:
            return
        if not manifest_path.is_file():
            raise AssertionError(f"Missing selected-model feature manifest: {manifest_path}")
        seen.add(manifest_path.resolve())
        result.append((owner, manifest_path, model_path, read_json(manifest_path)))

    for target in TARGETS:
        selected_path = root / target / "selected_model.json"
        if not selected_path.is_file():
            raise AssertionError(f"Missing selected model: {selected_path}")
        selected = read_json(selected_path)
        architecture = clean(
            selected.get("selected_architecture") or selected.get("selected_candidate")
        )
        if architecture in {"direct", "direct_component_blend"}:
            add(
                f"{target}:direct",
                root / target / "feature_manifest.json",
                root / target / "direct_model.txt",
            )
        if architecture in {"component", "direct_component_blend"}:
            deps = selected.get("component_dependencies") or []
            if not deps:
                raise AssertionError(f"Selected component target lacks dependencies: {target}")

            proxy_features: set[str] = set()
            for proxy_key in ("goal_line_volume_proxy_order", "red_zone_volume_proxy_order"):
                values = selected.get(proxy_key) or []
                if not isinstance(values, list):
                    raise AssertionError(f"{target} {proxy_key} must be a list")
                proxy_features.update(clean(value) for value in values if clean(value))

            for dep in deps:
                dep = clean(dep)
                cp = root / "components" / dep / "feature_manifest.json"
                ep = root / "efficiency" / dep / "feature_manifest.json"
                if cp.is_file():
                    add(f"{target}:component:{dep}", cp, cp.with_name("model.txt"))
                elif ep.is_file():
                    add(f"{target}:efficiency:{dep}", ep, ep.with_name("model.txt"))
                elif dep in proxy_features:
                    proxy_dependencies.append((target, dep))
                else:
                    raise AssertionError(f"Missing selected dependency manifest: {dep}")
    return result, proxy_dependencies


def compare_float(actual: pd.Series, expected: pd.Series, label: str) -> None:
    a = num(actual).to_numpy(dtype="float64")
    e = num(expected).to_numpy(dtype="float64")
    same_nan = np.isnan(a) & np.isnan(e)
    equal = np.isclose(a, e, rtol=0, atol=1e-6, equal_nan=True)
    bad = ~(equal | same_nan)
    if bad.any():
        idx = np.flatnonzero(bad)[:10]
        sample = [(float(a[i]) if not np.isnan(a[i]) else None, float(e[i]) if not np.isnan(e[i]) else None) for i in idx]
        raise AssertionError(f"{label} mismatch; sample={sample}")


def main() -> int:
    args = parse_args()
    season, week = args.season, args.week
    repo = common.repo_root()
    prop = common.prop_root()
    config = common.load_config()

    builder = prop / "scripts/project/build_current_features.py"
    output = prop / f"data/current/features/{season}_week_{week}_features.parquet"
    manifest_path = prop / f"data/current/features/{season}_week_{week}_feature_manifest.json"
    log_path = prop / f"logs/current_features_{season}_week_{week}.json"
    universe_path = prop / f"data/current/{season}_week_{week}_universe.parquet"
    roles_path = prop / f"data/current/{season}_week_{week}_roles.parquet"
    historical_path = repo / config["paths"]["historical_features"]
    hist_manifest_path = historical_path.with_name("feature_manifest.json")
    position_allowed_path = repo / config["paths"]["position_allowed"]
    weather_path = repo / config["paths"]["current_weather"].format(week=week)
    travel_path = repo / config["paths"]["current_travel"].format(season=season, week=week)

    print("CHECK 01: required builder, output, manifest, log, and required-read contract")
    for path in [
        builder, output, manifest_path, log_path, universe_path, roles_path,
        historical_path, hist_manifest_path, position_allowed_path, weather_path, travel_path,
    ]:
        if not path.is_file():
            raise AssertionError(f"Missing Issue 31 artifact/input: {path}")

    source_text = builder.read_text(encoding="utf-8")
    required_markers = [
        "stats_player_week_", "snap_counts_", "pbp_participation_",
        'config["paths"]["pbp_pattern"]', 'config["paths"]["team_stats_pattern"]',
        'config["paths"]["current_weather"]', 'config["paths"]["current_travel"]',
        'config["paths"]["historical_features"]', 'config["paths"]["position_allowed"]',
        "_week" + '"]' + ".lt(week)",
    ]
    missing_markers = [x for x in required_markers if x not in source_text]
    if missing_markers:
        raise AssertionError(f"Builder missing required-source/filter marker(s): {missing_markers}")

    current_manifest = read_json(manifest_path)
    log = read_json(log_path)
    if current_manifest.get("market_features_used") is not False:
        raise AssertionError("Current feature manifest must declare market_features_used=false")
    if log.get("status") != "passed":
        raise AssertionError("Current feature log status is not passed")
    if log.get("weather_join_key") != "game_id" or log.get("travel_join_key") != "game_id":
        raise AssertionError("Weather/travel join key contract is not game_id")

    print("CHECK 02: exact historical feature names, no targets/forbidden fields, and dtype parity")
    hist_manifest = read_json(hist_manifest_path)
    leading = list(hist_manifest["leading_columns"])
    feature_columns = list(hist_manifest["feature_columns"])
    expected_columns = leading + [c for c in feature_columns if c not in leading]

    frame = pd.read_parquet(output)
    if list(frame.columns) != expected_columns:
        missing = [c for c in expected_columns if c not in frame.columns]
        extra = [c for c in frame.columns if c not in expected_columns]
        raise AssertionError(
            f"Current feature headers differ from canonical historical feature schema; "
            f"missing={missing[:20]} extra={extra[:20]}"
        )
    bad_target = [c for c in frame.columns if c.startswith("target_")]
    if bad_target:
        raise AssertionError(f"Target columns leaked into current features: {bad_target}")
    bad_audit = [c for c in frame.columns if c.startswith("audit_")]
    if bad_audit:
        raise AssertionError(f"Historical audit columns must not be current model features: {bad_audit}")
    common.reject_forbidden_feature_columns(frame.columns, config)

    prior = season - 1
    try:
        ref = pd.read_parquet(
            historical_path,
            columns=expected_columns,
            filters=[("season", "==", prior)],
        )
    except Exception:
        ref = pd.read_parquet(historical_path, columns=expected_columns)
        ref = ref.loc[num(ref["season"]).eq(prior)].copy()
    if ref.empty:
        raise AssertionError(f"Historical features have no prior-season {prior} rows")
    dtype_bad = {
        c: (str(ref[c].dtype), str(frame[c].dtype))
        for c in expected_columns
        if str(ref[c].dtype) != str(frame[c].dtype)
    }
    if dtype_bad:
        raise AssertionError(f"Historical/current dtype mismatch: {dict(list(dtype_bad.items())[:20])}")

    print("CHECK 03: canonical grain, Issue 29/30 row parity, and current role overrides")
    common.ensure_unique(frame, GRAIN, "Issue 31 current features")
    roles = pd.read_parquet(roles_path)
    universe = pd.read_parquet(universe_path)
    if len(frame) != len(roles):
        raise AssertionError(f"Feature rows must equal eligible role rows: {len(frame)} != {len(roles)}")

    for x in [frame, roles, universe]:
        x["game_id"] = x["game_id"].map(norm_id)
        x["player_id"] = x["player_id"].map(norm_id)
        x["team"] = x["team"].map(norm_team)
    universe["opponent"] = universe["opponent"].map(norm_team)

    role_probe = frame[GRAIN + ["team"]].merge(
        roles[GRAIN + ["team", "primary_kicker_flag", "starter_flag"]],
        on=GRAIN + ["team"], how="left", validate="one_to_one",
    )
    if role_probe["primary_kicker_flag"].isna().any():
        raise AssertionError("Current feature grain does not map to every Issue 30 role row")
    if "role_primary_kicker_flag" in frame:
        expected = num(role_probe["primary_kicker_flag"]).fillna(0).astype(int)
        actual = num(frame["role_primary_kicker_flag"]).fillna(0).astype(int)
        if not actual.reset_index(drop=True).equals(expected.reset_index(drop=True)):
            raise AssertionError("role_primary_kicker_flag does not match Issue 30 roles")

    u_probe = frame[GRAIN].merge(
        universe[GRAIN + ["depth_rank", "injury_game_status", "home_flag", "opponent"]],
        on=GRAIN, how="left", validate="one_to_one",
    )
    if u_probe["opponent"].isna().any():
        raise AssertionError("Current feature grain does not map to Issue 29 universe")
    if "role_depth_rank_pregame" in frame:
        compare_float(frame["role_depth_rank_pregame"], u_probe["depth_rank"], "current depth rank")

    print("CHECK 04: Week 1 priors / strict completed-current-week source fence")
    if week == 1:
        season_to_date = [c for c in feature_columns if c.endswith("_season_to_date")]
        nonnull = {c: int(frame[c].notna().sum()) for c in season_to_date if frame[c].notna().any()}
        if nonnull:
            raise AssertionError(
                "Week 1 current-season season_to_date features must reset before any game; "
                f"nonnull={dict(list(nonnull.items())[:20])}"
            )
        if log.get("week1_prior_used") is not True or log.get("prior_season") != prior:
            raise AssertionError("Week 1 explicit prior-season policy missing from run log")
        if log.get("current_season_completed_weeks_allowed") != []:
            raise AssertionError("Week 1 must have no completed current-season weeks")
    else:
        expected_weeks = list(range(1, week))
        if log.get("current_season_completed_weeks_allowed") != expected_weeks:
            raise AssertionError("Current-source completed-week allow-list mismatch")

    source_audit = log.get("current_source_audit", {})
    for name in ["player_stats", "snap_counts", "participation", "pbp", "team_stats"]:
        if name not in source_audit:
            raise AssertionError(f"Missing current source audit: {name}")
        max_week = source_audit[name].get("max_source_week_used")
        if max_week is not None and int(max_week) >= week:
            raise AssertionError(f"Current source {name} used same/future week {max_week}")
        if week == 1 and int(source_audit[name].get("rows_used", 0)) != 0:
            raise AssertionError(f"Week 1 current source {name} used realized rows")

    print("CHECK 05: independently verify weather and travel by game_id")
    weather = pd.read_csv(weather_path, low_memory=False)
    travel = pd.read_csv(travel_path, low_memory=False)
    weather["game_id"] = weather["game_id"].map(norm_id)
    travel["game_id"] = travel["game_id"].map(norm_id)
    if weather["game_id"].duplicated().any() or travel["game_id"].duplicated().any():
        raise AssertionError("Weather/travel game_id must be unique")

    env = frame[["game_id", "home_flag"]].copy()
    env = env.merge(
        weather[["game_id", "temperature", "wind_speed"]],
        on="game_id", how="left", validate="many_to_one",
    )
    env = env.merge(
        travel[["game_id", "miles_traveled", "time_zones_crossed"]],
        on="game_id", how="left", validate="many_to_one",
    )
    if "environment_temperature" in frame:
        compare_float(frame["environment_temperature"], env["temperature"], "weather temperature game_id join")
    if "environment_wind" in frame:
        compare_float(frame["environment_wind"], env["wind_speed"], "weather wind game_id join")
    home = num(env["home_flag"]).fillna(0).eq(1)
    miles = num(env["miles_traveled"])
    zones = num(env["time_zones_crossed"])
    if "environment_team_miles_traveled" in frame:
        compare_float(
            frame["environment_team_miles_traveled"],
            pd.Series(np.where(home, 0.0, miles), index=frame.index),
            "team travel game_id join",
        )
    if "environment_opponent_miles_traveled" in frame:
        compare_float(
            frame["environment_opponent_miles_traveled"],
            pd.Series(np.where(home, miles, 0.0), index=frame.index),
            "opponent travel game_id join",
        )
    if "environment_team_time_zones_crossed" in frame:
        compare_float(
            frame["environment_team_time_zones_crossed"],
            pd.Series(np.where(home, 0.0, zones), index=frame.index),
            "team time-zone travel game_id join",
        )

    print("CHECK 06: selected direct/component/efficiency manifest schemas and hashes")
    manifests, proxy_dependencies = selected_manifests(repo)
    checked_hashes = 0
    checked_models = 0
    deferred_efficiency_derived = 0
    for owner, mpath, model_path, manifest in manifests:
        numeric = list(manifest.get("numeric_features", []))
        categorical = list(manifest.get("categorical_features", []))
        ordered = numeric + categorical
        derived = [clean(value) for value in manifest.get("derived_features", []) if clean(value)]
        canonical = [clean(value) for value in manifest.get("canonical_features", []) if clean(value)]

        if derived:
            if not canonical:
                raise AssertionError(
                    f"Selected efficiency manifest declares derived_features without canonical_features: {owner}"
                )
            if set(canonical) & set(derived):
                raise AssertionError(f"Selected manifest canonical/derived overlap: {owner}")
            if set(canonical + derived) != set(ordered):
                raise AssertionError(
                    f"Selected manifest canonical+derived schema mismatch: {owner}"
                )
            leaked = [c for c in derived if c in frame.columns]
            if leaked:
                raise AssertionError(
                    f"Model-local efficiency derived feature leaked into global current table "
                    f"{owner}: {leaked[:10]}"
                )
            deferred_efficiency_derived += len(derived)
        else:
            canonical = ordered

        missing = [c for c in canonical if c not in frame.columns]
        required = [clean(value) for value in manifest.get("required_features", []) if clean(value)]
        unknown_required = [c for c in required if c not in ordered]
        required_canonical = [c for c in required if c not in derived]
        missing_required = [c for c in required_canonical if c not in frame.columns]
        if unknown_required:
            raise AssertionError(
                f"Selected model required_features outside manifest schema {owner}: "
                f"{unknown_required[:10]}"
            )
        if missing or missing_required:
            raise AssertionError(
                f"Selected model {owner} missing canonical current feature(s): "
                f"{missing[:10]} required={missing_required[:10]}"
            )

        common.reject_forbidden_feature_columns(ordered, config)
        canonical_num = [c for c in numeric if c in canonical]
        canonical_cat = [c for c in categorical if c in canonical]
        bad_num = [c for c in canonical_num if not pd.api.types.is_numeric_dtype(frame[c].dtype) and not pd.api.types.is_bool_dtype(frame[c].dtype)]
        bad_cat = [c for c in canonical_cat if not (pd.api.types.is_object_dtype(frame[c].dtype) or pd.api.types.is_string_dtype(frame[c].dtype) or isinstance(frame[c].dtype, pd.CategoricalDtype))]
        if bad_num or bad_cat:
            raise AssertionError(f"Selected model canonical type mismatch {owner}: num={bad_num[:5]} cat={bad_cat[:5]}")

        calculated = manifest_hash(numeric, categorical)
        stored = manifest.get("feature_hash")
        if stored is not None:
            checked_hashes += 1
            if str(stored) != calculated:
                raise AssertionError(
                    f"Selected model stored feature hash mismatch {owner}: {stored} != {calculated}"
                )
        names = model_feature_names(model_path)
        if names is not None:
            checked_models += 1
            if names != ordered:
                raise AssertionError(f"Persisted LightGBM feature order mismatch: {owner}")

    for target, feature in proxy_dependencies:
        if feature not in frame.columns:
            raise AssertionError(
                f"Selected component proxy feature missing from current table: {target}:{feature}"
            )
        if feature.startswith("target_") or feature.startswith("audit_"):
            raise AssertionError(f"Invalid selected component proxy feature: {target}:{feature}")
        common.reject_forbidden_feature_columns([feature], config)

    manifest_checks = current_manifest.get("selected_model_schema_checks", [])
    if len(manifest_checks) != len(manifests):
        raise AssertionError(
            f"Current manifest selected-model check count mismatch: "
            f"{len(manifest_checks)} != {len(manifests)}"
        )

    proxy_checks = current_manifest.get("selected_proxy_feature_checks", [])
    expected_proxy = {(target, feature) for target, feature in proxy_dependencies}
    actual_proxy = {
        (clean(item.get("target")), clean(item.get("feature")))
        for item in proxy_checks
        if isinstance(item, dict) and item.get("present") is True
    }
    if actual_proxy != expected_proxy:
        raise AssertionError(
            f"Current manifest proxy dependency mismatch: actual={sorted(actual_proxy)} "
            f"expected={sorted(expected_proxy)}"
        )

    print("CHECK 07: market exclusion and current feature acceptance summary")
    market_path = prop / "evaluation/market_exclusion_audit.json"
    market = read_json(market_path)
    if market.get("passed") is not True:
        raise AssertionError("Issue 28 market exclusion audit is not passing")
    if log.get("no_target_columns") is not True or log.get("no_forbidden_columns") is not True:
        raise AssertionError("Issue 31 run log leakage declarations are not true")
    if log.get("historical_dtype_match") is not True:
        raise AssertionError("Issue 31 run log historical dtype match is not true")

    print(f"season={season}")
    print(f"week={week}")
    print(f"games={frame['game_id'].nunique()}")
    print(f"teams={frame['team'].nunique()}")
    print(f"rows={len(frame)}")
    print(f"features={len(feature_columns)}")
    print(f"columns={len(frame.columns)}")
    print(f"selected_manifests={len(manifests)}")
    print(f"stored_feature_hashes_checked={checked_hashes}")
    print(f"lightgbm_feature_orders_checked={checked_models}")
    print(f"efficiency_derived_features_deferred={deferred_efficiency_derived}")
    print(f"week1_prior_used={str(log.get('week1_prior_used')).lower()}")
    print("weather_join_key=game_id")
    print("travel_join_key=game_id")
    print("target_columns=0")
    print("market_features_used=false")
    print("ISSUE 31 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
