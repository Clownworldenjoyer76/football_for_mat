#!/usr/bin/env python3
"""Produce final current-week blended NFL player projections.

Issue 36 contract
-----------------
REQUIRED READS
  data/current/{season}_week_{week}_universe.parquet
  data/current/{season}_week_{week}_component_projections.parquet
  data/current/{season}_week_{week}_direct_projections.parquet
  models/{target}/selected_model.json
  models/calibration/{target}_calibration.json
  models/production_registry.json

ADDITIONAL PRODUCTION CONTEXT
  data/current/{season}_week_{week}_allocated_opportunity.parquet
  data/current/features/{season}_week_{week}_features.parquet
  config/target_eligibility.yaml
  persisted component/efficiency models required by selected component formulas

WRITES
  output/{season}/week_{week}_player_projections.csv
  output/{season}/week_{week}_active_player_projections.csv
  logs/week_projections_{season}_week_{week}.json

POLICY
  - Selected architecture and blend weights come only from Issue 25 selected_model.json.
  - Target/carry component formulas consume Issue 34 allocated shares.
  - Item 33 is the canonical owner of persisted component/efficiency scoring and
    all nine target component formulas; final assembly consumes those outputs.
  - Red-zone and goal-line Item 33 exposures are normalized exactly as declared
    by Issue 25 (positive predicted shares within team-game), limited to current
    target-eligible player rows.
  - Issue 26 calibration is applied to the selected point prediction. Quantile
    targets use calibrated q50 as the displayed point unless count calibration is
    also present; count targets use calibrated expected_count and calibrated p1+/p2+.
  - Ineligible target/player combinations remain in the audit output with zero and
    an explicit reason, and never appear in the active-only output.
  - Display rounding is applied only after all model/blend/calibration math.
  - No sportsbook/market-derived input is permitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    import lightgbm as lgb
except ModuleNotFoundError as exc:
    raise SystemExit("Issue 36 requires LightGBM in the active environment.") from exc

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
TRAIN_DIR = SCRIPTS_ROOT / "train"
for p in (SCRIPTS_ROOT, TRAIN_DIR, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import common

_CONFIG_CONTRACT = common.load_config()
import project_components as pc
import train_opportunity_models as opportunity
import train_efficiency_models as efficiency

GRAIN = ["season", "week", "game_id", "player_id"]
TEAM_GRAIN = ["season", "week", "game_id", "team"]
TARGETS = list(_CONFIG_CONTRACT["targets"].keys())
# SIX_TARGET_PRODUCTION_REGISTRY_MODE
OUTPUT_COLUMNS = [
    "season",
    "week",
    "game_id",
    "game_date",
    "game_time",
    "player_id",
    "player_name",
    "team",
    "opponent",
    "position",
    "target",
    "projection",
    "low",
    "high",
    "probability_1_plus",
    "probability_2_plus",
    "eligibility_status",
    "eligibility_reason",
    "role_status",
    "injury_game_status",
    "depth_rank",
    "model_architecture",
    "model_version",
    "feature_asof",
    "generated_at",
]

UNIVERSE_REQUIRED = [
    *GRAIN,
    "game_date",
    "game_time",
    "kickoff_timestamp",
    "player_name",
    "team",
    "opponent",
    "position",
    "position_group",
    "eligibility_status",
    "eligibility_reason",
    "role_status",
    "injury_game_status",
    "depth_rank",
]
COMPONENT_REQUIRED = list(pc.OUTPUT_COLUMNS)
DIRECT_COLUMNS = {target: f"direct_{target}" for target in TARGETS}
DIRECT_REQUIRED = [*GRAIN, *DIRECT_COLUMNS.values()]
ALLOCATION_REQUIRED = [
    *GRAIN,
    "team",
    "allocated_target_share",
    "allocated_carry_share",
    "allocated_def_participation",
]
CAL_CONTEXT_REQUIRED = [
    *GRAIN,
    "team",
    "position",
    "position_group",
    "history_no_nfl_history_flag",
    "history_history_games",
    "role_starter_promotion_flag",
    *pc.RED_ZONE_PASS_VOLUME_FEATURES,
    *pc.GOAL_LINE_RUSH_VOLUME_FEATURES,
]

EXTRA_OPPORTUNITY = [
    "player_red_zone_target_share",
    "player_goal_line_carry_share",
    "opponent_offensive_plays",
    "opponent_dropbacks",
    "player_defensive_participation",
]
EXTRA_EFFICIENCY = [
    "passing_td_rate",
    "rushing_td_per_goal_line_carry",
    "receiving_td_per_red_zone_target",
    "extra_point_conversion",
    "tackle_rate_per_defensive_play",
    "sack_rate_per_defensive_play",
]

COUNT_TARGETS = {
    "passing_tds",
    "rushing_tds",
    "receiving_tds",
    "tackles",
    "sacks",
}
ONE_DECIMAL_TARGETS = {
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "kicking_points",
    "tackles",
}
THREE_DECIMAL_TARGETS = {
    "passing_tds",
    "rushing_tds",
    "receiving_tds",
    "sacks",
}
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Produce final weekly player projections.")
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--week", type=int, required=True)
    return p.parse_args()


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(common.repo_root().resolve())).replace("\\", "/")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON missing: {path}")
    with path.open("r", encoding="utf-8-sig") as h:
        value = json.load(h)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required YAML missing: {path}")
    with path.open("r", encoding="utf-8-sig") as h:
        value = yaml.safe_load(h)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
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
            json.dump(payload, h, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str)
            h.write("\n")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    common.write_csv_atomic(frame, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda: h.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            "Market-exclusion preflight failed before final weekly projection. "
            f"stdout={cp.stdout[-2000:]!r} stderr={cp.stderr[-2000:]!r}"
        )
    return {"passed": True, "validator": repo_relative(path)}


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"} else text


def coalesce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing deterministic volume proxy columns: {missing}")
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    for c in columns:
        out = out.where(out.notna(), numeric(frame[c]))
    return out


def normalize_position_group(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("UNKNOWN")
        .str.strip()
        .str.upper()
        .replace("", "UNKNOWN")
    )


def load_selected_contracts(
    prop: Path,
    targets: list[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    supported = {"direct", "component", "direct_component_blend"}
    for target in targets:
        path = prop / "models" / target / "selected_model.json"
        payload = load_json(path)
        if payload.get("target") != target:
            raise ValueError(f"{path}: target mismatch")
        architecture = clean_text(payload.get("selected_architecture") or payload.get("selected_candidate"))
        if architecture not in supported:
            raise ValueError(f"{target}: unsupported production architecture {architecture!r}")
        if payload.get("market_features_used") is not False:
            raise ValueError(f"{target}: selected model violates market exclusion")
        if payload.get("test_used_for_selection") is not False:
            raise ValueError(f"{target}: reporting test used for selection")
        if architecture == "direct_component_blend":
            weights = payload.get("blend_weights")
            if not isinstance(weights, dict):
                raise ValueError(f"{target}: blend architecture missing blend_weights")
            wd = float(weights["direct"])
            wc = float(weights["component"])
            if wd < 0 or wc < 0 or abs((wd + wc) - 1.0) > 1e-9:
                raise ValueError(f"{target}: invalid blend weights direct={wd} component={wc}")
        result[target] = payload
    return result


def load_calibrations(
    prop: Path,
    selected: dict[str, dict[str, Any]],
    targets: list[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for target in targets:
        path = prop / "models" / "calibration" / f"{target}_calibration.json"
        payload = load_json(path)
        if payload.get("target") != target:
            raise ValueError(f"{path}: target mismatch")
        if payload.get("selected_architecture") != selected[target].get("selected_architecture"):
            raise ValueError(f"{target}: calibration architecture differs from selected model")
        if payload.get("market_features_used") is not False or payload.get("forbidden_features_used") is not False:
            raise ValueError(f"{target}: calibration violates market exclusion")
        mode = clean_text(payload.get("calibration_mode"))
        if mode not in {"quantiles", "count", "quantiles_and_count"}:
            raise ValueError(f"{target}: unsupported calibration mode {mode!r}")
        result[target] = payload
    return result


def _entry_version(entry: Any) -> str:
    if isinstance(entry, str):
        return clean_text(entry)
    if not isinstance(entry, dict):
        return ""
    for key in ("model_version", "production_version", "active_version", "version", "release"):
        value = clean_text(entry.get(key))
        if value:
            return value
    return ""


def resolve_registry_versions(
    registry: dict[str, Any],
    registry_path: Path,
) -> tuple[dict[str, Any], dict[str, str], list[str], list[str]]:
    _ = registry_path
    if list(registry.keys()) != TARGETS:
        raise ValueError("Production registry target set/order mismatch.")

    versions: dict[str, Any] = {}
    sources: dict[str, str] = {}
    production_targets: list[str] = []
    deferred_targets: list[str] = []

    for target in TARGETS:
        entry = registry[target]
        if not isinstance(entry, dict):
            raise ValueError(f"{target}: invalid production registry entry.")
        if list(entry.keys()) != ["production_approved", "version"]:
            raise ValueError(f"{target}: unexpected production registry entry schema.")

        approved = entry.get("production_approved")
        version = entry.get("version")
        if approved is True:
            version_text = clean_text(version)
            if not version_text:
                raise ValueError(f"{target}: approved target has blank production version.")
            versions[target] = version_text
            sources[target] = f"root.{target}"
            production_targets.append(target)
        elif approved is False and version is None:
            versions[target] = None
            sources[target] = f"root.{target}"
            deferred_targets.append(target)
        else:
            raise ValueError(
                f"{target}: invalid registry state approved={approved!r} version={version!r}"
            )

    if not production_targets:
        raise ValueError("No approved production targets in registry.")
    return versions, sources, production_targets, deferred_targets


TEAM_OPPONENT_CURRENT_SOURCE = {
    "player_defensive_opponent_plays_roll3": "team_offensive_plays_roll3_mean",
    "player_defensive_opponent_dropbacks_roll3": "team_dropbacks_roll3_mean",
    "player_defensive_opponent_rush_rate_roll3": "team_rush_rate_roll3_mean",
    "player_defensive_opponent_pass_rate_roll3": "team_pass_rate_roll3_mean",
}
TEAM_DEF_SACK_FEATURE = "player_defensive_team_def_sack_rate_roll3"


def strict_prior_team_def_sack_rate(
    config: dict[str, Any],
    repo: Path,
    season: int,
) -> pd.DataFrame:
    path = repo / str(config["paths"]["opponent_opportunity"])
    raw = common.read_parquet_required(
        path,
        ["season", "week", "team", "sacks", "opponent_dropbacks"],
    ).copy()
    raw["season"] = pd.to_numeric(raw["season"], errors="raise").astype(int)
    raw["week"] = pd.to_numeric(raw["week"], errors="raise").astype(int)
    raw = raw.loc[raw["season"].lt(season)].copy()
    raw["_team_key"] = raw["team"].map(opportunity.canonical_team)
    raw["_sacks"] = numeric(raw["sacks"])
    raw["_dropbacks"] = numeric(raw["opponent_dropbacks"])
    raw["_rate"] = np.where(
        raw["_sacks"].notna() & raw["_dropbacks"].notna() & raw["_dropbacks"].ne(0.0),
        raw["_sacks"] / raw["_dropbacks"],
        np.nan,
    )
    raw = raw.sort_values(["_team_key", "season", "week"], kind="mergesort")
    records: list[dict[str, Any]] = []
    for team, frame in raw.groupby("_team_key", sort=False):
        rates = frame["_rate"].dropna().to_numpy(dtype="float64")
        records.append({
            "_team_key": team,
            TEAM_DEF_SACK_FEATURE: (float(np.mean(rates[-3:])) if len(rates) else np.nan),
        })
    out = pd.DataFrame(records)
    common.ensure_unique(out, ["_team_key"], "Issue 36 strict-prior team defensive sack rate")
    return out


def team_opponent_inference_rows(
    features: pd.DataFrame,
    feature_names: list[str],
    team_def_rate: pd.DataFrame,
) -> pd.DataFrame:
    reconstructed = set(TEAM_OPPONENT_CURRENT_SOURCE) | {TEAM_DEF_SACK_FEATURE}
    passthrough = [name for name in feature_names if name not in reconstructed]
    common.require_columns(
        features,
        [*TEAM_GRAIN, "opponent", *passthrough, *TEAM_OPPONENT_CURRENT_SOURCE.values()],
        "Issue 36 current team-opponent features",
    )

    # These are genuine team-game fields and must remain invariant.
    opportunity.check_team_feature_invariance(features, passthrough)
    rows = opportunity.team_rows_from_features(features, passthrough)

    # Issue 15 defined player_defensive_opponent_* from the defender's
    # opponent team_form. Reconstruct that definition by joining the opposing
    # team's current lagged team-form values, rather than aggregating player rows.
    source_cols = list(TEAM_OPPONENT_CURRENT_SOURCE.values())
    opportunity.check_team_feature_invariance(features, source_cols)
    opponent_context = opportunity.team_rows_from_features(features, source_cols)
    opponent_context = opponent_context.rename(
        columns={
            "team": "_context_team",
            **{source: target for target, source in TEAM_OPPONENT_CURRENT_SOURCE.items()},
        }
    )
    rows = rows.merge(
        opponent_context[[
            "season", "week", "game_id", "_context_team",
            *TEAM_OPPONENT_CURRENT_SOURCE.keys(),
        ]],
        left_on=["season", "week", "game_id", "opponent"],
        right_on=["season", "week", "game_id", "_context_team"],
        how="left",
        validate="one_to_one",
    )

    rows["_team_key"] = rows["team"].map(opportunity.canonical_team)
    rows = rows.merge(team_def_rate, on="_team_key", how="left", validate="many_to_one")
    common.ensure_unique(rows, TEAM_GRAIN, "Issue 36 reconstructed team-opponent rows")
    return rows


def score_opportunity_component(
    prop: Path,
    features: pd.DataFrame,
    eligibility: dict[str, Any],
    component: str,
    team_def_rate: pd.DataFrame,
) -> pd.DataFrame:
    booster, _manifest, feature_names = pc.validate_booster_manifest(prop, "components", component)
    spec = opportunity.COMPONENTS[component]
    scope = str(spec["scope"])
    if scope == "team":
        opportunity.check_team_feature_invariance(features, feature_names)
        rows = opportunity.team_rows_from_features(features, feature_names)
    elif scope == "team_opponent":
        rows = team_opponent_inference_rows(features, feature_names, team_def_rate)
    elif scope == "player":
        rule = str(spec["eligible_rule"])
        positions = {str(x).strip().upper() for x in eligibility[rule]["eligible_positions"]}
        pos = features["position"].fillna("").astype(str).str.strip().str.upper()
        rows = features.loc[pos.isin(positions)].copy()
    else:
        raise ValueError(f"{component}: unsupported opportunity scope {scope!r}")
    X = opportunity.numeric_frame(rows, feature_names)
    pred = opportunity.transform_prediction(booster.predict(X), component)
    if not np.isfinite(pred).all():
        raise ValueError(f"{component}: nonfinite persisted-model prediction")
    key = GRAIN if scope == "player" else TEAM_GRAIN
    out = rows[key].copy()
    out[component] = pred
    common.ensure_unique(out, key, f"Issue 36 {component}")
    return out


def efficiency_history_columns(model_names: list[str]) -> list[str]:
    cols = [*GRAIN, "kickoff_timestamp", "position", "position_group"]
    for name in model_names:
        for f in efficiency.FEATURES[name]:
            if f not in efficiency.DERIVED_FEATURES and f not in cols:
                cols.append(f)
    return cols


def prepare_efficiency_raw_histories(
    config: dict[str, Any],
    history: pd.DataFrame,
    eligibility: dict[str, Any],
    model_names: list[str],
) -> dict[str, pd.DataFrame]:
    label_base = efficiency.prepare_label_base(config, history)
    result: dict[str, pd.DataFrame] = {}
    for name in model_names:
        raw = efficiency.build_component_label(label_base, name)
        raw = efficiency.apply_eligibility(raw, name, eligibility)
        result[name] = raw
    return result


def score_efficiency_model(
    prop: Path,
    current: pd.DataFrame,
    raw_history: pd.DataFrame,
    eligibility: dict[str, Any],
    model_name: str,
) -> pd.DataFrame:
    booster, _manifest, feature_names = pc.validate_booster_manifest(prop, "efficiency", model_name)
    rows = pc.efficiency_inference_frame(current, raw_history, model_name, eligibility)
    expected = list(efficiency.FEATURES[model_name])
    if feature_names != expected:
        raise ValueError(f"{model_name}: efficiency manifest differs from trainer order")
    X = efficiency.feature_matrix(rows, model_name)
    pred = efficiency.transform_prediction(booster.predict(X), model_name)
    if not np.isfinite(pred).all():
        raise ValueError(f"{model_name}: nonfinite persisted-model prediction")
    out = rows[GRAIN].copy()
    out[model_name] = pred
    common.ensure_unique(out, GRAIN, f"Issue 36 {model_name}")
    return out


def normalize_share_for_target(
    base: pd.DataFrame,
    raw_prediction: pd.DataFrame,
    raw_column: str,
    eligible_mask: pd.Series,
) -> pd.Series:
    work = base[[*GRAIN, "team"]].merge(raw_prediction, on=GRAIN, how="left", validate="one_to_one")
    raw = numeric(work[raw_column]).fillna(0.0).clip(0.0, 1.0)
    raw.loc[~eligible_mask.to_numpy()] = 0.0
    totals = raw.groupby(
        [work[c] for c in TEAM_GRAIN],
        sort=False,
    ).transform("sum")
    adjusted = raw.where(~totals.gt(0.0), raw / totals)
    return adjusted.clip(0.0, 1.0)


def build_component_points(
    config: dict[str, Any],
    prop: Path,
    repo: Path,
    component: pd.DataFrame,
    direct: pd.DataFrame,
    allocation: pd.DataFrame,
    features: pd.DataFrame,
    eligibility: dict[str, Any],
    season: int,
) -> tuple[dict[str, pd.Series], dict[str, Any]]:
    _ = (config, prop, repo, eligibility, season)

    direct_index = direct.set_index(GRAIN)
    key = pd.MultiIndex.from_frame(component[GRAIN])

    receiving_td_eligible = pd.Series(
        direct_index[DIRECT_COLUMNS["receiving_tds"]]
        .reindex(key)
        .notna()
        .to_numpy(),
        index=component.index,
    )
    rushing_td_eligible = pd.Series(
        direct_index[DIRECT_COLUMNS["rushing_tds"]]
        .reindex(key)
        .notna()
        .to_numpy(),
        index=component.index,
    )

    points, canonical_audit = pc.final_component_points(
        component,
        allocation,
        features,
        receiving_td_eligible=receiving_td_eligible,
        rushing_td_eligible=rushing_td_eligible,
    )
    audit = {
        **canonical_audit,
        "extra_opportunity_models_scored": [],
        "extra_efficiency_models_scored": [],
        "component_formula_owner": "project_components.py",
    }
    return points, audit



def select_point_prediction(
    target: str,
    selected: dict[str, Any],
    direct_values: pd.Series,
    component_values: pd.Series,
) -> pd.Series:
    architecture = str(selected["selected_architecture"])
    direct_values = numeric(direct_values)
    component_values = numeric(component_values)
    if architecture == "direct":
        return direct_values
    if architecture == "component":
        return component_values
    if architecture == "direct_component_blend":
        wd = float(selected["blend_weights"]["direct"])
        wc = float(selected["blend_weights"]["component"])
        return wd * direct_values + wc * component_values
    raise ValueError(f"{target}: unsupported selected architecture {architecture!r}")


def usage_signal(frame: pd.DataFrame, source: dict[str, Any]) -> np.ndarray:
    method = clean_text(source.get("method"))
    columns = list(source.get("columns", []))
    if method == "sum":
        total = np.zeros(len(frame), dtype="float64")
        any_finite = np.zeros(len(frame), dtype=bool)
        for column in columns:
            if column not in frame.columns:
                raise ValueError(f"Calibration usage source missing current feature: {column}")
            values = numeric(frame[column]).to_numpy(dtype="float64")
            finite = np.isfinite(values)
            total[finite] += values[finite]
            any_finite |= finite
        total[~any_finite] = np.nan
        return total
    if method == "single":
        if len(columns) != 1 or columns[0] not in frame.columns:
            raise ValueError(f"Calibration usage single source invalid: {columns}")
        return numeric(frame[columns[0]]).to_numpy(dtype="float64")
    return numeric(frame["selected_point_prediction"]).to_numpy(dtype="float64")


def apply_usage_buckets(values: np.ndarray, thresholds: dict[str, Any]) -> np.ndarray:
    low = float(thresholds["low_max"])
    medium = float(thresholds["medium_max"])
    labels = np.full(len(values), "low", dtype=object)
    finite = np.isfinite(values)
    labels[finite & (values > low)] = "medium"
    labels[finite & (values > medium)] = "high"
    labels[~finite] = "low"
    return labels


def risk_flags(frame: pd.DataFrame, threshold: int) -> dict[str, np.ndarray]:
    rookie = numeric(frame["history_no_nfl_history_flag"]).fillna(0).to_numpy(dtype="float64") >= 0.5
    promotion = numeric(frame["role_starter_promotion_flag"]).fillna(0).to_numpy(dtype="float64") >= 0.5
    history = numeric(frame["history_history_games"]).to_numpy(dtype="float64")
    low_history = ~np.isfinite(history) | (history < float(threshold))
    return {"rookie": rookie, "backup_promotion": promotion, "low_history": low_history}


def risk_multiplier(frame: pd.DataFrame, payload: dict[str, Any]) -> np.ndarray:
    threshold = int(payload.get("low_history_games_threshold", 4))
    flags = risk_flags(frame, threshold)
    result = np.ones(len(frame), dtype="float64")
    factors = payload.get("factors", {})
    for name, mask in flags.items():
        factor = float(factors[name]["multiplier"])
        result[mask] *= factor
    return np.minimum(result, float(payload.get("combined_cap", 3.0)))


def segment_factor(frame: pd.DataFrame, widening: dict[str, Any], interval_name: str) -> np.ndarray:
    result = np.full(len(frame), float(widening["global"][interval_name]), dtype="float64")
    segment = widening.get("segment_extra", {}).get(interval_name, {})
    keys = frame["position_group"].astype(str) + "|" + frame["usage_bucket"].astype(str)
    for key, details in segment.items():
        result[keys.eq(key).to_numpy()] *= float(details["multiplier"])
    return result


def quantile_outputs(frame: pd.DataFrame, payload: dict[str, Any]) -> dict[str, np.ndarray]:
    qcal = payload["quantile_calibration"]
    residual = {k: float(v) for k, v in qcal["residual_quantiles"].items()}
    point = numeric(frame["selected_point_prediction"]).to_numpy(dtype="float64")
    risk = risk_multiplier(frame, qcal["risk_widening"])
    center = float(residual["q50"])
    out: dict[str, np.ndarray] = {"q50": point + center}
    intervals = qcal["intervals"]
    widening = qcal["coverage_widening"]
    for interval_name in ("q10_q90", "q25_q75"):
        spec = intervals[interval_name]
        lower_name = str(spec["lower"])
        upper_name = str(spec["upper"])
        factors = segment_factor(frame, widening, interval_name)
        lo = float(residual[lower_name])
        hi = float(residual[upper_name])
        lower = point + center + (lo - center) * risk * factors
        upper = point + center + (hi - center) * risk * factors
        if bool(qcal.get("floor_at_zero")):
            lower = np.maximum(lower, 0.0)
            upper = np.maximum(upper, 0.0)
        out[lower_name] = np.minimum(lower, upper)
        out[upper_name] = np.maximum(lower, upper)
    if bool(qcal.get("floor_at_zero")):
        out["q50"] = np.maximum(out["q50"], 0.0)
    matrix = np.column_stack([out[name] for name in ["q10", "q25", "q50", "q75", "q90"]])
    matrix = np.maximum.accumulate(matrix, axis=1)
    for i, name in enumerate(["q10", "q25", "q50", "q75", "q90"]):
        out[name] = matrix[:, i]
    return out


def apply_mapping(values: np.ndarray, mapping: dict[str, Any]) -> np.ndarray:
    xp = np.asarray(mapping["knots_x"], dtype="float64")
    fp = np.asarray(mapping["knots_y"], dtype="float64")
    out = np.interp(
        np.asarray(values, dtype="float64"),
        xp,
        fp,
        left=float(mapping["left_value"]),
        right=float(mapping["right_value"]),
    )
    bounds = mapping.get("output_bounds", [None, None])
    if bounds[0] is not None:
        out = np.maximum(out, float(bounds[0]))
    if bounds[1] is not None:
        out = np.minimum(out, float(bounds[1]))
    return out


def count_outputs(frame: pd.DataFrame, payload: dict[str, Any]) -> dict[str, np.ndarray]:
    ccal = payload["count_calibration"]
    raw = np.maximum(numeric(frame["selected_point_prediction"]).to_numpy(dtype="float64"), 0.0)
    expected = apply_mapping(raw, ccal["expected_count"]["mapping"])
    poisson_p1 = 1.0 - np.exp(-expected)
    poisson_p2 = 1.0 - np.exp(-expected) * (1.0 + expected)
    p1 = apply_mapping(poisson_p1, ccal["probability_1_plus"]["mapping"])
    p2 = apply_mapping(poisson_p2, ccal["probability_2_plus"]["mapping"])
    return {
        "expected_count": np.maximum(expected, 0.0),
        "probability_1_plus": np.clip(p1, 0.0, 1.0),
        "probability_2_plus": np.clip(p2, 0.0, 1.0),
    }


def apply_point_prediction_blend(
    frame: pd.DataFrame,
    calibrated_point: np.ndarray,
    calibration: dict[str, Any],
) -> np.ndarray:
    spec = calibration.get("point_prediction_blend")
    if not isinstance(spec, dict):
        return np.asarray(calibrated_point, dtype="float64")
    alpha = float(spec.get("calibrated_weight", 1.0))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Invalid point calibration weight: {alpha}")
    raw = numeric(frame["selected_point_prediction"]).to_numpy(dtype="float64")
    base = np.asarray(calibrated_point, dtype="float64")
    output = raw + alpha * (base - raw)
    if bool(spec.get("floor_at_zero")):
        output = np.maximum(output, 0.0)
    return output


def calibrate_current_target(
    target: str,
    point: pd.Series,
    features: pd.DataFrame,
    calibration: dict[str, Any],
) -> pd.DataFrame:
    frame = features.copy().reset_index(drop=True)
    frame["selected_point_prediction"] = numeric(point).to_numpy(dtype="float64")
    frame["position_group"] = normalize_position_group(frame["position_group"])
    usage = usage_signal(frame, calibration["usage_bucket"]["source"])
    frame["usage_bucket"] = apply_usage_buckets(usage, calibration["usage_bucket"]["thresholds"])

    result = pd.DataFrame(index=frame.index)
    result["low"] = np.nan
    result["high"] = np.nan
    result["probability_1_plus"] = np.nan
    result["probability_2_plus"] = np.nan

    mode = str(calibration["calibration_mode"])
    qout: dict[str, np.ndarray] | None = None
    cout: dict[str, np.ndarray] | None = None
    if mode in {"quantiles", "quantiles_and_count"}:
        qout = quantile_outputs(frame, calibration)
        result["low"] = qout["q10"]
        result["high"] = qout["q90"]
    if mode in {"count", "quantiles_and_count"}:
        cout = count_outputs(frame, calibration)
        result["probability_1_plus"] = cout["probability_1_plus"]
        result["probability_2_plus"] = cout["probability_2_plus"]

    if cout is not None:
        base_point = cout["expected_count"]
    elif qout is not None:
        base_point = qout["q50"]
    else:
        raise ValueError(f"{target}: no calibration output")

    result["projection"] = apply_point_prediction_blend(
        frame,
        np.asarray(base_point, dtype="float64"),
        calibration,
    )
    return result[["projection", "low", "high", "probability_1_plus", "probability_2_plus"]]


def target_specific_reason(universe_row: pd.Series, target: str, eligible: bool) -> tuple[str, str]:
    if eligible:
        reason = clean_text(universe_row.get("eligibility_reason")) or "eligible_target_role"
        return "eligible", reason
    base_status = clean_text(universe_row.get("eligibility_status")).casefold()
    base_reason = clean_text(universe_row.get("eligibility_reason"))
    if base_status not in {"", "eligible", "active"} and base_reason:
        return "ineligible", base_reason
    return "ineligible", f"target_not_eligible_for_current_role_or_position:{target}"


def enforce_final_nonnegative_projection(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the production output support required by Issue 38.

    Calibration for historically signed yardage targets may yield a slightly
    negative q50 even when the production weekly projection contract requires
    a nonnegative displayed projection.  Floor only the final projection at
    zero.  If the calibrated upper interval is also negative, lift it to the
    floored projection so interval ordering remains valid.
    """
    out = frame.copy()
    projection = numeric(out["projection"])
    if projection.isna().any():
        raise ValueError("Issue 36 final projection contains null/nonfinite values")
    out["projection"] = projection.clip(lower=0.0)

    high = numeric(out["high"])
    bounded = high.notna()
    if bounded.any():
        out.loc[bounded, "high"] = np.maximum(
            high.loc[bounded].to_numpy(dtype="float64"),
            numeric(out.loc[bounded, "projection"]).to_numpy(dtype="float64"),
        )
    return out


def apply_display_rounding(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for target in ONE_DECIMAL_TARGETS:
        mask = out["target"].eq(target)
        out.loc[mask, "projection"] = numeric(out.loc[mask, "projection"]).round(1)
        out.loc[mask, "low"] = numeric(out.loc[mask, "low"]).round(1)
        out.loc[mask, "high"] = numeric(out.loc[mask, "high"]).round(1)
    for target in THREE_DECIMAL_TARGETS:
        mask = out["target"].eq(target)
        out.loc[mask, "projection"] = numeric(out.loc[mask, "projection"]).round(3)
    return out


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

    universe_path = prop / "data" / "current" / f"{season}_week_{week}_universe.parquet"
    component_path = prop / "data" / "current" / f"{season}_week_{week}_component_projections.parquet"
    direct_path = prop / "data" / "current" / f"{season}_week_{week}_direct_projections.parquet"
    allocation_path = prop / "data" / "current" / f"{season}_week_{week}_allocated_opportunity.parquet"
    features_path = prop / "data" / "current" / "features" / f"{season}_week_{week}_features.parquet"
    eligibility_path = prop / "config" / "target_eligibility.yaml"
    registry_path = prop / "models" / "production_registry.json"
    output_dir = prop / "output" / str(season)
    audit_output_path = output_dir / f"week_{week}_player_projections.csv"
    active_output_path = output_dir / f"week_{week}_active_player_projections.csv"
    log_path = prop / "logs" / f"week_projections_{season}_week_{week}.json"

    required = [
        universe_path,
        component_path,
        direct_path,
        allocation_path,
        features_path,
        eligibility_path,
        registry_path,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Issue 36 required production input missing: {path}")

    universe = pd.read_parquet(universe_path)
    component = pd.read_parquet(component_path)
    direct = pd.read_parquet(direct_path)
    allocation = pd.read_parquet(allocation_path)
    features = pd.read_parquet(features_path)
    # Match the accepted Issue 33 efficiency-inference contract: historical
    # and current kickoff timestamps must share one sortable timezone-aware
    # dtype before strict-prior history and current placeholders are combined.
    features = features.copy()
    features["kickoff_timestamp"] = pd.to_datetime(
        features["kickoff_timestamp"], errors="raise", utc=True
    )
    if features["kickoff_timestamp"].isna().any():
        raise ValueError("Issue 36 current features contain invalid kickoff_timestamp")
    common.require_columns(universe, UNIVERSE_REQUIRED, "Issue 36 universe")
    common.require_columns(component, COMPONENT_REQUIRED, "Issue 36 component projections")
    common.require_columns(direct, DIRECT_REQUIRED, "Issue 36 direct projections")
    common.require_columns(allocation, ALLOCATION_REQUIRED, "Issue 36 allocated opportunity")
    common.require_columns(features, CAL_CONTEXT_REQUIRED, "Issue 36 current feature context")
    common.ensure_unique(universe, GRAIN, "Issue 36 universe")
    for label, frame in [
        ("component", component),
        ("direct", direct),
        ("allocation", allocation),
        ("features", features),
    ]:
        common.ensure_unique(frame, GRAIN, f"Issue 36 {label}")

    # Issue 29 intentionally preserves ineligible players in the audit universe,
    # while Issues 31/33/34/35 operate only on the eligible current-player
    # subset.  Production projection inputs therefore must match the eligible
    # universe keys exactly; they must NOT be forced to match the full audit
    # universe row count.
    universe_eligible = universe.loc[
        universe["eligibility_status"].fillna("").astype(str).str.casefold().eq("eligible"),
        GRAIN,
    ].copy()
    if universe_eligible.empty:
        raise ValueError("Issue 36 universe contains no eligible current players")
    eligible_keys = set(map(tuple, universe_eligible.itertuples(index=False, name=None)))
    for label, frame in [
        ("component", component),
        ("direct", direct),
        ("allocation", allocation),
        ("features", features),
    ]:
        frame_keys = set(map(tuple, frame[GRAIN].itertuples(index=False, name=None)))
        if frame_keys != eligible_keys:
            missing = len(eligible_keys - frame_keys)
            extra = len(frame_keys - eligible_keys)
            raise ValueError(
                f"Issue 36 {label} keys differ from eligible universe "
                f"(missing={missing}, extra={extra})"
            )

    if set(pd.to_numeric(universe["season"], errors="raise").astype(int)) != {season}:
        raise ValueError("Issue 36 universe season mismatch")
    if set(pd.to_numeric(universe["week"], errors="raise").astype(int)) != {week}:
        raise ValueError("Issue 36 universe week mismatch")
    common.reject_forbidden_feature_columns(features.columns, config)
    if any(str(c).startswith("target_") for c in features.columns):
        raise ValueError("Issue 36 current features unexpectedly contain target columns")

    registry = load_json(registry_path)
    (
        model_versions,
        version_sources,
        production_targets,
        deferred_targets,
    ) = resolve_registry_versions(registry, registry_path)
    selected = load_selected_contracts(prop, production_targets)
    calibrations = load_calibrations(prop, selected, production_targets)
    eligibility = load_yaml(eligibility_path)

    # The direct file is the accepted current target-eligibility gate from Issue 35.
    direct_indexed = direct.set_index(GRAIN)
    component_points, component_audit = build_component_points(
        config,
        prop,
        repo,
        component,
        direct,
        allocation,
        features,
        eligibility,
        season,
    )

    base = universe[UNIVERSE_REQUIRED].merge(features, on=GRAIN, how="left", validate="one_to_one", suffixes=("", "_feature"))
    if "position_group_feature" in base.columns and base["position_group_feature"].isna().all():
        raise ValueError("Issue 36 feature context failed to join")
    # Prefer canonical current-feature calibration context where duplicate names exist.
    for col in [
        "position_group",
        "history_no_nfl_history_flag",
        "history_history_games",
        "role_starter_promotion_flag",
    ]:
        feature_col = f"{col}_feature"
        if feature_col in base.columns:
            base[col] = base[feature_col]

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    feature_asof = datetime.fromtimestamp(features_path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat()

    rows: list[pd.DataFrame] = []
    architecture_counts: dict[str, int] = {}
    eligible_counts: dict[str, int] = {}
    universe_key = pd.MultiIndex.from_frame(universe[GRAIN])
    component_key = pd.MultiIndex.from_frame(component[GRAIN])
    for target in production_targets:
        direct_col = DIRECT_COLUMNS[target]
        direct_values = direct_indexed[direct_col].reindex(universe_key).reset_index(drop=True)
        component_map = pd.Series(
            numeric(component_points[target]).to_numpy(dtype="float64"),
            index=component_key,
            dtype="float64",
        )
        component_values = component_map.reindex(universe_key).reset_index(drop=True)
        selected_point = select_point_prediction(target, selected[target], direct_values, component_values)
        eligible_mask = direct_values.notna()
        if selected_point.loc[eligible_mask].isna().any() or (~np.isfinite(selected_point.loc[eligible_mask].to_numpy(dtype="float64"))).any():
            raise ValueError(f"{target}: selected point missing/nonfinite on eligible rows")

        cal = calibrate_current_target(target, selected_point, base, calibrations[target])
        target_rows = universe[[
            "season", "week", "game_id", "game_date", "game_time", "player_id", "player_name",
            "team", "opponent", "position", "eligibility_status", "eligibility_reason",
            "role_status", "injury_game_status", "depth_rank",
        ]].copy().reset_index(drop=True)
        target_rows["target"] = target
        target_rows["projection"] = cal["projection"].to_numpy(dtype="float64")
        target_rows["low"] = cal["low"].to_numpy(dtype="float64")
        target_rows["high"] = cal["high"].to_numpy(dtype="float64")
        target_rows["probability_1_plus"] = cal["probability_1_plus"].to_numpy(dtype="float64")
        target_rows["probability_2_plus"] = cal["probability_2_plus"].to_numpy(dtype="float64")
        target_rows["model_architecture"] = str(selected[target]["selected_architecture"])
        target_rows["model_version"] = model_versions[target]
        target_rows["feature_asof"] = feature_asof
        target_rows["generated_at"] = generated_at

        statuses: list[str] = []
        reasons: list[str] = []
        for i, is_eligible in enumerate(eligible_mask.to_numpy(dtype=bool)):
            status, reason = target_specific_reason(universe.iloc[i], target, bool(is_eligible))
            statuses.append(status)
            reasons.append(reason)
        target_rows["eligibility_status"] = statuses
        target_rows["eligibility_reason"] = reasons

        ineligible = ~eligible_mask.to_numpy(dtype=bool)
        target_rows.loc[ineligible, "projection"] = 0.0
        target_rows.loc[ineligible, "low"] = 0.0
        target_rows.loc[ineligible, "high"] = 0.0
        target_rows.loc[ineligible, "probability_1_plus"] = 0.0
        target_rows.loc[ineligible, "probability_2_plus"] = 0.0

        if target in COUNT_TARGETS:
            probs = target_rows.loc[eligible_mask.to_numpy(dtype=bool), ["probability_1_plus", "probability_2_plus"]]
            if probs.isna().any().any():
                raise ValueError(f"{target}: eligible count rows missing calibrated probabilities")
            if ((probs < 0.0) | (probs > 1.0)).any().any():
                raise ValueError(f"{target}: calibrated probability outside [0,1]")
        else:
            # Non-count probability fields are intentionally null on eligible rows.
            target_rows.loc[eligible_mask.to_numpy(dtype=bool), ["probability_1_plus", "probability_2_plus"]] = np.nan

        architecture = str(selected[target]["selected_architecture"])
        architecture_counts[architecture] = architecture_counts.get(architecture, 0) + 1
        eligible_counts[target] = int(eligible_mask.sum())
        rows.append(target_rows[OUTPUT_COLUMNS])

    audit = pd.concat(rows, ignore_index=True)
    audit = enforce_final_nonnegative_projection(audit)
    audit = apply_display_rounding(audit)
    if list(audit.columns) != OUTPUT_COLUMNS:
        raise ValueError("Issue 36 exact output header order changed")
    if len(audit) != len(universe) * len(production_targets):
        raise ValueError(
            "Issue 36 audit output must contain player x approved-production-target rows"
        )
    if audit.duplicated([*GRAIN, "target"]).any():
        raise ValueError("Issue 36 audit output has duplicate player-target rows")

    active = audit.loc[audit["eligibility_status"].eq("eligible")].copy()
    if active.empty:
        raise ValueError("Issue 36 active-only output is empty")
    if active["eligibility_status"].ne("eligible").any():
        raise ValueError("Issue 36 active-only output contains ineligible rows")

    # Stable production ordering.
    target_rank = {target: i for i, target in enumerate(production_targets)}
    for frame in (audit, active):
        frame["_target_order"] = frame["target"].map(target_rank)
        frame.sort_values(
            ["game_date", "game_time", "game_id", "team", "player_name", "player_id", "_target_order"],
            kind="mergesort",
            inplace=True,
            na_position="last",
        )
        frame.drop(columns=["_target_order"], inplace=True)
        frame.reset_index(drop=True, inplace=True)

    write_csv_atomic(audit, audit_output_path)
    write_csv_atomic(active, active_output_path)

    payload = {
        "script": Path(__file__).name,
        "status": "passed",
        "season": season,
        "week": week,
        "players": int(len(universe)),
        "targets": len(production_targets),
        "production_targets": list(production_targets),
        "deferred_targets": list(deferred_targets),
        "audit_rows": int(len(audit)),
        "active_rows": int(len(active)),
        "columns": len(OUTPUT_COLUMNS),
        "eligible_rows_by_target": eligible_counts,
        "architecture_target_counts": architecture_counts,
        "display_rounding": {
            "yardage": 1,
            "td_and_sack_expected_counts": 3,
            "tackles_and_kicking": 1,
            "probabilities_rounded": False,
        },
        "ineligible_audit_projection_zero": True,
        "final_projection_floor_zero": True,
        "active_output_ineligible_rows": 0,
        "feature_asof": feature_asof,
        "generated_at": generated_at,
        "registry_sha256": sha256_file(registry_path),
        "registry_version_sources": version_sources,
        "market_exclusion_passed": bool(market["passed"]),
        "market_features_used": False,
        "audit_output": repo_relative(audit_output_path),
        "active_output": repo_relative(active_output_path),
        "log": repo_relative(log_path),
        "component_runtime": component_audit,
    }
    log_payload = {
        **payload,
        "inputs": {
            "universe": repo_relative(universe_path),
            "component_projections": repo_relative(component_path),
            "direct_projections": repo_relative(direct_path),
            "allocated_opportunity": repo_relative(allocation_path),
            "current_features": repo_relative(features_path),
            "production_registry": repo_relative(registry_path),
            "target_eligibility": repo_relative(eligibility_path),
            "selected_models": {
                target: repo_relative(prop / "models" / target / "selected_model.json")
                for target in production_targets
            },
            "calibrations": {
                target: repo_relative(prop / "models" / "calibration" / f"{target}_calibration.json")
                for target in production_targets
            },
        },
        "policy": {
            "selected_architecture_from_registry_assets_only": True,
            "issue34_allocated_target_and_carry_shares_used": True,
            "missing_component_dependencies_scored_from_persisted_models": True,
            "issue26_calibration_applied": True,
            "count_projection_is_calibrated_expected_count": True,
            "quantile_projection_is_calibrated_q50": True,
            "display_rounding_after_calibration": True,
            "ineligible_target_combinations_zero_in_audit": True,
            "ineligible_target_combinations_excluded_from_active": True,
            "probabilities_clipped_to_unit_interval": True,
            "market_exclusion_preflight": True,
        },
    }
    write_json_atomic(log_payload, log_path)

    print(json.dumps({"script": Path(__file__).name, "payload": payload}, sort_keys=True, separators=(",", ":")))
    print("FINAL WEEKLY PROJECTIONS BUILD: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
