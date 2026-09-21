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
    "projected_target_share", "projected_targets",
    "projected_yards_per_attempt", "projected_yards_per_carry",
    "projected_yards_per_target", "projected_red_zone_targets",
    "projected_goal_line_carries", "projected_fg_attempts",
    "projected_fg_make_probability", "projected_pat_attempts",
    "projected_pat_make_probability",
    "projected_opponent_plays", "projected_opponent_dropbacks",
    "projected_defensive_participation",
    "projected_tackle_rate", "projected_sack_rate",
    "component_passing_yards", "component_passing_tds",
    "component_rushing_yards", "component_rushing_tds",
    "component_receiving_yards", "component_receiving_tds",
    "component_kicking_points", "component_tackles", "component_sacks",
]

OPPORTUNITY_NEEDED = [
    "qb_pass_attempts", "team_pass_attempts", "team_rush_attempts",
    "player_carry_share", "player_target_share", "player_red_zone_target_share",
    "player_goal_line_carry_share", "field_goal_attempts", "extra_point_attempts",
    "opponent_offensive_plays", "opponent_dropbacks",
    "player_defensive_participation",
]
EFFICIENCY_NEEDED = [
    "passing_yards_per_attempt", "passing_td_rate",
    "rushing_yards_per_carry", "rushing_td_per_goal_line_carry",
    "receiving_yards_per_target", "receiving_td_per_red_zone_target",
    "field_goal_conversion", "extra_point_conversion",
    "tackle_rate_per_defensive_play", "sack_rate_per_defensive_play",
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



TEAM_OPPONENT_CURRENT_SOURCE = {
    "player_defensive_opponent_plays_roll3": "team_offensive_plays_roll3_mean",
    "player_defensive_opponent_dropbacks_roll3": "team_dropbacks_roll3_mean",
    "player_defensive_opponent_rush_rate_roll3": "team_rush_rate_roll3_mean",
    "player_defensive_opponent_pass_rate_roll3": "team_pass_rate_roll3_mean",
}
TEAM_DEF_SACK_FEATURE = "player_defensive_team_def_sack_rate_roll3"
TARGET_COMPONENT_COLUMNS = [
    "component_passing_yards",
    "component_passing_tds",
    "component_rushing_yards",
    "component_rushing_tds",
    "component_receiving_yards",
    "component_receiving_tds",
    "component_kicking_points",
    "component_tackles",
    "component_sacks",
]


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
    sacks = numeric(raw["sacks"])
    dropbacks = numeric(raw["opponent_dropbacks"])
    raw["_rate"] = np.where(
        sacks.notna() & dropbacks.notna() & dropbacks.ne(0.0),
        sacks / dropbacks,
        np.nan,
    )
    raw = raw.sort_values(
        ["_team_key", "season", "week"],
        kind="mergesort",
    )
    records: list[dict[str, Any]] = []
    for team, frame in raw.groupby("_team_key", sort=False):
        rates = frame["_rate"].dropna().to_numpy(dtype="float64")
        records.append(
            {
                "_team_key": team,
                TEAM_DEF_SACK_FEATURE: (
                    float(np.mean(rates[-3:])) if len(rates) else np.nan
                ),
            }
        )
    out = pd.DataFrame(records)
    common.ensure_unique(
        out,
        ["_team_key"],
        "Issue 33 strict-prior team defensive sack rate",
    )
    return out


def team_opponent_inference_rows(
    features: pd.DataFrame,
    feature_names: list[str],
    team_def_rate: pd.DataFrame,
) -> pd.DataFrame:
    reconstructed = set(TEAM_OPPONENT_CURRENT_SOURCE) | {
        TEAM_DEF_SACK_FEATURE
    }
    passthrough = [
        name for name in feature_names if name not in reconstructed
    ]
    common.require_columns(
        features,
        [
            *TEAM_GRAIN,
            "opponent",
            *passthrough,
            *TEAM_OPPONENT_CURRENT_SOURCE.values(),
        ],
        "Issue 33 current team-opponent features",
    )
    opportunity.check_team_feature_invariance(features, passthrough)
    rows = opportunity.team_rows_from_features(features, passthrough)

    source_cols = list(TEAM_OPPONENT_CURRENT_SOURCE.values())
    opportunity.check_team_feature_invariance(features, source_cols)
    opponent_context = opportunity.team_rows_from_features(
        features,
        source_cols,
    ).rename(
        columns={
            "team": "_context_team",
            **{
                source: target
                for target, source in TEAM_OPPONENT_CURRENT_SOURCE.items()
            },
        }
    )
    rows = rows.merge(
        opponent_context[
            [
                "season",
                "week",
                "game_id",
                "_context_team",
                *TEAM_OPPONENT_CURRENT_SOURCE.keys(),
            ]
        ],
        left_on=["season", "week", "game_id", "opponent"],
        right_on=["season", "week", "game_id", "_context_team"],
        how="left",
        validate="one_to_one",
    )
    rows["_team_key"] = rows["team"].map(opportunity.canonical_team)
    rows = rows.merge(
        team_def_rate,
        on="_team_key",
        how="left",
        validate="many_to_one",
    )
    common.ensure_unique(
        rows,
        TEAM_GRAIN,
        "Issue 33 reconstructed team-opponent rows",
    )
    return rows


def zero_safe_product(*series: pd.Series) -> pd.Series:
    if not series:
        raise ValueError("zero_safe_product requires at least one series")
    values = [numeric(item) for item in series]
    result = values[0].copy()
    for item in values[1:]:
        result = result * item
    zero_mask = pd.Series(False, index=result.index)
    for item in values:
        zero_mask |= item.eq(0.0)
    result.loc[zero_mask] = 0.0
    return result


def add_raw_target_components(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    out["component_passing_yards"] = zero_safe_product(
        out["projected_qb_pass_attempts"],
        out["projected_yards_per_attempt"],
    )
    out["component_passing_tds"] = zero_safe_product(
        out["projected_qb_pass_attempts"],
        out["_passing_td_rate"],
    )
    out["component_rushing_yards"] = zero_safe_product(
        out["projected_player_carries"],
        out["projected_yards_per_carry"],
    )
    out["component_rushing_tds"] = zero_safe_product(
        out["projected_goal_line_carries"],
        out["_rushing_td_rate"],
    )
    out["component_receiving_yards"] = zero_safe_product(
        out["projected_targets"],
        out["projected_yards_per_target"],
    )
    out["component_receiving_tds"] = zero_safe_product(
        out["projected_red_zone_targets"],
        out["_receiving_td_rate"],
    )
    out["component_kicking_points"] = (
        3.0
        * zero_safe_product(
            out["projected_fg_attempts"],
            out["projected_fg_make_probability"],
        )
        + zero_safe_product(
            out["projected_pat_attempts"],
            out["projected_pat_make_probability"],
        )
    )
    out["component_tackles"] = zero_safe_product(
        out["projected_opponent_plays"],
        out["projected_defensive_participation"],
        out["projected_tackle_rate"],
    )
    out["component_sacks"] = zero_safe_product(
        out["projected_opponent_plays"],
        out["projected_defensive_participation"],
        out["projected_sack_rate"],
    )

    for column in [
        "component_passing_tds",
        "component_rushing_tds",
        "component_receiving_tds",
        "component_kicking_points",
        "component_tackles",
        "component_sacks",
    ]:
        out[column] = numeric(out[column]).clip(lower=0.0)

    for column in TARGET_COMPONENT_COLUMNS:
        out[column] = numeric(out[column])
        if out[column].isna().any() or (~np.isfinite(out[column])).any():
            raise ValueError(
                f"Issue 33 nonfinite target component projection: {column}"
            )
    return out


def _normalized_component_exposure(
    base: pd.DataFrame,
    raw_exposure: pd.Series,
    team_volume: pd.Series,
    eligible_mask: pd.Series,
) -> pd.Series:
    raw = numeric(raw_exposure).fillna(0.0).clip(lower=0.0)
    raw.loc[~eligible_mask.to_numpy(dtype=bool)] = 0.0
    total = raw.groupby(
        [base[column] for column in TEAM_GRAIN],
        sort=False,
    ).transform("sum")
    volume = numeric(team_volume).fillna(0.0).clip(lower=0.0)
    out = pd.Series(0.0, index=base.index, dtype="float64")
    positive = total.gt(0.0)
    out.loc[positive] = (
        volume.loc[positive]
        * raw.loc[positive]
        / total.loc[positive]
    )
    return out


def _scale_component_for_allocated_exposure(
    raw_component: pd.Series,
    raw_exposure: pd.Series,
    allocated_exposure: pd.Series,
    *,
    fallback_unit_rate: pd.Series | None = None,
) -> pd.Series:
    component = numeric(raw_component)
    raw = numeric(raw_exposure).fillna(0.0).clip(lower=0.0)
    allocated = numeric(allocated_exposure).fillna(0.0).clip(lower=0.0)
    out = pd.Series(0.0, index=component.index, dtype="float64")

    positive_raw = raw.gt(1e-12)
    out.loc[positive_raw] = (
        component.loc[positive_raw]
        * allocated.loc[positive_raw]
        / raw.loc[positive_raw]
    )

    fallback = (~positive_raw) & allocated.gt(1e-12)
    if fallback.any():
        if fallback_unit_rate is None:
            raise ValueError(
                "Allocated component exposure is positive where raw exposure "
                "is zero and no canonical fallback rate is available."
            )
        unit = numeric(fallback_unit_rate)
        if unit.loc[fallback].isna().any():
            raise ValueError(
                "Canonical fallback component rate is missing for allocated "
                "positive exposure."
            )
        out.loc[fallback] = (
            allocated.loc[fallback] * unit.loc[fallback]
        )
    return out


def final_component_points(
    component: pd.DataFrame,
    allocation: pd.DataFrame,
    features: pd.DataFrame,
    *,
    receiving_td_eligible: pd.Series,
    rushing_td_eligible: pd.Series,
) -> tuple[dict[str, pd.Series], dict[str, Any]]:
    common.require_columns(
        component,
        OUTPUT_COLUMNS,
        "canonical Issue 33 component projections",
    )
    common.require_columns(
        allocation,
        [
            *GRAIN,
            "allocated_target_share",
            "allocated_carry_share",
            "allocated_def_participation",
        ],
        "Issue 34 allocation",
    )
    common.require_columns(
        features,
        [
            *GRAIN,
            *RED_ZONE_PASS_VOLUME_FEATURES,
            *GOAL_LINE_RUSH_VOLUME_FEATURES,
        ],
        "Issue 36 component allocation context",
    )

    base = component.merge(
        allocation[
            [
                *GRAIN,
                "allocated_target_share",
                "allocated_carry_share",
                "allocated_def_participation",
            ]
        ],
        on=GRAIN,
        how="left",
        validate="one_to_one",
    ).merge(
        features[
            [
                *GRAIN,
                *RED_ZONE_PASS_VOLUME_FEATURES,
                *GOAL_LINE_RUSH_VOLUME_FEATURES,
            ]
        ],
        on=GRAIN,
        how="left",
        validate="one_to_one",
    )
    if base[
        [
            "allocated_target_share",
            "allocated_carry_share",
            "allocated_def_participation",
        ]
    ].isna().any().any():
        raise ValueError("Issue 36 allocation merge is incomplete")

    if len(receiving_td_eligible) != len(base):
        raise ValueError("receiving_td_eligible length mismatch")
    if len(rushing_td_eligible) != len(base):
        raise ValueError("rushing_td_eligible length mismatch")

    allocated_carries = (
        numeric(base["projected_team_rush_attempts"]).clip(lower=0.0)
        * numeric(base["allocated_carry_share"]).clip(0.0, 1.0)
    )
    allocated_targets = (
        numeric(base["projected_team_pass_attempts"]).clip(lower=0.0)
        * numeric(base["allocated_target_share"]).clip(0.0, 1.0)
    )

    rz_volume = coalesce_numeric(
        base,
        RED_ZONE_PASS_VOLUME_FEATURES,
    ).clip(lower=0.0)
    gl_volume = coalesce_numeric(
        base,
        GOAL_LINE_RUSH_VOLUME_FEATURES,
    ).clip(lower=0.0)

    allocated_rz_targets = _normalized_component_exposure(
        base,
        base["projected_red_zone_targets"],
        rz_volume,
        receiving_td_eligible,
    )
    allocated_gl_carries = _normalized_component_exposure(
        base,
        base["projected_goal_line_carries"],
        gl_volume,
        rushing_td_eligible,
    )

    points: dict[str, pd.Series] = {
        "passing_yards": numeric(base["component_passing_yards"]),
        "passing_tds": numeric(base["component_passing_tds"]),
        "kicking_points": numeric(base["component_kicking_points"]),
    }
    points["rushing_yards"] = _scale_component_for_allocated_exposure(
        base["component_rushing_yards"],
        base["projected_player_carries"],
        allocated_carries,
        fallback_unit_rate=base["projected_yards_per_carry"],
    )
    points["receiving_yards"] = _scale_component_for_allocated_exposure(
        base["component_receiving_yards"],
        base["projected_targets"],
        allocated_targets,
        fallback_unit_rate=base["projected_yards_per_target"],
    )
    points["rushing_tds"] = _scale_component_for_allocated_exposure(
        base["component_rushing_tds"],
        base["projected_goal_line_carries"],
        allocated_gl_carries,
    )
    points["receiving_tds"] = _scale_component_for_allocated_exposure(
        base["component_receiving_tds"],
        base["projected_red_zone_targets"],
        allocated_rz_targets,
    )

    tackle_unit = zero_safe_product(
        base["projected_opponent_plays"],
        base["projected_tackle_rate"],
    )
    sack_unit = zero_safe_product(
        base["projected_opponent_plays"],
        base["projected_sack_rate"],
    )
    points["tackles"] = _scale_component_for_allocated_exposure(
        base["component_tackles"],
        base["projected_defensive_participation"],
        base["allocated_def_participation"],
        fallback_unit_rate=tackle_unit,
    )
    points["sacks"] = _scale_component_for_allocated_exposure(
        base["component_sacks"],
        base["projected_defensive_participation"],
        base["allocated_def_participation"],
        fallback_unit_rate=sack_unit,
    )

    for target in [
        "passing_tds",
        "rushing_tds",
        "receiving_tds",
        "kicking_points",
        "tackles",
        "sacks",
    ]:
        points[target] = numeric(points[target]).clip(lower=0.0)

    for target, values in points.items():
        values = numeric(values)
        if values.isna().any() or (~np.isfinite(values)).any():
            raise ValueError(
                f"Issue 36 canonical component adjustment is nonfinite: {target}"
            )
        points[target] = values

    return points, {
        "canonical_issue33_target_components_used": True,
        "issue33_component_columns": list(TARGET_COMPONENT_COLUMNS),
        "allocated_target_share_used": True,
        "allocated_carry_share_used": True,
        "allocated_def_participation_used": True,
        "red_zone_share_reconciled_from_issue33_exposure": True,
        "goal_line_share_reconciled_from_issue33_exposure": True,
        "duplicate_component_model_scoring": False,
    }


def component_inference_rows(
    features: pd.DataFrame,
    component: str,
    eligibility: dict[str, Any],
    team_def_rate: pd.DataFrame | None = None,
) -> pd.DataFrame:
    spec = opportunity.COMPONENTS[component]
    feature_names = list(spec["features"])
    scope = str(spec["scope"])
    if scope == "team":
        opportunity.check_team_feature_invariance(features, feature_names)
        return opportunity.team_rows_from_features(features, feature_names)
    if scope == "team_opponent":
        if team_def_rate is None:
            raise ValueError(
                f"{component}: team-opponent inference requires strict-prior "
                "team defensive context"
            )
        return team_opponent_inference_rows(
            features,
            feature_names,
            team_def_rate,
        )
    if scope == "player":
        missing = [c for c in feature_names if c not in features.columns]
        if missing:
            raise ValueError(
                f"{component}: missing current feature(s): {missing[:30]}"
            )
        rule = str(spec["eligible_rule"])
        positions = {
            str(x).strip().upper()
            for x in eligibility[rule]["eligible_positions"]
        }
        pos = (
            features["position"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )
        return features.loc[pos.isin(positions)].copy()
    raise ValueError(
        f"{component}: unsupported opportunity scope {scope!r}"
    )



def score_opportunity(
    root: Path,
    features: pd.DataFrame,
    eligibility: dict[str, Any],
    config: dict[str, Any],
    repo: Path,
    season: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    predictions: dict[str, pd.DataFrame] = {}
    audits: dict[str, Any] = {}
    need_team_opponent = any(
        str(opportunity.COMPONENTS[name]["scope"]) == "team_opponent"
        for name in OPPORTUNITY_NEEDED
    )
    team_def_rate = (
        strict_prior_team_def_sack_rate(config, repo, season)
        if need_team_opponent
        else None
    )

    for component in OPPORTUNITY_NEEDED:
        booster, manifest, feature_names = validate_booster_manifest(
            root,
            "components",
            component,
        )
        rows = component_inference_rows(
            features,
            component,
            eligibility,
            team_def_rate,
        )
        X = opportunity.numeric_frame(rows, feature_names)
        pred = opportunity.transform_prediction(
            booster.predict(X),
            component,
        )
        if not np.isfinite(pred).all():
            raise ValueError(
                f"{component}: nonfinite persisted-model prediction"
            )
        scope = str(opportunity.COMPONENTS[component]["scope"])
        key = GRAIN if scope == "player" else TEAM_GRAIN
        out = rows[key].copy()
        out[component] = pred
        common.ensure_unique(out, key, f"Issue 33 {component}")
        predictions[component] = out
        audits[component] = {
            "scope": str(manifest.get("scope")),
            "rows": int(len(out)),
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
    season = (
        int(args.season)
        if args.season is not None
        else int(config["seasons"]["current"])
    )
    week = int(args.week)
    if not 1 <= week <= 25:
        raise ValueError(f"Invalid week: {week}")

    market = run_market_preflight()
    repo = common.repo_root()
    prop = common.prop_root()

    features_path = (
        prop
        / "data"
        / "current"
        / "features"
        / f"{season}_week_{week}_features.parquet"
    )
    roles_path = (
        prop
        / "data"
        / "current"
        / f"{season}_week_{week}_roles.parquet"
    )
    output_path = (
        prop
        / "data"
        / "current"
        / f"{season}_week_{week}_component_projections.parquet"
    )
    log_path = (
        prop
        / "logs"
        / f"component_projections_{season}_week_{week}.json"
    )
    eligibility_path = prop / "config" / "target_eligibility.yaml"
    historical_path = repo / config["paths"]["historical_features"]

    required_paths = [
        features_path,
        roles_path,
        eligibility_path,
        historical_path,
    ]
    if week == 1:
        required_paths.append(
            prop
            / "data"
            / "current"
            / f"{season}_week_1_priors.parquet"
        )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Issue 33 required input missing: {path}"
            )

    features = pd.read_parquet(features_path)
    roles = pd.read_parquet(roles_path)
    common.require_columns(
        features,
        [
            *GRAIN,
            "team",
            "opponent",
            "position",
            "position_group",
            "kickoff_timestamp",
            *RED_ZONE_PASS_VOLUME_FEATURES,
            *GOAL_LINE_RUSH_VOLUME_FEATURES,
        ],
        "Issue 33 current features",
    )
    common.require_columns(
        roles,
        ROLE_REQUIRED,
        "Issue 33 current roles",
    )
    common.ensure_unique(
        features,
        GRAIN,
        "Issue 33 current features",
    )
    common.ensure_unique(
        roles,
        GRAIN,
        "Issue 33 current roles",
    )
    common.reject_forbidden_feature_columns(features.columns, config)
    if any(str(c).startswith("target_") for c in features.columns):
        raise ValueError(
            "Issue 33 current features unexpectedly contain target columns"
        )

    features = features.copy()
    features["season"] = pd.to_numeric(
        features["season"],
        errors="raise",
    ).astype(int)
    features["week"] = pd.to_numeric(
        features["week"],
        errors="raise",
    ).astype(int)
    features["kickoff_timestamp"] = pd.to_datetime(
        features["kickoff_timestamp"],
        errors="raise",
        utc=True,
    )
    if set(features["season"]) != {season} or set(features["week"]) != {week}:
        raise ValueError("Issue 33 current feature season/week mismatch")

    eligibility = load_yaml(eligibility_path)
    opp_pred, opp_audit = score_opportunity(
        prop,
        features,
        eligibility,
        config,
        repo,
        season,
    )

    hist_cols = efficiency_columns()
    history = pd.read_parquet(historical_path, columns=hist_cols)
    history["season"] = pd.to_numeric(
        history["season"],
        errors="raise",
    ).astype(int)
    history["week"] = pd.to_numeric(
        history["week"],
        errors="raise",
    ).astype(int)
    history["kickoff_timestamp"] = pd.to_datetime(
        history["kickoff_timestamp"],
        errors="raise",
        utc=True,
    )
    history = history.loc[history["season"].lt(season)].copy()
    if history.empty:
        raise ValueError(
            "Issue 33 has no strict-prior historical efficiency context"
        )
    common.ensure_unique(
        history,
        GRAIN,
        "Issue 33 historical efficiency context",
    )
    eff_pred, eff_audit = score_efficiency(
        prop,
        features,
        history,
        eligibility,
        config,
    )

    base = features[
        [
            *GRAIN,
            "team",
            "opponent",
            "position",
            *RED_ZONE_PASS_VOLUME_FEATURES,
            *GOAL_LINE_RUSH_VOLUME_FEATURES,
        ]
    ].copy()

    for comp, out_col in [
        ("team_pass_attempts", "projected_team_pass_attempts"),
        ("team_rush_attempts", "projected_team_rush_attempts"),
        ("field_goal_attempts", "_team_fg_attempts"),
        ("extra_point_attempts", "_team_pat_attempts"),
        ("opponent_offensive_plays", "projected_opponent_plays"),
        ("opponent_dropbacks", "projected_opponent_dropbacks"),
    ]:
        base = merge_team_prediction(
            base,
            opp_pred[comp],
            comp,
            out_col,
        )

    for comp, out_col in [
        ("qb_pass_attempts", "_raw_qb_pass_attempts"),
        ("player_carry_share", "_raw_carry_share"),
        ("player_target_share", "projected_target_share"),
        (
            "player_red_zone_target_share",
            "_raw_red_zone_target_share",
        ),
        (
            "player_goal_line_carry_share",
            "_raw_goal_line_carry_share",
        ),
        (
            "player_defensive_participation",
            "_raw_defensive_participation",
        ),
    ]:
        base = merge_player_prediction(
            base,
            opp_pred[comp],
            comp,
            out_col,
        )

    for name, out_col in [
        (
            "passing_yards_per_attempt",
            "projected_yards_per_attempt",
        ),
        ("passing_td_rate", "_passing_td_rate"),
        (
            "rushing_yards_per_carry",
            "projected_yards_per_carry",
        ),
        (
            "rushing_td_per_goal_line_carry",
            "_rushing_td_rate",
        ),
        (
            "receiving_yards_per_target",
            "projected_yards_per_target",
        ),
        (
            "receiving_td_per_red_zone_target",
            "_receiving_td_rate",
        ),
        (
            "field_goal_conversion",
            "_raw_fg_make_probability",
        ),
        (
            "extra_point_conversion",
            "_raw_pat_make_probability",
        ),
        (
            "tackle_rate_per_defensive_play",
            "projected_tackle_rate",
        ),
        (
            "sack_rate_per_defensive_play",
            "projected_sack_rate",
        ),
    ]:
        base = merge_player_prediction(
            base,
            eff_pred[name],
            name,
            out_col,
        )

    role_cols = roles[
        [
            *GRAIN,
            "primary_qb_flag",
            "primary_kicker_flag",
        ]
    ].copy()
    base = base.merge(
        role_cols,
        on=GRAIN,
        how="left",
        validate="one_to_one",
    )
    if base[
        ["primary_qb_flag", "primary_kicker_flag"]
    ].isna().any().any():
        raise ValueError(
            "Issue 33 role merge missing primary-role flags"
        )

    qb_primary = (
        numeric(base["primary_qb_flag"]).fillna(0).gt(0)
    )
    kicker_primary = (
        numeric(base["primary_kicker_flag"]).fillna(0).gt(0)
    )
    if int(qb_primary.sum()) != int(base["team"].nunique()):
        raise ValueError(
            "Issue 33 requires exactly one primary QB per team"
        )
    if int(kicker_primary.sum()) != int(base["team"].nunique()):
        raise ValueError(
            "Issue 33 requires exactly one primary kicker per team"
        )

    base["projected_qb_pass_attempts"] = 0.0
    base.loc[
        qb_primary,
        "projected_qb_pass_attempts",
    ] = numeric(
        base.loc[qb_primary, "_raw_qb_pass_attempts"]
    ).to_numpy()
    if base.loc[
        qb_primary,
        "projected_qb_pass_attempts",
    ].isna().any():
        raise ValueError(
            "Primary QB is missing qb_pass_attempts component prediction"
        )

    base["_raw_carry_share"] = (
        numeric(base["_raw_carry_share"])
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    base["projected_target_share"] = (
        numeric(base["projected_target_share"])
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    base["_raw_red_zone_target_share"] = (
        numeric(base["_raw_red_zone_target_share"])
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    base["_raw_goal_line_carry_share"] = (
        numeric(base["_raw_goal_line_carry_share"])
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    base["projected_player_carries"] = (
        numeric(base["projected_team_rush_attempts"]).clip(lower=0.0)
        * base["_raw_carry_share"]
    )
    base["projected_targets"] = (
        numeric(base["projected_team_pass_attempts"]).clip(lower=0.0)
        * base["projected_target_share"]
    )
    rz_volume = coalesce_numeric(
        base,
        RED_ZONE_PASS_VOLUME_FEATURES,
    ).clip(lower=0.0)
    gl_volume = coalesce_numeric(
        base,
        GOAL_LINE_RUSH_VOLUME_FEATURES,
    ).clip(lower=0.0)
    base["projected_red_zone_targets"] = (
        rz_volume * base["_raw_red_zone_target_share"]
    )
    base["projected_goal_line_carries"] = (
        gl_volume * base["_raw_goal_line_carry_share"]
    )

    base["projected_fg_attempts"] = 0.0
    base.loc[
        kicker_primary,
        "projected_fg_attempts",
    ] = numeric(
        base.loc[kicker_primary, "_team_fg_attempts"]
    ).to_numpy()
    base["projected_pat_attempts"] = 0.0
    base.loc[
        kicker_primary,
        "projected_pat_attempts",
    ] = numeric(
        base.loc[kicker_primary, "_team_pat_attempts"]
    ).to_numpy()

    base["projected_fg_make_probability"] = np.nan
    base.loc[
        kicker_primary,
        "projected_fg_make_probability",
    ] = numeric(
        base.loc[
            kicker_primary,
            "_raw_fg_make_probability",
        ]
    ).to_numpy()
    base["projected_pat_make_probability"] = np.nan
    base.loc[
        kicker_primary,
        "projected_pat_make_probability",
    ] = numeric(
        base.loc[
            kicker_primary,
            "_raw_pat_make_probability",
        ]
    ).to_numpy()

    if base.loc[
        kicker_primary,
        [
            "projected_fg_attempts",
            "projected_pat_attempts",
            "projected_fg_make_probability",
            "projected_pat_make_probability",
        ],
    ].isna().any().any():
        raise ValueError(
            "Primary kicker is missing kicking component/efficiency prediction"
        )

    base["projected_fg_make_probability"] = numeric(
        base["projected_fg_make_probability"]
    ).clip(0.0, 1.0)
    base["projected_pat_make_probability"] = numeric(
        base["projected_pat_make_probability"]
    ).clip(0.0, 1.0)

    def_eligible = base["_raw_defensive_participation"].notna()
    base["projected_defensive_participation"] = (
        numeric(base["_raw_defensive_participation"])
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    base["projected_tackle_rate"] = numeric(
        base["projected_tackle_rate"]
    ).clip(0.0, 1.0)
    base["projected_sack_rate"] = numeric(
        base["projected_sack_rate"]
    ).clip(0.0, 1.0)
    if base.loc[
        def_eligible,
        ["projected_tackle_rate", "projected_sack_rate"],
    ].isna().any().any():
        raise ValueError(
            "Eligible defender is missing tackle/sack efficiency prediction"
        )

    for column in [
        "_passing_td_rate",
        "_rushing_td_rate",
        "_receiving_td_rate",
    ]:
        base[column] = numeric(base[column]).clip(0.0, 1.0)

    nonnegative = [
        "projected_team_pass_attempts",
        "projected_qb_pass_attempts",
        "projected_team_rush_attempts",
        "projected_player_carries",
        "projected_target_share",
        "projected_targets",
        "projected_red_zone_targets",
        "projected_goal_line_carries",
        "projected_fg_attempts",
        "projected_pat_attempts",
        "projected_opponent_plays",
        "projected_opponent_dropbacks",
        "projected_defensive_participation",
    ]
    for column in nonnegative:
        base[column] = numeric(base[column])
        if base[column].isna().any() or base[column].lt(0.0).any():
            raise ValueError(
                f"Issue 33 invalid nonnegative projection column: {column}"
            )

    if base["projected_target_share"].gt(1.0).any():
        raise ValueError(
            "Issue 33 target-share prediction exceeds 1"
        )

    base = add_raw_target_components(base)

    output = base[OUTPUT_COLUMNS].copy()
    common.ensure_unique(
        output,
        GRAIN,
        "Issue 33 component projections",
    )
    if len(output) != len(features):
        raise ValueError(
            "Issue 33 output row count differs from current features"
        )
    common.reject_forbidden_feature_columns(output.columns, config)
    common.write_parquet_atomic(output, output_path)

    payload = {
        "script": Path(__file__).name,
        "status": "passed",
        "season": season,
        "week": week,
        "rows": int(len(output)),
        "games": int(output["game_id"].nunique()),
        "teams": int(output["team"].nunique()),
        "columns": len(OUTPUT_COLUMNS),
        "opportunity_models_scored": len(OPPORTUNITY_NEEDED),
        "efficiency_models_scored": len(EFFICIENCY_NEEDED),
        "target_components_projected": len(TARGET_COMPONENT_COLUMNS),
        "target_component_columns": list(TARGET_COMPONENT_COLUMNS),
        "strict_prior_efficiency_history_end_season": int(
            history["season"].max()
        ),
        "shares_reconciled": False,
        "share_reconciliation_stage": "current_week_allocation",
        "primary_qbs": int(qb_primary.sum()),
        "primary_kickers": int(kicker_primary.sum()),
        "market_exclusion_passed": bool(market["passed"]),
        "market_features_used": False,
        "output": repo_relative(output_path),
        "log": repo_relative(log_path),
    }
    log_payload = {
        **payload,
        "inputs": {
            "features": repo_relative(features_path),
            "roles": repo_relative(roles_path),
            "historical_features": repo_relative(historical_path),
            "eligibility": repo_relative(eligibility_path),
            **(
                {
                    "week1_priors": repo_relative(
                        prop
                        / "data"
                        / "current"
                        / f"{season}_week_1_priors.parquet"
                    )
                }
                if week == 1
                else {}
            ),
        },
        "opportunity_models": opp_audit,
        "efficiency_models": eff_audit,
        "policy": {
            "raw_share_components_only": True,
            "no_share_reconciliation_before_allocation": True,
            "primary_qb_assignment": True,
            "primary_kicker_assignment": True,
            "canonical_target_component_formula_owner": (
                "project_components.py"
            ),
            "sacks_denominator": "projected_opponent_plays",
            "market_exclusion_preflight": True,
        },
    }
    write_json_atomic(log_payload, log_path)
    print(
        json.dumps(
            {
                "script": Path(__file__).name,
                "payload": payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    print("COMPONENT PROJECTIONS: PASS")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
