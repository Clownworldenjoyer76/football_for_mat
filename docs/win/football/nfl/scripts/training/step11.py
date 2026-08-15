#!/usr/bin/env python3
"""
Step 11: train the leakage-free NFL candidate model stack (v2).

Inputs:
  docs/win/football/nfl/training/historical_core_<season>.csv

Only exact four-digit season files are used. The default range starts at 2021
and includes every contiguous season through the latest available file.

The v2 stack is isolated from the current production artifacts so it can be
validated before cutover.

Targets:
  spread residual: margin - spread_line
  total residual:  total_points - total_line
  moneyline probability: home win, ties excluded
  spread probability: home cover, pushes excluded
  total probability: OVER, pushes excluded

Feature policy:
  - pregame DRAT / EPRED values
  - current market prices and lines
  - lagged team statistics
  - lagged QB statistics
  - schedule/rest/venue/weather/travel
  - depth/injury
  - NO ml_*, ats_*, totals_* static rule-enrichment columns
  - NO outcomes, scores, targets, result columns, IDs, or season labels

Outputs:
  docs/win/football/nfl/models/step11_clean_feature_schema.json
  docs/win/football/nfl/models/step11_spread_residual_model_v2.cbm
  docs/win/football/nfl/models/step11_total_residual_model_v2.cbm
  docs/win/football/nfl/models/step11_moneyline_probability_model_v2.cbm
  docs/win/football/nfl/models/step11_spread_probability_model_v2.cbm
  docs/win/football/nfl/models/step11_total_probability_model_v2.cbm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

SCRIPT_DIR = Path(__file__).resolve().parent
NFL_ROOT = SCRIPT_DIR.parents[1]
TRAINING_DIR = NFL_ROOT / "training"
MODELS_DIR = NFL_ROOT / "models"
HISTORICAL_FILE_PATTERN = re.compile(r"^historical_core_(\d{4})\.csv$")
DEFAULT_START_SEASON = 2021
SCHEMA_PATH = MODELS_DIR / "step11_clean_feature_schema.json"
SPREAD_RESIDUAL_MODEL_PATH = MODELS_DIR / "step11_spread_residual_model_v2.cbm"
TOTAL_RESIDUAL_MODEL_PATH = MODELS_DIR / "step11_total_residual_model_v2.cbm"
MONEYLINE_PROB_MODEL_PATH = MODELS_DIR / "step11_moneyline_probability_model_v2.cbm"
SPREAD_PROB_MODEL_PATH = MODELS_DIR / "step11_spread_probability_model_v2.cbm"
TOTAL_PROB_MODEL_PATH = MODELS_DIR / "step11_total_probability_model_v2.cbm"

REGRESSOR_PARAMS = {
    "loss_function": "RMSE", "eval_metric": "RMSE", "iterations": 500,
    "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 12.0,
    "random_strength": 1.0, "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0, "one_hot_max_size": 10,
    "max_ctr_complexity": 1, "random_seed": 42, "thread_count": -1,
    "allow_writing_files": False, "verbose": False,
}
CLASSIFIER_PARAMS = {
    "loss_function": "Logloss", "eval_metric": "Logloss", "iterations": 500,
    "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 12.0,
    "random_strength": 1.0, "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0, "one_hot_max_size": 10,
    "max_ctr_complexity": 1, "random_seed": 42, "thread_count": -1,
    "allow_writing_files": False, "verbose": False,
}
TEAM_METRICS = [
    "off_epa_per_play", "def_epa_per_play", "off_success_rate",
    "def_success_rate", "yards_per_play", "yards_per_play_allowed",
    "points_per_drive", "points_per_drive_allowed", "red_zone_td_rate",
    "red_zone_td_rate_allowed", "early_down_epa", "third_down_conversion_rate",
]
QB_METRICS = ["epa_per_play", "cpoe", "air_yards", "sack_rate", "interception_rate", "fumble_rate"]
MARKET_FEATURES = [
    "away_moneyline", "home_moneyline", "spread_line", "away_spread_odds",
    "home_spread_odds", "total_line", "under_odds", "over_odds",
]
SCHEDULE_REST_VENUE_WEATHER_FEATURES = [
    "game_type", "week", "weekday", "gametime", "away_team", "home_team",
    "location", "away_rest", "home_rest", "div_game", "roof", "surface",
    "temp", "wind", "stadium_id", "stadium", "hist_surface",
    "hist_weather_icon", "hist_temperature", "hist_precip_probability",
    "hist_precip_type", "hist_wind_speed", "hist_wind_bearing", "rest_diff",
    "miles_traveled", "time_zones_crossed", "east_to_west", "west_to_east",
    "international_flag", "neutral_site_flag",
]
FORBIDDEN_EXACT = {
    "game_id", "season", "gameday", "away_score", "home_score", "margin",
    "total_points", "home_win", "home_ats_margin", "home_ats_result",
    "total_result", "away_qb_id", "home_qb_id", "away_qb_name",
    "home_qb_name", "away_coach", "home_coach", "hist_odds_total",
    "hist_home_spread", "hist_away_spread",
}
FORBIDDEN_PREFIXES = ("ml_", "ats_", "totals_")
FORCED_CATEGORICAL = {
    "game_type", "weekday", "gametime", "away_team", "home_team", "location",
    "roof", "surface", "stadium_id", "stadium", "hist_surface",
    "hist_weather_icon", "hist_precip_type",
}
BLANK_TOKENS = {"", "nan", "none", "null", "<na>", "nat"}
MISSING_CATEGORY = "__MISSING__"

def fail(message: str) -> None: raise RuntimeError(message)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()

def discover_training_seasons(start_season: int = DEFAULT_START_SEASON, end_season: int | None = None) -> list[int]:
    if not TRAINING_DIR.is_dir(): fail(f"Training directory not found: {TRAINING_DIR}")
    available = sorted({int(m.group(1)) for p in TRAINING_DIR.iterdir() if p.is_file() and (m := HISTORICAL_FILE_PATTERN.fullmatch(p.name))})
    if not available: fail(f"No historical_core_<YYYY>.csv files found in {TRAINING_DIR}")
    if start_season not in available: fail(f"Start season {start_season} missing; available={available}")
    resolved_end = max(available) if end_season is None else int(end_season)
    if resolved_end < start_season: fail("end-season cannot be earlier than start-season")
    seasons = list(range(start_season, resolved_end + 1))
    missing = [s for s in seasons if s not in available]
    if missing: fail(f"Historical season files must be contiguous; missing={missing}")
    return seasons

def team_feature_names() -> list[str]:
    return [c for m in TEAM_METRICS for c in (f"home_{m}", f"away_{m}", f"{m}_diff")]

def qb_feature_names() -> list[str]:
    return [c for m in QB_METRICS for c in (f"home_qb_{m}", f"away_qb_{m}", f"qb_{m}_diff")]

def unique_preserve(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))

def is_forbidden_feature(column: str) -> bool:
    return column in FORBIDDEN_EXACT or column.startswith(FORBIDDEN_PREFIXES) or column.endswith("_result")

def build_feature_list(columns: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    column_set = set(columns)
    drat = [c for c in columns if c.startswith("drat_") and not is_forbidden_feature(c)]
    epred = [c for c in columns if c.startswith("epred_") and not is_forbidden_feature(c)]
    team, qb = team_feature_names(), qb_feature_names()
    depth_injury = [c for c in columns if ("inj_" in c or "depth_starter_changes" in c) and not is_forbidden_feature(c)]
    families = {
        "drat": drat, "epred": epred, "market": list(MARKET_FEATURES),
        "team_lagged": team, "qb_lagged": qb,
        "schedule_rest_venue_weather_travel": list(SCHEDULE_REST_VENUE_WEATHER_FEATURES),
        "depth_injury": depth_injury,
    }
    features = unique_preserve(drat + epred + MARKET_FEATURES + team + qb + SCHEDULE_REST_VENUE_WEATHER_FEATURES + depth_injury)
    missing = [c for c in features if c not in column_set]
    if missing: fail(f"Required clean feature columns missing: {missing}")
    forbidden = [c for c in features if is_forbidden_feature(c)]
    if forbidden: fail(f"Forbidden/leakage columns entered feature matrix: {forbidden}")
    enrichment = [c for c in columns if c.startswith(FORBIDDEN_PREFIXES)]
    if enrichment: fail("Static enrichment columns still exist after Step 4: " + ", ".join(enrichment[:20]))
    return features, families

def read_inputs(seasons: list[int]) -> tuple[pd.DataFrame, dict[str, str], dict[int, Path]]:
    frames, hashes, paths = [], {}, {}
    reference = None
    for season in seasons:
        path = TRAINING_DIR / f"historical_core_{season}.csv"; paths[season] = path
        if not path.is_file(): fail(f"Missing input file: {path}")
        hashes[path.name] = sha256_file(path)
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8-sig", low_memory=False)
        if df.empty or len(df.columns) != len(set(df.columns)): fail(f"{path}: empty or duplicate columns")
        if reference is None: reference = list(df.columns)
        elif list(df.columns) != reference: fail(f"{path}: schema/order differs")
        required = {"game_id", "season", "margin", "total_points", "spread_line", "total_line", "home_ats_result", "total_result"}
        missing = sorted(required - set(df.columns))
        if missing: fail(f"{path}: missing required columns: {missing}")
        parsed = pd.to_numeric(df["season"], errors="coerce")
        if parsed.isna().any() or not (parsed.astype(int) == season).all(): fail(f"{path}: wrong-season rows")
        for target in ["margin", "total_points", "spread_line", "total_line"]:
            if pd.to_numeric(df[target], errors="coerce").isna().any(): fail(f"{path}: bad {target}")
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    ids = raw["game_id"].astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any(): fail("game_id values must be populated and unique")
    return raw, hashes, paths

def looks_numeric(series: pd.Series) -> bool:
    text = series.astype(str).str.strip(); nonblank = text[~text.str.casefold().isin(BLANK_TOKENS)]
    if nonblank.empty: return True
    return bool(pd.to_numeric(nonblank.str.replace("%", "", regex=False), errors="coerce").notna().all())

def infer_feature_types(raw: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    categorical, numeric = [], []
    for c in features:
        (categorical if c in FORCED_CATEGORICAL or not looks_numeric(raw[c]) else numeric).append(c)
    return categorical, numeric

def prepare_feature_matrix(raw: pd.DataFrame, features: list[str], categorical: list[str], numeric: list[str]) -> pd.DataFrame:
    X = raw[features].copy()
    for c in numeric:
        original = X[c].astype(str).str.strip(); cleaned = original.str.replace("%", "", regex=False)
        blank = cleaned.str.casefold().isin(BLANK_TOKENS); converted = pd.to_numeric(cleaned.mask(blank, np.nan), errors="coerce")
        if ((~blank) & converted.isna()).any(): fail(f"Numeric conversion failed for {c}")
        X[c] = converted
    for c in categorical:
        cleaned = X[c].astype(str).str.strip(); X[c] = cleaned.mask(cleaned.str.casefold().isin(BLANK_TOKENS), MISSING_CATEGORY)
    return X

def train_regressor(X: pd.DataFrame, y: pd.Series, cat_indices: list[int]) -> CatBoostRegressor:
    model = CatBoostRegressor(**REGRESSOR_PARAMS); model.fit(Pool(X, label=y, cat_features=cat_indices, feature_names=list(X.columns)), verbose=False); return model

def train_classifier(X: pd.DataFrame, y: pd.Series, cat_indices: list[int]) -> CatBoostClassifier:
    if set(y.astype(int).unique()) != {0, 1}: fail("Classifier target lacks both classes")
    model = CatBoostClassifier(**CLASSIFIER_PARAMS); model.fit(Pool(X, label=y.astype(int), cat_features=cat_indices, feature_names=list(X.columns)), verbose=False); return model

def json_safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating):
        value = float(value); return None if math.isnan(value) or math.isinf(value) else value
    return value

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--start-season", type=int, default=DEFAULT_START_SEASON); parser.add_argument("--end-season", type=int, default=None); args = parser.parse_args()
    seasons = discover_training_seasons(args.start_season, args.end_season); MODELS_DIR.mkdir(parents=True, exist_ok=True)
    raw, input_hashes, input_paths = read_inputs(seasons)
    features, families = build_feature_list(list(raw.columns)); categorical, numeric = infer_feature_types(raw, features)
    X = prepare_feature_matrix(raw, features, categorical, numeric); cat_indices = [i for i, c in enumerate(features) if c in set(categorical)]
    margin = pd.to_numeric(raw["margin"], errors="raise").astype(float); total_points = pd.to_numeric(raw["total_points"], errors="raise").astype(float)
    spread_line = pd.to_numeric(raw["spread_line"], errors="raise").astype(float); total_line = pd.to_numeric(raw["total_line"], errors="raise").astype(float)
    ats = raw["home_ats_result"].astype(str).str.strip().str.upper(); totals = raw["total_result"].astype(str).str.strip().str.upper()
    if sorted(set(ats) - {"WIN", "LOSS", "PUSH"}): fail("Unexpected ATS results")
    if sorted(set(totals) - {"OVER", "UNDER", "PUSH"}): fail("Unexpected total results")
    ml_mask = margin.ne(0.0); ats_mask = ats.isin(["WIN", "LOSS"]); total_mask = totals.isin(["OVER", "UNDER"])
    print(f"Step 11 v2 rows={len(raw)} features={len(features)} numeric={len(numeric)} categorical={len(categorical)}")
    spread_residual_model = train_regressor(X, margin - spread_line, cat_indices)
    total_residual_model = train_regressor(X, total_points - total_line, cat_indices)
    ml_model = train_classifier(X.loc[ml_mask], margin.loc[ml_mask].gt(0.0).astype(int), cat_indices)
    spread_model = train_classifier(X.loc[ats_mask], ats.loc[ats_mask].eq("WIN").astype(int), cat_indices)
    total_model = train_classifier(X.loc[total_mask], totals.loc[total_mask].eq("OVER").astype(int), cat_indices)
    spread_residual_model.save_model(SPREAD_RESIDUAL_MODEL_PATH); total_residual_model.save_model(TOTAL_RESIDUAL_MODEL_PATH)
    ml_model.save_model(MONEYLINE_PROB_MODEL_PATH); spread_model.save_model(SPREAD_PROB_MODEL_PATH); total_model.save_model(TOTAL_PROB_MODEL_PATH)
    schema = {
        "step": 11, "candidate_version": "v2_leak_free_market_residual_probability", "production_cutover": False,
        "created_utc": datetime.now(timezone.utc).isoformat(), "training_seasons": seasons, "training_rows": int(len(raw)),
        "targets": {"spread_residual": "margin - spread_line", "total_residual": "total_points - total_line", "moneyline_probability": "home win; ties excluded", "spread_probability": "home cover; pushes excluded", "total_probability": "OVER; pushes excluded"},
        "model_files": {"spread_residual": SPREAD_RESIDUAL_MODEL_PATH.name, "total_residual": TOTAL_RESIDUAL_MODEL_PATH.name, "moneyline_probability": MONEYLINE_PROB_MODEL_PATH.name, "spread_probability": SPREAD_PROB_MODEL_PATH.name, "total_probability": TOTAL_PROB_MODEL_PATH.name},
        "regressor_params": REGRESSOR_PARAMS, "classifier_params": CLASSIFIER_PARAMS, "feature_count": len(features), "feature_order": features,
        "numeric_features": numeric, "categorical_features": categorical, "categorical_feature_indices": cat_indices, "feature_families": families,
        "forbidden_exact": sorted(FORBIDDEN_EXACT), "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
        "preprocessing": {"numeric": "strip; remove %; blank tokens -> NaN; parse numeric", "categorical": f"strip; blank tokens -> {MISSING_CATEGORY}", "missing_category_token": MISSING_CATEGORY},
        "input_files": [input_paths[s].name for s in seasons], "input_sha256": input_hashes,
    }
    SCHEMA_PATH.write_text(json.dumps(json_safe(schema), indent=2) + "\n", encoding="utf-8")
    print("Step 11 v2 complete."); print(f"Schema: {SCHEMA_PATH}")
    return 0

if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr); raise
