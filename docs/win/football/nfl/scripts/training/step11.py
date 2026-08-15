#!/usr/bin/env python3
"""
Step 11: train the NFL market-relative candidate stack (v3).

The sportsbook no-vig probability is the starting point for every classifier.
CatBoost is trained with that market logit as its baseline and therefore learns
only an additive log-odds correction. Current sportsbook prices used to build
that baseline are deliberately excluded from the feature matrix.

Targets:
  spread residual: margin - spread_line
  total residual: total_points - total_line
  moneyline adjustment: home win vs no-vig moneyline baseline
  spread adjustment: home cover vs no-vig spread-price baseline
  total adjustment: OVER vs no-vig total-price baseline

Outputs are isolated v3 artifacts. Nothing here changes the live production
model files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
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
CANDIDATE_VERSION = "v3_market_relative_shrinkage"

SCHEMA_PATH = MODELS_DIR / "step11_market_relative_feature_schema_v3.json"
SPREAD_RESIDUAL_MODEL_PATH = MODELS_DIR / "step11_spread_residual_model_v3.cbm"
TOTAL_RESIDUAL_MODEL_PATH = MODELS_DIR / "step11_total_residual_model_v3.cbm"
MONEYLINE_ADJUSTMENT_MODEL_PATH = MODELS_DIR / "step11_moneyline_adjustment_model_v3.cbm"
SPREAD_ADJUSTMENT_MODEL_PATH = MODELS_DIR / "step11_spread_adjustment_model_v3.cbm"
TOTAL_ADJUSTMENT_MODEL_PATH = MODELS_DIR / "step11_total_adjustment_model_v3.cbm"

REGRESSOR_PARAMS = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 400,
    "learning_rate": 0.025,
    "depth": 5,
    "l2_leaf_reg": 20.0,
    "random_strength": 1.5,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0,
    "one_hot_max_size": 10,
    "max_ctr_complexity": 1,
    "random_seed": 42,
    "thread_count": -1,
    "allow_writing_files": False,
    "verbose": False,
}

CLASSIFIER_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "iterations": 300,
    "learning_rate": 0.02,
    "depth": 4,
    "l2_leaf_reg": 30.0,
    "random_strength": 2.0,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0,
    "one_hot_max_size": 10,
    "max_ctr_complexity": 1,
    "random_seed": 42,
    "thread_count": -1,
    "allow_writing_files": False,
    "verbose": False,
}

TEAM_METRICS = [
    "off_epa_per_play", "def_epa_per_play", "off_success_rate",
    "def_success_rate", "yards_per_play", "yards_per_play_allowed",
    "points_per_drive", "points_per_drive_allowed", "red_zone_td_rate",
    "red_zone_td_rate_allowed", "early_down_epa", "third_down_conversion_rate",
]
QB_METRICS = [
    "epa_per_play", "cpoe", "air_yards", "sack_rate",
    "interception_rate", "fumble_rate",
]
MARKET_LINE_FEATURES = ["spread_line", "total_line"]
BASELINE_PRICE_COLUMNS = [
    "away_moneyline", "home_moneyline",
    "away_spread_odds", "home_spread_odds",
    "under_odds", "over_odds",
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
PROB_EPS = 1e-6


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_training_seasons(start_season: int, end_season: int | None) -> list[int]:
    if not TRAINING_DIR.is_dir():
        fail(f"Training directory not found: {TRAINING_DIR}")
    available = sorted({
        int(match.group(1))
        for path in TRAINING_DIR.iterdir()
        if path.is_file() and (match := HISTORICAL_FILE_PATTERN.fullmatch(path.name))
    })
    if not available:
        fail(f"No historical_core_<YYYY>.csv files found in {TRAINING_DIR}")
    if start_season not in available:
        fail(f"Start season {start_season} missing; available={available}")
    resolved_end = max(available) if end_season is None else int(end_season)
    if resolved_end < start_season:
        fail("end-season cannot be earlier than start-season")
    seasons = list(range(start_season, resolved_end + 1))
    missing = [season for season in seasons if season not in available]
    if missing:
        fail(f"Historical season files must be contiguous; missing={missing}")
    return seasons


def team_feature_names() -> list[str]:
    return [column for metric in TEAM_METRICS for column in (f"home_{metric}", f"away_{metric}", f"{metric}_diff")]


def qb_feature_names() -> list[str]:
    return [column for metric in QB_METRICS for column in (f"home_qb_{metric}", f"away_qb_{metric}", f"qb_{metric}_diff")]


def unique_preserve(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def is_forbidden_feature(column: str) -> bool:
    return column in FORBIDDEN_EXACT or column in BASELINE_PRICE_COLUMNS or column.startswith(FORBIDDEN_PREFIXES) or column.endswith("_result")


def build_feature_list(columns: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    column_set = set(columns)
    drat = [c for c in columns if c.startswith("drat_") and not is_forbidden_feature(c)]
    epred = [c for c in columns if c.startswith("epred_") and not is_forbidden_feature(c)]
    team = team_feature_names()
    qb = qb_feature_names()
    depth_injury = [c for c in columns if ("inj_" in c or "depth_starter_changes" in c) and not is_forbidden_feature(c)]
    families = {
        "drat": drat,
        "epred": epred,
        "market_lines_only": list(MARKET_LINE_FEATURES),
        "team_lagged": team,
        "qb_lagged": qb,
        "schedule_rest_venue_weather_travel": list(SCHEDULE_REST_VENUE_WEATHER_FEATURES),
        "depth_injury": depth_injury,
    }
    features = unique_preserve(drat + epred + MARKET_LINE_FEATURES + team + qb + SCHEDULE_REST_VENUE_WEATHER_FEATURES + depth_injury)
    missing = [c for c in features if c not in column_set]
    if missing:
        fail(f"Required v3 feature columns missing: {missing}")
    forbidden = [c for c in features if is_forbidden_feature(c)]
    if forbidden:
        fail(f"Forbidden/leakage/price columns entered v3 feature matrix: {forbidden}")
    enrichment = [c for c in columns if c.startswith(FORBIDDEN_PREFIXES)]
    if enrichment:
        fail("Static enrichment columns still exist after Step 4: " + ", ".join(enrichment[:20]))
    return features, families


def read_inputs(seasons: list[int]) -> tuple[pd.DataFrame, dict[str, str]]:
    frames: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    reference_columns: list[str] | None = None
    required = {"game_id", "season", "margin", "total_points", "spread_line", "total_line", "home_ats_result", "total_result", *BASELINE_PRICE_COLUMNS}
    for season in seasons:
        path = TRAINING_DIR / f"historical_core_{season}.csv"
        if not path.is_file():
            fail(f"Missing input file: {path}")
        hashes[path.name] = sha256_file(path)
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8-sig", low_memory=False)
        if frame.empty or len(frame.columns) != len(set(frame.columns)):
            fail(f"{path}: empty or duplicate columns")
        if reference_columns is None:
            reference_columns = list(frame.columns)
        elif list(frame.columns) != reference_columns:
            fail(f"{path}: schema/order differs")
        missing = sorted(required - set(frame.columns))
        if missing:
            fail(f"{path}: missing required columns: {missing}")
        parsed_season = pd.to_numeric(frame["season"], errors="coerce")
        if parsed_season.isna().any() or not (parsed_season.astype(int) == season).all():
            fail(f"{path}: wrong-season rows")
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    game_ids = raw["game_id"].astype(str).str.strip()
    if game_ids.eq("").any() or game_ids.duplicated().any():
        fail("game_id values must be populated and unique")
    return raw, hashes


def looks_numeric(series: pd.Series) -> bool:
    text = series.astype(str).str.strip()
    nonblank = text[~text.str.casefold().isin(BLANK_TOKENS)]
    if nonblank.empty:
        return True
    converted = pd.to_numeric(nonblank.str.replace("%", "", regex=False), errors="coerce")
    return bool(converted.notna().all())


def infer_feature_types(raw: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    categorical: list[str] = []
    numeric: list[str] = []
    for column in features:
        if column in FORCED_CATEGORICAL or not looks_numeric(raw[column]):
            categorical.append(column)
        else:
            numeric.append(column)
    return categorical, numeric


def prepare_feature_matrix(raw: pd.DataFrame, features: list[str], categorical: list[str], numeric: list[str]) -> pd.DataFrame:
    matrix = raw[features].copy()
    for column in numeric:
        original = matrix[column].astype(str).str.strip()
        cleaned = original.str.replace("%", "", regex=False)
        blank = cleaned.str.casefold().isin(BLANK_TOKENS)
        converted = pd.to_numeric(cleaned.mask(blank, np.nan), errors="coerce")
        if ((~blank) & converted.isna()).any():
            fail(f"Numeric conversion failed for {column}")
        matrix[column] = converted
    for column in categorical:
        cleaned = matrix[column].astype(str).str.strip()
        matrix[column] = cleaned.mask(cleaned.str.casefold().isin(BLANK_TOKENS), MISSING_CATEGORY)
    return matrix


def numeric_column(raw: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(raw[column], errors="coerce")
    if values.isna().any():
        fail(f"Invalid numeric values in {column}")
    return values.astype(float)


def american_implied_probability(odds: pd.Series) -> pd.Series:
    values = pd.to_numeric(odds, errors="coerce").astype(float)
    if values.isna().any() or (values == 0.0).any():
        fail("American odds contain missing/zero values")
    probability = pd.Series(np.where(values > 0.0, 100.0 / (values + 100.0), (-values) / ((-values) + 100.0)), index=odds.index, dtype=float)
    if (~probability.between(0.0, 1.0, inclusive="neither")).any():
        fail("American odds produced invalid implied probabilities")
    return probability


def no_vig_positive_probability(positive_odds: pd.Series, negative_odds: pd.Series) -> pd.Series:
    positive = american_implied_probability(positive_odds)
    negative = american_implied_probability(negative_odds)
    denominator = positive + negative
    if (denominator <= 0.0).any():
        fail("Invalid two-way market implied-probability denominator")
    return (positive / denominator).clip(PROB_EPS, 1.0 - PROB_EPS)


def probability_logit(probability: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=float)
    values = np.clip(values, PROB_EPS, 1.0 - PROB_EPS)
    return np.log(values / (1.0 - values))


def train_regressor(X: pd.DataFrame, y: pd.Series, cat_indices: list[int]) -> CatBoostRegressor:
    model = CatBoostRegressor(**REGRESSOR_PARAMS)
    pool = Pool(X, label=y, cat_features=cat_indices, feature_names=list(X.columns))
    model.fit(pool, verbose=False)
    return model


def train_adjustment_classifier(X: pd.DataFrame, y: pd.Series, baseline_probability: pd.Series, cat_indices: list[int]) -> CatBoostClassifier:
    labels = y.astype(int)
    if set(labels.unique()) != {0, 1}:
        fail("Classifier target lacks both classes")
    baseline = probability_logit(baseline_probability)
    pool = Pool(X, label=labels, baseline=baseline, cat_features=cat_indices, feature_names=list(X.columns))
    model = CatBoostClassifier(**CLASSIFIER_PARAMS)
    model.fit(pool, verbose=False)
    return model


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=DEFAULT_START_SEASON)
    parser.add_argument("--end-season", type=int, default=None)
    args = parser.parse_args()
    seasons = discover_training_seasons(args.start_season, args.end_season)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    raw, input_hashes = read_inputs(seasons)
    features, families = build_feature_list(list(raw.columns))
    categorical, numeric = infer_feature_types(raw, features)
    matrix = prepare_feature_matrix(raw, features, categorical, numeric)
    categorical_set = set(categorical)
    cat_indices = [index for index, column in enumerate(features) if column in categorical_set]
    margin = numeric_column(raw, "margin")
    total_points = numeric_column(raw, "total_points")
    spread_line = numeric_column(raw, "spread_line")
    total_line = numeric_column(raw, "total_line")
    ats = raw["home_ats_result"].astype(str).str.strip().str.upper()
    totals = raw["total_result"].astype(str).str.strip().str.upper()
    if set(ats) - {"WIN", "LOSS", "PUSH"}:
        fail("Unexpected ATS results")
    if set(totals) - {"OVER", "UNDER", "PUSH"}:
        fail("Unexpected total results")
    market_home_win = no_vig_positive_probability(raw["home_moneyline"], raw["away_moneyline"])
    market_home_cover = no_vig_positive_probability(raw["home_spread_odds"], raw["away_spread_odds"])
    market_over = no_vig_positive_probability(raw["over_odds"], raw["under_odds"])
    ml_mask = margin.ne(0.0)
    spread_mask = ats.isin(["WIN", "LOSS"])
    total_mask = totals.isin(["OVER", "UNDER"])
    print(f"Step 11 v3 rows={len(raw)} features={len(features)} numeric={len(numeric)} categorical={len(categorical)}")
    spread_residual_model = train_regressor(matrix, margin - spread_line, cat_indices)
    total_residual_model = train_regressor(matrix, total_points - total_line, cat_indices)
    moneyline_adjustment_model = train_adjustment_classifier(matrix.loc[ml_mask], margin.loc[ml_mask].gt(0.0), market_home_win.loc[ml_mask], cat_indices)
    spread_adjustment_model = train_adjustment_classifier(matrix.loc[spread_mask], ats.loc[spread_mask].eq("WIN"), market_home_cover.loc[spread_mask], cat_indices)
    total_adjustment_model = train_adjustment_classifier(matrix.loc[total_mask], totals.loc[total_mask].eq("OVER"), market_over.loc[total_mask], cat_indices)
    spread_residual_model.save_model(SPREAD_RESIDUAL_MODEL_PATH)
    total_residual_model.save_model(TOTAL_RESIDUAL_MODEL_PATH)
    moneyline_adjustment_model.save_model(MONEYLINE_ADJUSTMENT_MODEL_PATH)
    spread_adjustment_model.save_model(SPREAD_ADJUSTMENT_MODEL_PATH)
    total_adjustment_model.save_model(TOTAL_ADJUSTMENT_MODEL_PATH)
    schema = {
        "step": 11,
        "candidate_version": CANDIDATE_VERSION,
        "production_cutover": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_seasons": seasons,
        "training_rows": int(len(raw)),
        "targets": {
            "spread_residual": "margin - spread_line",
            "total_residual": "total_points - total_line",
            "moneyline_adjustment": "home win with no-vig home moneyline logit as CatBoost baseline",
            "spread_adjustment": "home cover with no-vig home spread-price logit as CatBoost baseline",
            "total_adjustment": "OVER with no-vig OVER-price logit as CatBoost baseline",
        },
        "market_baselines": {
            "moneyline": {"positive_side": "home", "positive_price": "home_moneyline", "negative_price": "away_moneyline"},
            "spread": {"positive_side": "home cover", "positive_price": "home_spread_odds", "negative_price": "away_spread_odds"},
            "total": {"positive_side": "OVER", "positive_price": "over_odds", "negative_price": "under_odds"},
            "formula": "normalize the two quoted American implied probabilities to sum to 1; feed logit(no-vig p) as CatBoost Pool baseline",
            "price_columns_excluded_from_feature_matrix": BASELINE_PRICE_COLUMNS,
        },
        "model_files": {
            "spread_residual": SPREAD_RESIDUAL_MODEL_PATH.name,
            "total_residual": TOTAL_RESIDUAL_MODEL_PATH.name,
            "moneyline_adjustment": MONEYLINE_ADJUSTMENT_MODEL_PATH.name,
            "spread_adjustment": SPREAD_ADJUSTMENT_MODEL_PATH.name,
            "total_adjustment": TOTAL_ADJUSTMENT_MODEL_PATH.name,
        },
        "regressor_params": REGRESSOR_PARAMS,
        "classifier_params": CLASSIFIER_PARAMS,
        "feature_count": len(features),
        "feature_order": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "categorical_feature_indices": cat_indices,
        "feature_families": families,
        "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
        "input_sha256": input_hashes,
    }
    SCHEMA_PATH.write_text(json.dumps(json_safe(schema), indent=2) + "\n", encoding="utf-8")
    print("Step 11 v3 complete.")
    print(f"Schema: {SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
