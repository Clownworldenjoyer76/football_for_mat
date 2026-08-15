#!/usr/bin/env python3
"""
Step 13: expanding-season out-of-fold backtest for the v3 market-relative stack.

For each held-out season, models are trained only on earlier seasons. The
classifier's baseline is the sportsbook no-vig logit; the saved adjustment is
therefore incremental to the market rather than an unconstrained standalone
probability.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

SCRIPT_DIR = Path(__file__).resolve().parent
NFL_ROOT = SCRIPT_DIR.parents[1]
TRAINING_DIR = NFL_ROOT / "training"
MODELS_DIR = NFL_ROOT / "models"
BACKTEST_DIR = TRAINING_DIR / "backtests"
SCHEMA_PATH = MODELS_DIR / "step11_market_relative_feature_schema_v3.json"
OUTPUT_PATH = BACKTEST_DIR / "step13_market_relative_backtest_v3.csv"
HISTORICAL_FILE_PATTERN = re.compile(r"^historical_core_(\d{4})\.csv$")
BLANK_TOKENS = {"", "nan", "none", "null", "<na>", "nat"}
MISSING_CATEGORY = "__MISSING__"
PROB_EPS = 1e-6


def fail(message: str) -> None:
    raise RuntimeError(message)


def discover_seasons() -> list[int]:
    seasons = sorted({int(match.group(1)) for path in TRAINING_DIR.iterdir() if path.is_file() and (match := HISTORICAL_FILE_PATTERN.fullmatch(path.name))})
    if len(seasons) < 2:
        fail("Need at least two historical season files")
    if seasons != list(range(min(seasons), max(seasons) + 1)):
        fail(f"Historical season files must be contiguous: {seasons}")
    return seasons


def read_season(season: int) -> pd.DataFrame:
    path = TRAINING_DIR / f"historical_core_{season}.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8-sig", low_memory=False)
    if frame.empty:
        fail(f"Empty training file: {path}")
    return frame


def prepare_matrix(raw: pd.DataFrame, schema: dict) -> pd.DataFrame:
    features = schema["feature_order"]
    missing = [column for column in features if column not in raw.columns]
    if missing:
        fail(f"Missing v3 feature columns: {missing}")
    matrix = raw[features].copy()
    for column in schema["numeric_features"]:
        original = matrix[column].astype(str).str.strip()
        cleaned = original.str.replace("%", "", regex=False)
        blank = cleaned.str.casefold().isin(BLANK_TOKENS)
        converted = pd.to_numeric(cleaned.mask(blank, np.nan), errors="coerce")
        if ((~blank) & converted.isna()).any():
            fail(f"Numeric conversion failed for {column}")
        matrix[column] = converted
    for column in schema["categorical_features"]:
        cleaned = matrix[column].astype(str).str.strip()
        matrix[column] = cleaned.mask(cleaned.str.casefold().isin(BLANK_TOKENS), MISSING_CATEGORY)
    return matrix


def numeric(raw: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(raw[column], errors="coerce")
    if values.isna().any():
        fail(f"Invalid numeric values in {column}")
    return values.astype(float)


def american_implied_probability(odds: pd.Series) -> pd.Series:
    values = pd.to_numeric(odds, errors="coerce").astype(float)
    if values.isna().any() or (values == 0.0).any():
        fail("American odds contain missing/zero values")
    return pd.Series(np.where(values > 0.0, 100.0 / (values + 100.0), (-values) / ((-values) + 100.0)), index=odds.index, dtype=float)


def no_vig_positive_probability(positive_odds: pd.Series, negative_odds: pd.Series) -> pd.Series:
    positive = american_implied_probability(positive_odds)
    negative = american_implied_probability(negative_odds)
    return (positive / (positive + negative)).clip(PROB_EPS, 1.0 - PROB_EPS)


def logit(probability: pd.Series | np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=float), PROB_EPS, 1.0 - PROB_EPS)
    return np.log(values / (1.0 - values))


def train_regressor(X: pd.DataFrame, y: pd.Series, schema: dict) -> CatBoostRegressor:
    model = CatBoostRegressor(**schema["regressor_params"])
    model.fit(Pool(X, label=y, cat_features=schema["categorical_feature_indices"], feature_names=schema["feature_order"]), verbose=False)
    return model


def train_adjustment(X: pd.DataFrame, y: pd.Series, baseline_probability: pd.Series, schema: dict) -> CatBoostClassifier:
    labels = y.astype(int)
    if set(labels.unique()) != {0, 1}:
        fail("Classifier target lacks both classes")
    model = CatBoostClassifier(**schema["classifier_params"])
    model.fit(Pool(X, label=labels, baseline=logit(baseline_probability), cat_features=schema["categorical_feature_indices"], feature_names=schema["feature_order"]), verbose=False)
    return model


def predict_adjusted_probability(model: CatBoostClassifier, X: pd.DataFrame, baseline_probability: pd.Series, schema: dict) -> np.ndarray:
    pool = Pool(X, baseline=logit(baseline_probability), cat_features=schema["categorical_feature_indices"], feature_names=schema["feature_order"])
    return np.asarray(model.predict_proba(pool)[:, 1], dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Accepted for workflow compatibility; output is always rebuilt")
    _ = parser.parse_args()
    if not SCHEMA_PATH.is_file():
        fail(f"Missing Step 11 v3 schema: {SCHEMA_PATH}")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    if schema.get("candidate_version") != "v3_market_relative_shrinkage":
        fail("Unexpected Step 11 schema candidate_version")
    seasons = discover_seasons()
    season_frames = {season: read_season(season) for season in seasons}
    reference = list(season_frames[seasons[0]].columns)
    for season in seasons[1:]:
        if list(season_frames[season].columns) != reference:
            fail(f"Season {season} schema differs from first season")
    outputs: list[pd.DataFrame] = []
    for holdout_season in seasons[1:]:
        training_seasons = [season for season in seasons if season < holdout_season]
        train = pd.concat([season_frames[season] for season in training_seasons], ignore_index=True)
        test = season_frames[holdout_season].copy().reset_index(drop=True)
        X_train = prepare_matrix(train, schema)
        X_test = prepare_matrix(test, schema)
        train_margin = numeric(train, "margin")
        train_total = numeric(train, "total_points")
        train_spread_line = numeric(train, "spread_line")
        train_total_line = numeric(train, "total_line")
        test_margin = numeric(test, "margin")
        test_total = numeric(test, "total_points")
        test_spread_line = numeric(test, "spread_line")
        test_total_line = numeric(test, "total_line")
        train_ats = train["home_ats_result"].astype(str).str.strip().str.upper()
        train_totals = train["total_result"].astype(str).str.strip().str.upper()
        test_ats = test["home_ats_result"].astype(str).str.strip().str.upper()
        test_totals = test["total_result"].astype(str).str.strip().str.upper()
        train_market_ml = no_vig_positive_probability(train["home_moneyline"], train["away_moneyline"])
        train_market_spread = no_vig_positive_probability(train["home_spread_odds"], train["away_spread_odds"])
        train_market_total = no_vig_positive_probability(train["over_odds"], train["under_odds"])
        test_market_ml = no_vig_positive_probability(test["home_moneyline"], test["away_moneyline"])
        test_market_spread = no_vig_positive_probability(test["home_spread_odds"], test["away_spread_odds"])
        test_market_total = no_vig_positive_probability(test["over_odds"], test["under_odds"])
        ml_mask = train_margin.ne(0.0)
        spread_mask = train_ats.isin(["WIN", "LOSS"])
        total_mask = train_totals.isin(["OVER", "UNDER"])
        spread_residual_model = train_regressor(X_train, train_margin - train_spread_line, schema)
        total_residual_model = train_regressor(X_train, train_total - train_total_line, schema)
        ml_model = train_adjustment(X_train.loc[ml_mask], train_margin.loc[ml_mask].gt(0.0), train_market_ml.loc[ml_mask], schema)
        spread_model = train_adjustment(X_train.loc[spread_mask], train_ats.loc[spread_mask].eq("WIN"), train_market_spread.loc[spread_mask], schema)
        total_model = train_adjustment(X_train.loc[total_mask], train_totals.loc[total_mask].eq("OVER"), train_market_total.loc[total_mask], schema)
        spread_residual_pred = np.asarray(spread_residual_model.predict(X_test), dtype=float)
        total_residual_pred = np.asarray(total_residual_model.predict(X_test), dtype=float)
        full_ml = predict_adjusted_probability(ml_model, X_test, test_market_ml, schema)
        full_spread = predict_adjusted_probability(spread_model, X_test, test_market_spread, schema)
        full_total = predict_adjusted_probability(total_model, X_test, test_market_total, schema)
        frame = pd.DataFrame({
            "season": pd.to_numeric(test["season"], errors="raise").astype(int),
            "week": test["week"],
            "gameday": test.get("gameday", ""),
            "gametime": test.get("gametime", ""),
            "game_id": test["game_id"],
            "away_team": test["away_team"],
            "home_team": test["home_team"],
            "training_seasons": f"{min(training_seasons)}-{max(training_seasons)}",
            "training_rows": len(train),
            "away_moneyline": numeric(test, "away_moneyline"),
            "home_moneyline": numeric(test, "home_moneyline"),
            "spread_line": test_spread_line,
            "away_spread_odds": numeric(test, "away_spread_odds"),
            "home_spread_odds": numeric(test, "home_spread_odds"),
            "total_line": test_total_line,
            "under_odds": numeric(test, "under_odds"),
            "over_odds": numeric(test, "over_odds"),
            "predicted_spread_residual": spread_residual_pred,
            "actual_spread_residual": test_margin - test_spread_line,
            "predicted_margin": test_spread_line + spread_residual_pred,
            "actual_margin": test_margin,
            "predicted_total_residual": total_residual_pred,
            "actual_total_residual": test_total - test_total_line,
            "predicted_total": test_total_line + total_residual_pred,
            "actual_total_points": test_total,
            "market_home_win_probability": np.asarray(test_market_ml, dtype=float),
            "full_home_win_probability": full_ml,
            "home_win_logit_adjustment": logit(full_ml) - logit(test_market_ml),
            "market_home_cover_probability": np.asarray(test_market_spread, dtype=float),
            "full_home_cover_probability": full_spread,
            "home_cover_logit_adjustment": logit(full_spread) - logit(test_market_spread),
            "market_over_probability": np.asarray(test_market_total, dtype=float),
            "full_over_probability": full_total,
            "over_logit_adjustment": logit(full_total) - logit(test_market_total),
            "actual_home_win": np.where(test_margin > 0.0, 1.0, np.where(test_margin < 0.0, 0.0, np.nan)),
            "actual_home_ats_result": test_ats,
            "actual_total_result": test_totals,
        })
        outputs.append(frame)
        print(f"Step 13 v3 fold holdout={holdout_season} train={min(training_seasons)}-{max(training_seasons)} train_rows={len(train)} holdout_rows={len(test)}")
    output = pd.concat(outputs, ignore_index=True)
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = OUTPUT_PATH.with_suffix(".tmp")
    output.to_csv(temp_path, index=False)
    temp_path.replace(OUTPUT_PATH)
    print(f"Step 13 v3 complete: heldout_seasons={seasons[1]}-{seasons[-1]} rows={len(output)} -> {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
