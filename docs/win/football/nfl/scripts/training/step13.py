#!/usr/bin/env python3
"""
Step 13: leakage-free chronological backtest for the Step 11 v2 candidate stack.

Each held-out season is predicted using only earlier seasons. No game from the
held-out season participates in model fitting.

Inputs:
  docs/win/football/nfl/training/historical_core_<season>.csv
  docs/win/football/nfl/models/step11_clean_feature_schema.json

Output:
  docs/win/football/nfl/training/backtests/step13_chronological_backtest_v2.csv
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

SCRIPT_DIR = Path(__file__).resolve().parent
NFL_ROOT = SCRIPT_DIR.parents[1]
TRAINING_DIR = NFL_ROOT / "training"
MODELS_DIR = NFL_ROOT / "models"
BACKTEST_DIR = TRAINING_DIR / "backtests"
SCHEMA_PATH = MODELS_DIR / "step11_clean_feature_schema.json"
OUTPUT_PATH = BACKTEST_DIR / "step13_chronological_backtest_v2.csv"
BLANK_TOKENS = {"", "nan", "none", "null", "<na>", "nat"}
MISSING_CATEGORY = "__MISSING__"
REQUIRED_RESULT_COLUMNS = [
    "game_id", "season", "week", "gameday", "gametime", "away_team", "home_team",
    "spread_line", "total_line", "away_moneyline", "home_moneyline",
    "away_spread_odds", "home_spread_odds", "under_odds", "over_odds",
    "margin", "total_points", "home_ats_result", "total_result",
]
OUTPUT_COLUMNS = [
    "season", "week", "gameday", "gametime", "game_id", "away_team", "home_team",
    "training_seasons", "training_rows", "away_moneyline", "home_moneyline",
    "spread_line", "away_spread_odds", "home_spread_odds", "total_line",
    "under_odds", "over_odds", "predicted_spread_residual", "actual_spread_residual",
    "predicted_margin", "actual_margin", "predicted_total_residual", "actual_total_residual",
    "predicted_total", "actual_total_points", "predicted_home_score", "predicted_away_score",
    "raw_home_win_probability", "raw_home_cover_probability", "raw_over_probability",
    "actual_home_win", "actual_home_ats_result", "actual_total_result",
]

def fail(message: str) -> None: raise RuntimeError(message)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()

def load_schema() -> dict[str, Any]:
    if not SCHEMA_PATH.is_file(): fail(f"Missing Step 11 v2 schema: {SCHEMA_PATH}")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    required = {"candidate_version", "training_seasons", "regressor_params", "classifier_params", "feature_count", "feature_order", "numeric_features", "categorical_features", "categorical_feature_indices", "input_sha256"}
    missing = sorted(required - set(schema))
    if missing: fail(f"Step 11 v2 schema missing keys: {missing}")
    if schema.get("candidate_version") != "v2_leak_free_market_residual_probability": fail("Unexpected Step 11 candidate_version")
    seasons = [int(v) for v in schema["training_seasons"]]
    if len(seasons) < 2 or seasons != list(range(seasons[0], seasons[-1] + 1)): fail(f"Training seasons invalid: {seasons}")
    features = list(schema["feature_order"]); numeric = list(schema["numeric_features"]); categorical = list(schema["categorical_features"]); cat_indices = list(schema["categorical_feature_indices"])
    if len(features) != int(schema["feature_count"]) or len(features) != len(set(features)): fail("Invalid feature_order")
    if set(numeric) & set(categorical) or set(numeric) | set(categorical) != set(features): fail("Invalid feature type coverage")
    expected = [i for i, c in enumerate(features) if c in set(categorical)]
    if cat_indices != expected: fail("categorical_feature_indices mismatch")
    return schema

def read_inputs(schema: dict[str, Any], seasons: list[int]) -> pd.DataFrame:
    frames, reference = [], None
    hashes = dict(schema.get("input_sha256", {}))
    for season in seasons:
        path = TRAINING_DIR / f"historical_core_{season}.csv"
        if not path.is_file(): fail(f"Missing input file: {path}")
        expected_hash = hashes.get(path.name)
        if expected_hash and sha256_file(path) != expected_hash: fail(f"{path.name}: SHA256 differs from Step 11 input")
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8-sig", low_memory=False)
        if df.empty or len(df.columns) != len(set(df.columns)): fail(f"{path}: empty or duplicate columns")
        if reference is None: reference = list(df.columns)
        elif list(df.columns) != reference: fail(f"{path}: schema/order differs")
        missing = sorted(set(REQUIRED_RESULT_COLUMNS) - set(df.columns))
        if missing: fail(f"{path}: missing backtest columns: {missing}")
        feature_missing = [c for c in schema["feature_order"] if c not in df.columns]
        if feature_missing: fail(f"{path}: missing Step 11 v2 features: {feature_missing}")
        parsed = pd.to_numeric(df["season"], errors="coerce")
        if parsed.isna().any() or not (parsed.astype(int) == season).all(): fail(f"{path}: wrong-season rows")
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True); ids = raw["game_id"].astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any(): fail("game_id values must be populated and unique")
    return raw

def prepare_feature_matrix(raw: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    features = list(schema["feature_order"]); numeric = list(schema["numeric_features"]); categorical = list(schema["categorical_features"]); X = raw[features].copy()
    for c in numeric:
        original = X[c].astype(str).str.strip(); cleaned = original.str.replace("%", "", regex=False); blank = cleaned.str.casefold().isin(BLANK_TOKENS); converted = pd.to_numeric(cleaned.mask(blank, np.nan), errors="coerce")
        if ((~blank) & converted.isna()).any(): fail(f"Numeric conversion failed for {c}")
        X[c] = converted
    for c in categorical:
        cleaned = X[c].astype(str).str.strip(); X[c] = cleaned.mask(cleaned.str.casefold().isin(BLANK_TOKENS), MISSING_CATEGORY)
    return X

def validate_results(raw: pd.DataFrame) -> None:
    for c in ["margin", "total_points", "spread_line", "total_line"]:
        if pd.to_numeric(raw[c], errors="coerce").isna().any(): fail(f"{c}: blank/non-numeric rows")
    ats = raw["home_ats_result"].astype(str).str.strip().str.upper(); totals = raw["total_result"].astype(str).str.strip().str.upper()
    if sorted(set(ats) - {"WIN", "LOSS", "PUSH"}): fail("Unexpected ATS results")
    if sorted(set(totals) - {"OVER", "UNDER", "PUSH"}): fail("Unexpected total results")

def train_regressor(X: pd.DataFrame, y: pd.Series, cat_indices: list[int], params: dict[str, Any]) -> CatBoostRegressor:
    model = CatBoostRegressor(**params); model.fit(Pool(X, label=y, cat_features=cat_indices, feature_names=list(X.columns)), verbose=False); return model

def train_classifier(X: pd.DataFrame, y: pd.Series, cat_indices: list[int], params: dict[str, Any]) -> CatBoostClassifier:
    if set(y.astype(int).unique()) != {0, 1}: fail("Classifier training fold lacks both classes")
    model = CatBoostClassifier(**params); model.fit(Pool(X, label=y.astype(int), cat_features=cat_indices, feature_names=list(X.columns)), verbose=False); return model

def predict_positive_probability(model: CatBoostClassifier, X: pd.DataFrame) -> np.ndarray:
    probability = np.asarray(model.predict_proba(X), dtype=float); classes = [int(v) for v in model.classes_]
    if probability.ndim != 2 or 1 not in classes: fail("Invalid classifier probability output")
    output = probability[:, classes.index(1)]
    if not np.isfinite(output).all() or ((output < 0) | (output > 1)).any(): fail("Invalid probabilities")
    return output

def make_rows(holdout: pd.DataFrame, training_seasons: list[int], training_rows: int, spread_residual: np.ndarray, total_residual: np.ndarray, ml_probability: np.ndarray, spread_probability: np.ndarray, total_probability: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for pos, (_, row) in enumerate(holdout.iterrows()):
        spread_line = float(row["spread_line"]); total_line = float(row["total_line"]); margin = float(row["margin"]); total_points = float(row["total_points"])
        predicted_margin = spread_line + float(spread_residual[pos]); predicted_total = total_line + float(total_residual[pos])
        rows.append({
            "season": int(float(row["season"])), "week": int(float(row["week"])), "gameday": str(row["gameday"]).strip(), "gametime": str(row["gametime"]).strip(), "game_id": str(row["game_id"]).strip(), "away_team": str(row["away_team"]).strip(), "home_team": str(row["home_team"]).strip(),
            "training_seasons": f"{training_seasons[0]}-{training_seasons[-1]}", "training_rows": training_rows,
            "away_moneyline": row["away_moneyline"], "home_moneyline": row["home_moneyline"], "spread_line": spread_line, "away_spread_odds": row["away_spread_odds"], "home_spread_odds": row["home_spread_odds"], "total_line": total_line, "under_odds": row["under_odds"], "over_odds": row["over_odds"],
            "predicted_spread_residual": float(spread_residual[pos]), "actual_spread_residual": margin - spread_line, "predicted_margin": predicted_margin, "actual_margin": margin,
            "predicted_total_residual": float(total_residual[pos]), "actual_total_residual": total_points - total_line, "predicted_total": predicted_total, "actual_total_points": total_points,
            "predicted_home_score": (predicted_total + predicted_margin) / 2.0, "predicted_away_score": (predicted_total - predicted_margin) / 2.0,
            "raw_home_win_probability": float(ml_probability[pos]), "raw_home_cover_probability": float(spread_probability[pos]), "raw_over_probability": float(total_probability[pos]),
            "actual_home_win": 1 if margin > 0 else 0 if margin < 0 else "", "actual_home_ats_result": str(row["home_ats_result"]).strip().upper(), "actual_total_result": str(row["total_result"]).strip().upper(),
        })
    return rows

def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix(path.suffix + ".tmp"); df.to_csv(temp, index=False); os.replace(temp, path)

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--reset", action="store_true", help="Accepted for workflow compatibility; v2 always rebuilds atomically."); parser.parse_args()
    schema = load_schema(); seasons = [int(v) for v in schema["training_seasons"]]; raw = read_inputs(schema, seasons); validate_results(raw); X = prepare_feature_matrix(raw, schema)
    season_num = pd.to_numeric(raw["season"], errors="raise").astype(int); margin = pd.to_numeric(raw["margin"], errors="raise").astype(float); total_points = pd.to_numeric(raw["total_points"], errors="raise").astype(float); spread_line = pd.to_numeric(raw["spread_line"], errors="raise").astype(float); total_line = pd.to_numeric(raw["total_line"], errors="raise").astype(float)
    ats = raw["home_ats_result"].astype(str).str.strip().str.upper(); totals = raw["total_result"].astype(str).str.strip().str.upper(); spread_target = margin - spread_line; total_target = total_points - total_line
    cat_indices = list(schema["categorical_feature_indices"]); reg_params = dict(schema["regressor_params"]); cls_params = dict(schema["classifier_params"]); results = []
    for holdout_season in seasons[1:]:
        train_mask = season_num < holdout_season; holdout_mask = season_num == holdout_season; train_seasons = sorted(season_num.loc[train_mask].unique().tolist())
        spread_model = train_regressor(X.loc[train_mask], spread_target.loc[train_mask], cat_indices, reg_params); total_res_model = train_regressor(X.loc[train_mask], total_target.loc[train_mask], cat_indices, reg_params)
        ml_train = train_mask & margin.ne(0.0); ats_train = train_mask & ats.isin(["WIN", "LOSS"]); total_train = train_mask & totals.isin(["OVER", "UNDER"])
        ml_model = train_classifier(X.loc[ml_train], margin.loc[ml_train].gt(0.0).astype(int), cat_indices, cls_params); ats_model = train_classifier(X.loc[ats_train], ats.loc[ats_train].eq("WIN").astype(int), cat_indices, cls_params); total_model = train_classifier(X.loc[total_train], totals.loc[total_train].eq("OVER").astype(int), cat_indices, cls_params)
        holdout_X = X.loc[holdout_mask]
        results.extend(make_rows(raw.loc[holdout_mask], train_seasons, int(train_mask.sum()), np.asarray(spread_model.predict(holdout_X), dtype=float), np.asarray(total_res_model.predict(holdout_X), dtype=float), predict_positive_probability(ml_model, holdout_X), predict_positive_probability(ats_model, holdout_X), predict_positive_probability(total_model, holdout_X)))
        print(f"Step 13 v2 fold holdout={holdout_season} train={train_seasons[0]}-{train_seasons[-1]} train_rows={int(train_mask.sum())} holdout_rows={int(holdout_mask.sum())}")
    output = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    if output.empty or output["game_id"].duplicated().any(): fail("Invalid Step 13 v2 output")
    output["_date"] = pd.to_datetime(output["gameday"], errors="raise"); output["_time"] = pd.to_timedelta(output["gametime"].astype(str) + ":00", errors="raise")
    output = output.sort_values(["season", "week", "_date", "_time", "game_id"], kind="mergesort").drop(columns=["_date", "_time"]).reset_index(drop=True)
    atomic_write_csv(output, OUTPUT_PATH); print(f"Step 13 v2 complete: heldout_seasons={seasons[1]}-{seasons[-1]} rows={len(output)} -> {OUTPUT_PATH}"); return 0

if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr); raise
