#!/usr/bin/env python3
"""
Step 14 v2: calibrate direct out-of-fold probability models chronologically.

Production calibrators are fit on all out-of-fold rows, while validation is
strictly expanding chronological: each evaluation season uses only earlier OOF
seasons to fit its calibrator.

Outputs:
  docs/win/football/nfl/models/step14_probability_calibration_v2.json
  docs/win/football/nfl/training/backtests/step14_chronological_probabilities_v2.csv
"""
from __future__ import annotations
import argparse, hashlib, json, math, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
NFL_ROOT = SCRIPT_DIR.parents[1]
TRAINING_DIR = NFL_ROOT / "training"
MODELS_DIR = NFL_ROOT / "models"
DEFAULT_BACKTEST_PATH = TRAINING_DIR / "backtests" / "step13_chronological_backtest_v2.csv"
DEFAULT_OUTPUT_PATH = MODELS_DIR / "step14_probability_calibration_v2.json"
DEFAULT_PROBABILITY_OUTPUT = TRAINING_DIR / "backtests" / "step14_chronological_probabilities_v2.csv"
EPS = 1e-6
REQUIRED_COLUMNS = {
    "season", "week", "gameday", "gametime", "game_id", "away_moneyline",
    "home_moneyline", "away_spread_odds", "home_spread_odds", "under_odds",
    "over_odds", "actual_margin", "actual_home_ats_result", "actual_total_result",
    "raw_home_win_probability", "raw_home_cover_probability", "raw_over_probability",
}

def fail(message: str) -> None: raise RuntimeError(message)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()

def sigmoid(z: np.ndarray | float) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=float), -700.0, 700.0); return 1.0 / (1.0 + np.exp(-z))

def logit(probability: np.ndarray | pd.Series) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), EPS, 1.0 - EPS); return np.log(p / (1.0 - p))

def fit_logistic_1d(x: np.ndarray, y: np.ndarray, max_iter: int = 250, tolerance: float = 1e-10, ridge: float = 1e-6) -> tuple[float, float]:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 20: fail("Invalid calibration sample")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or set(np.unique(y)) != {0.0, 1.0}: fail("Calibration data invalid")
    X = np.column_stack([np.ones(len(x)), x]); beta = np.array([0.0, 1.0]); penalty = np.diag([0.0, ridge])
    for _ in range(max_iter):
        p = sigmoid(X @ beta); w = np.clip(p * (1.0 - p), 1e-8, None); gradient = X.T @ (y - p) - penalty @ beta; hessian = X.T @ (w[:, None] * X) + penalty
        try: step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc: fail(f"Calibration Hessian singular: {exc}")
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < tolerance: beta = beta_new; break
        beta = beta_new
    else: fail("Calibration logistic regression did not converge")
    return float(beta[0]), float(beta[1])

def apply_calibration(raw_probability: np.ndarray | pd.Series, intercept: float, slope: float) -> np.ndarray:
    return sigmoid(intercept + slope * logit(raw_probability))

def probability_metrics(probability: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    p = np.asarray(probability, dtype=float); target = np.asarray(y, dtype=float)
    if not len(p): return {"rows": 0}
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return {
        "rows": int(len(p)), "positive_rate": float(np.mean(target)), "mean_probability": float(np.mean(p)),
        "brier_score": float(np.mean((p - target) ** 2)),
        "log_loss": float(-np.mean(target * np.log(p) + (1.0 - target) * np.log(1.0 - p))),
        "accuracy_at_0_5": float(np.mean((p >= 0.5) == (target >= 0.5))),
        "predicted_positive_count": int(np.sum(p >= 0.5)), "predicted_negative_count": int(np.sum(p < 0.5)),
        "probability_min": float(np.min(p)), "probability_max": float(np.max(p)),
    }

def calibration_bins(probability: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    p = np.asarray(probability, dtype=float); target = np.asarray(y, dtype=float); output = []; edges = np.linspace(0.0, 1.0, 11)
    for i in range(10):
        low, high = edges[i], edges[i + 1]; mask = (p >= low) & (p < high if i < 9 else p <= high)
        if mask.any(): output.append({"min_probability": float(low), "max_probability": float(high), "rows": int(mask.sum()), "mean_probability": float(np.mean(p[mask])), "actual_rate": float(np.mean(target[mask]))})
    return output

def numeric(series: pd.Series, name: str) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce")
    if result.isna().any(): fail(f"{name} contains blank/non-numeric values")
    return result.astype(float)

def build_outcome_series(df: pd.DataFrame, market: str) -> pd.Series:
    outcome = pd.Series(np.nan, index=df.index, dtype=float)
    if market == "moneyline":
        margin = numeric(df["actual_margin"], "actual_margin"); outcome.loc[margin > 0] = 1.0; outcome.loc[margin < 0] = 0.0; return outcome
    if market == "spread":
        status = df["actual_home_ats_result"].astype(str).str.strip().str.upper()
        if sorted(set(status) - {"WIN", "LOSS", "PUSH"}): fail("Unexpected ATS results")
        outcome.loc[status == "WIN"] = 1.0; outcome.loc[status == "LOSS"] = 0.0; return outcome
    if market == "total":
        status = df["actual_total_result"].astype(str).str.strip().str.upper()
        if sorted(set(status) - {"OVER", "UNDER", "PUSH"}): fail("Unexpected total results")
        outcome.loc[status == "OVER"] = 1.0; outcome.loc[status == "UNDER"] = 0.0; return outcome
    fail(f"Unknown market: {market}")

def calibrate_market(df: pd.DataFrame, market: str, raw_probability_column: str) -> tuple[dict[str, Any], pd.Series]:
    raw = numeric(df[raw_probability_column], raw_probability_column)
    if ((raw <= 0) | (raw >= 1)).any(): fail(f"{raw_probability_column} must be strictly between 0 and 1")
    outcome = build_outcome_series(df, market); season = pd.to_numeric(df["season"], errors="raise").astype(int); valid = outcome.notna(); seasons = sorted(season.loc[valid].unique().tolist())
    if len(seasons) < 2: fail(f"{market}: not enough OOF seasons")
    production_intercept, production_slope = fit_logistic_1d(logit(raw.loc[valid]), outcome.loc[valid].to_numpy(dtype=float))
    chronological = pd.Series(np.nan, index=df.index, dtype=float); folds = []
    for eval_season in seasons[1:]:
        train_mask = valid & (season < eval_season); eval_mask = valid & (season == eval_season)
        if not train_mask.any() or not eval_mask.any(): continue
        intercept, slope = fit_logistic_1d(logit(raw.loc[train_mask]), outcome.loc[train_mask].to_numpy(dtype=float)); predicted = apply_calibration(raw.loc[eval_mask], intercept, slope); chronological.loc[eval_mask] = predicted
        folds.append({"evaluation_season": int(eval_season), "calibration_training_seasons": sorted(int(v) for v in season.loc[train_mask].unique()), "training_rows": int(train_mask.sum()), "evaluation_rows": int(eval_mask.sum()), "intercept": intercept, "slope": slope, "raw_metrics": probability_metrics(raw.loc[eval_mask].to_numpy(dtype=float), outcome.loc[eval_mask].to_numpy(dtype=float)), "calibrated_metrics": probability_metrics(predicted, outcome.loc[eval_mask].to_numpy(dtype=float))})
    chrono_mask = chronological.notna() & valid
    section = {
        "input_probability": raw_probability_column,
        "positive_outcome": {"moneyline": "home win", "spread": "home cover", "total": "OVER"}[market],
        "excluded": {"moneyline": "ties", "spread": "pushes", "total": "pushes"}[market],
        "production_calibration": {"method": "sigmoid(intercept + slope * logit(raw_probability))", "intercept": production_intercept, "slope": production_slope, "rows": int(valid.sum()), "source_seasons": seasons},
        "diagnostics": {
            "raw_all_oof": probability_metrics(raw.loc[valid].to_numpy(dtype=float), outcome.loc[valid].to_numpy(dtype=float)),
            "chronological_raw_same_rows": probability_metrics(raw.loc[chrono_mask].to_numpy(dtype=float), outcome.loc[chrono_mask].to_numpy(dtype=float)),
            "chronological_calibrated": probability_metrics(chronological.loc[chrono_mask].to_numpy(dtype=float), outcome.loc[chrono_mask].to_numpy(dtype=float)),
            "chronological_calibration_bins": calibration_bins(chronological.loc[chrono_mask].to_numpy(dtype=float), outcome.loc[chrono_mask].to_numpy(dtype=float)),
            "folds": folds, "excluded_rows": int((~valid).sum()),
        },
    }
    return section, chronological

def american_profit(odds: Any) -> float | None:
    try: price = float(odds)
    except (TypeError, ValueError): return None
    if not math.isfinite(price) or price == 0: return None
    return price / 100.0 if price > 0 else 100.0 / abs(price)

def quoted_ev(probability: float, odds: Any) -> float | None:
    profit = american_profit(odds); return None if profit is None else probability * profit - (1.0 - probability)

def total_betting_audit(df: pd.DataFrame, over_probability: pd.Series) -> dict[str, Any]:
    rows = []
    for index, row in df.iterrows():
        if pd.isna(over_probability.loc[index]): continue
        p_over = float(over_probability.loc[index]); p_under = 1.0 - p_over; over_ev = quoted_ev(p_over, row["over_odds"]); under_ev = quoted_ev(p_under, row["under_odds"])
        if over_ev is None or under_ev is None: continue
        candidates = [("OVER", over_ev, p_over, row["over_odds"]), ("UNDER", under_ev, p_under, row["under_odds"])]; candidates = [c for c in candidates if c[1] > 0]
        if not candidates: rows.append({"side": "NONE", "profit": 0.0, "ev": max(over_ev, under_ev)}); continue
        side, ev, probability, odds = max(candidates, key=lambda item: item[1]); actual = str(row["actual_total_result"]).strip().upper()
        if actual == "PUSH": profit, result = 0.0, "PUSH"
        elif actual == side: profit, result = american_profit(odds), "WIN"
        else: profit, result = -1.0, "LOSS"
        if profit is not None: rows.append({"side": side, "profit": profit, "result": result, "ev": ev, "probability": probability})
    if not rows: return {"evaluated_rows": 0}
    audit = pd.DataFrame(rows); bets = audit[audit["side"].isin(["OVER", "UNDER"])].copy()
    if bets.empty: return {"evaluated_rows": int(len(audit)), "bets": 0, "over_bets": 0, "under_bets": 0, "no_bet_rows": int((audit["side"] == "NONE").sum())}
    settled = bets[bets["result"].isin(["WIN", "LOSS"])]
    return {
        "evaluated_rows": int(len(audit)), "bets": int(len(bets)), "over_bets": int((bets["side"] == "OVER").sum()), "under_bets": int((bets["side"] == "UNDER").sum()), "no_bet_rows": int((audit["side"] == "NONE").sum()),
        "wins": int((bets["result"] == "WIN").sum()), "losses": int((bets["result"] == "LOSS").sum()), "pushes": int((bets["result"] == "PUSH").sum()),
        "win_rate_ex_pushes": float((settled["result"] == "WIN").mean()) if len(settled) else None,
        "flat_profit_units": float(bets["profit"].sum()), "flat_roi": float(bets["profit"].sum() / len(bets)), "average_model_probability": float(bets["probability"].mean()), "average_quoted_ev": float(bets["ev"].mean()),
    }
def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix(path.suffix + ".tmp"); df.to_csv(temp, index=False); os.replace(temp, path)
def json_safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list): return [json_safe(v) for v in value]
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating):
        value = float(value); return None if not math.isfinite(value) else value
    return value

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--backtest", type=Path, default=DEFAULT_BACKTEST_PATH); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH); parser.add_argument("--probability-output", type=Path, default=DEFAULT_PROBABILITY_OUTPUT); args = parser.parse_args()
    backtest_path = args.backtest.resolve(); output_path = args.output.resolve(); probability_output = args.probability_output.resolve()
    if not backtest_path.is_file(): fail(f"Backtest file not found: {backtest_path}")
    df = pd.read_csv(backtest_path, low_memory=False); missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing: fail(f"Backtest missing columns: {missing}")
    if df.empty or df["game_id"].isna().any() or df["game_id"].duplicated().any(): fail("Backtest rows/game_id invalid")
    ml_section, ml_chrono = calibrate_market(df, "moneyline", "raw_home_win_probability"); spread_section, spread_chrono = calibrate_market(df, "spread", "raw_home_cover_probability"); total_section, total_chrono = calibrate_market(df, "total", "raw_over_probability")
    probability_frame = df.copy(); probability_frame["chrono_home_win_probability"] = ml_chrono; probability_frame["chrono_away_win_probability"] = 1.0 - ml_chrono; probability_frame["chrono_home_cover_probability"] = spread_chrono; probability_frame["chrono_away_cover_probability"] = 1.0 - spread_chrono; probability_frame["chrono_over_probability"] = total_chrono; probability_frame["chrono_under_probability"] = 1.0 - total_chrono; atomic_write_csv(probability_frame, probability_output)
    total_audit = total_betting_audit(df, total_chrono)
    artifact = {
        "step": "14_probability_calibration_v2", "candidate_version": "v2_leak_free_market_residual_probability", "production_cutover": False, "created_utc": datetime.now(timezone.utc).isoformat(), "source_backtest": backtest_path.name, "source_backtest_sha256": sha256_file(backtest_path), "source_rows": int(len(df)), "source_seasons": sorted(int(v) for v in pd.to_numeric(df["season"], errors="raise").unique()),
        "method": "Platt calibration on logit of direct OOF classifier probability", "live_formula": "sigmoid(intercept + slope * logit(raw_classifier_probability))", "validation_method": "expanding chronological calibration: each evaluation season uses only earlier OOF seasons",
        "calibrations": {"moneyline": ml_section, "spread": spread_section, "total": total_section},
        "chronological_betting_diagnostics": {"total_positive_quoted_ev_zero_threshold": total_audit, "note": "Choose the total side with the larger positive quoted-price EV."},
        "chronological_probability_output": probability_output.name,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True); output_path.write_text(json.dumps(json_safe(artifact), indent=2) + "\n", encoding="utf-8")
    print("Step 14 v2 probability calibration complete.")
    for market, section in artifact["calibrations"].items():
        d = section["diagnostics"]["chronological_calibrated"]; p = section["production_calibration"]; print(f"{market}: chrono_rows={d.get('rows', 0)} brier={d.get('brier_score')} logloss={d.get('log_loss')} production_intercept={p['intercept']:.8f} production_slope={p['slope']:.8f}")
    print("TOTAL positive-EV audit: " + json.dumps(total_audit, sort_keys=True)); print(f"Wrote: {output_path}"); print(f"Wrote: {probability_output}")

if __name__ == "__main__": main()
