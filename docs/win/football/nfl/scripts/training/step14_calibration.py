#!/usr/bin/env python3
"""
Step 14: nested chronological shrinkage and validation for v3.

The Step 13 classifier output is a market baseline plus a learned logit
adjustment. This step never performs unconstrained recalibration. Instead it
chooses an adjustment multiplier alpha from a fixed grid, where alpha=0 is the
pure no-vig market and alpha=1 is the full CatBoost correction.

For each evaluation season, alpha is selected using only earlier OOF seasons.
The resulting probabilities are therefore nested chronological estimates.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
NFL_ROOT = SCRIPT_DIR.parents[1]
TRAINING_DIR = NFL_ROOT / "training"
MODELS_DIR = NFL_ROOT / "models"
BACKTEST_DIR = TRAINING_DIR / "backtests"
INPUT_PATH = BACKTEST_DIR / "step13_market_relative_backtest_v3.csv"
OUTPUT_PATH = BACKTEST_DIR / "step14_market_relative_probabilities_v3.csv"
REPORT_PATH = MODELS_DIR / "step14_market_relative_validation_v3.json"
ALPHA_GRID = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
EDGE_THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.05]
PROB_EPS = 1e-6
MIN_BRIER_GAIN = 0.001
MIN_LOGLOSS_GAIN = 0.002
MARKETS = {
    "moneyline": {"market_probability": "market_home_win_probability", "adjustment": "home_win_logit_adjustment", "positive_price": "home_moneyline", "negative_price": "away_moneyline", "positive_label": "HOME", "negative_label": "AWAY"},
    "spread": {"market_probability": "market_home_cover_probability", "adjustment": "home_cover_logit_adjustment", "positive_price": "home_spread_odds", "negative_price": "away_spread_odds", "positive_label": "HOME", "negative_label": "AWAY"},
    "total": {"market_probability": "market_over_probability", "adjustment": "over_logit_adjustment", "positive_price": "over_odds", "negative_price": "under_odds", "positive_label": "OVER", "negative_label": "UNDER"},
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def logit(probability: np.ndarray | pd.Series) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=float), PROB_EPS, 1.0 - PROB_EPS)
    return np.log(values / (1.0 - values))


def sigmoid(value: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def apply_alpha(market_probability: pd.Series, adjustment: pd.Series, alpha: float) -> np.ndarray:
    return sigmoid(logit(market_probability) + float(alpha) * np.asarray(adjustment, dtype=float))


def resolved_market_rows(frame: pd.DataFrame, market: str) -> tuple[pd.DataFrame, np.ndarray]:
    if market == "moneyline":
        outcome = pd.to_numeric(frame["actual_home_win"], errors="coerce")
        mask = outcome.isin([0.0, 1.0])
        return frame.loc[mask].copy(), outcome.loc[mask].astype(int).to_numpy()
    if market == "spread":
        result = frame["actual_home_ats_result"].astype(str).str.upper().str.strip()
        mask = result.isin(["WIN", "LOSS"])
        return frame.loc[mask].copy(), result.loc[mask].eq("WIN").astype(int).to_numpy()
    if market == "total":
        result = frame["actual_total_result"].astype(str).str.upper().str.strip()
        mask = result.isin(["OVER", "UNDER"])
        return frame.loc[mask].copy(), result.loc[mask].eq("OVER").astype(int).to_numpy()
    fail(f"Unknown market: {market}")


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    probability = np.clip(np.asarray(p, dtype=float), PROB_EPS, 1.0 - PROB_EPS)
    labels = np.asarray(y, dtype=int)
    if len(labels) == 0:
        return {"rows": 0}
    brier = float(np.mean((probability - labels) ** 2))
    logloss = float(-np.mean(labels * np.log(probability) + (1 - labels) * np.log(1.0 - probability)))
    predictions = (probability >= 0.5).astype(int)
    return {
        "rows": int(len(labels)),
        "positive_rate": float(np.mean(labels)),
        "mean_probability": float(np.mean(probability)),
        "brier_score": brier,
        "log_loss": logloss,
        "accuracy_at_0_5": float(np.mean(predictions == labels)),
        "predicted_positive_count": int(np.sum(predictions == 1)),
        "predicted_negative_count": int(np.sum(predictions == 0)),
        "probability_min": float(np.min(probability)),
        "probability_max": float(np.max(probability)),
    }


def choose_alpha(frame: pd.DataFrame, market: str) -> tuple[float, list[dict[str, Any]]]:
    resolved, y = resolved_market_rows(frame, market)
    cfg = MARKETS[market]
    if len(resolved) == 0:
        fail(f"No resolved rows for {market}")
    market_probability = pd.to_numeric(resolved[cfg["market_probability"]], errors="raise")
    adjustment = pd.to_numeric(resolved[cfg["adjustment"]], errors="raise")
    candidates: list[dict[str, Any]] = []
    for alpha in ALPHA_GRID:
        candidate_metrics = metrics(y, apply_alpha(market_probability, adjustment, alpha))
        candidates.append({"alpha": alpha, **candidate_metrics})
    best = min(candidates, key=lambda row: (row["log_loss"], row["brier_score"], row["alpha"]))
    return float(best["alpha"]), candidates


def american_profit_per_unit(odds: float) -> float:
    if odds > 0.0:
        return odds / 100.0
    if odds < 0.0:
        return 100.0 / abs(odds)
    fail("American odds cannot be zero")


def quoted_ev(probability: float, odds: float) -> float:
    profit = american_profit_per_unit(odds)
    return probability * profit - (1.0 - probability)


def betting_audit(frame: pd.DataFrame, market: str, probability_column: str, threshold: float) -> dict[str, Any]:
    resolved, y = resolved_market_rows(frame, market)
    cfg = MARKETS[market]
    model_positive = pd.to_numeric(resolved[probability_column], errors="coerce").to_numpy(dtype=float)
    market_positive = pd.to_numeric(resolved[cfg["market_probability"]], errors="coerce").to_numpy(dtype=float)
    positive_odds = pd.to_numeric(resolved[cfg["positive_price"]], errors="coerce").to_numpy(dtype=float)
    negative_odds = pd.to_numeric(resolved[cfg["negative_price"]], errors="coerce").to_numpy(dtype=float)
    bets = wins = losses = positive_bets = negative_bets = 0
    flat_profit = 0.0
    edge_values: list[float] = []
    ev_values: list[float] = []
    for index in range(len(resolved)):
        p_pos = float(model_positive[index])
        p_market = float(market_positive[index])
        edge_pos = p_pos - p_market
        edge_neg = -edge_pos
        ev_pos = quoted_ev(p_pos, float(positive_odds[index]))
        ev_neg = quoted_ev(1.0 - p_pos, float(negative_odds[index]))
        side: str | None = None
        edge = ev = odds = 0.0
        outcome_win = False
        minimum_edge = max(float(threshold), 1e-9)
        if edge_pos >= minimum_edge and ev_pos > 1e-9:
            side = "positive"
            edge, ev, odds = edge_pos, ev_pos, float(positive_odds[index])
            outcome_win = bool(y[index] == 1)
        elif edge_neg >= minimum_edge and ev_neg > 1e-9:
            side = "negative"
            edge, ev, odds = edge_neg, ev_neg, float(negative_odds[index])
            outcome_win = bool(y[index] == 0)
        if side is None:
            continue
        bets += 1
        edge_values.append(edge)
        ev_values.append(ev)
        if side == "positive":
            positive_bets += 1
        else:
            negative_bets += 1
        if outcome_win:
            wins += 1
            flat_profit += american_profit_per_unit(odds)
        else:
            losses += 1
            flat_profit -= 1.0
    return {
        "fair_edge_threshold": float(threshold),
        "evaluated_rows": int(len(resolved)),
        "bets": int(bets),
        "positive_side": cfg["positive_label"],
        "positive_bets": int(positive_bets),
        "negative_side": cfg["negative_label"],
        "negative_bets": int(negative_bets),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": None if bets == 0 else float(wins / bets),
        "flat_profit_units": float(flat_profit),
        "flat_roi": None if bets == 0 else float(flat_profit / bets),
        "average_fair_edge": None if not edge_values else float(np.mean(edge_values)),
        "average_quoted_ev": None if not ev_values else float(np.mean(ev_values)),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
    return value


def main() -> int:
    if not INPUT_PATH.is_file():
        fail(f"Missing Step 13 v3 backtest: {INPUT_PATH}")
    raw = pd.read_csv(INPUT_PATH, low_memory=False)
    if raw.empty:
        fail("Step 13 v3 backtest is empty")
    seasons = sorted(pd.to_numeric(raw["season"], errors="raise").astype(int).unique().tolist())
    if len(seasons) < 2:
        fail("Need at least two OOF seasons for nested chronological shrinkage")
    output = raw.copy()
    report: dict[str, Any] = {
        "step": "14_market_relative_validation_v3",
        "candidate_version": "v3_market_relative_shrinkage",
        "production_cutover": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_backtest": INPUT_PATH.name,
        "source_rows": int(len(raw)),
        "source_seasons": seasons,
        "method": "market no-vig probability + alpha * CatBoost logit adjustment",
        "alpha_grid": ALPHA_GRID,
        "selection_method": "minimum log loss on prior OOF seasons, then Brier, then smaller alpha",
        "validation_method": "nested chronological: each evaluation season selects alpha using only earlier OOF seasons",
        "gate": {"minimum_brier_gain": MIN_BRIER_GAIN, "minimum_logloss_gain": MIN_LOGLOSS_GAIN, "minimum_fold_wins": "ceil(two thirds of chronological evaluation folds)"},
        "markets": {},
    }
    for market, cfg in MARKETS.items():
        resolved_all, y_all = resolved_market_rows(raw, market)
        market_all = pd.to_numeric(resolved_all[cfg["market_probability"]], errors="raise")
        adjustment_all = pd.to_numeric(resolved_all[cfg["adjustment"]], errors="raise")
        production_alpha, alpha_candidates = choose_alpha(raw, market)
        production_probability = apply_alpha(market_all, adjustment_all, production_alpha)
        chronological_frames: list[pd.DataFrame] = []
        fold_reports: list[dict[str, Any]] = []
        for evaluation_season in seasons[1:]:
            selection_seasons = [season for season in seasons if season < evaluation_season]
            selection_frame = raw[raw["season"].isin(selection_seasons)]
            evaluation_frame = raw[raw["season"].eq(evaluation_season)].copy()
            alpha, candidate_rows = choose_alpha(selection_frame, market)
            eval_resolved, eval_y = resolved_market_rows(evaluation_frame, market)
            market_probability = pd.to_numeric(eval_resolved[cfg["market_probability"]], errors="raise")
            adjustment = pd.to_numeric(eval_resolved[cfg["adjustment"]], errors="raise")
            selected_probability = apply_alpha(market_probability, adjustment, alpha)
            market_metrics = metrics(eval_y, market_probability.to_numpy(dtype=float))
            selected_metrics = metrics(eval_y, selected_probability)
            fold_reports.append({
                "evaluation_season": int(evaluation_season),
                "selection_seasons": selection_seasons,
                "selected_alpha": alpha,
                "selection_candidates": candidate_rows,
                "market_metrics": market_metrics,
                "selected_metrics": selected_metrics,
                "logloss_gain_vs_market": market_metrics["log_loss"] - selected_metrics["log_loss"],
                "brier_gain_vs_market": market_metrics["brier_score"] - selected_metrics["brier_score"],
            })
            eval_copy = eval_resolved.copy()
            eval_copy[f"chrono_{market}_alpha"] = alpha
            eval_copy[f"chrono_{market}_probability"] = selected_probability
            chronological_frames.append(eval_copy)
        chronological = pd.concat(chronological_frames, ignore_index=True)
        chrono_resolved, chrono_y = resolved_market_rows(chronological, market)
        chrono_market = pd.to_numeric(chrono_resolved[cfg["market_probability"]], errors="raise").to_numpy(dtype=float)
        chrono_selected = pd.to_numeric(chrono_resolved[f"chrono_{market}_probability"], errors="raise").to_numpy(dtype=float)
        market_chrono_metrics = metrics(chrono_y, chrono_market)
        selected_chrono_metrics = metrics(chrono_y, chrono_selected)
        brier_gain = market_chrono_metrics["brier_score"] - selected_chrono_metrics["brier_score"]
        logloss_gain = market_chrono_metrics["log_loss"] - selected_chrono_metrics["log_loss"]
        fold_wins = sum(1 for fold in fold_reports if fold["logloss_gain_vs_market"] > 0.0 and fold["brier_gain_vs_market"] > 0.0)
        required_fold_wins = int(math.ceil((2.0 / 3.0) * len(fold_reports)))
        gate_pass = bool(production_alpha > 0.0 and brier_gain >= MIN_BRIER_GAIN and logloss_gain >= MIN_LOGLOSS_GAIN and fold_wins >= required_fold_wins)
        betting = {f"fair_edge_{int(threshold * 100)}pct": betting_audit(chronological, market, f"chrono_{market}_probability", threshold) for threshold in EDGE_THRESHOLDS}
        report["markets"][market] = {
            "production_alpha": production_alpha,
            "production_alpha_candidates": alpha_candidates,
            "market_baseline_all_oof": metrics(y_all, market_all.to_numpy(dtype=float)),
            "full_adjustment_all_oof": metrics(y_all, apply_alpha(market_all, adjustment_all, 1.0)),
            "production_selected_all_oof": metrics(y_all, production_probability),
            "chronological_market_same_rows": market_chrono_metrics,
            "chronological_selected": selected_chrono_metrics,
            "chronological_brier_gain_vs_market": brier_gain,
            "chronological_logloss_gain_vs_market": logloss_gain,
            "chronological_fold_wins": fold_wins,
            "required_fold_wins": required_fold_wins,
            "recommended_for_cutover": gate_pass,
            "folds": fold_reports,
            "betting_audits": betting,
        }
        probability_column = f"chrono_{market}_probability"
        alpha_column = f"chrono_{market}_alpha"
        output[probability_column] = np.nan
        output[alpha_column] = np.nan
        keyed = chronological[["game_id", probability_column, alpha_column]].drop_duplicates("game_id")
        probability_map = keyed.set_index("game_id")[probability_column]
        alpha_map = keyed.set_index("game_id")[alpha_column]
        output[probability_column] = output["game_id"].map(probability_map)
        output[alpha_column] = output["game_id"].map(alpha_map)
        output[f"chrono_{market}_fair_edge_positive"] = output[probability_column] - pd.to_numeric(output[cfg["market_probability"]], errors="coerce")
    report["overall_recommended_for_cutover"] = bool(all(report["markets"][market]["recommended_for_cutover"] for market in MARKETS))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8")
    temp_path = OUTPUT_PATH.with_suffix(".tmp")
    output.to_csv(temp_path, index=False)
    temp_path.replace(OUTPUT_PATH)
    print("Step 14 v3 market-relative validation complete.")
    for market in MARKETS:
        result = report["markets"][market]
        print(f"{market}: alpha={result['production_alpha']} chrono_brier_gain={result['chronological_brier_gain_vs_market']:.6f} chrono_logloss_gain={result['chronological_logloss_gain_vs_market']:.6f} fold_wins={result['chronological_fold_wins']}/{len(result['folds'])} cutover={result['recommended_for_cutover']}")
    print(f"Wrote: {REPORT_PATH}")
    print(f"Wrote: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
