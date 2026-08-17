#!/usr/bin/env python3
"""
Step 14 calibration: build live probability mappings from the Step 13 backtest.

Repository location:
    docs/win/football/nfl/scripts/training/step14_calibration.py

Default input:
    docs/win/football/nfl/training/backtests/step13_chronological_backtest.csv

Default output:
    docs/win/football/nfl/models/step14_probability_calibration.json

Calibration definitions:
  ML:
    x = predicted_margin
    y = 1 when actual_margin > 0, 0 when actual_margin < 0
    tied games are excluded

  ATS:
    x = predicted_margin - spread_line
    y = 1 when actual_home_ats_result == WIN,
        0 when actual_home_ats_result == LOSS
    pushes are excluded

  TOTAL:
    x = predicted_total - total_line
    y = 1 when actual_total_result == OVER,
        0 when actual_total_result == UNDER
    pushes are excluded

Each calibration is a one-variable logistic (Platt-style) mapping:
    probability = 1 / (1 + exp(-(intercept + slope * x)))
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
NFL_ROOT = SCRIPT_DIR.parents[1]
TRAINING_DIR = NFL_ROOT / "training"
MODELS_DIR = NFL_ROOT / "models"
DEFAULT_BACKTEST_PATH = (
    TRAINING_DIR / "backtests" / "step13_chronological_backtest.csv"
)
DEFAULT_OUTPUT_PATH = MODELS_DIR / "step14_probability_calibration.json"

REQUIRED_COLUMNS = {
    "season",
    "week",
    "gameday",
    "gametime",
    "game_id",
    "spread_line",
    "total_line",
    "predicted_margin",
    "actual_margin",
    "predicted_total",
    "actual_total_points",
    "actual_home_ats_result",
    "actual_total_result",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def fit_logistic_1d(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 200,
    tolerance: float = 1e-12,
) -> tuple[float, float]:
    """Unregularized two-parameter logistic regression via Newton/IRLS."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        fail("Calibration x/y dimensions are invalid")
    if len(x) < 2:
        fail("Not enough calibration rows")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        fail("Calibration data contain non-finite values")
    if set(np.unique(y)) != {0.0, 1.0}:
        fail("Calibration target must contain both 0 and 1 outcomes")

    X = np.column_stack([np.ones(len(x), dtype=float), x])
    beta = np.zeros(2, dtype=float)

    for _ in range(max_iter):
        eta = X @ beta
        p = sigmoid(eta)
        w = np.clip(p * (1.0 - p), 1e-12, None)

        gradient = X.T @ (y - p)
        hessian = X.T @ (w[:, None] * X)

        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            fail(f"Calibration Hessian is singular: {exc}")

        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < tolerance:
            beta = beta_new
            break
        beta = beta_new
    else:
        fail("Calibration logistic regression did not converge")

    return float(beta[0]), float(beta[1])


def diagnostics(
    x: np.ndarray,
    y: np.ndarray,
    intercept: float,
    slope: float,
) -> dict[str, float | int]:
    probability = sigmoid(intercept + slope * x)
    eps = 1e-15
    clipped = np.clip(probability, eps, 1.0 - eps)
    brier = float(np.mean((probability - y) ** 2))
    log_loss = float(
        -np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))
    )
    return {
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "brier_score_in_sample": brier,
        "log_loss_in_sample": log_loss,
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "x_mean": float(np.mean(x)),
    }


def numeric(series: pd.Series, name: str) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce")
    if result.isna().any():
        examples = series[result.isna()].head(5).tolist()
        fail(f"{name} contains blank/non-numeric values: {examples}")
    return result.astype(float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backtest",
        type=Path,
        default=DEFAULT_BACKTEST_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    args = parser.parse_args()

    backtest_path = args.backtest.resolve()
    output_path = args.output.resolve()

    if not backtest_path.is_file():
        fail(f"Backtest file not found: {backtest_path}")

    df = pd.read_csv(backtest_path, low_memory=False)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        fail(f"Backtest is missing required columns: {missing}")
    if df.empty:
        fail("Backtest contains zero rows")
    if df["game_id"].isna().any() or df["game_id"].duplicated().any():
        fail("Backtest game_id must be populated and unique")

    predicted_margin = numeric(df["predicted_margin"], "predicted_margin")
    actual_margin = numeric(df["actual_margin"], "actual_margin")
    spread_line = numeric(df["spread_line"], "spread_line")
    predicted_total = numeric(df["predicted_total"], "predicted_total")
    total_line = numeric(df["total_line"], "total_line")

    ml_mask = actual_margin != 0.0
    ml_x = predicted_margin.loc[ml_mask].to_numpy(dtype=float)
    ml_y = (actual_margin.loc[ml_mask] > 0.0).astype(int).to_numpy()

    ats_status = df["actual_home_ats_result"].astype(str).str.strip().str.upper()
    bad_ats = sorted(set(ats_status) - {"WIN", "LOSS", "PUSH"})
    if bad_ats:
        fail(f"Unexpected actual_home_ats_result values: {bad_ats}")
    ats_mask = ats_status.isin(["WIN", "LOSS"])
    ats_x = (
        predicted_margin.loc[ats_mask] - spread_line.loc[ats_mask]
    ).to_numpy(dtype=float)
    ats_y = (ats_status.loc[ats_mask] == "WIN").astype(int).to_numpy()

    total_status = df["actual_total_result"].astype(str).str.strip().str.upper()
    bad_total = sorted(set(total_status) - {"OVER", "UNDER", "PUSH"})
    if bad_total:
        fail(f"Unexpected actual_total_result values: {bad_total}")
    total_mask = total_status.isin(["OVER", "UNDER"])
    total_x = (
        predicted_total.loc[total_mask] - total_line.loc[total_mask]
    ).to_numpy(dtype=float)
    total_y = (total_status.loc[total_mask] == "OVER").astype(int).to_numpy()

    ml_intercept, ml_slope = fit_logistic_1d(ml_x, ml_y)
    ats_intercept, ats_slope = fit_logistic_1d(ats_x, ats_y)
    total_intercept, total_slope = fit_logistic_1d(total_x, total_y)

    if ml_slope <= 0 or ats_slope <= 0 or total_slope <= 0:
        fail(
            "One or more fitted calibration slopes are not positive; "
            "inspect Step 13 before using calibration"
        )

    artifact = {
        "step": "14_probability_calibration",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_backtest": backtest_path.name,
        "source_backtest_sha256": sha256_file(backtest_path),
        "source_rows": int(len(df)),
        "source_seasons": sorted(
            int(v) for v in pd.to_numeric(df["season"], errors="raise").unique()
        ),
        "method": "one_variable_logistic_platt",
        "probability_formula": "1 / (1 + exp(-(intercept + slope * x)))",
        "calibrations": {
            "moneyline": {
                "x_definition": "predicted_margin",
                "positive_outcome": "home win (actual_margin > 0)",
                "excluded": "actual_margin == 0 (ties)",
                "intercept": ml_intercept,
                "slope": ml_slope,
                "home_win_probability": "sigmoid(intercept + slope * predicted_margin)",
                "away_win_probability": "1 - home_win_probability",
                "diagnostics": diagnostics(
                    ml_x, ml_y, ml_intercept, ml_slope
                ),
                "excluded_rows": int((~ml_mask).sum()),
            },
            "spread": {
                "x_definition": "predicted_margin - spread_line",
                "positive_outcome": "home cover (actual_home_ats_result == WIN)",
                "excluded": "actual_home_ats_result == PUSH",
                "intercept": ats_intercept,
                "slope": ats_slope,
                "home_cover_probability": "sigmoid(intercept + slope * (predicted_margin - spread_line))",
                "away_cover_probability": "1 - home_cover_probability",
                "diagnostics": diagnostics(
                    ats_x, ats_y, ats_intercept, ats_slope
                ),
                "excluded_rows": int((~ats_mask).sum()),
            },
            "total": {
                "x_definition": "predicted_total - total_line",
                "positive_outcome": "OVER (actual_total_result == OVER)",
                "excluded": "actual_total_result == PUSH",
                "intercept": total_intercept,
                "slope": total_slope,
                "over_probability": "sigmoid(intercept + slope * (predicted_total - total_line))",
                "under_probability": "1 - over_probability",
                "diagnostics": diagnostics(
                    total_x, total_y, total_intercept, total_slope
                ),
                "excluded_rows": int((~total_mask).sum()),
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
        handle.write("\n")

    print("Step 14 probability calibration complete.")
    print(f"Source rows: {len(df)}")
    print(
        "ML: "
        f"rows={len(ml_y)}, intercept={ml_intercept:.12f}, "
        f"slope={ml_slope:.12f}"
    )
    print(
        "ATS: "
        f"rows={len(ats_y)}, intercept={ats_intercept:.12f}, "
        f"slope={ats_slope:.12f}"
    )
    print(
        "TOTAL: "
        f"rows={len(total_y)}, intercept={total_intercept:.12f}, "
        f"slope={total_slope:.12f}"
    )
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
