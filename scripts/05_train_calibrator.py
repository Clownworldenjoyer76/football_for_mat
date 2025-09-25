#!/usr/bin/env python3
"""
Train probability calibrators for OVER outcomes using historical props.

Input CSV must include at least:
  - market (str)         e.g. 'passing_yards'
  - prob_over (float)    uncalibrated probability our pipeline produced
  - over_actual (int/float/bool)  1 if actual result went over the prop line, else 0

Usage examples:
  python scripts/05_train_calibrator.py
  python scripts/05_train_calibrator.py --market qb_passing_yards --method isotonic
  python scripts/05_train_calibrator.py --csv data/props/history_props_with_outcomes.csv --outdir models/calibration
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sys, pathlib as _pl

# allow `from scripts.lib...` when executed as a script
sys.path.append(str(_pl.Path(__file__).resolve().parents[1]))

from scripts.lib.calibration import (
    fit_calibrator,
    apply_calibrator,
    evaluate_calibration,
    cross_validated_method_choice,
    save_calibrator,
)

MIN_ROWS_PER_MARKET = 50   # skip markets with fewer than this many clean rows
EPS = 1e-6                 # clip probabilities away from 0/1 for stable losses


def _coerce_bool01(x: pd.Series) -> pd.Series:
    """
    Coerce a series to {0,1}:
      - accepts ints/floats/bools/strings ("0","1","true","false","yes","no")
      - NaN stays NaN (caller decides drop/fill)
    """
    s = pd.to_numeric(x, errors="coerce")
    if s.isna().all():
        lower = x.astype(str).str.strip().str.lower()
        map_ = {
            "1": 1, "true": 1, "t": 1, "yes": 1, "y": 1, "over": 1,
            "0": 0, "false": 0, "f": 0, "no": 0, "n": 0, "under": 0,
        }
        s = lower.map(map_).astype("float64")
    s = s.fillna(np.nan)
    s = (s > 0.5).astype("float64")
    return s


def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sanitized copy with numeric probs ∈ (0,1) and target in {0,1}."""
    out = df.copy()

    out["prob_over"] = pd.to_numeric(out["prob_over"], errors="coerce")
    out["prob_over"].replace([np.inf, -np.inf], np.nan, inplace=True)

    out["over_actual"] = _coerce_bool01(out["over_actual"])
    out["over_actual"].replace([np.inf, -np.inf], np.nan, inplace=True)

    out = out.dropna(subset=["prob_over", "over_actual"])
    out["prob_over"] = out["prob_over"].clip(EPS, 1.0 - EPS)
    out["over_actual"] = out["over_actual"].astype(int)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default="data/props/history_props_with_outcomes.csv",
        help="Historical props file including actual outcomes.",
    )
    ap.add_argument("--market", default=None, help="Filter to a single market (optional)")
    ap.add_argument(
        "--method",
        default="auto",
        choices=["auto", "isotonic", "sigmoid"],
        help="Calibration method (auto = choose via CV per market).",
    )
    ap.add_argument(
        "--outdir",
        default="models/calibration",
        help="Where to write calibrator artifacts and summary CSV.",
    )
    ap.add_argument(
        "--min_rows",
        type=int,
        default=MIN_ROWS_PER_MARKET,
        help=f"Minimum clean rows per market (default {MIN_ROWS_PER_MARKET}).",
    )
    args = ap.parse_args()

    src = Path(args.csv)
    if not src.exists():
        raise SystemExit(f"ERROR: input CSV not found: {src}")

    df = pd.read_csv(src)
    required = {"market", "prob_over", "over_actual"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: required columns missing from {src}: {sorted(missing)}")

    if args.market:
        df = df[df["market"] == args.market].copy()

    df = _sanitize(df)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    markets = sorted(df["market"].dropna().unique().tolist())

    if not markets:
        pd.DataFrame(
            columns=[
                "market", "n", "method",
                "brier_before", "brier_after",
                "logloss_before", "logloss_after",
                "artifact", "reason",
            ]
        ).to_csv(outdir / "calibration_summary.csv", index=False)
        print("No markets found after sanitization; wrote empty calibration_summary.csv")
        return

    for mk in markets:
        grp = df[df["market"] == mk]
        n = int(len(grp))
        if n < args.min_rows:
            print(f"Skipping {mk}: only {n} clean rows (< {args.min_rows})")
            results.append({"market": mk, "n": n, "method": "", "brier_before": "", "brier_after": "", "logloss_before": "", "logloss_after": "", "artifact": "", "reason": "too_few_rows"})
            continue

        p_hat = grp["prob_over"].astype(float).to_numpy()
        y = grp["over_actual"].astype(int).to_numpy()

        # --- Option 1: SKIP single-class markets (prevents sklearn error) ---
        unique_classes = np.unique(y)
        if unique_classes.size < 2:
            print(f"Skipping {mk}: single-class outcomes (all {unique_classes[0]})")
            results.append({"market": mk, "n": n, "method": "", "brier_before": "", "brier_after": "", "logloss_before": "", "logloss_after": "", "artifact": "", "reason": "single_class"})
            continue

        # Choose/assign method
        method = args.method
        if method == "auto":
            method = cross_validated_method_choice(p_hat, y)

        # Fit & evaluate
        model = fit_calibrator(p_hat, y, method=method)
        p_cal = apply_calibrator(model, p_hat)
        rep = evaluate_calibration(p_hat, p_cal, y)

        # Save artifact
        out_path = outdir / f"{mk}_{method}.joblib"
        save_calibrator(model, out_path)

        results.append(
            {
                "market": mk,
                "n": rep.n,
                "method": method,
                "brier_before": rep.brier_before,
                "brier_after": rep.brier_after,
                "logloss_before": rep.logloss_before,
                "logloss_after": rep.logloss_after,
                "artifact": out_path.as_posix(),
                "reason": "",
            }
        )

    pd.DataFrame(results).to_csv(outdir / "calibration_summary.csv", index=False)

    trained = [r for r in results if r.get("artifact")]
    if trained:
        print(f"✓ Wrote calibrators to {outdir} for {len(trained)} market(s).")
    else:
        print(f"No calibrators trained (too few rows or single-class). Wrote summary at {outdir}/calibration_summary.csv")


if __name__ == "__main__":
    main()
