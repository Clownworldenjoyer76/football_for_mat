
#!/usr/bin/env python3
"""
Train a probability calibrator for OVER outcomes given historical predictions.
Input CSV must contain at least:
  - market (str)         e.g. 'passing_yards'
  - prob_over (float)    uncalibrated probability our pipeline produced at prediction time
  - over_actual (int)    1 if actual result went over the prop line, else 0
You can point to a single market via --market, or train one per market found.
"""
import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
import pandas as pd
from scripts.lib.calibration import fit_calibrator, apply_calibrator, evaluate_calibration, cross_validated_method_choice, save_calibrator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/props/history_props_with_outcomes.csv", help="historical props file including actual outcomes")
    ap.add_argument("--market", default=None, help="filter to a single market (optional)")
    ap.add_argument("--method", default="auto", choices=["auto","isotonic","sigmoid"], help="calibration method")
    ap.add_argument("--outdir", default="models/calibration", help="where to write calibrators")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.market:
        df = df[df["market"] == args.market].copy()
    assert {"market","prob_over","over_actual"}.issubset(df.columns)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    results = []
    for mk, grp in df.groupby("market"):
        p_hat = grp["prob_over"].astype(float).values
        y = grp["over_actual"].astype(int).values
        method = args.method if args.method != "auto" else cross_validated_method_choice(p_hat, y)
        model = fit_calibrator(p_hat, y, method=method)
        p_cal = apply_calibrator(model, p_hat)
        rep = evaluate_calibration(p_hat, p_cal, y)
        rep.method = method
        # save
        out_path = outdir / f"{mk}_{method}.joblib"
        save_calibrator(model, out_path)
        results.append({"market": mk, "n": rep.n, "method": method, "brier_before": rep.brier_before, "brier_after": rep.brier_after, "logloss_before": rep.logloss_before, "logloss_after": rep.logloss_after, "artifact": out_path.as_posix()})
    pd.DataFrame(results).to_csv(Path(args.outdir) / "calibration_summary.csv", index=False)
    print("✓ Wrote calibrators to", outdir)

if __name__ == "__main__":
    main()
