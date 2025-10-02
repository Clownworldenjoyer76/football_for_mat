

# === Season Guard (auto-added) ===
# Ensures outputs are for TARGET_SEASON (default 2025). Non-invasive: runs at process exit.
try:
    import os, atexit, sys
    import pandas as _pd
    TARGET_SEASON = int(os.getenv("TARGET_SEASON", "2025"))
    _SEASON_GUARD_PATHS = ["data/props/props_current_calibrated.csv"]

    def _season_guard_check():
        for _p in _SEASON_GUARD_PATHS:
            try:
                if not _p or not isinstance(_p, str): 
                    continue
                if not os.path.exists(_p):
                    continue
                _df = _pd.read_csv(_p)
                if "season" not in _df.columns:
                    continue
                _mx = _pd.to_numeric(_df["season"], errors="coerce").max()
                if _mx != TARGET_SEASON:
                    print(f"ERROR: {_p} max(season)={int(_mx)} != {TARGET_SEASON}. Rebuild inputs for {TARGET_SEASON}.", file=sys.stderr)
                    sys.exit(1)
            except Exception as _e:
                print(f"WARNING: season guard issue for {_p}: {_e}", file=sys.stderr)

    atexit.register(_season_guard_check)
except Exception as _e:
    pass
# === End Season Guard ===


#!/usr/bin/env python3
"""
Apply a previously trained calibrator to current props.
Input CSV must contain:
  - market (str)
  - prob_over (float)
Will write:
  - prob_over_cal (calibrated)
  - prob_under_cal (1 - prob_over_cal)
"""
import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
import pandas as pd
import joblib

def load_best_calibrator(calib_dir: Path, market: str):
    # prefer isotonic if available; otherwise sigmoid
    iso = calib_dir / f"{market}_isotonic.joblib"
    sig = calib_dir / f"{market}_sigmoid.joblib"
    if iso.exists(): return joblib.load(iso)
    if sig.exists(): return joblib.load(sig)
    raise FileNotFoundError(f"No calibrator found for {market} in {calib_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_in", default="data/props/props_current.csv", help="props file with uncalibrated prob_over")
    ap.add_argument("--csv_out", default="data/props/props_current_calibrated.csv", help="output path")
    ap.add_argument("--calib_dir", default="models/calibration", help="where calibrator .joblib files are stored")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_in)
    assert {"market","prob_over"}.issubset(df.columns)

    calib_dir = Path(args.calib_dir)
    # apply per-market
    out = []
    for mk, grp in df.groupby("market"):
        model = load_best_calibrator(calib_dir, mk)
        # model is ("isotonic", estimator) or ("sigmoid", estimator)
        kind, est = model
        # Apply using the same logic as in the lib
        import numpy as np
        eps = 1e-6
        p = grp["prob_over"].astype(float).values
        if kind == "isotonic":
            p_cal = np.clip(est.predict(p), 0, 1)
        else:
            z = np.log((p + eps) / (1 - p + eps)).reshape(-1, 1)
            p_cal = est.predict_proba(z)[:, 1]
        tmp = grp.copy()
        tmp["prob_over_cal"] = p_cal
        tmp["prob_under_cal"] = 1 - p_cal
        out.append(tmp)
    pd.concat(out, axis=0, ignore_index=True).to_csv(args.csv_out, index=False)
    print("✓ Wrote calibrated props ->", args.csv_out)

if __name__ == "__main__":
    main()
