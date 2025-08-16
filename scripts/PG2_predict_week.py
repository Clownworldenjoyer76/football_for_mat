#!/usr/bin/env python3
# Predict upcoming week props using pregame models (no current-week stats)
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import argparse

HIST = Path("data/pregame/history_rolling_ma.csv.gz")
OUTD = Path("data/pregame"); OUTD.mkdir(parents=True, exist_ok=True)
OUTP = Path("data/props"); OUTP.mkdir(parents=True, exist_ok=True)

TARGETS = {
  "qb_passing_yards": ("QB", "passing_yards"),
  "rb_rushing_yards": ("RB", "rushing_yards"),
  "wr_rec_yards":     ("WR", "receiving_yards"),
  "wrte_receptions":  ("WR", "receptions"),
}

def depth_chart_candidates(df, pos_tag, top_n=1 if "QB" else 3):
    pos = pos_tag
    if pos_tag == "WR":  # WR/TE combined output; include both, pick by snaps
        mask = df["position"].str.upper().str.match(r"WR|TE")
    else:
        mask = df["position"].str.upper().str.startswith(pos)
    d = df[mask].copy()
    # recent usage = last game's snap_pct (already lagged mean available; use plyr_snap_pct_ma8 as stable signal)
    snap_col = "plyr_snap_pct_ma8" if "plyr_snap_pct_ma8" in d.columns else ("snap_pct" if "snap_pct" in d.columns else None)
    if snap_col:
        d = d.sort_values(["team", snap_col], ascending=[True, False])
    # pick top by team
    take = d.groupby("team").head(top_n).copy()
    return take

def build_fixture(df, season, week):
    # upcoming opponents: use the latest known opponent mapping per team from the rolling dataset for that season/week-1
    # If you have a schedules file, join it here instead. Otherwise, keep placeholders.
    # Here we assume df already contains "team" and "opponent" from the last played week; we’ll keep opponent for feature context.
    cand = df[(df["season"] == season) & (df["week"] < week)].copy()
    # take last row per player (latest lag features)
    latest = cand.sort_values(["player_id","season","week"]).groupby("player_id").tail(1)
    return latest

def load_model(name):
    path = Path(f"models/pregame/{name}.joblib")
    if not path.exists():
        return None, None
    model, feats = joblib.load(path)
    return model, feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    args = ap.parse_args()

    if not HIST.exists():
        raise SystemExit(f"missing {HIST}")
    hist = pd.read_csv(HIST, low_memory=False)

    preds_written = 0
    for key, (pos_tag, ycol) in TARGETS.items():
        model, feats = load_model(key)
        if model is None:
            print(f"skip {key}: model missing"); 
            continue

        latest = build_fixture(hist, args.season, args.week)
        # choose “depth chart” by snaps
        top_n = 1 if pos_tag == "QB" else (3 if pos_tag == "WR" else 2)  # QB1, WR/TE top-3, RB top-2
        cand = depth_chart_candidates(latest, pos_tag, top_n=top_n)

        X = cand[feats].fillna(0)
        yhat = model.predict(X)

        out = cand.copy()
        out["y_pred"] = yhat
        out["season"] = args.season
        out["week"] = args.week
        out["market"] = key
        keep = [c for c in ["season","week","player_id","player_name","team","opponent","position","y_pred"] if c in out.columns]
        out = out[keep]
        fp = OUTP / f"{key}.csv"
        out.to_csv(fp, index=False)
        print(f"✓ wrote {fp} ({len(out)} rows)")
        preds_written += len(out)

    # quick summary
    Path("output").mkdir(parents=True, exist_ok=True)
    with open("output/pregame_summary.txt","w") as f:
        f.write(f"season={args.season} week={args.week} preds={preds_written}\n")

if __name__ == "__main__":
    main()
