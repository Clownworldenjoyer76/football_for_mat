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
LOGD = Path("output"); LOGD.mkdir(parents=True, exist_ok=True)

TARGETS = {
  "qb_passing_yards": ("QB", "passing_yards"),
  "rb_rushing_yards": ("RB", "rushing_yards"),
  "wr_rec_yards":     ("WR", "receiving_yards"),
  "wrte_receptions":  ("WR", "receptions"),
}

def load_model(name):
    path = Path(f"models/pregame/{name}.joblib")
    if not path.exists():
        return None, None
    return joblib.load(path)  # returns (model, feat_cols)

def depth_chart_candidates(df, pos_tag, top_n):
    pos = df["position"].astype(str).str.upper().fillna("")
    if pos_tag == "QB":
        mask = pos.str.startswith("QB")
    elif pos_tag == "RB":
        mask = pos.str.startswith("RB")
    else:  # WR/TE combined
        mask = pos.str.match(r"WR|TE")
    d = df[mask].copy()
    snap_col = "plyr_snap_pct_ma8" if "plyr_snap_pct_ma8" in d.columns else ("snap_pct" if "snap_pct" in d.columns else None)
    if snap_col:
        d = d.sort_values(["team", snap_col], ascending=[True, False])
    return d.groupby("team", as_index=False, group_keys=False).head(top_n)

def build_fixture(hist, season, week):
    cand = hist[(hist.get("season", -1) == season) & (hist.get("week", 10) < week)].copy()
    if cand.empty:
        return cand
    # latest row per player_id
    if "player_id" not in cand.columns:
        return pd.DataFrame()
    return (cand.sort_values(["player_id", "season", "week"])
                 .groupby("player_id", as_index=False, group_keys=False)
                 .tail(1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    args = ap.parse_args()

    if not HIST.exists():
        raise SystemExit(f"missing {HIST}")
    hist = pd.read_csv(HIST, low_memory=False)

    summary = []
    total_preds = 0

    for key, (pos_tag, ycol) in TARGETS.items():
        model, model_feats = load_model(key)
        if model is None:
            summary.append(f"{key}: skip (model missing)")
            # still write empty file with header
            pd.DataFrame(columns=["season","week","player_id","player_name","team","opponent","position","y_pred"])\
              .to_csv(OUTP / f"{key}.csv", index=False)
            continue

        latest = build_fixture(hist, args.season, args.week)
        if latest.empty:
            summary.append(f"{key}: skip (no history rows for season={args.season}, week<{args.week})")
            pd.DataFrame(columns=["season","week","player_id","player_name","team","opponent","position","y_pred"])\
              .to_csv(OUTP / f"{key}.csv", index=False)
            continue

        # choose “depth chart” by snaps
        top_n = 1 if pos_tag == "QB" else (2 if pos_tag == "RB" else 3)
        cand = depth_chart_candidates(latest, pos_tag, top_n=top_n)
        if cand.empty:
            summary.append(f"{key}: skip (no candidates after depth filter)")
            pd.DataFrame(columns=["season","week","player_id","player_name","team","opponent","position","y_pred"])\
              .to_csv(OUTP / f"{key}.csv", index=False)
            continue

        # intersect model features with available columns
        avail = [c for c in model_feats if c in cand.columns]
        if not avail:
            summary.append(f"{key}: skip (no overlapping features with model)")
            pd.DataFrame(columns=["season","week","player_id","player_name","team","opponent","position","y_pred"])\
              .to_csv(OUTP / f"{key}.csv", index=False)
            continue

        X = cand[avail].fillna(0)
        if X.shape[0] == 0:
            summary.append(f"{key}: skip (empty X after filtering)")
            pd.DataFrame(columns=["season","week","player_id","player_name","team","opponent","position","y_pred"])\
              .to_csv(OUTP / f"{key}.csv", index=False)
            continue

        yhat = model.predict(X)

        out = cand.copy()
        out["y_pred"] = yhat
        out["season"] = args.season
        out["week"] = args.week
        keep = [c for c in ["season","week","player_id","player_name","team","opponent","position","y_pred"] if c in out.columns or c in ["season","week","y_pred"]]
        out = out[keep]
        out.to_csv(OUTP / f"{key}.csv", index=False)
        total_preds += len(out)

        summary.append(f"{key}: ok ({len(out)} rows, {len(avail)} features)")

    LOGD.mkdir(parents=True, exist_ok=True)
    with open(LOGD / "pregame_summary.txt", "w") as f:
        f.write(f"season={args.season} week={args.week} total_preds={total_preds}\n")
        f.write("\n".join(summary) + "\n")
    print("\n".join(summary))
    print(f"wrote {LOGD/'pregame_summary.txt'}")

if __name__ == "__main__":
    main()
