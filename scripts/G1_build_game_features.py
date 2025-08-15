#!/usr/bin/env python3
# Aggregate player projections -> team/game features
from pathlib import Path
import pandas as pd

PRED_DIR = Path("data/predictions")
OUT_DIR  = Path("data/games"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_preds(name):
    fp = PRED_DIR / f"{name}.csv"
    if not fp.exists():
        raise SystemExit(f"missing {fp}")
    df = pd.read_csv(fp)
    need = [c for c in ["season","week","team","opponent","y_pred"] if c in df.columns]
    return df[need].rename(columns={"y_pred": name})

def choose_qb(df_qb):
    # one QB per team: take the largest projection (likely the starter)
    grp = df_qb.groupby(["season","week","team","opponent"], as_index=False)[ "qb_passing_yards" ].max()
    return grp

def sum_rb(df_rb):
    # sum all RB rushing yards per team (simple, robust)
    grp = df_rb.groupby(["season","week","team","opponent"], as_index=False)[ "rb_rushing_yards" ].sum()
    return grp

def canon_pair(a, b):
    # we don't have home/away; create deterministic home/away by name
    return (a, b) if a < b else (b, a)

def main():
    qb = load_preds("qb_passing_yards")
    rb = load_preds("rb_rushing_yards")

    qb = choose_qb(qb)
    rb = sum_rb(rb)

    team_feats = pd.merge(qb, rb, on=["season","week","team","opponent"], how="outer").fillna(0.0)
    team_feats["team_total_yards"] = team_feats["qb_passing_yards"] + team_feats["rb_rushing_yards"]

    # make one row per game (home/away assigned deterministically)
    rows = []
    seen = set()
    for _, r in team_feats.iterrows():
        s,w,t,o = int(r.season), int(r.week), r.team, r.opponent
        home, away = canon_pair(t, o)
        key = (s,w,home,away)
        if key not in seen:
            seen.add(key)
            rows.append({"season":s,"week":w,"home_team":home,"away_team":away})

    games = pd.DataFrame(rows)

    # attach features for each side
    hf = team_feats.rename(columns={
        "team":"home_team", "opponent":"away_team",
        "team_total_yards":"home_total_yards",
        "qb_passing_yards":"home_pass_yds",
        "rb_rushing_yards":"home_rush_yds",
    })
    af = team_feats.rename(columns={
        "team":"away_team", "opponent":"home_team",
        "team_total_yards":"away_total_yards",
        "qb_passing_yards":"away_pass_yds",
        "rb_rushing_yards":"away_rush_yds",
    })
    feat = games.merge(hf, on=["season","week","home_team","away_team"], how="left") \
                .merge(af, on=["season","week","home_team","away_team"], how="left") \
                .fillna(0.0)

    feat.to_csv(OUT_DIR / "game_features.csv", index=False)
    print(f"✓ wrote {OUT_DIR/'game_features.csv'} ({len(feat)} rows)")

if __name__ == "__main__":
    main()
