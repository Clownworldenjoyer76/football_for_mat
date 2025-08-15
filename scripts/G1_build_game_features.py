#!/usr/bin/env python3
# Aggregate player projections -> team/game features (robust to column names)
from pathlib import Path
import pandas as pd

PRED_DIR = Path("data/predictions")
OUT_DIR  = Path("data/games"); OUT_DIR.mkdir(parents=True, exist_ok=True)

TEAM_CANDIDATES = ["team","posteam","recent_team","team_name","player_team","franchise"]
OPP_CANDIDATES  = ["opponent","opp_team","defteam","opponent_team","def_team"]

def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    return None

def _ensure_team_cols(df):
    """Return df with guaranteed 'team' and 'opponent' columns (best-effort)."""
    tcol = _pick_col(df, TEAM_CANDIDATES)
    ocol = _pick_col(df, OPP_CANDIDATES)

    # If neither, try to infer from home/away style
    if tcol is None and "home_team" in df.columns and "away_team" in df.columns and "is_home" in df.columns:
        df = df.copy()
        df["team"] = df.apply(lambda r: r["home_team"] if r.get("is_home", False) else r["away_team"], axis=1)
        # crude opponent from the other side
        df["opponent"] = df.apply(lambda r: r["away_team"] if r.get("is_home", False) else r["home_team"], axis=1)
        return df

    # If we have team and opponent under different names, rename
    df = df.copy()
    if tcol and tcol != "team":
        df = df.rename(columns={tcol: "team"})
    if ocol and ocol != "opponent":
        df = df.rename(columns={ocol: "opponent"})

    # If still missing any, create empty so groupby won’t crash but we’ll see it in print
    if "team" not in df.columns: df["team"] = pd.NA
    if "opponent" not in df.columns: df["opponent"] = pd.NA
    return df

def load_preds(market_name):
    fp = PRED_DIR / f"{market_name}.csv"
    if not fp.exists():
        raise SystemExit(f"missing predictions file: {fp}")
    df = pd.read_csv(fp)

    # normalize the target column to the market name
    ycol = "y_pred" if "y_pred" in df.columns else None
    if ycol is None:
        # try common alternates
        for c in [market_name, "pred", "prediction", "pred_mean", "mean_pred"]:
            if c in df.columns:
                ycol = c
                break
    if ycol is None:
        raise SystemExit(f"{fp} missing y_pred/pred column. Found cols: {df.columns.tolist()}")

    # ensure season/week
    need_base = []
    for k in ["season","week"]:
        if k in df.columns: need_base.append(k)

    df = _ensure_team_cols(df)

    # Debug: show columns we’ll use (first 3 rows)
    print(f"[{market_name}] cols: {df.columns.tolist()}")
    print(f"[{market_name}] sample:\n{df.head(3)}")

    cols = need_base + ["team","opponent", ycol]
    df = df[[c for c in cols if c in df.columns]].copy()
    return df.rename(columns={ycol: market_name})

def choose_qb(df_qb):
    # one QB per team: largest projection
    return (df_qb
            .dropna(subset=["team","opponent"])
            .groupby([c for c in ["season","week","team","opponent"] if c in df_qb.columns],
                     as_index=False)["qb_passing_yards"]
            .max())

def sum_rb(df_rb):
    # sum all RB rushing yards per team
    return (df_rb
            .dropna(subset=["team","opponent"])
            .groupby([c for c in ["season","week","team","opponent"] if c in df_rb.columns],
                     as_index=False)["rb_rushing_yards"]
            .sum())

def canon_pair(a, b):
    # no home/away info → deterministic order by name
    a = "" if pd.isna(a) else str(a)
    b = "" if pd.isna(b) else str(b)
    return (a, b) if a <= b else (b, a)

def main():
    qb = load_preds("qb_passing_yards")
    rb = load_preds("rb_rushing_yards")

    if not {"team","opponent"}.issubset(qb.columns):
        raise SystemExit("QB predictions missing team/opponent after normalization.")
    if not {"team","opponent"}.issubset(rb.columns):
        raise SystemExit("RB predictions missing team/opponent after normalization.")

    qb = choose_qb(qb)
    rb = sum_rb(rb)

    keys = [k for k in ["season","week","team","opponent"] if k in qb.columns and k in rb.columns]
    team_feats = pd.merge(qb, rb, on=keys, how="outer").fillna(0.0)
    team_feats["team_total_yards"] = team_feats["qb_passing_yards"] + team_feats["rb_rushing_yards"]

    # build one row per game
    rows, seen = [], set()
    # allow season/week optional (older exports may not have both)
    for _, r in team_feats.iterrows():
        s = int(r["season"]) if "season" in team_feats.columns and pd.notna(r["season"]) else pd.NA
        w = int(r["week"])   if "week"   in team_feats.columns and pd.notna(r["week"])   else pd.NA
        t, o = r["team"], r["opponent"]
        home, away = canon_pair(t, o)
        key = (s, w, home, away)
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

    feat = (games.merge(hf, on=[c for c in ["season","week","home_team","away_team"] if c in games.columns], how="left")
                 .merge(af, on=[c for c in ["season","week","home_team","away_team"] if c in games.columns], how="left")
                 .fillna(0.0))

    out_fp = OUT_DIR / "game_features.csv"
    feat.to_csv(out_fp, index=False)
    print(f"wrote {out_fp} ({len(feat)} rows)")

if __name__ == "__main__":
    main()
