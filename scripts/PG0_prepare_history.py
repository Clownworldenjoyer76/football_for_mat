#!/usr/bin/env python3
# Build pregame rolling features (no leakage) + write summary
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("data/pregame"); OUT.mkdir(parents=True, exist_ok=True)
ROLL_G = 8  # rolling window

# Source candidates (first that exists wins)
CANDIDATES = [
    ("parquet", "data/features/weekly_all.parquet"),
    ("csv",     "data/features/weekly_all.csv.gz"),
    ("csv",     "data/features/weekly_all.csv"),
    ("parquet", "data/features/weekly.parquet"),
    ("csv",     "data/features/weekly.csv.gz"),
    ("csv",     "data/features/weekly.csv"),
]

HIST_CSV = OUT / "history_rolling_ma.csv.gz"
HIST_SUM = OUT / "_history_summary.txt"

def load_weekly():
    for kind, p in CANDIDATES:
        fp = Path(p)
        if fp.exists():
            df = pd.read_parquet(fp) if kind == "parquet" else pd.read_csv(fp, low_memory=False)
            return df, fp
    raise SystemExit(f"[PG0] No weekly file found. Tried: {[p for _,p in CANDIDATES]}")

def ensure_team_opp(df):
    df = df.copy()
    if "team" not in df.columns:
        for alt in ["posteam", "recent_team", "team_name"]:
            if alt in df.columns: df = df.rename(columns={alt: "team"}); break
    if "opponent" not in df.columns:
        for alt in ["defteam", "opp_team", "opponent_team"]:
            if alt in df.columns: df = df.rename(columns={alt: "opponent"}); break
    if "is_home" not in df.columns:
        if "home_away" in df.columns:
            df["is_home"] = (df["home_away"].astype(str).str.upper() != "AWAY").astype(int)
        else:
            df["is_home"] = 1
    return df

def to_num(df, cols):
    for c in cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

def roll_player(df, cols, w):
    use = [c for c in cols if c in df.columns]
    if not use: return pd.DataFrame(index=df.index)
    r = (df.sort_values(["player_id","season","week"])
           .groupby("player_id")[use].rolling(w, min_periods=1).mean().shift(1)
           .reset_index(level=0, drop=True))
    r.columns = [f"plyr_{c}_ma{w}" for c in r.columns]
    return r

def roll_opp_allowed(df, cols, w):
    base = [c for c in cols if c in df.columns]
    need = {"opponent","position","season","week"}
    if not base or not need.issubset(df.columns): return pd.DataFrame(index=df.index)
    r = (df.sort_values(["opponent","position","season","week"])
           .groupby(["opponent","position"])[base].rolling(w, min_periods=1).mean().shift(1)
           .reset_index(level=[0,1], drop=True))
    r.columns = [f"opp_{c}_ma{w}" for c in r.columns]
    return r

def write_summary(df_in, df_out, src_path, used_player_cols, used_opp_cols, windows):
    lines = []
    lines.append(f"source: {src_path}")
    lines.append(f"in_rows: {len(df_in)}, in_cols: {df_in.shape[1]}")
    lines.append(f"out_rows: {len(df_out)}, out_cols: {df_out.shape[1]}")
    if {"season","week"}.issubset(df_out.columns):
        sw = (df_out.groupby(["season","week"]).size().reset_index(name="rows")
              .sort_values(["season","week"]).tail(10))
        lines.append("last_10_season_week_rows:")
        for _, r in sw.iterrows():
            lines.append(f"  {int(r['season'])}-W{int(r['week'])}: {int(r['rows'])}")
    if "position" in df_out.columns:
        pos = df_out["position"].astype(str).str.upper().value_counts().to_dict()
        lines.append(f"by_position_rows: {pos}")
    lines.append("")
    lines.append(f"rolling_windows: {windows}")
    lines.append(f"player_cols_used: {used_player_cols}")
    lines.append(f"opp_cols_used: {used_opp_cols}")
    # NaN check on generated rolling cols
    roll_cols = [c for c in df_out.columns if c.startswith("plyr_") or c.startswith("opp_")]
    if roll_cols:
        na_rate = (df_out[roll_cols].isna().mean().sort_values(ascending=False).head(15)).to_dict()
        lines.append("top_missing_rates (first 15 rolling cols):")
        for k,v in na_rate.items():
            lines.append(f"  {k}: {v:.3f}")
    HIST_SUM.write_text("\n".join(lines) + "\n")

def main():
    df, src_path = load_weekly()
    df = ensure_team_opp(df)

    # keep essentials + common stat cols (auto-detect from your file)
    keep_meta = [c for c in ["season","week","player_id","player_name","position","team","opponent","is_home"] if c in df.columns]
    stat_candidates = [
        "attempts","completions","passing_yards","rushing_yards",
        "receptions","receiving_yards","targets","snap_pct","snaps"
    ]
    stat_cols = [c for c in stat_candidates if c in df.columns]
    df = df[keep_meta + stat_cols].copy()

    # types
    for c in ["season","week"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    to_num(df, stat_cols)

    # build rolling features
    windows = [ROLL_G]  # single window = 8
    pieces = [df.reset_index(drop=True)]
    for w in windows:
        pieces.append(roll_player(df, stat_cols, w))
        pieces.append(roll_opp_allowed(df, ["passing_yards","rushing_yards","receiving_yards","receptions"], w))
    out = pd.concat(pieces, axis=1)

    # write data + summary
    out.to_csv(HIST_CSV, index=False)
    write_summary(df, out, src_path, stat_cols, ["passing_yards","rushing_yards","receiving_yards","receptions"], windows)

    print(f"[PG0] wrote {HIST_CSV} rows={len(out)} cols={out.shape[1]}")
    print(f"[PG0] wrote {HIST_SUM}")

if __name__ == "__main__":
    main()
