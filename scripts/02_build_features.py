#!/usr/bin/env python3
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import yaml
from utils.paths import DATA_RAW, DATA_FEATURES, ensure_dirs

def roll_mean(series, window):
    return (
        series.groupby(level=0)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

def add_roll_col(df, pid_col, col, w):
    # index by player_id for stable alignment
    s = (
        df.set_index(pid_col)[col]
        .groupby(level=0)
        .rolling(w, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df[f"{col}_l{w}"] = s.values

def clip_numeric(df, qlow=0.005, qhi=0.995):
    num_cols = df.select_dtypes(include="number").columns
    return df.assign(
        **{c: df[c].clip(df[c].quantile(qlow), df[c].quantile(qhi)) for c in num_cols}
    )

def build_wr_receiving_yards(weekly, cfg):
    wr = weekly[weekly["position"].isin(["WR", "TE"])].copy()
    wr.sort_values(["player_id", "season", "week"], inplace=True)
    for w in cfg["rolling_windows"]:
        for col in ["targets", "routes_run", "snaps", "air_yards", "receiving_yards", "receptions"]:
            if col in wr.columns:
                add_roll_col(wr, "player_id", col, w)

    wr["y_next_recv_yds"] = wr.groupby("player_id")["receiving_yards"].shift(-1)

    if "snaps" in wr.columns:
        wr = wr[wr["snaps"].fillna(0) >= cfg["guards"]["min_snaps"]]

    wr = wr.dropna(subset=["y_next_recv_yds"])
    wr = clip_numeric(wr, *cfg["guards"]["clip_quantiles"])
    feats = [c for c in wr.columns if any(c.endswith(f"_l{w}") for w in cfg["rolling_windows"])]
    out = wr[["player_id", "player_name", "season", "week"] + feats + ["y_next_recv_yds"]].copy()
    out.to_parquet(DATA_FEATURES / "receiving_yards_features.parquet", index=False)

def build_rb_rushing_yards(weekly, cfg):
    rb = weekly[weekly["position"] == "RB"].copy()
    rb.sort_values(["player_id", "season", "week"], inplace=True)
    for w in cfg["rolling_windows"]:
        for col in ["rush_attempts", "snaps", "rushing_yards", "targets"]:
            if col in rb.columns:
                add_roll_col(rb, "player_id", col, w)

    rb["y_next_rush_yds"] = rb.groupby("player_id")["rushing_yards"].shift(-1)

    if "snaps" in rb.columns:
        rb = rb[rb["snaps"].fillna(0) >= cfg["guards"]["min_snaps"]]

    rb = rb.dropna(subset=["y_next_rush_yds"])
    rb = clip_numeric(rb, *cfg["guards"]["clip_quantiles"])
    feats = [c for c in rb.columns if any(c.endswith(f"_l{w}") for w in cfg["rolling_windows"])]
    out = rb[["player_id", "player_name", "season", "week"] + feats + ["y_next_rush_yds"]].copy()
    out.to_parquet(DATA_FEATURES / "rushing_yards_features.parquet", index=False)

def build_anytime_td(weekly, cfg):
    skill = weekly[weekly["position"].isin(["RB", "WR", "TE"])].copy()
    skill.sort_values(["player_id", "season", "week"], inplace=True)
    for w in cfg["rolling_windows"]:
        for col in [
            "targets",
            "routes_run",
            "snaps",
            "rush_attempts",
            "receiving_tds",
            "rushing_tds",
            "red_zone_targets",
            "red_zone_attempts",
        ]:
            if col in skill.columns:
                add_roll_col(skill, "player_id", col, w)

    td_next = (
        skill.groupby("player_id")["receiving_tds"].shift(-1).fillna(0)
        + skill.groupby("player_id")["rushing_tds"].shift(-1).fillna(0)
    )
    skill["y_next_td"] = (td_next > 0).astype(int)

    if "snaps" in skill.columns:
        skill = skill[skill["snaps"].fillna(0) >= cfg["guards"]["min_snaps"]]

    skill = clip_numeric(skill, *cfg["guards"]["clip_quantiles"])
    feats = [c for c in skill.columns if any(c.endswith(f"_l{w}") for w in cfg["rolling_windows"])]
    out = skill[["player_id", "player_name", "season", "week"] + feats + ["y_next_td"]].copy()
    out.to_parquet(DATA_FEATURES / "anytime_td_features.parquet", index=False)

def main():
    ensure_dirs()
    cfg = yaml.safe_load(open("config/features.yml"))
    weekly = pd.read_parquet(DATA_RAW / "weekly.parquet")

    build_wr_receiving_yards(weekly, cfg)
    build_rb_rushing_yards(weekly, cfg)
    build_anytime_td(weekly, cfg)

    print("✅ Features written")

if __name__ == "__main__":
    main()
