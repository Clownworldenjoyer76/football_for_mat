#!/usr/bin/env python
"""
NFL all-4 historical testing + pipeline feature builder.

Uses:
  1) ready_drat
  2) ready_epred
  3) ready_odds
  4) ready_schedule

Important:
- Source files are READ ONLY.
- EPRED projected points, projected totals, and PtDiff are NOT used.
- Final scores are used only to create the historical winner target.
- All outputs are written to the testing folder.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "This script requires pandas and numpy.\n"
        "Install them with: python -m pip install pandas numpy"
    ) from exc


DEFAULT_DRAT = r"C:\Users\ntmal\Downloads\mat\football_for_mat\docs\win\football\pred\ready\ready_drat"
DEFAULT_EPRED = r"C:\Users\ntmal\Downloads\mat\football_for_mat\docs\win\football\pred\ready\ready_epred"
DEFAULT_ODDS = r"C:\Users\ntmal\Downloads\mat\football_for_mat\docs\win\football\pred\ready\ready_odds"
DEFAULT_SCHEDULE = r"C:\Users\ntmal\Downloads\mat\football_for_mat\docs\win\football\pred\ready\ready_schedule"
DEFAULT_OUT = r"C:\Users\ntmal\Downloads\mat\football_for_mat\docs\win\football\pred\testing"

# Conservative thresholds for "actionable" output.
MIN_GAMES = 50
MIN_SEASONS = 3
MIN_ABS_LIFT_PP = 3.0
MIN_SEASON_CONSISTENCY = 2 / 3
MIN_FORWARD_CHECKS = 2
MIN_FORWARD_SUCCESS = 2 / 3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--drat", default=DEFAULT_DRAT)
    p.add_argument("--epred", default=DEFAULT_EPRED)
    p.add_argument("--odds", default=DEFAULT_ODDS)
    p.add_argument("--schedule", default=DEFAULT_SCHEDULE)
    p.add_argument("--out", default=DEFAULT_OUT)
    return p.parse_args()


def read_folder(folder: Path, source_name: str) -> pd.DataFrame:
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig", dtype=str, low_memory=False)
        df["__source_file"] = f.name
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    if "game_id" not in result.columns:
        raise ValueError(f"{source_name}: game_id column not found")

    result["game_id"] = result["game_id"].astype(str).str.strip()
    result = result[result["game_id"].ne("") & result["game_id"].notna()].copy()
    return result


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def clean_team(s):
    return s.astype("string").str.strip()


def moneyline_to_num(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.upper()
    s = s.replace({"EV": "100", "EVEN": "100"})
    extracted = s.str.extract(r"^([+-]?\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def spread_to_num(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    s = (
        s.str.replace("½", ".5", regex=False)
         .str.replace("¼", ".25", regex=False)
         .str.replace("¾", ".75", regex=False)
         .str.replace("−", "-", regex=False)
    )
    u = s.str.upper()
    s = s.mask(u.isin(["PK", "PICK", "PICKEM", "PICK'EM"]), "0")
    extracted = s.str.extract(r"^([+-]?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def implied_prob(ml: pd.Series) -> pd.Series:
    x = pd.to_numeric(ml, errors="coerce")
    out = pd.Series(np.nan, index=x.index, dtype=float)
    pos = x > 0
    neg = x < 0
    out.loc[pos] = 100.0 / (x.loc[pos] + 100.0)
    out.loc[neg] = (-x.loc[neg]) / ((-x.loc[neg]) + 100.0)
    return out


def bucket_numeric(series, bins, labels):
    return pd.cut(
        series,
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    ).astype("string").fillna("Unknown")


def prob_bucket(p):
    return bucket_numeric(
        p,
        [-np.inf, .55, .60, .65, .70, .75, .80, .85, .90, np.inf],
        ["<55%", "55-59.9%", "60-64.9%", "65-69.9%", "70-74.9%",
         "75-79.9%", "80-84.9%", "85-89.9%", "90%+"],
    )


def rating_bucket(x):
    return bucket_numeric(
        x,
        [-np.inf, -6, -3, 0, 3, 6, 10, np.inf],
        ["<-6", "-6 to -3", "-3 to 0", "0 to 2.9", "3 to 5.9", "6 to 9.9", "10+"],
    )


def model_diff_bucket(x):
    return bucket_numeric(
        x,
        [-np.inf, 3, 6, 10, 15, np.inf],
        ["<3 pts", "3-5.9 pts", "6-9.9 pts", "10-14.9 pts", "15+ pts"],
    )


def edge_bucket(x):
    return bucket_numeric(
        x,
        [-np.inf, -10, -5, -2, 2, 5, 10, np.inf],
        ["<-10 pts", "-10 to -5 pts", "-5 to -2 pts", "-2 to +2 pts",
         "+2 to +5 pts", "+5 to +10 pts", "+10 pts+"],
    )


def spread_bucket(x):
    return bucket_numeric(
        x,
        [-np.inf, -10, -7, -4, -3, 0, .000001, 3, 4, 7, 10, np.inf],
        ["Fav -10+", "Fav -7 to -9.5", "Fav -4 to -6.5", "Fav -3 to -3.5",
         "Fav -0.5 to -2.5", "Pickem", "Dog +0.5 to +2.5", "Dog +3 to +3.5",
         "Dog +4 to +6.5", "Dog +7 to +9.5", "Dog +10+"],
    )


def matchup_bucket(x):
    return bucket_numeric(x, [-np.inf, 25, 50, 75, np.inf],
                          ["<25", "25-49.9", "50-74.9", "75+"])


def week_bucket(x):
    return bucket_numeric(x, [-np.inf, 4, 9, 14, np.inf],
                          ["Weeks 1-3", "Weeks 4-8", "Weeks 9-13", "Weeks 14+"])


def wind_bucket(x):
    return bucket_numeric(x, [-np.inf, 10, 15, 20, np.inf],
                          ["<10 mph", "10-14.9 mph", "15-19.9 mph", "20+ mph"])


def temp_bucket(x):
    return bucket_numeric(x, [-np.inf, 32, 50, 70, np.inf],
                          ["<32F", "32-49F", "50-69F", "70F+"])


def precip_bucket(x):
    p = x.astype(float).copy()
    mask = p.notna() & (p <= 1)
    p.loc[mask] = p.loc[mask] * 100
    return bucket_numeric(p, [-np.inf, 10, 30, 60, np.inf],
                          ["<10%", "10-29%", "30-59%", "60%+"])


def total_bucket(x):
    return bucket_numeric(x, [-np.inf, 40, 44, 48, 52, np.inf],
                          ["<40", "40-43.5", "44-47.5", "48-51.5", "52+"])


def canonicalize_source(df: pd.DataFrame, schedule: pd.DataFrame, source: str):
    """
    Align a source to schedule home/away orientation using team names.
    Returns aligned dataframe and issue dataframe.
    """
    need = ["game_id", "home_team", "away_team"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")

    tmp = df.merge(
        schedule[["game_id", "home_team", "away_team"]].rename(
            columns={"home_team": "__sched_home", "away_team": "__sched_away"}
        ),
        on="game_id",
        how="inner",
    )

    tmp["home_team"] = clean_team(tmp["home_team"])
    tmp["away_team"] = clean_team(tmp["away_team"])
    tmp["__sched_home"] = clean_team(tmp["__sched_home"])
    tmp["__sched_away"] = clean_team(tmp["__sched_away"])

    same = (
        tmp["home_team"].eq(tmp["__sched_home"])
        & tmp["away_team"].eq(tmp["__sched_away"])
    )
    rev = (
        tmp["home_team"].eq(tmp["__sched_away"])
        & tmp["away_team"].eq(tmp["__sched_home"])
    )

    issues = tmp.loc[~(same | rev), [
        "game_id", "home_team", "away_team", "__sched_home", "__sched_away", "__source_file"
    ]].copy()
    issues["source"] = source
    issues["issue"] = "team orientation mismatch"

    tmp = tmp.loc[same | rev].copy()
    tmp["__reversed"] = rev.loc[tmp.index]
    return tmp, issues


def dedupe_by_game_id(df: pd.DataFrame, source: str):
    dup = df[df.duplicated("game_id", keep=False)].copy()
    issues = pd.DataFrame()
    if not dup.empty:
        issues = dup[["game_id", "__source_file"]].copy()
        issues["source"] = source
        issues["issue"] = "duplicate game_id; last row retained"
    df = df.drop_duplicates("game_id", keep="last").copy()
    return df, issues


def build_features(drat, epred, odds, schedule):
    # Required schedule columns
    for c in ["game_id", "home_team", "away_team", "season", "season_type", "week"]:
        if c not in schedule.columns:
            raise ValueError(f"schedule missing required column: {c}")

    schedule = schedule.copy()
    schedule["home_team"] = clean_team(schedule["home_team"])
    schedule["away_team"] = clean_team(schedule["away_team"])
    schedule["season"] = to_num(schedule["season"])
    schedule["week"] = to_num(schedule["week"])

    schedule, sched_dup = dedupe_by_game_id(schedule, "schedule")

    drat, d_mismatch = canonicalize_source(drat, schedule, "drat")
    epred, e_mismatch = canonicalize_source(epred, schedule, "epred")
    odds, o_mismatch = canonicalize_source(odds, schedule, "odds")

    drat, d_dup = dedupe_by_game_id(drat, "drat")
    epred, e_dup = dedupe_by_game_id(epred, "epred")
    odds, o_dup = dedupe_by_game_id(odds, "odds")

    issues = pd.concat(
        [x for x in [sched_dup, d_mismatch, e_mismatch, o_mismatch, d_dup, e_dup, o_dup]
         if x is not None and not x.empty],
        ignore_index=True,
    ) if any(x is not None and not x.empty for x in
             [sched_dup, d_mismatch, e_mismatch, o_mismatch, d_dup, e_dup, o_dup]) else pd.DataFrame()

    # -------- DRAT aligned fields --------
    rev = drat["__reversed"]

    d_home_prob = np.where(rev, to_num(drat["away_prob"]), to_num(drat["home_prob"]))
    d_away_prob = np.where(rev, to_num(drat["home_prob"]), to_num(drat["away_prob"]))

    d_home_ml = np.where(
        rev,
        moneyline_to_num(drat["away_moneyline"]),
        moneyline_to_num(drat["home_moneyline"]),
    )
    d_away_ml = np.where(
        rev,
        moneyline_to_num(drat["home_moneyline"]),
        moneyline_to_num(drat["away_moneyline"]),
    )

    d_aligned = pd.DataFrame({
        "game_id": drat["game_id"].values,
        "drat_home_prob": d_home_prob,
        "drat_away_prob": d_away_prob,
        "home_moneyline": d_home_ml,
        "away_moneyline": d_away_ml,
    })

    # -------- EPRED aligned fields --------
    rev = epred["__reversed"]

    e_home_raw = np.where(rev, to_num(epred["away_prob"]), to_num(epred["home_prob"]))
    e_away_raw = np.where(rev, to_num(epred["home_prob"]), to_num(epred["away_prob"]))

    non_tie_sum = e_home_raw + e_away_raw
    e_home_prob = np.where(non_tie_sum > 0, e_home_raw / non_tie_sum, np.nan)
    e_away_prob = np.where(non_tie_sum > 0, e_away_raw / non_tie_sum, np.nan)

    e_home_rating = np.where(rev, to_num(epred["away_rating"]), to_num(epred["home_rating"]))
    e_away_rating = np.where(rev, to_num(epred["home_rating"]), to_num(epred["away_rating"]))

    e_aligned = pd.DataFrame({
        "game_id": epred["game_id"].values,
        "epred_home_prob": e_home_prob,
        "epred_away_prob": e_away_prob,
        "epred_home_rating": e_home_rating,
        "epred_away_rating": e_away_rating,
        "matchupQuality": to_num(epred["matchupQuality"]).values
        if "matchupQuality" in epred.columns else np.nan,
    })

    # -------- ODDS aligned fields --------
    rev = odds["__reversed"]

    o_home_score = np.where(rev, to_num(odds["away_score"]), to_num(odds["home_score"]))
    o_away_score = np.where(rev, to_num(odds["home_score"]), to_num(odds["away_score"]))

    o_home_spread = np.where(
        rev, spread_to_num(odds["away_spread"]), spread_to_num(odds["home_spread"])
    )
    o_away_spread = np.where(
        rev, spread_to_num(odds["home_spread"]), spread_to_num(odds["away_spread"])
    )

    def optional_num(col):
        return to_num(odds[col]).values if col in odds.columns else np.nan

    def optional_text(col):
        return odds[col].astype("string").values if col in odds.columns else ""

    o_aligned = pd.DataFrame({
        "game_id": odds["game_id"].values,
        "__home_score": o_home_score,
        "__away_score": o_away_score,
        "odds_home_spread": o_home_spread,
        "odds_away_spread": o_away_spread,
        "odds_total": optional_num("odds_total"),
        "surface": optional_text("surface"),
        "weather_icon": optional_text("weather_icon"),
        "temperature": optional_num("temperature"),
        "precip_probability": optional_num("precip_probability"),
        "precip_type": optional_text("precip_type"),
        "wind_speed": optional_num("wind_speed"),
        "wind_bearing": optional_num("wind_bearing"),
    })

    # -------- Join all four --------
    base = schedule[
        ["game_id", "home_team", "away_team", "season", "season_type", "week"]
    ].copy()

    f = (
        base.merge(d_aligned, on="game_id", how="inner")
            .merge(e_aligned, on="game_id", how="inner")
            .merge(o_aligned, on="game_id", how="inner")
    )

    # Only games with needed model probabilities and final scores.
    required = [
        "drat_home_prob", "drat_away_prob",
        "epred_home_prob", "epred_away_prob",
        "__home_score", "__away_score",
    ]
    f = f.dropna(subset=required).copy()

    # Historical target only.
    f["winner_side"] = np.where(
        f["__home_score"] > f["__away_score"], "Home",
        np.where(f["__away_score"] > f["__home_score"], "Away", "Tie")
    )
    f["home_won"] = np.where(
        f["winner_side"].eq("Home"), 1,
        np.where(f["winner_side"].eq("Away"), 0, np.nan)
    )

    # No-vig market probability from DRAT moneylines.
    ih = implied_prob(f["home_moneyline"])
    ia = implied_prob(f["away_moneyline"])
    isum = ih + ia
    f["market_home_prob_novig"] = np.where(isum > 0, ih / isum, np.nan)
    f["market_away_prob_novig"] = np.where(isum > 0, ia / isum, np.nan)

    f["drat_pick"] = np.where(f["drat_home_prob"] >= f["drat_away_prob"], "Home", "Away")
    f["epred_pick"] = np.where(f["epred_home_prob"] >= f["epred_away_prob"], "Home", "Away")

    market_by_prob = np.where(
        f["market_home_prob_novig"].notna(),
        np.where(f["market_home_prob_novig"] >= .5, "Home", "Away"),
        None,
    )
    market_by_spread = np.where(
        f["odds_home_spread"].notna(),
        np.where(f["odds_home_spread"] < 0, "Home",
                 np.where(f["odds_home_spread"] > 0, "Away", "Even")),
        "Even",
    )
    f["market_pick"] = np.where(pd.notna(market_by_prob), market_by_prob, market_by_spread)

    f["drat_pick_prob"] = np.where(
        f["drat_pick"].eq("Home"), f["drat_home_prob"], f["drat_away_prob"]
    )
    f["epred_pick_prob"] = np.where(
        f["epred_pick"].eq("Home"), f["epred_home_prob"], f["epred_away_prob"]
    )
    f["market_pick_prob"] = np.where(
        f["market_pick"].eq("Home"), f["market_home_prob_novig"],
        np.where(f["market_pick"].eq("Away"), f["market_away_prob_novig"], np.nan)
    )

    f["epred_rating_diff_home"] = f["epred_home_rating"] - f["epred_away_rating"]

    f["drat_epred_agree"] = np.where(f["drat_pick"].eq(f["epred_pick"]), "Agree", "Disagree")
    f["drat_market_agree"] = np.where(
        f["market_pick"].isin(["Home", "Away"]),
        np.where(f["drat_pick"].eq(f["market_pick"]), "Agree", "Disagree"),
        "Unknown",
    )
    f["epred_market_agree"] = np.where(
        f["market_pick"].isin(["Home", "Away"]),
        np.where(f["epred_pick"].eq(f["market_pick"]), "Agree", "Disagree"),
        "Unknown",
    )
    f["all_three_agree"] = np.where(
        f["market_pick"].isin(["Home", "Away"])
        & f["drat_pick"].eq(f["epred_pick"])
        & f["drat_pick"].eq(f["market_pick"]),
        "Yes", "No"
    )

    f["drat_epred_home_diff_pp"] = (
        (f["drat_home_prob"] - f["epred_home_prob"]).abs() * 100
    )
    f["drat_epred_mean_home_prob"] = (
        f["drat_home_prob"] + f["epred_home_prob"]
    ) / 2
    f["all3_mean_home_prob"] = (
        f["drat_home_prob"] + f["epred_home_prob"] + f["market_home_prob_novig"]
    ) / 3

    f["drat_market_edge_home_pp"] = (
        f["drat_home_prob"] - f["market_home_prob_novig"]
    ) * 100
    f["epred_market_edge_home_pp"] = (
        f["epred_home_prob"] - f["market_home_prob_novig"]
    ) * 100

    f["week_bucket"] = week_bucket(f["week"])
    f["matchup_bucket"] = matchup_bucket(f["matchupQuality"])
    f["model_diff_bucket"] = model_diff_bucket(f["drat_epred_home_diff_pp"])
    f["odds_total_bucket"] = total_bucket(f["odds_total"])
    f["surface_bucket"] = np.where(
        f["surface"].astype("string").str.contains("dome|indoor", case=False, na=False),
        "Dome/Indoor",
        np.where(f["surface"].astype("string").str.strip().eq(""), "Unknown", "Outdoor"),
    )
    f["wind_bucket"] = wind_bucket(f["wind_speed"])
    f["temperature_bucket"] = temp_bucket(f["temperature"])
    f["precip_bucket"] = precip_bucket(f["precip_probability"])

    # Remove actual scores from output matrix; they were only used for target creation.
    f = f.drop(columns=["__home_score", "__away_score"])

    return f, issues


def selected_value(df, side_col, home_col, away_col):
    return np.where(df[side_col].eq("Home"), df[home_col],
                    np.where(df[side_col].eq("Away"), df[away_col], np.nan))


def build_selection_rows(f: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add_family(family, side, prob, mask=None):
        if mask is None:
            mask = pd.Series(True, index=f.index)
        sub = f.loc[mask].copy()
        if sub.empty:
            return

        sub["Family"] = family
        sub["SelectedSide"] = side.loc[sub.index]
        sub["SelectedProb"] = prob.loc[sub.index]
        sub = sub[sub["SelectedSide"].isin(["Home", "Away"]) & sub["winner_side"].ne("Tie")].copy()
        if sub.empty:
            return

        sub["Won"] = (sub["SelectedSide"] == sub["winner_side"]).astype(int)
        sub["SelectedTeam"] = np.where(
            sub["SelectedSide"].eq("Home"), sub["home_team"], sub["away_team"]
        )
        sub["MarketProbSelected"] = selected_value(
            sub, "SelectedSide", "market_home_prob_novig", "market_away_prob_novig"
        )
        sub["SelectedSpread"] = selected_value(
            sub, "SelectedSide", "odds_home_spread", "odds_away_spread"
        )
        sub["RatingGapSelected"] = np.where(
            sub["SelectedSide"].eq("Home"),
            sub["epred_rating_diff_home"],
            -sub["epred_rating_diff_home"],
        )

        sub["MarketRole"] = np.where(
            sub["MarketProbSelected"].notna(),
            np.where(sub["MarketProbSelected"] > .5, "Market Favorite",
                     np.where(sub["MarketProbSelected"] < .5, "Market Underdog", "Market Even")),
            np.where(sub["SelectedSpread"] < 0, "Market Favorite",
                     np.where(sub["SelectedSpread"] > 0, "Market Underdog", "Market Even"))
        )

        sub["SelectedVsDRAT"] = np.where(sub["SelectedSide"].eq(sub["drat_pick"]), "Agree", "Disagree")
        sub["SelectedVsEPRED"] = np.where(sub["SelectedSide"].eq(sub["epred_pick"]), "Agree", "Disagree")
        sub["SelectedVsMarket"] = np.where(
            sub["market_pick"].isin(["Home", "Away"]),
            np.where(sub["SelectedSide"].eq(sub["market_pick"]), "Agree", "Disagree"),
            "Unknown",
        )

        sub["ModelMarketEdgePP"] = (
            sub["SelectedProb"] - sub["MarketProbSelected"]
        ) * 100

        sub["ProbBucket"] = prob_bucket(sub["SelectedProb"])
        sub["SpreadBucket"] = spread_bucket(sub["SelectedSpread"])
        sub["RatingGapBucket"] = rating_bucket(sub["RatingGapSelected"])
        sub["ModelMarketEdgeBucket"] = edge_bucket(sub["ModelMarketEdgePP"])

        keep = [
            "Family", "game_id", "season", "week", "season_type",
            "SelectedSide", "SelectedTeam", "Won", "SelectedProb",
            "ProbBucket", "MarketRole", "SpreadBucket", "RatingGapBucket",
            "matchup_bucket", "week_bucket", "surface_bucket", "wind_bucket",
            "temperature_bucket", "precip_bucket", "odds_total_bucket",
            "drat_epred_agree", "SelectedVsDRAT", "SelectedVsEPRED",
            "SelectedVsMarket", "model_diff_bucket", "ModelMarketEdgeBucket",
        ]
        rows.append(sub[keep].rename(columns={
            "matchup_bucket": "MatchupBucket",
            "week_bucket": "WeekBucket",
            "surface_bucket": "SurfaceBucket",
            "wind_bucket": "WindBucket",
            "temperature_bucket": "TemperatureBucket",
            "precip_bucket": "PrecipBucket",
            "odds_total_bucket": "OddsTotalBucket",
            "drat_epred_agree": "DratEpredAgreement",
            "model_diff_bucket": "ModelDiffBucket",
        }))

    add_family("DRAT", f["drat_pick"], f["drat_pick_prob"])
    add_family("EPRED", f["epred_pick"], f["epred_pick_prob"])

    market_mask = f["market_pick"].isin(["Home", "Away"])
    add_family("MARKET", f["market_pick"], f["market_pick_prob"], market_mask)

    de_mask = f["drat_pick"].eq(f["epred_pick"])
    de_side = f["drat_pick"]
    de_prob = np.where(
        de_side.eq("Home"),
        (f["drat_home_prob"] + f["epred_home_prob"]) / 2,
        (f["drat_away_prob"] + f["epred_away_prob"]) / 2,
    )
    add_family("DRAT_EPRED_CONSENSUS", de_side, pd.Series(de_prob, index=f.index), de_mask)

    a3_mask = f["all_three_agree"].eq("Yes")
    a3_side = f["drat_pick"]
    a3_prob = np.where(
        a3_side.eq("Home"),
        (f["drat_home_prob"] + f["epred_home_prob"] + f["market_home_prob_novig"]) / 3,
        (f["drat_away_prob"] + f["epred_away_prob"] + f["market_away_prob_novig"]) / 3,
    )
    add_family("ALL3_CONSENSUS", a3_side, pd.Series(a3_prob, index=f.index), a3_mask)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def wilson_interval(wins, n, z=1.96):
    if n <= 0:
        return np.nan, np.nan
    p = wins / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt((p*(1-p)/n) + (z*z/(4*n*n))) / den
    return center * 100, half * 100


def test_condition(family_df, condition_df, family_baseline):
    n = len(condition_df)
    wins = int(condition_df["Won"].sum())
    pct = 100 * wins / n
    lift = pct - family_baseline

    low_center, half = wilson_interval(wins, n)
    ci_low = low_center - half
    ci_high = low_center + half

    # Season consistency vs same family's seasonal baseline.
    fam_season = family_df.groupby("season")["Won"].agg(["count", "sum"])
    fam_season["pct"] = 100 * fam_season["sum"] / fam_season["count"]

    grp_season = condition_df.groupby("season")["Won"].agg(["count", "sum"])
    grp_season["pct"] = 100 * grp_season["sum"] / grp_season["count"]

    joined = grp_season.join(fam_season[["pct"]], rsuffix="_fam")
    joined = joined[joined["count"] >= 5].copy()
    joined["lift"] = joined["pct"] - joined["pct_fam"]

    eligible_seasons = len(joined)
    if eligible_seasons:
        same_dir = (
            ((lift > 0) & (joined["lift"] > 0))
            | ((lift < 0) & (joined["lift"] < 0))
        ).sum()
        season_consistency = same_dir / eligible_seasons
    else:
        season_consistency = np.nan

    # Rolling walk-forward direction check using season aggregates only.
    seasons = sorted(fam_season.index.tolist())
    prior_grp_n = prior_grp_w = prior_fam_n = prior_fam_w = 0
    forward_checks = forward_successes = 0

    for season in seasons:
        fg = fam_season.loc[season]
        gg = grp_season.loc[season] if season in grp_season.index else None

        if (
            prior_grp_n >= 20
            and gg is not None
            and gg["count"] >= 5
            and prior_fam_n > 0
        ):
            prior_lift = 100 * prior_grp_w / prior_grp_n - 100 * prior_fam_w / prior_fam_n
            current_lift = 100 * gg["sum"] / gg["count"] - 100 * fg["sum"] / fg["count"]

            if abs(prior_lift) > 1e-12 and abs(current_lift) > 1e-12:
                forward_checks += 1
                if math.copysign(1, prior_lift) == math.copysign(1, current_lift):
                    forward_successes += 1

        if gg is not None:
            prior_grp_n += int(gg["count"])
            prior_grp_w += int(gg["sum"])
        prior_fam_n += int(fg["count"])
        prior_fam_w += int(fg["sum"])

    forward_pct = (
        forward_successes / forward_checks if forward_checks else np.nan
    )

    return {
        "Games": n,
        "Wins": wins,
        "Losses": n - wins,
        "Win_Pct": round(pct, 2),
        "Wilson95_Low": round(ci_low, 2),
        "Wilson95_High": round(ci_high, 2),
        "Family_Baseline_WinPct": round(family_baseline, 2),
        "Lift_vs_Family": round(lift, 2),
        "Eligible_Seasons": eligible_seasons,
        "Season_Consistency_Pct": (
            round(season_consistency * 100, 2)
            if pd.notna(season_consistency) else np.nan
        ),
        "Forward_Checks": forward_checks,
        "Forward_Successes": forward_successes,
        "Forward_Success_Pct": (
            round(forward_pct * 100, 2) if pd.notna(forward_pct) else np.nan
        ),
    }


def run_tests(sel: pd.DataFrame):
    single_features = [
        "ProbBucket",
        "SelectedSide",
        "MarketRole",
        "SpreadBucket",
        "RatingGapBucket",
        "MatchupBucket",
        "WeekBucket",
        "SurfaceBucket",
        "WindBucket",
        "TemperatureBucket",
        "PrecipBucket",
        "OddsTotalBucket",
        "DratEpredAgreement",
        "SelectedVsMarket",
        "ModelDiffBucket",
        "ModelMarketEdgeBucket",
    ]

    # Deliberate interactions only; no brute-force feature-pair explosion.
    interactions = [
        ("ProbBucket", "RatingGapBucket"),
        ("ProbBucket", "MarketRole"),
        ("ProbBucket", "SpreadBucket"),
        ("RatingGapBucket", "MarketRole"),
        ("RatingGapBucket", "SpreadBucket"),
        ("DratEpredAgreement", "MarketRole"),
        ("DratEpredAgreement", "ModelDiffBucket"),
        ("SelectedVsMarket", "ProbBucket"),
        ("SelectedVsMarket", "RatingGapBucket"),
        ("SelectedVsMarket", "ModelMarketEdgeBucket"),
        ("WeekBucket", "ProbBucket"),
        ("MarketRole", "OddsTotalBucket"),
        ("SurfaceBucket", "WindBucket"),
        ("TemperatureBucket", "WindBucket"),
    ]

    result_rows = []
    baselines = []

    for family, fam in sel.groupby("Family", sort=False):
        fam = fam.copy()
        baseline = 100 * fam["Won"].mean()
        baselines.append({
            "Family": family,
            "Games": len(fam),
            "Wins": int(fam["Won"].sum()),
            "Losses": int(len(fam) - fam["Won"].sum()),
            "Win_Pct": round(baseline, 2),
        })

        for col in single_features:
            for value, grp in fam.groupby(col, dropna=False):
                stats = test_condition(fam, grp, baseline)
                result_rows.append({
                    "Family": family,
                    "Test": col,
                    "Condition": f"{col}={value}",
                    **stats,
                })

        for a, b in interactions:
            for keys, grp in fam.groupby([a, b], dropna=False):
                stats = test_condition(fam, grp, baseline)
                result_rows.append({
                    "Family": family,
                    "Test": f"{a} + {b}",
                    "Condition": f"{a}={keys[0]} | {b}={keys[1]}",
                    **stats,
                })

    results = pd.DataFrame(result_rows)
    baselines = pd.DataFrame(baselines).sort_values("Win_Pct", ascending=False)

    actionable = results[
        (results["Games"] >= MIN_GAMES)
        & (results["Eligible_Seasons"] >= MIN_SEASONS)
        & (results["Lift_vs_Family"].abs() >= MIN_ABS_LIFT_PP)
        & (results["Season_Consistency_Pct"] >= MIN_SEASON_CONSISTENCY * 100)
        & (results["Forward_Checks"] >= MIN_FORWARD_CHECKS)
        & (results["Forward_Success_Pct"] >= MIN_FORWARD_SUCCESS * 100)
    ].copy()

    actionable["Recommended_Use"] = np.where(
        actionable["Lift_vs_Family"] > 0,
        "Increase reliability / confidence",
        "Decrease reliability / confidence",
    )

    actionable = actionable.sort_values(
        ["Lift_vs_Family", "Games"],
        key=lambda s: s.abs() if s.name == "Lift_vs_Family" else s,
        ascending=[False, False],
    )

    return baselines, results, actionable


def probability_benchmarks(f: pd.DataFrame):
    specs = {
        "DRAT": "drat_home_prob",
        "EPRED": "epred_home_prob",
        "MARKET_NO_VIG": "market_home_prob_novig",
        "DRAT_EPRED_MEAN": "drat_epred_mean_home_prob",
        "ALL3_MEAN": "all3_mean_home_prob",
    }

    rows = []
    for name, col in specs.items():
        sub = f[["home_won", col]].dropna()
        if sub.empty:
            continue
        y = sub["home_won"].astype(float)
        p = sub[col].astype(float)
        pred = (p >= .5).astype(int)
        rows.append({
            "Model": name,
            "Games": len(sub),
            "Accuracy_Pct": round(100 * (pred == y).mean(), 2),
            "Brier": round(float(((p - y) ** 2).mean()), 5),
        })

    return pd.DataFrame(rows).sort_values(["Brier", "Accuracy_Pct"], ascending=[True, False])


def build_implementation(actionable: pd.DataFrame):
    if actionable.empty:
        return pd.DataFrame(columns=[
            "Family", "Feature_Test", "Historical_Condition", "Direction",
            "Historical_Win_Pct", "Lift_Points", "Games", "Forward_Record",
            "Pipeline_Action",
        ])

    out = actionable.copy()
    out["Direction"] = np.where(out["Lift_vs_Family"] > 0, "Positive", "Negative")
    out["Forward_Record"] = (
        out["Forward_Successes"].astype(str) + "/" + out["Forward_Checks"].astype(str)
    )
    out["Pipeline_Action"] = np.where(
        out["Lift_vs_Family"] > 0,
        "Add positive reliability adjustment when condition is present",
        "Add negative reliability adjustment when condition is present",
    )
    return out[[
        "Family", "Test", "Condition", "Direction", "Win_Pct", "Lift_vs_Family",
        "Games", "Season_Consistency_Pct", "Forward_Record", "Pipeline_Action",
    ]].rename(columns={
        "Test": "Feature_Test",
        "Condition": "Historical_Condition",
        "Win_Pct": "Historical_Win_Pct",
        "Lift_vs_Family": "Lift_Points",
    })


def main():
    args = parse_args()

    drat_dir = Path(args.drat)
    epred_dir = Path(args.epred)
    odds_dir = Path(args.odds)
    schedule_dir = Path(args.schedule)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading CSV files...", flush=True)
    drat = read_folder(drat_dir, "drat")
    epred = read_folder(epred_dir, "epred")
    odds = read_folder(odds_dir, "odds")
    schedule = read_folder(schedule_dir, "schedule")

    print("[2/6] Joining all four datasets and building pregame features...", flush=True)
    features, issues = build_features(drat, epred, odds, schedule)
    if features.empty:
        raise SystemExit("No usable games remained after the four-way join.")

    features.to_csv(out_dir / "historical_game_features.csv", index=False)
    issues.to_csv(out_dir / "join_issues.csv", index=False)

    print(f"      usable joined games: {len(features)}", flush=True)

    print("[3/6] Building DRAT / EPRED / market / consensus selection rows...", flush=True)
    selections = build_selection_rows(features)
    selections.to_csv(out_dir / "selection_rows.csv", index=False)

    print("[4/6] Testing features and meaningful interactions across seasons...", flush=True)
    baselines, full_tests, actionable = run_tests(selections)
    baselines.to_csv(out_dir / "source_baselines.csv", index=False)
    full_tests.to_csv(out_dir / "full_test_results.csv", index=False)
    actionable.to_csv(out_dir / "actionable_findings.csv", index=False)

    print("[5/6] Comparing probability sources...", flush=True)
    benchmarks = probability_benchmarks(features)
    benchmarks.to_csv(out_dir / "probability_benchmarks.csv", index=False)

    print("[6/6] Writing pipeline implementation output...", flush=True)
    implementation = build_implementation(actionable)
    implementation.to_csv(out_dir / "pipeline_implementation.csv", index=False)

    lines = []
    lines.append("NFL FOUR-DATASET HISTORICAL TESTING")
    lines.append("=" * 38)
    lines.append(f"Usable joined historical games: {len(features)}")
    lines.append(f"Actionable repeatable findings: {len(actionable)}")
    lines.append("")
    lines.append("SOURCE / CONSENSUS BASELINES")
    for _, r in baselines.iterrows():
        lines.append(
            f"{r['Family']}: {int(r['Wins'])}-{int(r['Losses'])} = {r['Win_Pct']}%"
        )

    lines.append("")
    lines.append("PROBABILITY BENCHMARKS")
    for _, r in benchmarks.iterrows():
        lines.append(
            f"{r['Model']}: accuracy {r['Accuracy_Pct']}% | Brier {r['Brier']}"
        )

    lines.append("")
    lines.append("TOP ACTIONABLE FINDINGS")
    if actionable.empty:
        lines.append("None met the repeatability thresholds.")
    else:
        top = actionable.assign(
            __abs_lift=actionable["Lift_vs_Family"].abs()
        ).sort_values(["__abs_lift", "Games"], ascending=[False, False]).head(20)

        for _, r in top.iterrows():
            lines.append(
                f"{r['Family']} | {r['Condition']} | "
                f"{int(r['Wins'])}-{int(r['Losses'])} | {r['Win_Pct']}% | "
                f"lift {r['Lift_vs_Family']} pts | "
                f"forward {int(r['Forward_Successes'])}/{int(r['Forward_Checks'])}"
            )

    (out_dir / "HEADLINE_RESULTS.txt").write_text("\n".join(lines), encoding="utf-8")

    print("", flush=True)
    print("COMPLETE", flush=True)
    print(f"Headline: {out_dir / 'HEADLINE_RESULTS.txt'}", flush=True)
    print(f"Actionable findings: {out_dir / 'actionable_findings.csv'}", flush=True)
    print(f"Pipeline implementation: {out_dir / 'pipeline_implementation.csv'}", flush=True)
    print(f"Historical feature matrix: {out_dir / 'historical_game_features.csv'}", flush=True)


if __name__ == "__main__":
    main()
