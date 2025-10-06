#!/usr/bin/env python3
# Build per-market feature CSVs from data/props/props_current.csv (STRICT)

import os, sys
from pathlib import Path
import pandas as pd

PROPS_CUR = Path("data/props/props_current.csv")
WEEKLY_LATEST = Path("data/weekly/latest.csv")
OUT_DIR = Path("data/features")

# Canonical market names → (output filename, allowed positions or None)
MARKETS = [
    # existing four
    ("qb_passing_yards",       "qb_passing_yards.csv",       {"QB"}),
    ("rb_rushing_yards",       "rb_rushing_yards.csv",       {"RB"}),
    ("wr_rec_yards",           "wr_rec_yards.csv",           {"WR"}),
    ("wrte_receptions",        "wrte_receptions.csv",        {"WR","TE"}),
    # new (no strict position filter unless obvious)
    ("qb_passing_tds",         "qb_passing_tds.csv",         {"QB"}),
    ("qb_interceptions",       "qb_interceptions.csv",       {"QB"}),
    ("player_tds",             "player_tds.csv",             None),
    ("player_sacks",           "player_sacks.csv",           None),
    ("player_tackles",         "player_tackles.csv",         None),
    ("player_tackles_assists", "player_tackles_assists.csv", None),
    ("player_field_goals_made","player_field_goals_made.csv",{"K"}),  # kicker if present
]

REQ_META = ["season","week","game_id","player_id","player_name","team","opponent"]
CANDIDATE_ID_COLS   = ["player_id","playerId","id"]
CANDIDATE_NAME_COLS = ["player_name","player","name","playerFullName","runner"]
CANDIDATE_TEAM_COLS = ["team","recent_team","team_abbr","teamAbbr","team_short","team_id"]
CANDIDATE_OPP_COLS  = ["opponent","opponent_team","opp","opp_abbr","opponent_abbr","oppAbbr"]
CANDIDATE_WEEK_COLS = ["week","week_number","weekNumber"]
CANDIDATE_GAME_COLS = ["game_id","gameId","gsis_id","gameIdStr"]
CANDIDATE_POS_COLS  = ["position","pos","player_position"]

def die(msg: str) -> None:
    print(f"[features:ERROR] {msg}", file=sys.stderr)
    sys.exit(1)

def info(msg: str) -> None:
    print(f"[features] {msg}")

def env_season() -> int:
    s = os.getenv("TARGET_SEASON") or os.getenv("SEASON") or ""
    if not s.isdigit():
        die("TARGET_SEASON env is required (strict mode).")
    return int(s)

def pick_first_col(df: pd.DataFrame, candidates, default_name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    df[default_name] = None
    return default_name

def read_week_fallback() -> int | None:
    if WEEKLY_LATEST.exists():
        try:
            wdf = pd.read_csv(WEEKLY_LATEST)
            for c in ["week","current_week","wk","week_number"]:
                if c in wdf.columns and pd.notnull(wdf[c]).any():
                    return int(pd.to_numeric(wdf[c], errors="coerce").dropna().iloc[0])
        except Exception:
            pass
    return None

def normalize_metadata(df: pd.DataFrame) -> pd.DataFrame:
    id_col   = pick_first_col(df, CANDIDATE_ID_COLS,   "player_id")
    name_col = pick_first_col(df, CANDIDATE_NAME_COLS, "player_name")
    team_col = pick_first_col(df, CANDIDATE_TEAM_COLS, "team")
    opp_col  = pick_first_col(df, CANDIDATE_OPP_COLS,  "opponent")
    week_col = pick_first_col(df, CANDIDATE_WEEK_COLS, "week")
    game_col = pick_first_col(df, CANDIDATE_GAME_COLS, "game_id")

    out = df.rename(columns={
        id_col:"player_id", name_col:"player_name", team_col:"team",
        opp_col:"opponent", week_col:"week", game_col:"game_id"
    }).copy()

    if "week" in out and out["week"].isna().all():
        wk = read_week_fallback()
        if wk is not None:
            out["week"] = wk

    for c in REQ_META:
        if c not in out.columns:
            out[c] = None
    return out

def copy_or_na(out: pd.DataFrame, target: str, alts: list[str]):
    if target not in out.columns:
        for a in alts:
            if a in out.columns:
                out[target] = out[a]
                break
        else:
            out[target] = pd.NA

def build_for_market(props: pd.DataFrame, market_exact: str, season: int, pos_allow: set[str] | None) -> pd.DataFrame:
    if "market" not in props.columns:
        die("props_current.csv has no 'market' column (strict mode).")

    df = props[(props["market"].astype(str) == market_exact) & (props["season"].astype(int) == season)].copy()

    # Optional position filter
    if pos_allow:
        pos_col = None
        for c in CANDIDATE_POS_COLS:
            if c in df.columns:
                pos_col = c
                break
        if pos_col is not None:
            df = df[df[pos_col].astype(str).str.upper().isin({p.upper() for p in pos_allow})].copy()

    df = normalize_metadata(df)

    # Stable numeric/context features when present
    copy_or_na(df, "line",        ["point","prop_line","projection"])
    copy_or_na(df, "odds_over",   ["over_odds","odds_o","american_odds_over","overAmerican","prob_over","price_american"])
    copy_or_na(df, "odds_under",  ["under_odds","odds_u","american_odds_under","underAmerican","prob_under"])
    copy_or_na(df, "book",        ["sportsbook","book_name","book","source"])

    cols = REQ_META + ["line","odds_over","odds_under","book"]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    return df[cols].drop_duplicates()

def main() -> int:
    season = env_season()

    if not PROPS_CUR.exists():
        die(f"Missing input file: {PROPS_CUR}")

    try:
        props = pd.read_csv(PROPS_CUR, low_memory=False)
    except Exception as e:
        die(f"Unable to read {PROPS_CUR}: {e}")

    if "season" not in props.columns:
        die("props_current.csv is missing a 'season' column (strict mode).")

    seasons_present = sorted(set(pd.to_numeric(props["season"], errors="coerce").dropna().astype(int).tolist()))
    if season not in seasons_present:
        die(f"TARGET_SEASON={season} not present in props_current.csv. Seasons present: {seasons_present}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    empty = []
    for market_exact, out_name, pos_allow in MARKETS:
        df_feat = build_for_market(props, market_exact, season, pos_allow)
        out_path = OUT_DIR / out_name
        df_feat.to_csv(out_path, index=False)
        info(f"Wrote {out_path} rows={len(df_feat)} (market='{market_exact}', season={season})")
        if len(df_feat) == 0:
            empty.append(out_name)

    if empty:
        uniq_markets = sorted(props["market"].astype(str).dropna().unique().tolist())
        die(
            "No rows produced for:\n  - " + "\n  - ".join(empty) +
            f"\nCheck props_current.csv contains those exact markets for season {season}."
            f"\nMarkets present: {uniq_markets[:12]}{' ...' if len(uniq_markets)>12 else ''}"
        )

    info("All feature files written successfully (strict mode).")
    return 0

if __name__ == "__main__":
    sys.exit(main())
