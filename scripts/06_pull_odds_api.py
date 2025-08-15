#!/usr/bin/env python3
"""
Pull NFL player prop odds from The Odds API and save CSVs per market.

Env:
  ODDS_API_KEY  (set in GitHub Secrets)

Outputs (in data/odds/):
  api_qb_passing_yards.csv
  api_rb_rushing_yards.csv
  api_wr_rec_yards.csv
  api_wrte_receptions.csv
Also writes data/odds/_api_summary.txt for quick debugging.

Free-tier friendly: one request with multiple markets.
"""
from pathlib import Path
import os, sys, json
import requests
import pandas as pd

OUT = Path("data/odds"); OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = OUT / "_api_summary.txt"

API_KEY = os.getenv("ODDS_API_KEY", "").strip()
if not API_KEY:
    print("Missing ODDS_API_KEY env var", file=sys.stderr)
    # still create empty files to avoid breaking downstream
    for m in ["qb_passing_yards","rb_rushing_yards","wr_rec_yards","wrte_receptions"]:
        (OUT / f"api_{m}.csv").write_text("player_name,team,prop,line,over_odds,under_odds,book,season,week\n")
    sys.exit(0)

# Map your markets -> The Odds API markets
MARKET_MAP = {
    "qb_passing_yards": ("player_pass_yds",),
    "rb_rushing_yards": ("player_rush_yds",),
    "wr_rec_yards":     ("player_rec_yds",),
    "wrte_receptions":  ("player_receptions",),
}

SPORT = "americanfootball_nfl"
REGIONS = "us"             # US books
BOOKMAKERS = "draftkings"  # limit to DK; comma-sep for more books if you want
MARKETS = ",".join(sorted({v[0] for v in MARKET_MAP.values()}))
URL = (
    f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds/"
    f"?apiKey={API_KEY}&regions={REGIONS}&markets={MARKETS}&bookmakers={BOOKMAKERS}&oddsFormat=american"
)

def nearest_half(x):
    try:
        return round(float(x) * 2) / 2
    except Exception:
        return None

def normalize_name(s):
    if not isinstance(s, str): return ""
    return s.replace(" Jr.", " Jr").replace(".", "").strip()

def pull():
    r = requests.get(URL, timeout=30)
    return r.status_code, r.text

def parse_payload(txt):
    # payload: list of games, each has bookmakers[] -> markets[] -> outcomes[]
    try:
        data = json.loads(txt)
    except Exception:
        return {}
    rows_by_market = {k: [] for k in MARKET_MAP.keys()}
    for game in data or []:
        bks = game.get("bookmakers") or []
        for bk in bks:
            if (bk.get("key") or "").lower() != "draftkings":
                continue
            for mkt in bk.get("markets") or []:
                mkey = (mkt.get("key") or "").lower()
                # reverse-map to our market names
                ours = [our for our, api_m in MARKET_MAP.items() if api_m[0] == mkey]
                if not ours:
                    continue
                our_market = ours[0]
                for o in mkt.get("outcomes") or []:
                    # outcomes have name, price (american), and sometimes a point/line
                    player = normalize_name(o.get("name"))
                    line = nearest_half(o.get("point"))
                    price = o.get("price")
                    side = (o.get("description") or o.get("name") or "").lower()
                    # build an entry keyed by player+line and fill over/under as we see them
                    key = (player, line)
                    rows = rows_by_market[our_market]
                    if not rows or (rows and (rows[-1].get("_key") != key)):
                        rows.append({"_key": key, "player_name": player, "team": None,
                                     "prop": our_market, "line": line,
                                     "over_odds": None, "under_odds": None,
                                     "book": "DK", "season": None, "week": None})
                    # ensure we update the correct row (last might not match if players repeat)
                    if rows and rows[-1]["_key"] != key:
                        rows.append({"_key": key, "player_name": player, "team": None,
                                     "prop": our_market, "line": line,
                                     "over_odds": None, "under_odds": None,
                                     "book": "DK", "season": None, "week": None})
                    # set over/under by description if present; fallback: guess by “Over/Under” in desc
                    if "over" in side:
                        rows[-1]["over_odds"] = price
                    elif "under" in side:
                        rows[-1]["under_odds"] = price
                    else:
                        # if only one side present, leave the other as None
                        pass
    # finalize dict -> DataFrames
    out = {}
    for market, rows in rows_by_market.items():
        if not rows:
            out[market] = pd.DataFrame(columns=["player_name","team","prop","line","over_odds","under_odds","book","season","week"])
            continue
        df = pd.DataFrame(rows).drop(columns=["_key"], errors="ignore")
        # drop rows with no line or both odds missing
        df = df[~df["line"].isna() & ~(df["over_odds"].isna() & df["under_odds"].isna())]
        out[market] = df
    return out

def main():
    code, txt = pull()
    summary_lines = [f"HTTP {code}", f"bytes {len(txt or '')}"]
    if code != 200:
        # write empty files to keep pipeline green
        for m in MARKET_MAP.keys():
            (OUT / f"api_{m}.csv").write_text("player_name,team,prop,line,over_odds,under_odds,book,season,week\n")
        SUMMARY.write_text("\n".join(summary_lines) + "\n")
        return

    parsed = parse_payload(txt)
    for mkt, df in parsed.items():
        outp = OUT / f"api_{mkt}.csv"
        df.to_csv(outp, index=False)
        summary_lines.append(f"{mkt}: {len(df)} rows -> {outp.name}")

    SUMMARY.write_text("\n".join(summary_lines) + "\n")

if __name__ == "__main__":
    main()
