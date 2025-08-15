#!/usr/bin/env python3
# Pull NFL player prop odds from The Odds API into data/odds/api_*.csv
# Env: ODDS_API_KEY
from pathlib import Path
import os, sys, json, requests, pandas as pd

OUT = Path("data/odds"); OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = OUT / "_api_summary.txt"

API_KEY = os.getenv("ODDS_API_KEY", "").strip()
MARKET_MAP = {
    "qb_passing_yards": ("player_pass_yds",),
    "rb_rushing_yards": ("player_rush_yds",),
    "wr_rec_yards":     ("player_rec_yds",),
    "wrte_receptions":  ("player_receptions",),
}
SPORT = "americanfootball_nfl"
REGIONS = "us"
BOOKMAKERS = "draftkings"
MARKETS = ",".join(sorted({v[0] for v in MARKET_MAP.values()}))
URL = (f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds/"
       f"?apiKey={API_KEY}&regions={REGIONS}&markets={MARKETS}"
       f"&bookmakers={BOOKMAKERS}&oddsFormat=american")

def nearest_half(x):
    try: return round(float(x)*2)/2
    except: return None

def normalize_name(s):
    if not isinstance(s, str): return ""
    return s.replace(" Jr.", " Jr").replace(".", "").strip()

def write_empty():
    hdr = "player_name,team,prop,line,over_odds,under_odds,book,season,week\n"
    for m in MARKET_MAP.keys():
        (OUT / f"api_{m}.csv").write_text(hdr)

def parse_payload(txt):
    try:
        data = json.loads(txt)
    except Exception:
        return {k: pd.DataFrame(columns=["player_name","team","prop","line","over_odds","under_odds","book","season","week"])
                for k in MARKET_MAP.keys()}

    rows = {k: [] for k in MARKET_MAP.keys()}
    for game in data or []:
        for bk in game.get("bookmakers") or []:
            if (bk.get("key") or "").lower() != "draftkings":
                continue
            for mkt in bk.get("markets") or []:
                mkey = (mkt.get("key") or "").lower()
                ours = [our for our, api_m in MARKET_MAP.items() if api_m[0] == mkey]
                if not ours: continue
                our_market = ours[0]
                for o in mkt.get("outcomes") or []:
                    player = normalize_name(o.get("name"))
                    line   = nearest_half(o.get("point"))
                    price  = o.get("price")
                    desc   = (o.get("description") or o.get("name") or "").lower()
                    rec = {"player_name": player, "team": None, "prop": our_market,
                           "line": line, "over_odds": None, "under_odds": None,
                           "book": "DK", "season": None, "week": None}
                    if "over" in desc:
                        rec["over_odds"] = price
                    elif "under" in desc:
                        rec["under_odds"] = price
                    rows[our_market].append(rec)

    out = {}
    for m, r in rows.items():
        if not r:
            out[m] = pd.DataFrame(columns=["player_name","team","prop","line","over_odds","under_odds","book","season","week"])
            continue
        df = pd.DataFrame(r)
        # group by player+line, combine over/under if they appeared as separate outcomes
        df = (df.groupby(["player_name","line","prop","book"], as_index=False)
                .agg({"team":"first","season":"first","week":"first",
                      "over_odds":"max","under_odds":"max"}))
        df = df[~df["line"].isna() & ~(df["over_odds"].isna() & df["under_odds"].isna())]
        out[m] = df
    return out

def main():
    if not API_KEY:
        SUMMARY.write_text("HTTP 0\nmissing ODDS_API_KEY\n")
        write_empty(); return

    try:
        r = requests.get(URL, timeout=30)
        code, txt = r.status_code, r.text
    except Exception as e:
        SUMMARY.write_text(f"HTTP 0\nerror {e}\n"); write_empty(); return

    lines = [f"HTTP {code}", f"bytes {len(txt or '')}"]
    if code != 200:
        SUMMARY.write_text("\n".join(lines) + "\n"); write_empty(); return

    parsed = parse_payload(txt)
    for mkt, df in parsed.items():
        df.to_csv(OUT / f"api_{mkt}.csv", index=False)
        lines.append(f"{mkt}: {len(df)} rows")
    SUMMARY.write_text("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
