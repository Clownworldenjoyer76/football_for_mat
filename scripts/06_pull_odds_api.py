#!/usr/bin/env python3
# Auto-discovers valid player-prop market keys, then pulls odds.
# Env: ODDS_API_KEY
from pathlib import Path
import os, sys, json, requests, pandas as pd

OUT = Path("data/odds"); OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = OUT / "_api_summary.txt"
MARKETS_JSON = OUT / "_api_markets.json"
MARKETS_CSV  = OUT / "_api_markets.csv"

API_KEY = os.getenv("ODDS_API_KEY", "").strip()
SPORT   = "americanfootball_nfl"
REGIONS = "us"
BOOKS   = "draftkings"
ODDS_BASE = "https://api.the-odds-api.com/v4"

def wsum(lines): SUMMARY.write_text("\n".join(lines) + "\n")

def nearest_half(x):
    try: return round(float(x)*2)/2
    except: return None

def norm_name(s):
    if not isinstance(s, str): return ""
    return s.replace(" Jr.", " Jr").replace(".", "").strip()

def get(url, timeout=30):
    r = requests.get(url, timeout=timeout)
    return r.status_code, r.text

def discover_markets():
    # Markets catalog endpoint
    url = f"{ODDS_BASE}/sports/{SPORT}/odds-markets?apiKey={API_KEY}"
    code, txt = get(url)
    lines = [f"[discover] HTTP {code}", f"bytes {len(txt or '')}"]
    if code != 200:
        wsum(lines + ["[discover] failed"])
        return [], lines
    try:
        data = json.loads(txt)
    except Exception:
        wsum(lines + ["[discover] json parse fail"])
        return [], lines

    # Save raw + csv
    MARKETS_JSON.write_text(json.dumps(data, indent=2))
    rows = []
    for m in data:
        # typical fields: key, title, description, outcomes
        rows.append({
            "key": m.get("key"),
            "title": m.get("title"),
            "description": m.get("description"),
        })
    pd.DataFrame(rows).to_csv(MARKETS_CSV, index=False)

    # Pick plausible keys by title/description
    wanted = {
        "qb_passing_yards":    ["passing yards"],
        "rb_rushing_yards":    ["rushing yards"],
        "wr_rec_yards":        ["receiving yards"],
        "wrte_receptions":     ["receptions"],
    }
    picks = {}
    for k in wanted:
        picks[k] = []

    for m in data:
        t = (m.get("title") or "").lower()
        d = (m.get("description") or "").lower()
        key = m.get("key")
        for ours, needles in wanted.items():
            if any(n in t or n in d for n in needles):
                picks[ours].append(key)

    chosen = {}
    for ours, keys in picks.items():
        # Choose the first candidate if multiple
        chosen[ours] = keys[0] if keys else None

    lines += ["[discover] chosen markets:"] + [f"  {k}: {v}" for k,v in chosen.items()]
    wsum(lines)
    return chosen, lines

def pull_odds(chosen_keys):
    # Build combined markets param from discovered keys (drop missing)
    mk = ",".join(sorted({v for v in chosen_keys.values() if v}))
    if not mk:
        return ["[pull] no discovered market keys"]
    url = (f"{ODDS_BASE}/sports/{SPORT}/odds/"
           f"?apiKey={API_KEY}&regions={REGIONS}&bookmakers={BOOKS}"
           f"&markets={mk}&oddsFormat=american")

    code, txt = get(url)
    lines = [f"[pull] HTTP {code}", f"bytes {len(txt or '')}", f"[pull] markets={mk}"]
    if code != 200:
        # write response body for debugging
        OUT.joinpath("_api_error_body.txt").write_text(txt or "")
        lines.append("[pull] non-200; wrote _api_error_body.txt")
        # still create empty CSVs
        for ours in chosen_keys.keys():
            (OUT / f"api_{ours}.csv").write_text("player_name,team,prop,line,over_odds,under_odds,book,season,week\n")
        wsum(lines)
        return lines

    try:
        payload = json.loads(txt)
    except Exception:
        lines.append("[pull] json parse fail")
        wsum(lines)
        return lines

    # Flatten by market
    by_market = {k: [] for k in chosen_keys.keys()}
    for game in payload or []:
        for bk in game.get("bookmakers") or []:
            if (bk.get("key") or "").lower() != "draftkings":
                continue
            for mkt in bk.get("markets") or []:
                mkey = (mkt.get("key") or "").lower()
                # map back to our market name
                ours = next((o for o,k in chosen_keys.items() if k and k.lower()==mkey), None)
                if not ours: continue
                for o in mkt.get("outcomes") or []:
                    player = norm_name(o.get("name"))
                    line   = nearest_half(o.get("point"))
                    price  = o.get("price")
                    desc   = (o.get("description") or o.get("name") or "").lower()
                    rec = {"player_name": player, "team": None, "prop": ours,
                           "line": line, "over_odds": None, "under_odds": None,
                           "book": "DK", "season": None, "week": None}
                    if "over" in desc:  rec["over_odds"]  = price
                    elif "under" in desc: rec["under_odds"] = price
                    by_market[ours].append(rec)

    # Combine over/under per player+line
    for ours, rows in by_market.items():
        if not rows:
            pd.DataFrame(columns=["player_name","team","prop","line","over_odds","under_odds","book","season","week"])\
              .to_csv(OUT / f"api_{ours}.csv", index=False)
            lines.append(f"[pull] {ours}: 0 rows")
            continue
        df = pd.DataFrame(rows)
        df = (df.groupby(["player_name","line","prop","book"], as_index=False)
                .agg({"team":"first","season":"first","week":"first",
                      "over_odds":"max","under_odds":"max"}))
        df = df[~df["line"].isna() & ~(df["over_odds"].isna() & df["under_odds"].isna())]
        df.to_csv(OUT / f"api_{ours}.csv", index=False)
        lines.append(f"[pull] {ours}: {len(df)} rows")

    wsum(lines)
    return lines

def main():
    if not API_KEY:
        wsum(["[init] missing ODDS_API_KEY"]); 
        # create empties so pipeline stays green
        for ours in ["qb_passing_yards","rb_rushing_yards","wr_rec_yards","wrte_receptions"]:
            (OUT / f"api_{ours}.csv").write_text("player_name,team,prop,line,over_odds,under_odds,book,season,week\n")
        sys.exit(0)

    chosen, _ = discover_markets()
    # if discovery failed, still write empties
    if not chosen or all(v is None for v in chosen.values()):
        wsum(["[discover] no markets found; writing empty CSVs"])
        for ours in ["qb_passing_yards","rb_rushing_yards","wr_rec_yards","wrte_receptions"]:
            (OUT / f"api_{ours}.csv").write_text("player_name,team,prop,line,over_odds,under_odds,book,season,week\n")
        sys.exit(0)

    pull_odds(chosen)

if __name__ == "__main__":
    main()
