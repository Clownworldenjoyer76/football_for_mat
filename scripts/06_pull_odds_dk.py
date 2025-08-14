#!/usr/bin/env python3
"""
DraftKings NFL player props -> data/odds/dk_<market>.csv
Also writes debug snapshots: data/odds/_dk_eventgroup_<id>.json.gz
"""
from pathlib import Path
import time, json, gzip, re
import requests
import pandas as pd

OUT_DIR = Path("data/odds"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# Try several DK NFL eventGroupIds (DK rotates these)
EVENTGROUP_IDS = [88808, 88670846, 42648, 84240]

# Our markets -> keywords to match anywhere (category, subcategory, offer name, outcome label)
MARKETS = {
    "qb_passing_yards": ["player passing yards", "passing yards"],
    "rb_rushing_yards": ["player rushing yards", "rushing yards"],
    "wr_rec_yards":     ["player receiving yards", "receiving yards"],
    "wrte_receptions":  ["player receptions", "receptions"],
}

UA = {"User-Agent": "Mozilla/5.0 (OddsPuller/1.0)"}

def nearest_half(x):
    try: return round(float(x) * 2) / 2
    except: return None

def american_to_float(x):
    try: return float(x)
    except: return None

def fetch_eventgroup(egid: int):
    url = f"https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/{egid}?format=json"
    r = requests.get(url, headers=UA, timeout=25)
    r.raise_for_status()
    j = r.json()
    # write debug snapshot
    dbg = OUT_DIR / f"_dk_eventgroup_{egid}.json.gz"
    with gzip.open(dbg, "wt", encoding="utf-8") as f:
        json.dump(j, f)
    return j

def text_match(s: str, needles):
    s = (s or "").lower()
    return any(n in s for n in needles)

def yield_offers(j):
    """Yield (category_name, subcat_name, offer_list) across the tree."""
    eg = j.get("eventGroup", {})
    cats = eg.get("offerCategories", []) or []
    for cat in cats:
        cname = (cat.get("name") or "")
        subs = cat.get("offerSubcategoryDescriptors", []) or []
        for sub in subs:
            sname = (sub.get("name") or "")
            offers = sub.get("offerSubcategory", {}).get("offers", []) or []
            for offer_list in offers:
                yield cname, sname, offer_list

def parse_offer_list(offer_list):
    """Return dict with player, line, over/under odds if present."""
    player = None; line = None; over = None; under = None; team = None
    # First pass: discover player, line
    for outcome in offer_list:
        lbl = (outcome.get("label") or "")
        player = player or outcome.get("participant") or lbl
        team = team or outcome.get("teamAbbreviation") or outcome.get("team")
        line = line or outcome.get("line") or outcome.get("handicap")
        if line is None:
            m = re.search(r"(\d+(?:\.\d+)?)", lbl)
            if m: line = float(m.group(1))
    # Second pass: map over/under
    for outcome in offer_list:
        lbl = (outcome.get("label") or "").lower()
        odds = outcome.get("oddsAmerican")
        if "over" in lbl:
            over = american_to_float(odds)
        if "under" in lbl:
            under = american_to_float(odds)
    return {
        "player_name": (player or "").replace(".", "").replace(" Jr.", " Jr").strip(),
        "team": team,
        "line": nearest_half(line),
        "over_odds": over,
        "under_odds": under,
    }

def collect_market(df_offers, needles):
    rows = []
    for cname, sname, offer_list in df_offers:
        if not (text_match(cname, needles) or text_match(sname, needles)):
            # fallback: also scan first outcome label text
            if not offer_list or not text_match((offer_list[0].get("label") or ""), needles):
                continue
        row = parse_offer_list(offer_list)
        # Must have both sides and a line
        if row["line"] is None or (row["over_odds"] is None and row["under_odds"] is None):
            continue
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["player_name","team","prop","line","over_odds","under_odds","book","season","week"])
    df = pd.DataFrame(rows)
    df["book"] = "DK"; df["season"] = pd.NA; df["week"] = pd.NA
    return (df.sort_values(["player_name","line"])
              .drop_duplicates(subset=["player_name","line"], keep="last"))

def main():
    # Try each event group; stop at first that yields any props across markets
    any_found = False
    for egid in EVENTGROUP_IDS:
        try:
            j = fetch_eventgroup(egid)
        except Exception as e:
            print(f"[warn] eg {egid} fetch failed: {e}")
            time.sleep(1.0); continue

        offers_iter = list(yield_offers(j))
        if not offers_iter:
            print(f"[info] eg {egid} has no offers nodes")
            continue

        found_this = False
        for market, needles in MARKETS.items():
            df = collect_market(offers_iter, needles)
            out = OUT_DIR / f"dk_{market}.csv"
            df.assign(prop=market).to_csv(out, index=False)
            print(f"✓ {market}: {len(df)} rows -> {out}")
            found_this = found_this or (len(df) > 0)
        any_found = any_found or found_this
        if any_found:
            break  # stop after first working event group

    if not any_found:
        print("[warn] No DK player prop rows found. Inspect data/odds/_dk_eventgroup_*.json.gz")

if __name__ == "__main__":
    main()
