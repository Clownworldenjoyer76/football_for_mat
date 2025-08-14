#!/usr/bin/env python3
"""
DraftKings NFL player props -> data/odds/dk_<market>.csv
More robust:
- Pull event group JSON (v5) + per-category JSON (v4) for all offerCategoryIds
- Match markets by keywords across category/subcategory/outcomes
- Write debug snapshots for inspection: data/odds/_dk_eventgroup_*.json.gz and _dk_category_<id>.json.gz
"""
from pathlib import Path
import time, json, gzip, re
import requests
import pandas as pd

OUT_DIR = Path("data/odds"); OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENTGROUP_IDS = [88808, 88670846, 42648, 84240]  # DK rotates these
MARKETS = {
    "qb_passing_yards": ["player passing yards","passing yards"],
    "rb_rushing_yards": ["player rushing yards","rushing yards"],
    "wr_rec_yards":     ["player receiving yards","receiving yards"],
    "wrte_receptions":  ["player receptions","receptions"],
}
UA = {"User-Agent": "Mozilla/5.0 (OddsPuller/1.1)"}

def nearest_half(x):
    try: return round(float(x) * 2) / 2
    except: return None

def american_to_float(x):
    try: return float(x)
    except: return None

def normalize_name(s):
    if not isinstance(s, str): return ""
    return s.replace(" Jr.", " Jr").replace(".", "").strip()

def fetch_json(url: str):
    r = requests.get(url, headers=UA, timeout=25)
    r.raise_for_status()
    return r.json()

def dump_gz(obj, path: Path):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f)

def text_match(s: str, needles):
    s = (s or "").lower()
    return any(n in s for n in needles)

def yield_offers_from_v5(j):
    eg = j.get("eventGroup", {}) or {}
    for cat in eg.get("offerCategories", []) or []:
        cname = (cat.get("name") or "")
        for sub in cat.get("offerSubcategoryDescriptors", []) or []:
            sname = (sub.get("name") or "")
            offers = sub.get("offerSubcategory", {}).get("offers", []) or []
            for offer_list in offers:
                yield cname, sname, offer_list

def yield_offers_from_v4(cat_json):
    # v4 categories format tends to be flatter but still has offer lists
    cname = (cat_json.get("name") or "")
    for sub in cat_json.get("subcategories", []) or []:
        sname = (sub.get("name") or "")
        for offer_list in sub.get("offers", []) or []:
            yield cname, sname, offer_list

def parse_offer_list(offer_list):
    player = None; team = None; line = None; over = None; under = None
    for outcome in offer_list:
        lbl = (outcome.get("label") or "")
        player = player or outcome.get("participant") or lbl
        team = team or outcome.get("teamAbbreviation") or outcome.get("team")
        line = line or outcome.get("line") or outcome.get("handicap")
        if line is None and lbl:
            m = re.search(r"(\d+(?:\.\d+)?)", lbl)
            if m: line = float(m.group(1))
    for outcome in offer_list:
        lbl = (outcome.get("label") or "").lower()
        odds = outcome.get("oddsAmerican")
        if "over" in lbl:  over = american_to_float(odds)
        if "under" in lbl: under = american_to_float(odds)
    return normalize_name(player), team, nearest_half(line), over, under

def collect_rows(offers_iter, market, needles):
    rows = []
    for cname, sname, offer_list in offers_iter:
        # match on category, subcategory, or first outcome label
        ok = text_match(cname, needles) or text_match(sname, needles)
        if not ok and offer_list:
            ok = text_match((offer_list[0].get("label") or ""), needles)
        if not ok: 
            continue
        player, team, line, over, under = parse_offer_list(offer_list)
        if line is None or (over is None and under is None) or not player:
            continue
        rows.append({
            "player_name": player, "team": team, "prop": market,
            "line": line, "over_odds": over, "under_odds": under,
            "book": "DK", "season": pd.NA, "week": pd.NA
        })
    if not rows:
        return pd.DataFrame(columns=["player_name","team","prop","line","over_odds","under_odds","book","season","week"])
    df = pd.DataFrame(rows)
    return (df.sort_values(["player_name","line"])
              .drop_duplicates(subset=["player_name","prop","line"], keep="last"))

def main():
    any_found = False
    for egid in EVENTGROUP_IDS:
        try:
            # v5 event group
            v5 = fetch_json(f"https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/{egid}?format=json")
            dump_gz(v5, OUT_DIR / f"_dk_eventgroup_{egid}.json.gz")
        except Exception as e:
            print(f"[warn] v5 fetch eg {egid} failed: {e}"); time.sleep(0.8); continue

        # start with whatever v5 exposes
        offers_v5 = list(yield_offers_from_v5(v5))

        # also try each offerCategoryId via v4 per-category endpoint for deeper props
        cat_ids = []
        for cat in (v5.get("eventGroup", {}) or {}).get("offerCategories", []) or []:
            cid = cat.get("offerCategoryId")
            if isinstance(cid, int): cat_ids.append(cid)
        offers_v4 = []
        for cid in sorted(set(cat_ids)):
            try:
                v4 = fetch_json(f"https://sportsbook.draftkings.com/sites/US-SB/api/v4/eventgroups/{egid}/categories/{cid}?format=json")
                dump_gz(v4, OUT_DIR / f"_dk_category_{egid}_{cid}.json.gz")
                offers_v4.extend(list(yield_offers_from_v4(v4)))
                time.sleep(0.2)
            except Exception as e:
                print(f"[warn] v4 fetch eg {egid} cat {cid} failed: {e}")
                time.sleep(0.2)

        all_offers = offers_v5 + offers_v4
        found_this_eg = False
        for market, needles in MARKETS.items():
            df = collect_rows(all_offers, market, needles)
            out = OUT_DIR / f"dk_{market}.csv"
            df.to_csv(out, index=False)
            print(f"✓ {market}: {len(df)} rows -> {out}")
            if len(df) > 0: found_this_eg = True
        if found_this_eg:
            any_found = True
            break

    if not any_found:
        print("[warn] No DK player-prop rows found. Inspect data/odds/_dk_eventgroup_*.json.gz and _dk_category_*.json.gz")

if __name__ == "__main__":
    main()
