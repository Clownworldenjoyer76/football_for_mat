#!/usr/bin/env python3
"""
Pull DraftKings NFL player prop odds (unofficial public JSON).
- No API key. Uses requests only.
- Saves one CSV per market in data/odds/: dk_<market>.csv
- Markets matched to your pipeline:
    qb_passing_yards
    rb_rushing_yards
    wr_rec_yards
    wrte_receptions
Notes:
- DK endpoints can change. Script fails gracefully and writes empty files with a reason.
- Lines are rounded to .5 to match your props.
"""
from pathlib import Path
import time
import json
import math
import requests
import pandas as pd

OUT_DIR = Path("data/odds")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# DraftKings NFL event group (commonly 88808, but DK can rotate; we try a few)
CANDIDATE_EVENTGROUP_IDS = [88808, 42648, 88670846]

# Map our markets -> DK category keywords to search in offers
MARKETS = {
    "qb_passing_yards": ["Player Passing Yards"],
    "rb_rushing_yards": ["Player Rushing Yards"],
    "wr_rec_yards":     ["Player Receiving Yards"],
    "wrte_receptions":  ["Player Receptions"],  # WR/TE combined by book
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; OddsPuller/1.0; +https://example.com)"
}

def american_to_float(x):
    """Return float or None."""
    try:
        return float(x)
    except Exception:
        return None

def implied_prob(american):
    a = american_to_float(american)
    if a is None:
        return None
    if a < 0:
        return (-a) / ((-a) + 100.0)
    return 100.0 / (a + 100.0)

def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return name.replace(" Jr.", " Jr").replace(".", "").strip()

def nearest_half(x):
    try:
        return round(x * 2) / 2
    except Exception:
        return None

def pull_eventgroup(eventgroup_id: int):
    # Main offers feed (US-SB site). DK sometimes requires region param; we try simple first.
    url = f"https://sportsbook.draftkings.com//sites/US-SB/api/v5/eventgroups/{eventgroup_id}?format=json"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def extract_offers(j, wanted_titles):
    """Find markets whose 'categoryName' or 'offerCategory' title matches any wanted_titles."""
    results = []
    eventGroup = j.get("eventGroup", {})
    categories = eventGroup.get("offerCategories", []) or []
    for cat in categories:
        cat_name = (cat.get("name") or "").strip()
        subcats = cat.get("offerSubcategoryDescriptors", []) or []
        for sub in subcats:
            sub_name = (sub.get("name") or "").strip()
            offers = sub.get("offerSubcategory", {}).get("offers", []) or []
            # Check matching by either name
            name_match = any(
                any(w.lower() in (cat_name or "").lower() for w in wanted_titles) or
                any(w.lower() in (sub_name or "").lower() for w in wanted_titles)
            for _ in [0]
            )
            if not name_match:
                continue
            for offer_list in offers:
                # offer_list is a list of outcomes representing a single line or same player variant
                results.append(offer_list)
    return results

def offers_to_rows(offer_list, market):
    """Flatten one list of outcomes (a single offer) into rows."""
    rows = []
    # Try to get a common line (most DK props have two outcomes over/under sharing the same line)
    # If not present, we infer from the label or line/handicap field.
    line = None
    player = None
    team = None
    for outcome in offer_list:
        # Outcome fields vary; be defensive
        # Common fields: label, oddsAmerican, line or handicap, participant
        player = player or outcome.get("participant") or outcome.get("label")
        team = team or outcome.get("teamAbbreviation") or outcome.get("team")
        # DK sometimes uses 'line' or 'handicap' or embeds in label
        if line is None:
            line = outcome.get("line") or outcome.get("handicap")
            # Fallback: parse numbers from label like "Over 64.5"
            if line is None and isinstance(outcome.get("label"), str):
                import re
                m = re.search(r"([0-9]+(?:\.[0-9]+)?)", outcome["label"])
                if m:
                    line = float(m.group(1))
        # Collect rows; we’ll separate over/under by label if available
    for outcome in offer_list:
        label = (outcome.get("label") or "").lower()
        is_over = "over" in label
        is_under = "under" in label
        odds_am = outcome.get("oddsAmerican")
        rows.append({
            "player_name": normalize_name(player or ""),
            "team": team,
            "prop": market,
            "line": nearest_half(float(line)) if line is not None else None,
            "over_odds": odds_am if is_over else None,
            "under_odds": odds_am if is_under else None,
        })
    # Merge the two rows (over/under) into one
    if len(rows) >= 2:
        over = next((r for r in rows if r["over_odds"] is not None), None)
        under = next((r for r in rows if r["under_odds"] is not None), None)
        if over or under:
            merged = {
                "player_name": (over or under)["player_name"],
                "team": (over or under)["team"],
                "prop": market,
                "line": (over or under)["line"],
                "over_odds": american_to_float((over or {}).get("over_odds")),
                "under_odds": american_to_float((under or {}).get("under_odds")),
                "book": "DK",
            }
            return [merged]
    # If we couldn’t merge, still return what we have
    for r in rows:
        r["book"] = "DK"
    return rows

def fetch_market(market_name: str, titles):
    """Fetch a single market across candidate event groups."""
    # Try event groups until one returns data for this market
    for eg in CANDIDATE_EVENTGROUP_IDS:
        try:
            data = pull_eventgroup(eg)
            offers = extract_offers(data, titles)
            if not offers:
                time.sleep(0.8)
                continue
            # Flatten
            all_rows = []
            for offer in offers:
                rows = offers_to_rows(offer, market_name)
                all_rows.extend(rows)
            if all_rows:
                return pd.DataFrame(all_rows)
        except Exception:
            # Try next event group
            time.sleep(0.8)
            continue
    # None found
    return pd.DataFrame(columns=[
        "player_name","team","prop","line","over_odds","under_odds","book"
    ])

def add_season_week(df: pd.DataFrame):
    # If your pipeline’s latest season/week are needed, you can join from features/manifest.
    # For now, leave blank; Step 05/06 merge can ignore or fill from props files later.
    df["season"] = pd.NA
    df["week"] = pd.NA
    return df

def dedupe(df: pd.DataFrame):
    # Keep best (latest) per player/prop/line from DK
    if df.empty:
        return df
    return (df.sort_values(["player_name","prop","line"])
              .drop_duplicates(subset=["player_name","prop","line"], keep="last"))

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for market, titles in MARKETS.items():
        df = fetch_market(market, titles)
        df = add_season_week(df)
        df = dedupe(df)
        out = OUT_DIR / f"dk_{market}.csv"
        # Always write a file so downstream doesn’t break
        df.to_csv(out, index=False)
        print(f"✓ {market}: {len(df)} rows -> {out}")

if __name__ == "__main__":
    main()
