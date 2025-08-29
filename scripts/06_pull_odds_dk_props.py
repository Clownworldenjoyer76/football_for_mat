#!/usr/bin/env python3
"""
Fetch NFL player prop odds from DraftKings' public eventgroup JSON and write:
  data/odds/dk_player_props_latest.csv

Notes
-----
- No API key required. Uses DK's public NFL event group feed (eventGroupId=88808).
- Output schema:
    event_id, game, start_time_utc, player, market, line, side, price_american, book, ts_utc
- Markets covered depend on what's live (passing yards, rushing yards, receptions, TDs, etc.)
"""

from __future__ import annotations
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import requests
import pandas as pd


NFL_EVENTGROUP_ID = 88808  # NFL
DK_EVENTGROUP_URL = f"https://sportsbook.draftkings.com/sites/US-SB/api/v4/eventgroups/{NFL_EVENTGROUP_ID}?format=json"
OUT_DIR = "data/odds"
OUT_FILE = os.path.join(OUT_DIR, "dk_player_props_latest.csv")


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fetch_json(url: str) -> Dict[str, Any]:
    headers = {
        "accept": "application/json, text/plain, */*",
        "user-agent": "Mozilla/5.0",
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def build_event_lookup(payload: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    events = payload.get("eventGroup", {}).get("events", [])
    return {ev.get("eventId"): ev for ev in events}


def american_price(outcome: Dict[str, Any]) -> Optional[int]:
    if "americanOdds" in outcome:
        return outcome.get("americanOdds")
    odds = outcome.get("odds")
    if isinstance(odds, dict) and "americanOdds" in odds:
        return odds.get("americanOdds")
    if "price" in outcome:
        return outcome.get("price")
    return None


def numeric_line(offer: Dict[str, Any], outcome: Dict[str, Any]) -> Optional[float]:
    for k in ("line", "lineString", "oddsAmericanLine", "points"):
        v = outcome.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    for k in ("line", "lineString"):
        v = offer.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    return None


def normalize_market_name(cat_name: str, subcat_name: str, label: str) -> str:
    base = subcat_name or cat_name or ""
    base = base.lower().strip()
    replacements = {
        "player passing yards": "passing_yards",
        "player rushing yards": "rushing_yards",
        "player receiving yards": "receiving_yards",
        "player receptions": "receptions",
        "player rushing attempts": "rushing_attempts",
        "player passing tds": "passing_tds",
        "player rushing tds": "rushing_tds",
        "player receiving tds": "receiving_tds",
        "player passing completions": "completions",
        "player interceptions": "interceptions",
        "player longest reception": "longest_reception",
        "player longest rush": "longest_rush",
        "player field goals made": "fg_made",
        "player tackles + assists": "tackles_assists",
    }
    for k, v in replacements.items():
        if k in base:
            return v
    if "over" in label.lower() or "under" in label.lower():
        return "player_prop"
    return base or "player_prop"


def extract_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    eg = payload.get("eventGroup", {})
    categories = eg.get("offerCategories", []) or []
    event_by_id = build_event_lookup(payload)

    for cat in categories:
        cat_name = cat.get("name", "")
        subcats = cat.get("offerSubcategoryDescriptors", []) or []
        for sub in subcats:
            sub_name = sub.get("name", "")
            desc = sub.get("offerSubcategory") or {}
            offers = desc.get("offers", []) or []
            for offer_group in offers:
                for offer in offer_group:
                    event_id = offer.get("eventId")
                    ev = event_by_id.get(event_id) or {}
                    game_name = ev.get("name")
                    start_time = ev.get("startDate")
                    start_time_utc = start_time if start_time else None
                    outcomes = offer.get("outcomes", []) or []
                    for oc in outcomes:
                        label = oc.get("label", "") or ""
                        player = oc.get("participant", "") or label
                        price = american_price(oc)
                        line = numeric_line(offer, oc)
                        side = None
                        lab_low = label.lower()
                        if "over" in lab_low:
                            side = "over"
                        elif "under" in lab_low:
                            side = "under"
                        market = normalize_market_name(cat_name, sub_name, label)
                        rows.append({
                            "event_id": event_id,
                            "game": game_name,
                            "start_time_utc": start_time_utc,
                            "player": player,
                            "market": market,
                            "line": line,
                            "side": side,
                            "price_american": price,
                            "book": "DraftKings",
                            "ts_utc": iso_utc_now(),
                        })
    return rows


def main() -> None:
    print(f"[dk] fetching {DK_EVENTGROUP_URL}", flush=True)
    data = fetch_json(DK_EVENTGROUP_URL)
    rows = extract_rows(data)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["player"].astype(str).str.len() > 0]
        df = df[df["price_american"].notna()]
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)
    print(f"[dk] wrote {OUT_FILE} ({len(df)} rows)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[dk] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
