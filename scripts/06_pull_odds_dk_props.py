#!/usr/bin/env python3
"""
DraftKings NFL player props → data/odds/dk_player_props_latest.csv

What this does
--------------
1) Auto-discovers the current NFL eventGroupId from DK's master list.
2) Pulls the NFL event group JSON.
3) Extracts player prop offers into a normalized CSV:
   event_id, game, start_time_utc, player, market, line, side, price_american, book, ts_utc

Requirements: requests, pandas
"""

from __future__ import annotations
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import requests
import pandas as pd


MASTER_GROUPS_URL = "https://sportsbook.draftkings.com/sites/US-SB/api/v4/eventgroups?format=json"
GROUP_URL_TEMPLATE = "https://sportsbook.draftkings.com/sites/US-SB/api/v4/eventgroups/{gid}?format=json"

OUT_DIR = "data/odds"
OUT_FILE = os.path.join(OUT_DIR, "dk_player_props_latest.csv")


# ------------------------------- utils --------------------------------
def iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def http_get_json(url: str) -> Dict[str, Any]:
    headers = {
        "accept": "application/json, text/plain, */*",
        "user-agent": "Mozilla/5.0",
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


# -------------------------- DK helpers --------------------------------
def find_nfl_group_id() -> int:
    """
    Find the current NFL eventGroupId from the master groups feed.
    We accept several label variants DK has used historically.
    """
    payload = http_get_json(MASTER_GROUPS_URL)
    groups = payload.get("eventGroupList", []) or payload.get("eventGroups", []) or []

    wanted = {"nfl", "national football league"}
    for g in groups:
        name = f"{g.get('name','')} {g.get('eventGroupName','')}".lower()
        if any(w in name for w in wanted):
            gid = g.get("eventGroupId") or g.get("groupId") or g.get("id")
            if gid is not None:
                return int(gid)

    # Fallback: search any group that contains "football" and "league"
    for g in groups:
        name = f"{g.get('name','')} {g.get('eventGroupName','')}".lower()
        if "football" in name:
            gid = g.get("eventGroupId") or g.get("groupId") or g.get("id")
            if gid is not None:
                return int(gid)

    raise RuntimeError("Could not locate NFL eventGroupId from DraftKings master list.")


def build_event_lookup(group_payload: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    events = group_payload.get("eventGroup", {}).get("events", []) or []
    return {ev.get("eventId"): ev for ev in events if ev.get("eventId") is not None}


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
    base = (subcat_name or cat_name or "").lower().strip()
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


def extract_player_props(group_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    eg = group_payload.get("eventGroup", {}) or {}
    categories = eg.get("offerCategories", []) or []

    event_by_id = build_event_lookup(group_payload)

    for cat in categories:
        cat_name = cat.get("name", "") or ""
        subcats = cat.get("offerSubcategoryDescriptors", []) or []
        for sub in subcats:
            sub_name = sub.get("name", "") or ""
            desc = sub.get("offerSubcategory") or {}
            offers = desc.get("offers", []) or []

            # Only keep subcategories that look like player-related markets
            # (DK's taxonomy varies; this is intentionally permissive)
            looks_player = any(x in (cat_name + " " + sub_name).lower() for x in [
                "player", "qb", "rb", "wr", "te", "defensive", "kicker"
            ])

            if not looks_player:
                continue

            for offer_group in offers:
                for offer in offer_group:
                    event_id = offer.get("eventId")
                    ev = event_by_id.get(event_id) or {}
                    game_name = ev.get("name")
                    start_time_utc = ev.get("startDate")  # DK already uses ISO-8601 Z

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


# ------------------------------ main ----------------------------------
def main() -> None:
    print("[dk] discovering NFL eventGroupId...", flush=True)
    gid = find_nfl_group_id()
    print(f"[dk] NFL eventGroupId = {gid}", flush=True)

    url = GROUP_URL_TEMPLATE.format(gid=gid)
    print(f"[dk] fetching {url}", flush=True)
    group_payload = http_get_json(url)

    rows = extract_player_props(group_payload)
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
