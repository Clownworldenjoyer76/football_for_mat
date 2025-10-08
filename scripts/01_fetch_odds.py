#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_fetch_odds.py

Fetches:
  1) "Featured" markets in one shot (h2h/spreads/totals)  -> data/odds/raw/featured_raw_<UTCSTAMP>.json
  2) Optional *player* props per-event (if event IDs are provided) -> data/odds/raw/player_raw_<UTCSTAMP>.json

Environment (typical from workflow):
  ODDS_API_KEY          : required (Odds API key)
  ODDS_SPORT            : default "americanfootball_nfl"
  ODDS_REGIONS          : default "us"
  ODDS_MARKETS          : featured markets (comma list). Default "h2h,spreads,totals"
  ODDS_BOOKS            : optional bookmaker keys (comma list)
  ODDS_ODDS_FORMAT      : default "american"
  ODDS_DATE_FORMAT      : default "iso"
  ODDS_PLAYER_MARKETS   : player markets (comma list) for per-event fetch; if empty, player step is still executed and will write an empty JSON.
  ODDS_EVENT_IDS        : comma list of event IDs to fetch player props for (use a small set to conserve quota).
  ODDS_SLEEP_MS         : optional sleep between API calls (player loop), e.g. "150"

CLI flags are optional mirrors of the env above.

This script ALWAYS writes the raw files (even if the arrays are empty) so downstream
cache building can proceed without extra API calls.
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import requests  # type: ignore
except Exception:
    requests = None

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "odds" / "raw"
PROC_DIR = ROOT / "data" / "odds" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

API_BASE = "https://api.the-odds-api.com/v4"
SPORT_DEFAULT = "americanfootball_nfl"

def utc_stamp() -> str:
    import datetime as dt
    return dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def parse_csv_env(name: str, default: str = "") -> List[str]:
    raw = os.getenv(name, default).strip()
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]

def fetch_url(url: str, params: Dict[str, Any]) -> Any:
    if requests is None:
        raise RuntimeError("The 'requests' package is not available.")
    r = requests.get(url, params=params, timeout=30)
    # If the provider rejects some markets, bubble up the exact message.
    try:
        r.raise_for_status()
    except Exception as e:
        # Add URL and params to error for easier debugging in Actions logs
        msg = f"HTTP {r.status_code} for URL: {r.url}\nBody: {r.text}"
        raise SystemExit(msg) from e
    try:
        return r.json()
    except Exception:
        return r.text

def fetch_featured(api_key: str,
                   sport: str,
                   regions: str,
                   markets: List[str],
                   books: Optional[str],
                   odds_fmt: str,
                   date_fmt: str) -> Dict[str, Any]:
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": ",".join(markets) if markets else "h2h,spreads,totals",
        "oddsFormat": odds_fmt,
        "dateFormat": date_fmt,
    }
    if books:
        params["bookmakers"] = books
    url = f"{API_BASE}/sports/{sport}/odds"
    data = fetch_url(url, params)
    return {
        "meta": {
            "kind": "featured",
            "sport": sport,
            "regions": regions,
            "markets": markets,
            "bookmakers": books,
            "oddsFormat": odds_fmt,
            "dateFormat": date_fmt,
        },
        "data": data if isinstance(data, list) else [],
    }

def fetch_players_for_events(api_key: str,
                             sport: str,
                             event_ids: List[str],
                             player_markets: List[str],
                             regions: str,
                             books: Optional[str],
                             odds_fmt: str,
                             date_fmt: str,
                             sleep_ms: int) -> Dict[str, Any]:
    """
    Fetch player props one event at a time. Writes a compact envelope:
      { meta: {...}, events: [ {eventId: "...", markets:[...]}, ... ] }
    If event_ids or player_markets are empty, returns an empty envelope.
    """
    envelope: Dict[str, Any] = {
        "meta": {
            "kind": "player",
            "sport": sport,
            "regions": regions,
            "markets": player_markets,
            "bookmakers": books,
            "oddsFormat": odds_fmt,
            "dateFormat": date_fmt,
            "event_count": len(event_ids),
        },
        "events": [],
    }
    if not event_ids or not player_markets:
        return envelope

    for eid in event_ids:
        params = {
            "apiKey": api_key,
            "regions": regions,
            "markets": ",".join(player_markets),
            "oddsFormat": odds_fmt,
            "dateFormat": date_fmt,
        }
        if books:
            params["bookmakers"] = books
        url = f"{API_BASE}/sports/{sport}/events/{eid}/odds"
        try:
            data = fetch_url(url, params)
            envelope["events"].append({
                "eventId": eid,
                "payload": data if isinstance(data, dict) else data,
            })
        except SystemExit as e:
            # Record the failure inline but keep going so we still produce a JSON
            envelope["events"].append({
                "eventId": eid,
                "error": str(e),
            })
        # rate limit
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)
    return envelope

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch featured and optional player props (per-event) from The Odds API.")
    ap.add_argument("--sport", default=os.getenv("ODDS_SPORT", SPORT_DEFAULT))
    ap.add_argument("--regions", default=os.getenv("ODDS_REGIONS", "us"))
    ap.add_argument("--markets", default=os.getenv("ODDS_MARKETS", "h2h,spreads,totals"))
    ap.add_argument("--books", default=os.getenv("ODDS_BOOKS", ""))
    ap.add_argument("--odds-format", default=os.getenv("ODDS_ODDS_FORMAT", "american"))
    ap.add_argument("--date-format", default=os.getenv("ODDS_DATE_FORMAT", "iso"))
    ap.add_argument("--player-markets", default=os.getenv("ODDS_PLAYER_MARKETS", ""))
    ap.add_argument("--event-ids", default=os.getenv("ODDS_EVENT_IDS", ""))
    ap.add_argument("--sleep-ms", type=int, default=int(os.getenv("ODDS_SLEEP_MS", "0")))
    args = ap.parse_args(argv)

    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        print("ERROR: missing ODDS_API_KEY", file=sys.stderr)
        return 2

    sport = args.sport
    regions = args.regions
    feat_markets = [m for m in parse_csv_env("ODDS_MARKETS", args.markets) if m]
    # Hard-guard featured markets to the 3 allowed types
    ALLOWED_FEATURED = {"h2h", "spreads", "totals"}
    feat_markets = [m for m in feat_markets if m in ALLOWED_FEATURED]
    if not feat_markets:
        feat_markets = ["h2h", "spreads", "totals"]

    books = args.books.strip() or None
    odds_fmt = args.odds_format
    date_fmt = args.date_format

    # --- 1) Featured ---
    ts = utc_stamp()
    featured = fetch_featured(api_key, sport, regions, feat_markets, books, odds_fmt, date_fmt)
    fpath = RAW_DIR / f"featured_raw_{ts}.json"
    write_json(fpath, featured)
    print(f"[featured] wrote {fpath}  items={len(featured.get('data', []))}")

    # --- 2) Player (per-event) ---
    player_markets = [m for m in parse_csv_env("ODDS_PLAYER_MARKETS", args.player_markets) if m]
    event_ids = [e for e in parse_csv_env("ODDS_EVENT_IDS", args.event_ids) if e]
    players_env = fetch_players_for_events(
        api_key=api_key,
        sport=sport,
        event_ids=event_ids,
        player_markets=player_markets,
        regions=regions,
        books=books,
        odds_fmt=odds_fmt,
        date_fmt=date_fmt,
        sleep_ms=args.sleep_ms,
    )
    ppath = RAW_DIR / f"player_raw_{ts}.json"
    write_json(ppath, players_env)
    print(f"[players] wrote {ppath}  events={len(players_env.get('events', []))}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
