#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_fetch_odds.py
Fetch NFL featured odds (h2h/spreads/totals) and/or per-event player props
from The Odds API, with defensive parsing and clear file outputs.

Environment (all optional unless noted):
  ODDS_API_KEY       (required unless --dry-run)
  ODDS_MODE          "featured" | "players" | "both" (default: "featured")
  SPORT              default "americanfootball_nfl"
  ODDS_REGIONS       default "us"
  ODDS_BOOKS         comma list of books (default set below)
  ODDS_MARKETS       for FEATURED only (default: "h2h,spreads,totals")
  PLAYER_MARKETS     for PLAYERS only (default list below)
  ODDS_MAX_EVENTS    int cap for player-prop event calls (default: 2)
  ODDS_FORMAT        "american" (default)
  DATE_FORMAT        "iso" (default)

Outputs:
  data/odds/raw/featured_raw_<UTCYYYYMMDD_HHMMSS>.json
  data/odds/processed/featured_<UTCYYYYMMDD_HHMMSS>.csv
  data/odds/raw/player_raw_<EVENTID>_<UTCYYYYMMDD_HHMMSS>.json
"""

from __future__ import annotations

import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

# Lazy import in case runner doesn't have requests preinstalled
try:
    import requests
except Exception:
    requests = None

# -----------------------------
# Config & paths
# -----------------------------
API_BASE_SPORT = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
API_BASE_EVENT = "https://api.the-odds-api.com/v4/sports/{sport}/events/{event_id}/odds"

DEFAULT_SPORT = os.getenv("SPORT", "americanfootball_nfl")
DEFAULT_REGIONS = os.getenv("ODDS_REGIONS", "us")
DEFAULT_BOOKS = os.getenv(
    "ODDS_BOOKS",
    "draftkings,fanduel,betmgm,caesars,espnbet,bet365,pointsbet,betway",
)
DEFAULT_ODDS_FORMAT = os.getenv("ODDS_FORMAT", "american")
DEFAULT_DATE_FORMAT = os.getenv("DATE_FORMAT", "iso")

# Featured markets (ONLY valid for /sports/{sport}/odds)
DEFAULT_FEATURED_MARKETS = os.getenv(
    "ODDS_MARKETS",
    "h2h,spreads,totals",
)

# Player-prop markets (ONLY valid for /events/{eventId}/odds)
DEFAULT_PLAYER_MARKETS = os.getenv(
    "PLAYER_MARKETS",
    ",".join([
        "player_field_goals",
        "player_kicking_points",
        "player_pass_interceptions",
        "player_pass_rush_yds",
        "player_pass_tds",
        "player_pass_yds",
        "player_receptions",
        "player_reception_yds",
        "player_rush_attempts",
        "player_rush_yds",
        "player_sacks",
        "player_solo_tackles",
        "player_tackles_assists",
        "player_anytime_td",
    ])
)

DEFAULT_MODE = os.getenv("ODDS_MODE", "featured").strip().lower()
DEFAULT_MAX_EVENTS = int(os.getenv("ODDS_MAX_EVENTS", "2"))

REPO = Path(__file__).resolve().parents[1]
RAW_DIR = REPO / "data" / "odds" / "raw"
PROC_DIR = REPO / "data" / "odds" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def utc_stamp() -> str:
    import datetime as dt
    return dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def ensure_requests():
    if requests is None:
        print("ERROR: requests is not available; install it or use --dry-run.", file=sys.stderr)
        sys.exit(3)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def get_featured(api_key: str, sport: str, regions: str, markets: str, books: str) -> Any:
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "bookmakers": books,
        "oddsFormat": DEFAULT_ODDS_FORMAT,
        "dateFormat": DEFAULT_DATE_FORMAT,
    }
    url = API_BASE_SPORT.format(sport=sport)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def get_event_props(api_key: str, sport: str, event_id: str, regions: str, markets: str, books: str) -> Any:
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "bookmakers": books,
        "oddsFormat": DEFAULT_ODDS_FORMAT,
        "dateFormat": DEFAULT_DATE_FORMAT,
    }
    url = API_BASE_EVENT.format(sport=sport, event_id=event_id)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


# ---------- flatteners ----------
def flatten_featured(blob: dict) -> List[Dict[str, Any]]:
    """Flatten 1 featured-event object into rows."""
    rows: List[Dict[str, Any]] = []
    if not isinstance(blob, dict):
        return rows

    ev_id = blob.get("id", "")
    home = blob.get("home_team", "")
    away = blob.get("away_team", "")
    commence = blob.get("commence_time", "")
    sport_key = blob.get("sport_key", "")

    for bm in blob.get("bookmakers", []):
        if not isinstance(bm, dict):
            # Defensive: skip malformed entries
            continue
        book = bm.get("key", "")
        book_title = bm.get("title", "")
        last_update = bm.get("last_update", "")
        for market in bm.get("markets", []):
            if not isinstance(market, dict):
                continue
            mkey = market.get("key", "")
            for outc in market.get("outcomes", []):
                if not isinstance(outc, dict):
                    continue
                name = outc.get("name", "")
                price = outc.get("price")
                point = outc.get("point")
                rows.append({
                    "event_id": ev_id,
                    "sport_key": sport_key,
                    "commence_time_utc": commence,
                    "home_team": home,
                    "away_team": away,
                    "book": book,
                    "book_title": book_title,
                    "market": mkey,
                    "runner": name,
                    "price_american": price,
                    "point": point,
                    "last_update": last_update,
                })
    return rows


def flatten_player(event_id: str, blob: dict) -> List[Dict[str, Any]]:
    """
    Flatten per-event player props blob into rows.
    Defensive against non-dict bookmaker/market/outcome entries.
    """
    rows: List[Dict[str, Any]] = []
    if not isinstance(blob, dict):
        return rows

    for bm in blob.get("bookmakers", []):
        if not isinstance(bm, dict):
            # <<< FIX: guard against strings / malformed entries >>>
            print(f"[WARN] Skipping malformed bookmaker entry: {bm!r}")
            continue
        book = bm.get("key", "")
        book_title = bm.get("title", "")
        last_update = bm.get("last_update", "")

        for market in bm.get("markets", []):
            if not isinstance(market, dict):
                continue
            mkey = market.get("key", "")
            for outc in market.get("outcomes", []):
                if not isinstance(outc, dict):
                    continue
                name = outc.get("name", "")
                price = outc.get("price")
                point = outc.get("point")
                rows.append({
                    "event_id": event_id,
                    "book": book,
                    "book_title": book_title,
                    "market": mkey,
                    "runner": name,
                    "price_american": price,
                    "point": point,
                    "last_update": last_update,
                })
    return rows


# -----------------------------
# Main
# -----------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Fetch featured odds and/or player props from The Odds API.")
    p.add_argument("--sport", default=DEFAULT_SPORT)
    p.add_argument("--mode", default=DEFAULT_MODE, choices=["featured", "players", "both"])
    p.add_argument("--regions", default=DEFAULT_REGIONS)
    p.add_argument("--books", default=DEFAULT_BOOKS)
    p.add_argument("--featured-markets", default=DEFAULT_FEATURED_MARKETS)
    p.add_argument("--player-markets", default=DEFAULT_PLAYER_MARKETS)
    p.add_argument("--max-events", type=int, default=DEFAULT_MAX_EVENTS)
    p.add_argument("--sleep", type=float, default=0.0, help="sleep seconds between API calls")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key and not args.dry_run:
        print("ERROR: Missing ODDS_API_KEY", file=sys.stderr)
        return 2

    ts = utc_stamp()

    # Dry-run produces one tiny featured-like blob & one tiny per-event blob to verify plumbing.
    if args.dry_run:
        featured_sample = [{
            "id": "evt123",
            "sport_key": args.sport,
            "commence_time": "2025-09-07T17:00:00Z",
            "home_team": "Team A",
            "away_team": "Team B",
            "bookmakers": [{
                "key": "draftkings",
                "title": "DraftKings",
                "last_update": ts,
                "markets": [
                    {"key": "h2h", "outcomes": [{"name": "Team A", "price": -120}, {"name": "Team B", "price": +100}]}
                ],
            }],
        }]
        fr_json = RAW_DIR / f"featured_raw_{ts}.json"
        write_json(fr_json, featured_sample)
        rows = []
        for ev in featured_sample:
            rows.extend(flatten_featured(ev))
        write_csv(
            PROC_DIR / f"featured_{ts}.csv",
            rows,
            ["event_id","sport_key","commence_time_utc","home_team","away_team","book","book_title","market","runner","price_american","point","last_update"],
        )

        per_event_sample = {
            "bookmakers": [{
                "key": "draftkings",
                "title": "DraftKings",
                "last_update": ts,
                "markets": [
                    {"key": "player_receptions", "outcomes": [{"name": "Player X Over", "price": -115, "point": 5.5}]}
                ],
            }]
        }
        write_json(RAW_DIR / f"player_raw_evt123_{ts}.json", per_event_sample)
        print("[DRY] wrote featured + player samples")
        return 0

    ensure_requests()

    # 1) FEATURED
    if args.mode in ("featured", "both"):
        try:
            feat = get_featured(api_key, args.sport, args.regions, args.featured_markets, args.books)
        except requests.HTTPError as e:
            print("ERROR: Featured fetch failed:", str(e), file=sys.stderr)
            return 1

        # Write raw JSON
        raw_path = RAW_DIR / f"featured_raw_{ts}.json"
        write_json(raw_path, feat)

        # Flatten -> CSV
        all_rows: List[Dict[str, Any]] = []
        if isinstance(feat, list):
            for ev in feat:
                all_rows.extend(flatten_featured(ev))
        elif isinstance(feat, dict):
            all_rows.extend(flatten_featured(feat))

        csv_path = PROC_DIR / f"featured_{ts}.csv"
        write_csv(
            csv_path,
            all_rows,
            ["event_id","sport_key","commence_time_utc","home_team","away_team",
             "book","book_title","market","runner","price_american","point","last_update"],
        )
        print(f"[OK] wrote featured JSON -> {raw_path}")
        print(f"[OK] wrote featured CSV  -> {csv_path}")
        print(f"[OK] featured events: {len(feat) if isinstance(feat, list) else (1 if feat else 0)}")

    # 2) PLAYERS (per event)
    if args.mode in ("players", "both"):
        # If we didn't just fetch featured, try to get a small event list from featured endpoint
        try:
            feat_for_events = get_featured(api_key, args.sport, args.regions, "h2h", args.books)
        except Exception as e:
            print("[WARN] Could not fetch event list for player props; skipping. Reason:", e, file=sys.stderr)
            feat_for_events = []

        # Build a list of event ids
        event_ids: List[str] = []
        if isinstance(feat_for_events, list):
            for ev in feat_for_events:
                if isinstance(ev, dict) and ev.get("id"):
                    event_ids.append(str(ev["id"]))
        event_ids = event_ids[: max(0, args.max_events)]

        if not event_ids:
            print("[WARN] No events available for player props (event list empty).")
            # Not fatal — we still may have run featured above.
            return 0

        # Query per-event props (limited by --max-events)
        for idx, ev_id in enumerate(event_ids, start=1):
            try:
                blob = get_event_props(
                    api_key=api_key,
                    sport=args.sport,
                    event_id=ev_id,
                    regions=args.regions,
                    markets=args.player_markets,
                    books=args.books,
                )
            except requests.HTTPError as e:
                print(f"ERROR: Player props request failed for event {ev_id}: {e}", file=sys.stderr)
                continue

            # Write the raw per-event player JSON
            out_json = RAW_DIR / f"player_raw_{ev_id}_{ts}.json"
            write_json(out_json, blob)
            print(f"[OK] wrote player JSON for event {idx}/{len(event_ids)} -> {out_json}")

            # (Optional) If you want a quick CSV preview of rows, uncomment below:
            # rows = flatten_player(ev_id, blob)
            # if rows:
            #     out_csv = PROC_DIR / f"player_{ev_id}_{ts}.csv"
            #     write_csv(out_csv, rows, ["event_id","book","book_title","market","runner","price_american","point","last_update"])
            #     print(f"[OK] wrote player CSV -> {out_csv}")

            # Be courteous to the API
            if args.sleep > 0:
                time.sleep(args.sleep)

    return 0


if __name__ == "__main__":
    sys.exit(main())
