#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_fetch_odds.py
Fetch NFL odds from The Odds API.

What this script does
---------------------
1) FEATURED MARKETS (single request)
   - Calls:   /v4/sports/{sport}/odds
   - Markets: h2h, spreads, totals (configurable)
   - Writes:  data/odds/raw/featured_raw_<UTCSTAMP>.json

2) PLAYER MARKETS (per-event, capped)
   - Calls:   /v4/sports/{sport}/events  (to list upcoming events)
              /v4/sports/{sport}/events/{eventId}/odds  (for each event)
   - Markets: configurable list of player_* markets
   - Caps:    max events (ODDS_MAX_EVENTS) + limited bookmakers
   - Writes:  data/odds/raw/player_raw_<UTCSTAMP>.json  (one aggregated JSON blob)
   - ALSO:    Builds data/props/props_current.csv from the fetched player markets

Environment variables
---------------------
ODDS_API_KEY            (required unless DRY_RUN=1)
TARGET_SEASON           (defaults to current UTC year if unset)
SPORT                   (default: americanfootball_nfl)
ODDS_FEATURED_MARKETS   (default: "h2h,spreads,totals")
ODDS_PLAYER_MARKETS     (default: see DEFAULT_PLAYER_MARKETS below)
ODDS_BOOKMAKERS         (default: "draftkings")  # keep small to save calls
ODDS_REGIONS            (default: "us")
ODDS_ODDS_FORMAT        (default: "american")
ODDS_DATE_FORMAT        (default: "iso")
ODDS_MAX_EVENTS         (default: "8")           # cap per-event fetches
ODDS_SLEEP_MS           (default: "250")         # throttle between requests
DRY_RUN                 (default: "0")           # 1 = no network; writes sample

Outputs
-------
data/odds/raw/featured_raw_<UTCYYYYMMDD_HHMMSS>.json
data/odds/raw/player_raw_<UTCYYYYMMDD_HHMMSS>.json
data/props/props_current.csv
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import datetime as dt
from typing import Any, Dict, List, Optional
from pathlib import Path
import csv

try:
    import requests
except Exception:
    requests = None

# ---------- Defaults ----------
SPORT_DEFAULT = "americanfootball_nfl"
FEATURED_DEFAULT = "h2h,spreads,totals"
DEFAULT_PLAYER_MARKETS = ",".join([
    "player_field_goals_made",
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

DEFAULT_BOOKMAKERS = "draftkings"
DEFAULT_REGIONS = "us"
DEFAULT_ODDS_FORMAT = "american"
DEFAULT_DATE_FORMAT = "iso"
DEFAULT_MAX_EVENTS = 8
DEFAULT_SLEEP_MS = 250

API_BASE = "https://api.the-odds-api.com/v4"

# ---------- Paths ----------
REPO = Path(__file__).resolve().parents[1]
RAW_DIR = REPO / "data" / "odds" / "raw"
PROC_DIR = REPO / "data" / "odds" / "processed"
PROPS_DIR = REPO / "data" / "props"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)
PROPS_DIR.mkdir(parents=True, exist_ok=True)

PROPS_CSV = PROPS_DIR / "props_current.csv"

# Internal mapping → your strict features/props names
PLAYER_MARKET_MAP = {
    "player_pass_yds":        "qb_passing_yards",
    "player_rush_yds":        "rb_rushing_yards",
    "player_reception_yds":   "wr_rec_yards",
    "player_receptions":      "wrte_receptions",
    "player_pass_tds":        "qb_passing_tds",
    "player_pass_interceptions": "qb_interceptions",
    "player_anytime_td":      "player_tds",
    "player_sacks":           "player_sacks",
    "player_solo_tackles":    "player_tackles",          # alias down to tackles
    "player_tackles_assists": "player_tackles_assists",
    "player_field_goals_made":"player_field_goals_made",
    "player_kicking_points":  "player_kicking_points",
    "player_pass_rush_yds":   "player_pass_rush_yds",
    "player_rush_attempts":   "player_rush_attempts",
}

PROPS_FIELDS = [
    "season","week","game_id","event_id",
    "commence_time_utc","home_team","away_team",
    "market","runner","team","opponent",
    "book","book_title","line","odds_over","odds_under","last_update"
]

# ---------- Helpers ----------
def now_stamp() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def this_season() -> int:
    return dt.datetime.utcnow().year

def sleep_ms(ms: int) -> None:
    time.sleep(max(ms, 0) / 1000.0)

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

def env_int(name: str, default: int) -> int:
    v = (os.getenv(name) or "").strip()
    return int(v) if v.isdigit() else default

def http_get(url: str, params: Dict[str, Any]) -> Any:
    if requests is None:
        raise RuntimeError("requests not available")
    r = requests.get(url, params=params, timeout=30)
    try:
        r.raise_for_status()
    except Exception as e:
        # show helpful diagnostic
        print("Error: HTTP", getattr(r, "status_code", "?"), "from Odds API", file=sys.stderr)
        print("Error: URL:", r.url, file=sys.stderr)
        try:
            print("Error: Response:", r.text[:800], file=sys.stderr)
        except Exception:
            pass
        raise
    return r.json()

# ---------- Normalizers ----------
def normalize_player_event(event_blob: Dict[str, Any], bookmakers_limit: Optional[List[str]], season: int) -> List[Dict[str, Any]]:
    """
    Convert one events/{id}/odds payload into our props rows.
    """
    rows: List[Dict[str, Any]] = []

    ev = event_blob.get("event") or {}
    odds = event_blob.get("odds") or {}

    eid = ev.get("id")
    home, away = ev.get("home_team"), ev.get("away_team")
    commence = ev.get("commence_time")

    bms = odds.get("bookmakers") or []
    for bm in bms:
        book = bm.get("key")
        if bookmakers_limit and book not in bookmakers_limit:
            continue
        book_title = bm.get("title")
        last_update = bm.get("last_update")

        for market in bm.get("markets") or []:
            raw_mk = market.get("key")
            mapped = PLAYER_MARKET_MAP.get(raw_mk)
            if not mapped:
                continue

            # Group o/u by (player, line)
            grouped: Dict[str, Dict[str, Any]] = {}
            for out in market.get("outcomes") or []:
                player = out.get("description") or out.get("name")
                if not player:
                    continue
                line = out.get("point")
                nm = (out.get("name") or "").lower()
                k = f"{player}|{line}"
                rec = grouped.setdefault(k, {"runner": player, "line": line, "over": None, "under": None, "price": None})
                # Some markets provide Over/Under, some provide a single "price"
                if nm == "over":
                    rec["over"] = out.get("price")
                elif nm == "under":
                    rec["under"] = out.get("price")
                else:
                    # single sided (e.g., any time TD might be one price)
                    rec["price"] = out.get("price")

            for rec in grouped.values():
                odds_over = rec["over"]
                odds_under = rec["under"]
                if odds_over is None and odds_under is None:
                    odds_over = rec["price"]

                rows.append({
                    "season": season, "week": None,
                    "game_id": eid, "event_id": eid,
                    "commence_time_utc": commence,
                    "home_team": home, "away_team": away,
                    "market": mapped, "runner": rec["runner"],
                    "team": None, "opponent": None,
                    "book": book, "book_title": book_title,
                    "line": rec["line"],
                    "odds_over": odds_over, "odds_under": odds_under,
                    "last_update": last_update,
                })
    return rows

# ---------- Main fetchers ----------
def fetch_featured(api_key: str, sport: str, markets: str, regions: str,
                   odds_format: str, date_format: str,
                   bookmakers: Optional[str]) -> Dict[str, Any]:
    url = f"{API_BASE}/sports/{sport}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    return http_get(url, params)

def fetch_events(api_key: str, sport: str, date_format: str) -> List[Dict[str, Any]]:
    url = f"{API_BASE}/sports/{sport}/events"
    params = {"apiKey": api_key, "dateFormat": date_format}
    return http_get(url, params)

def fetch_event_player_odds(api_key: str, sport: str, event_id: str,
                            markets: str, regions: str,
                            odds_format: str, date_format: str,
                            bookmakers: Optional[str]) -> Dict[str, Any]:
    url = f"{API_BASE}/sports/{sport}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    return http_get(url, params)

# ---------- CLI ----------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Fetch Odds API featured + player markets.")
    p.add_argument("--sport", default=os.getenv("SPORT", SPORT_DEFAULT))
    args = p.parse_args(argv)

    dry_run = (os.getenv("DRY_RUN", "0").strip() == "1")
    api_key = (os.getenv("ODDS_API_KEY") or "").strip()
    if not api_key and not dry_run:
        print("ERROR: Missing ODDS_API_KEY", file=sys.stderr)
        return 2

    target_season = os.getenv("TARGET_SEASON")
    try:
        season = int(target_season) if target_season else this_season()
    except Exception:
        season = this_season()

    featured_markets = (os.getenv("ODDS_FEATURED_MARKETS") or FEATURED_DEFAULT).replace(" ", "")
    player_markets = (os.getenv("ODDS_PLAYER_MARKETS") or DEFAULT_PLAYER_MARKETS).replace(" ", "")
    bookmakers = (os.getenv("ODDS_BOOKMAKERS") or DEFAULT_BOOKMAKERS).replace(" ", "")
    regions = (os.getenv("ODDS_REGIONS") or DEFAULT_REGIONS).strip()
    odds_fmt = (os.getenv("ODDS_ODDS_FORMAT") or DEFAULT_ODDS_FORMAT).strip()
    date_fmt = (os.getenv("ODDS_DATE_FORMAT") or DEFAULT_DATE_FORMAT).strip()
    max_events = env_int("ODDS_MAX_EVENTS", DEFAULT_MAX_EVENTS)
    sleep_ms_between = env_int("ODDS_SLEEP_MS", DEFAULT_SLEEP_MS)

    ts = now_stamp()

    # ---------------- Featured markets ----------------
    if dry_run:
        featured_blob = [{"id": "EVT_SAMPLE", "home_team": "Team A", "away_team": "Team B", "bookmakers": []}]
    else:
        featured_blob = fetch_featured(
            api_key=api_key,
            sport=args.sport,
            markets=featured_markets,
            regions=regions,
            odds_format=odds_fmt,
            date_format=date_fmt,
            bookmakers=bookmakers,
        )
    feat_path = RAW_DIR / f"featured_raw_{ts}.json"
    write_json(feat_path, featured_blob)
    print(f"[featured] wrote {feat_path}")

    # ---------------- Player markets (per-event; capped) ----------------
    player_rows: List[Dict[str, Any]] = []
    player_agg: Dict[str, Any] = {"events": []}

    if dry_run:
        sample_event = {"id": "EVT_SAMPLE", "home_team": "Team A", "away_team": "Team B", "commence_time": f"{dt.datetime.utcnow():%Y-%m-%dT%H:%M:%SZ}"}
        player_agg["events"].append({
            "event": sample_event,
            "odds": {"bookmakers": []}
        })
    else:
        events = fetch_events(api_key=api_key, sport=args.sport, date_format=date_fmt)
        # Cap to save tokens
        events = events[:max_events]
        allow_books = [b.strip() for b in bookmakers.split(",")] if bookmakers else None

        for i, ev in enumerate(events, 1):
            eid = ev.get("id")
            if not eid:
                continue
            try:
                blob = fetch_event_player_odds(
                    api_key=api_key, sport=args.sport, event_id=eid,
                    markets=player_markets, regions=regions,
                    odds_format=odds_fmt, date_format=date_fmt,
                    bookmakers=bookmakers
                )
                # The Odds API returns just bookmakers/markets; attach event basics we already know
                enriched = {"event": ev, "odds": blob}
                player_agg["events"].append(enriched)
                # normalize to props rows
                player_rows.extend(normalize_player_event(enriched, allow_books, season))
                print(f"[player] {i}/{len(events)} event_id={eid} rows+={len(player_rows)}")
                sleep_ms(sleep_ms_between)
            except Exception as e:
                print(f"[player:WARN] failed event {eid}: {e}", file=sys.stderr)

    player_path = RAW_DIR / f"player_raw_{ts}.json"
    write_json(player_path, player_agg)
    print(f"[player] wrote {player_path} (events={len(player_agg.get('events', []))})")

    # ---------------- Write props_current.csv ----------------
    write_csv(PROPS_CSV, player_rows, PROPS_FIELDS)
    print(f"[props] wrote {PROPS_CSV} rows={len(player_rows)} season={season}")

    # Also drop a tiny processed marker for visibility / debugging
    proc_csv = PROC_DIR / f"odds_summary_{ts}.csv"
    write_csv(proc_csv, [], ["placeholder"])
    print(f"[proc] wrote {proc_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
