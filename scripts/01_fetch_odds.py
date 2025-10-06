#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch NFL odds (including player props) from The Odds API and normalize to CSV.

Inputs:
  - Env: ODDS_API_KEY  (required unless --dry-run)
  - Optional CLI: --sport --regions --markets --books --odds-format --date-format --sleep --dry-run

Outputs:
  - data/odds/raw/odds_raw_<UTCYYYYMMDD_HHMMSS>.json
  - data/odds/processed/odds_<UTCYYYYMMDD_HHMMSS>.csv
"""

from __future__ import annotations
import os, sys, json, time, argparse, datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv

try:
    import requests
except Exception:
    requests = None

# ---------- Defaults ----------
DEFAULT_SPORT = "americanfootball_nfl"
DEFAULT_REGIONS = "us"
# Include core game markets + all requested player markets
DEFAULT_MARKETS = ",".join([
    "h2h", "spreads", "totals",
    # player props (your list)
    "player_pass_yards",
    "player_rush_yards",
    "player_rec_yards",
    "player_receptions",
    "player_pass_tds",
    "player_interceptions",
    # optional if available at your provider:
    "player_touchdowns",
    "player_sacks",
    "player_tackles",
    "player_tackle_assists",
    "player_field_goals_made",
])
DEFAULT_ODDS_FORMAT = "american"
DEFAULT_DATE_FORMAT = "iso"
DEFAULT_BOOKS = "draftkings,fanduel,betmgm,caesars,espnbet,bet365,pointsbet,betway"

API_BASE = "https://api.the-odds-api.com/v4/sports/{sport}/odds"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "odds" / "raw"
PROC_DIR = ROOT / "data" / "odds" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

def now_stamp() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

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

def fetch_odds(api_key: str, sport: str, regions: str, markets: str,
               odds_format: str, date_format: str, books: Optional[str],
               sleep_between: float = 0.0) -> List[Dict[str, Any]]:
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }
    if books:
        params["bookmakers"] = books
    url = API_BASE.format(sport=sport)
    resp = requests.get(url, params=params, timeout=45)
    resp.raise_for_status()
    time.sleep(sleep_between)
    return resp.json()

def normalize(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Flatten Odds API JSON into a single table. We keep a generic schema that
    works for both game and player markets.
    """
    rows: List[Dict[str, Any]] = []
    for ev in events:
        game_id = ev.get("id", "")
        sport_key = ev.get("sport_key", "")
        commence_time = ev.get("commence_time", "")
        home_team = ev.get("home_team", "")
        away_team = ev.get("away_team", "")

        for bm in ev.get("bookmakers", []) or []:
            book = bm.get("key", "")
            book_title = bm.get("title", "")
            last_update = bm.get("last_update", "")
            for m in bm.get("markets", []) or []:
                market_key = m.get("key", "")
                for oc in m.get("outcomes", []) or []:
                    # For player props, 'name' is typically the player or 'Over'/'Under'
                    name = oc.get("name", "")
                    price = oc.get("price", None)
                    point = oc.get("point", None)
                    rows.append({
                        "game_id": game_id,
                        "sport_key": sport_key,
                        "commence_time_utc": commence_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "book": book,
                        "book_title": book_title,
                        "market": market_key,
                        "runner": name,
                        "price_american": price,
                        "point": point,
                        "last_update": last_update,
                    })
    return rows

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Fetch and normalize NFL markets & player props.")
    p.add_argument("--sport", default=DEFAULT_SPORT)
    p.add_argument("--regions", default=DEFAULT_REGIONS)
    p.add_argument("--markets", default=DEFAULT_MARKETS)
    p.add_argument("--books", default=DEFAULT_BOOKS)
    p.add_argument("--odds-format", default=DEFAULT_ODDS_FORMAT)
    p.add_argument("--date-format", default=DEFAULT_DATE_FORMAT)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key and not args.dry_run:
        print("ERROR: Missing env ODDS_API_KEY", file=sys.stderr)
        return 2

    ts = now_stamp()

    if args.dry_run:
        sample = [{"id":"sample","sport_key":args.sport,"commence_time":"2025-09-10T00:20:00Z",
                   "home_team":"A","away_team":"B","bookmakers":[{"key":"draftkings","title":"DraftKings",
                   "last_update":ts,"markets":[
                       {"key":"player_pass_yards","outcomes":[{"name":"Some QB","price":-110,"point":255.5}]},
                       {"key":"player_receptions","outcomes":[{"name":"Some WR","price":-115,"point":5.5}]}
                   ]}]}]
        raw = RAW_DIR / f"odds_raw_{ts}.json"
        write_json(raw, sample)
        rows = normalize(sample)
        csv_path = PROC_DIR / f"odds_{ts}.csv"
        write_csv(csv_path, rows, list(rows[0].keys()))
        print("Wrote (dry-run):", raw, "and", csv_path)
        return 0

    if requests is None:
        print("ERROR: requests not available (no --dry-run).", file=sys.stderr)
        return 3

    data = fetch_odds(api_key=api_key, sport=args.sport, regions=args.regions,
                      markets=args.markets, odds_format=args.odds_format,
                      date_format=args.date_format, books=args.books,
                      sleep_between=args.sleep)

    raw_path = RAW_DIR / f"odds_raw_{ts}.json"
    write_json(raw_path, data)

    rows = normalize(data)
    if not rows:
        print("WARN: No rows returned from provider.", file=sys.stderr)

    fields = ["game_id","sport_key","commence_time_utc","home_team","away_team",
              "book","book_title","market","runner","price_american","point","last_update"]
    csv_path = PROC_DIR / f"odds_{ts}.csv"
    if rows:
        write_csv(csv_path, rows, fields)
    else:
        # still write an empty file with header for debugging
        write_csv(csv_path, [], fields)

    print("Wrote raw JSON:", raw_path)
    print("Wrote CSV:     ", csv_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
