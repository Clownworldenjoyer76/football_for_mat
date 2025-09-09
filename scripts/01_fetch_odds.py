#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full script: /mnt/data/football_for_mat-main/scripts/01_fetch_odds.py

Fetch NFL odds from The Odds API and normalize to CSV.

Inputs:
  - Environment var: ODDS_API_KEY
  - CLI args: --sport, --regions, --markets, --books, --odds-format, --date-format, --sleep, --dry-run

Outputs:
  - data/odds/raw/odds_raw_<UTCYYYYMMDD_HHMMSS>.json
  - data/odds/processed/odds_<UTCYYYYMMDD_HHMMSS>.csv
"""
import os
import sys
import json
import time
import argparse
import datetime as dt
from typing import List, Dict, Any, Optional
import csv
from pathlib import Path

try:
    import requests
except Exception:
    requests = None

DEFAULT_SPORT = "americanfootball_nfl"
DEFAULT_REGIONS = "us"
DEFAULT_MARKETS = "h2h,spreads,totals"
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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def fetch_odds(api_key: str,
               sport: str,
               regions: str,
               markets: str,
               odds_format: str,
               date_format: str,
               books: Optional[str],
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
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(sleep_between)
    return resp.json()


def normalize(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ev in data:
        game_id = ev.get("id", "")
        sport_key = ev.get("sport_key", "")
        commence_time = ev.get("commence_time", "")
        home_team = ev.get("home_team", "")
        away_team = ev.get("away_team", "")
        for bm in ev.get("bookmakers", []):
            book = bm.get("key", "")
            book_title = bm.get("title", "")
            last_update = bm.get("last_update", "")
            for mark in bm.get("markets", []):
                market_key = mark.get("key", "")
                for outcome in mark.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = outcome.get("price", None)
                    point = outcome.get("point", None)
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
    p = argparse.ArgumentParser(description="Fetch and normalize NFL odds.")
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
        print("ERROR: Missing environment variable ODDS_API_KEY", file=sys.stderr)
        return 2

    ts = now_stamp()

    if args.dry_run:
        sample = [
            {
                "id": "sample123",
                "sport_key": args.sport,
                "commence_time": "2025-09-10T00:20:00Z",
                "home_team": "Philadelphia Eagles",
                "away_team": "Dallas Cowboys",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "title": "DraftKings",
                        "last_update": ts,
                        "markets": [
                            {"key": "h2h", "outcomes": [
                                {"name": "Philadelphia Eagles", "price": -160},
                                {"name": "Dallas Cowboys", "price": +140}
                            ]},
                            {"key": "spreads", "outcomes": [
                                {"name": "Philadelphia Eagles", "price": -110, "point": -3.5},
                                {"name": "Dallas Cowboys", "price": -110, "point": +3.5}
                            ]},
                            {"key": "totals", "outcomes": [
                                {"name": "Over", "price": -105, "point": 47.5},
                                {"name": "Under", "price": -115, "point": 47.5}
                            ]}
                        ]
                    }
                ]
            }
        ]
        raw_path = RAW_DIR / f"odds_raw_{ts}.json"
        write_json(raw_path, sample)
        rows = normalize(sample)
        fields = [
            "game_id","sport_key","commence_time_utc","home_team","away_team",
            "book","book_title","market","runner","price_american","point","last_update"
        ]
        csv_path = PROC_DIR / f"odds_{ts}.csv"
        write_csv(csv_path, rows, fields)
        print(f"Wrote (dry-run) raw JSON: {raw_path}")
        print(f"Wrote (dry-run) CSV:      {csv_path}")
        return 0

    if requests is None:
        print("ERROR: requests not available and --dry-run not set", file=sys.stderr)
        return 3

    data = fetch_odds(
        api_key=api_key,
        sport=args.sport,
        regions=args.regions,
        markets=args.markets,
        odds_format=args.odds_format,
        date_format=args.date_format,
        books=args.books,
        sleep_between=args.sleep
    )

    raw_path = RAW_DIR / f"odds_raw_{ts}.json"
    write_json(raw_path, data)

    rows = normalize(data)
    fields = [
        "game_id","sport_key","commence_time_utc","home_team","away_team",
        "book","book_title","market","runner","price_american","point","last_update"
    ]
    csv_path = PROC_DIR / f"odds_{ts}.csv"
    write_csv(csv_path, rows, fields)

    print(f"Wrote raw JSON: {raw_path}")
    print(f"Wrote CSV:      {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
