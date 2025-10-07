#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch NFL odds (incl. supported player props) from The Odds API and normalize to CSV.

Defaults (safe for The Odds API and your requirements):
  Markets = h2h, spreads, totals,
            player_passing_yards, player_rushing_yards, player_receiving_yards,
            player_receptions, player_pass_tds, player_interceptions

Override markets via:
  - env ODDS_MARKETS="comma,separated,list"
  - or CLI --markets "comma,separated,list"

Inputs:
  - Env (required unless --dry-run): ODDS_API_KEY
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

# ---------- Defaults (restricted to known-valid markets) ----------
DEFAULT_SPORT = "americanfootball_nfl"
DEFAULT_REGIONS = "us"

_VALID_PLAYER_MARKETS = [
    "player_passing_yards",
    "player_rushing_yards",
    "player_receiving_yards",
    "player_receptions",
    "player_pass_tds",
    "player_interceptions",
]

DEFAULT_MARKETS = ",".join(["h2h", "spreads", "totals", *_VALID_PLAYER_MARKETS])

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

def _effective_markets(cli_markets: Optional[str]) -> str:
    # Priority: CLI --markets > env ODDS_MARKETS > DEFAULT_MARKETS
    m = (cli_markets or os.getenv("ODDS_MARKETS") or DEFAULT_MARKETS).strip()
    parts = [p.strip() for p in m.split(",") if p.strip()]
    return ",".join(parts)

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
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        body = ""
        try:
            body = e.response.text[:500]
        except Exception:
            pass
        print(f"Error:  HTTP {status} from Odds API", file=sys.stderr)
        print(f"Error:  URL: {resp.url}", file=sys.stderr)
        if status == 422:
            print("Error:  The provider rejected one or more requested 'markets'.", file=sys.stderr)
            print(f"Error:  Markets sent: {markets}", file=sys.stderr)
            print(f"Error:  Response (truncated): {body}", file=sys.stderr)
            print("[HINT] Keep to known-valid markets or set env ODDS_MARKETS to a smaller/edited list.", file=sys.stderr)
        raise
    time.sleep(sleep_between)
    return resp.json()

def normalize(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
    p = argparse.ArgumentParser(description="Fetch and normalize NFL markets & supported player props.")
    p.add_argument("--sport", default=DEFAULT_SPORT)
    p.add_argument("--regions", default=DEFAULT_REGIONS)
    p.add_argument("--markets", default=None, help="Comma list; overrides env ODDS_MARKETS/DEFAULT_MARKETS")
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

    markets = _effective_markets(args.markets)
    ts = now_stamp()

    if args.dry_run:
        sample = [{
            "id":"sample","sport_key":args.sport,"commence_time":"2025-09-10T00:20:00Z",
            "home_team":"A","away_team":"B","bookmakers":[{"key":"draftkings","title":"DraftKings",
            "last_update":ts,"markets":[
                {"key":"player_passing_yards","outcomes":[{"name":"Some QB","price":-110,"point":255.5}]},
                {"key":"player_receptions","outcomes":[{"name":"Some WR","price":-115,"point":5.5}]}
            ]}]}]
        raw = RAW_DIR / f"odds_raw_{ts}.json"
        write_json(raw, sample)
        rows = normalize(sample)
        csv_path = PROC_DIR / f"odds_{ts}.csv"
        write_csv(csv_path, rows, list(rows[0].keys()))
        print("Wrote (dry-run):", raw, "and", csv_path)
        print("Markets (dry-run):", markets)
        return 0

    if requests is None:
        print("ERROR: requests not available (no --dry-run).", file=sys.stderr)
        return 3

    data = fetch_odds(api_key=api_key, sport=args.sport, regions=args.regions,
                      markets=markets, odds_format=args.odds_format,
                      date_format=args.date_format, books=args.books,
                      sleep_between=args.sleep)

    raw_path = RAW_DIR / f"odds_raw_{ts}.json"
    write_json(raw_path, data)

    rows = normalize(data)
    fields = ["game_id","sport_key","commence_time_utc","home_team","away_team",
              "book","book_title","market","runner","price_american","point","last_update"]
    csv_path = PROC_DIR / f"odds_{ts}.csv"
    write_csv(csv_path, rows, fields if rows else fields)  # write header even if empty

    print("Markets:", markets)
    print("Wrote raw JSON:", raw_path)
    print("Wrote CSV:     ", csv_path, f"(rows={len(rows)})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
