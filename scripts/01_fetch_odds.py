#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch NFL featured markets AND per-event player props, then write:
  - data/odds/raw/featured_raw_<UTCYYYYMMDD_HHMMSS>.json
  - data/odds/processed/featured_<UTCYYYYMMDD_HHMMSS>.csv
  - data/odds/raw/player_raw_<EVENTID>.json    (one file per event with any player props)
  - data/odds/processed/player_<UTCYYYYMMDD_HHMMSS>.csv  (flattened rows; optional)

Env:
  ODDS_API_KEY        (required)
  FEATURED_MARKETS    default: "h2h,spreads,totals"
  PLAYER_MARKETS      comma list of valid player markets, e.g.
                      "player_pass_yds,player_rush_yds,player_reception_yds,player_receptions,player_pass_tds,player_interceptions"
  REGIONS             default: "us"
  BOOKMAKERS          default: "draftkings,fanduel,betmgm,caesars,espnbet,bet365,pointsbet,betway"
  ODDS_FORMAT         default: "american"
  DATE_FORMAT         default: "iso"
  DRY_RUN             default: "0" (set "1" to avoid network calls)

This script is intentionally self-contained to avoid “missing script” issues.
"""

import os, sys, json, time, csv
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime

try:
    import requests
except Exception:
    requests = None

API_SPORT = "americanfootball_nfl"
API_BASE_FEATURED = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
API_BASE_EVENT = "https://api.the-odds-api.com/v4/sports/{sport}/events/{event_id}/odds"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "odds" / "raw"
PROC_DIR = ROOT / "data" / "odds" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

def now_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

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

def env(name: str, default: Optional[str]=None) -> str:
    v = os.getenv(name)
    return v if v is not None and v != "" else (default or "")

def must_env(name: str) -> str:
    v = env(name, "")
    if not v:
        print(f"ERROR: missing env {name}", file=sys.stderr)
        sys.exit(2)
    return v

def http_get(url: str, params: Dict[str, str]) -> Any:
    resp = requests.get(url, params=params, timeout=45)
    # If provider rejects markets, raise now so caller can report clearly.
    resp.raise_for_status()
    return resp.json()

def flatten_featured(json_blob: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ev in json_blob:
        ev_id = ev.get("id", "")
        sport_key = ev.get("sport_key", "")
        commence = ev.get("commence_time", "")
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        for bm in ev.get("bookmakers", []):
            book = bm.get("key", "")
            book_title = bm.get("title", "")
            last_update = bm.get("last_update", "")
            for m in bm.get("markets", []):
                mkey = m.get("key", "")
                for out in m.get("outcomes", []):
                    rows.append({
                        "event_id": ev_id,
                        "sport_key": sport_key,
                        "commence_time_utc": commence,
                        "home_team": home,
                        "away_team": away,
                        "book": book,
                        "book_title": book_title,
                        "market": mkey,
                        "runner": out.get("name", ""),
                        "price_american": out.get("price", None),
                        "point": out.get("point", None),
                        "last_update": last_update,
                    })
    return rows

def flatten_player(ev_id: str, json_blob: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for bm in json_blob or []:
        book = bm.get("key", "")
        book_title = bm.get("title", "")
        last_update = bm.get("last_update", "")
        for m in bm.get("markets", []):
            mkey = m.get("key", "")
            for out in m.get("outcomes", []):
                rows.append({
                    "event_id": ev_id,
                    "book": book,
                    "book_title": book_title,
                    "market": mkey,
                    "runner": out.get("name",""),
                    "price_american": out.get("price", None),
                    "point": out.get("point", None),
                    "last_update": last_update,
                })
    return rows

def fetch_featured(api_key: str, markets: str, regions: str, odds_format: str, date_format: str, books: str, dry: bool) -> List[Dict[str, Any]]:
    if dry:
        # tiny fake sample with one event
        return [{
            "id": "sample_ev",
            "sport_key": API_SPORT,
            "commence_time": "2025-09-10T00:20:00Z",
            "home_team": "Home",
            "away_team": "Away",
            "bookmakers": [{
                "key": "draftkings", "title": "DraftKings", "last_update": now_stamp(),
                "markets": [{"key":"h2h","outcomes":[{"name":"Home","price":-120},{"name":"Away","price":+100}]}]
            }]
        }]
    url = API_BASE_FEATURED.format(sport=API_SPORT)
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
        "bookmakers": books
    }
    return http_get(url, params)

def fetch_player_for_event(api_key: str, event_id: str, markets: str, regions: str, odds_format: str, date_format: str, books: str, dry: bool) -> List[Dict[str, Any]]:
    if dry:
        return [{
            "key":"draftkings","title":"DraftKings","last_update":now_stamp(),
            "markets":[{"key":"player_pass_yds","outcomes":[{"name":"QB A","price":-115,"point":250.5},{"name":"QB A","price":-105,"point":250.5}]}]
        }]
    url = API_BASE_EVENT.format(sport=API_SPORT, event_id=event_id)
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
        "bookmakers": books
    }
    return http_get(url, params)

def main() -> int:
    if requests is None:
        print("ERROR: requests not available", file=sys.stderr)
        return 3

    api_key = must_env("ODDS_API_KEY")
    featured_markets = env("FEATURED_MARKETS", "h2h,spreads,totals")
    player_markets   = env("PLAYER_MARKETS", "")
    regions    = env("REGIONS", "us")
    books      = env("BOOKMAKERS", "draftkings,fanduel,betmgm,caesars,espnbet,bet365,pointsbet,betway")
    odds_fmt   = env("ODDS_FORMAT", "american")
    date_fmt   = env("DATE_FORMAT", "iso")
    dry_run    = env("DRY_RUN", "0") == "1"

    ts = now_stamp()

    # ---------------- Featured markets ----------------
    try:
        featured = fetch_featured(api_key, featured_markets, regions, odds_fmt, date_fmt, books, dry_run)
    except requests.HTTPError as e:
        print("ERROR: Featured request failed:", str(e), file=sys.stderr)
        print("HINT: Keep FEATURED_MARKETS to valid values: h2h,spreads,totals", file=sys.stderr)
        return 1

    raw_featured_path = RAW_DIR / f"featured_raw_{ts}.json"
    write_json(raw_featured_path, featured)
    feat_rows = flatten_featured(featured)
    feat_csv_path = PROC_DIR / f"featured_{ts}.csv"
    write_csv(feat_csv_path, feat_rows, [
        "event_id","sport_key","commence_time_utc","home_team","away_team",
        "book","book_title","market","runner","price_american","point","last_update"
    ])
    print(f"[OK] wrote featured JSON -> {raw_featured_path}")
    print(f"[OK] wrote featured CSV  -> {feat_csv_path}")
    print(f"[OK] featured events: {len({r['event_id'] for r in feat_rows})}")

    # collect unique event ids
    event_ids = sorted({ev.get("id","") for ev in featured if ev.get("id")})
    if not event_ids:
        print("[WARN] No event IDs discovered from featured feed; skipping player props.")
        return 0

    if not player_markets.strip():
        print("[INFO] PLAYER_MARKETS env empty; skipping player props.")
        return 0

    # ---------------- Player props per event ----------------
    total_player_rows = 0
    all_player_rows: List[Dict[str, Any]] = []

    for ev_id in event_ids:
        try:
            blob = fetch_player_for_event(api_key, ev_id, player_markets, regions, odds_fmt, date_fmt, books, dry_run)
        except requests.HTTPError as e:
            # If markets invalid, fail fast once with a useful message.
            print(f"ERROR: Player props request failed for event {ev_id}: {e}", file=sys.stderr)
            print("HINT: Ensure PLAYER_MARKETS contains only valid player markets.", file=sys.stderr)
            return 1

        # write raw per event
        raw_p = RAW_DIR / f"player_raw_{ev_id}.json"
        write_json(raw_p, blob)
        rows = flatten_player(ev_id, blob)
        total_player_rows += len(rows)
        all_player_rows.extend(rows)

    # optional convenience CSV of all player rows (cache remains per-event JSON)
    ply_csv_path = PROC_DIR / f"player_{ts}.csv"
    write_csv(ply_csv_path, all_player_rows, [
        "event_id","book","book_title","market","runner","price_american","point","last_update"
    ])

    print(f"[OK] wrote {len(event_ids)} player_raw_*.json; total rows={total_player_rows}")
    print(f"[OK] wrote player CSV -> {ply_csv_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
