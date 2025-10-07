#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_fetch_odds.py  — Featured markets fetch (event-level, NOT player props)

Fetches featured markets (h2h, spreads, totals) from The Odds API
via /v4/sports/{sport}/odds.

ENV (recommended):
  ODDS_API_KEY        -> required
  ODDS_SPORT          -> default: americanfootball_nfl
  ODDS_REGIONS        -> default: us
  ODDS_MARKETS        -> default: h2h,spreads,totals   (player props NOT allowed here)
  ODDS_BOOKS          -> default: draftkings,fanduel,betmgm,caesars,espnbet,bet365,pointsbet,betway
  ODDS_FORMAT         -> default: american
  ODDS_DATE_FORMAT    -> default: iso
  ODDS_SLEEP_SECS     -> default: 0.0 (between requests)

Outputs:
  data/odds/raw/featured_raw_<UTCYYYYMMDD_HHMMSS>.json
  data/odds/processed/featured_<UTCYYYYMMDD_HHMMSS>.csv
"""
from __future__ import annotations
import os
import sys
import csv
import json
import time
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import requests
except Exception:
    requests = None

# ---------- config ----------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "odds" / "raw"
PROC_DIR = ROOT / "data" / "odds" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

API_BASE = "https://api.the-odds-api.com/v4/sports/{sport}/odds"

DEFAULTS = dict(
    sport="americanfootball_nfl",
    regions="us",
    markets="h2h,spreads,totals",  # IMPORTANT: only featured markets here
    odds_format="american",
    date_format="iso",
    books="draftkings,fanduel,betmgm,caesars,espnbet,bet365,pointsbet,betway",
    sleep_secs=0.0,
)

FIELDS = [
    "event_id","sport_key","commence_time_utc","home_team","away_team",
    "book","book_title","market","runner","price_american","point","last_update"
]

# ---------- helpers ----------
def now_stamp() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_csv(p: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

def env(k: str, default: Optional[str] = None) -> str:
    v = os.getenv(k)
    return v if v is not None and v != "" else (default or "")

def fetch_featured(api_key: str, sport: str, regions: str, markets: str,
                   odds_format: str, date_format: str, books: str,
                   sleep_between: float = 0.0) -> List[Dict[str, Any]]:
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,          # must be only featured markets
        "oddsFormat": odds_format,
        "dateFormat": date_format,
        "bookmakers": books,
    }
    url = API_BASE.format(sport=sport)
    r = requests.get(url, params=params, timeout=30)
    try:
        r.raise_for_status()
    except Exception as e:
        print("Error:  HTTP", getattr(r, "status_code", "?"), "from Odds API", file=sys.stderr)
        print("Error:  URL:", r.url, file=sys.stderr)
        try:
            print("Error:  Response:", r.text[:400], file=sys.stderr)
        except Exception:
            pass
        raise
    time.sleep(sleep_between)
    return r.json()

def normalize_featured(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ev in data:
        eid  = ev.get("id","")
        skey = ev.get("sport_key","")
        t0   = ev.get("commence_time","")
        home = ev.get("home_team","")
        away = ev.get("away_team","")
        for bm in ev.get("bookmakers", []):
            bkey = bm.get("key","")
            bttl = bm.get("title","")
            upd  = bm.get("last_update","")
            for mk in bm.get("markets", []):
                mkey = mk.get("key","")
                for out in mk.get("outcomes", []):
                    rows.append({
                        "event_id": eid,
                        "sport_key": skey,
                        "commence_time_utc": t0,
                        "home_team": home,
                        "away_team": away,
                        "book": bkey,
                        "book_title": bttl,
                        "market": mkey,
                        "runner": out.get("name",""),
                        "price_american": out.get("price", ""),
                        "point": out.get("point", ""),
                        "last_update": upd,
                    })
    return rows

# ---------- main ----------
def main() -> int:
    if requests is None:
        print("ERROR: requests not available.", file=sys.stderr)
        return 3

    api_key = env("ODDS_API_KEY")
    if not api_key:
        print("ERROR: Missing ODDS_API_KEY", file=sys.stderr)
        return 2

    sport       = env("ODDS_SPORT",       DEFAULTS["sport"])
    regions     = env("ODDS_REGIONS",     DEFAULTS["regions"])
    markets     = env("ODDS_MARKETS",     DEFAULTS["markets"])
    odds_format = env("ODDS_FORMAT",      DEFAULTS["odds_format"])
    date_format = env("ODDS_DATE_FORMAT", DEFAULTS["date_format"])
    books       = env("ODDS_BOOKS",       DEFAULTS["books"])
    sleep_secs  = float(env("ODDS_SLEEP_SECS", str(DEFAULTS["sleep_secs"])) or 0.0)

    # Guard: ensure we didn't sneak in player_* markets here
    banned_tokens = ("player_", "anytime_td")
    if any(tok in markets for tok in banned_tokens):
        print("ERROR: player prop markets are not allowed in this featured fetch.", file=sys.stderr)
        return 2

    ts = now_stamp()
    data = fetch_featured(api_key, sport, regions, markets, odds_format, date_format, books, sleep_secs)
    raw_path = RAW_DIR / f"featured_raw_{ts}.json"
    write_json(raw_path, data)

    rows = normalize_featured(data)
    out_csv = PROC_DIR / f"featured_{ts}.csv"
    write_csv(out_csv, rows, FIELDS)

    print(f"[featured] Wrote raw -> {raw_path}")
    print(f"[featured] Wrote csv -> {out_csv}  rows={len(rows)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
