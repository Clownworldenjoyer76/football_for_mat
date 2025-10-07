#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01b_fetch_player_props.py — Player props fetch (per-event)

Flow:
  1) GET /v4/sports/{sport}/events   -> event list for upcoming/current slate
  2) For each event.id:
       GET /v4/sports/{sport}/events/{event_id}/odds
         with markets=<comma list of player_* markets>

ENV (recommended):
  ODDS_API_KEY        -> required
  ODDS_SPORT          -> default: americanfootball_nfl
  ODDS_REGIONS        -> default: us
  ODDS_BOOKS          -> default: draftkings,fanduel,betmgm,caesars,espnbet,bet365,pointsbet,betway
  ODDS_FORMAT         -> default: american
  ODDS_DATE_FORMAT    -> default: iso
  ODDS_SLEEP_SECS     -> default: 0.0 (delay between HTTP calls)
  ODDS_PLAYER_MARKETS -> default: comma-joined list below

Player markets requested by user:
  player_field_goals,player_kicking_points,player_pass_interceptions,player_pass_rush_yds,
  player_pass_tds,player_pass_yds,player_receptions,player_reception_yds,player_rush_attempts,
  player_rush_yds,player_sacks,player_solo_tackles,player_tackles_assists,player_anytime_td

Outputs:
  data/odds/raw/player_props_raw_<UTCYYYYMMDD_HHMMSS>.json
  data/odds/processed/player_props_<UTCYYYYMMDD_HHMMSS>.csv
"""
from __future__ import annotations
import os
import sys
import csv
import json
import time
import datetime as dt
from pathlib import Path
from typing import Any, Dict, Iterable, List

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

EVENTS_URL   = "https://api.the-odds-api.com/v4/sports/{sport}/events"
EVENT_ODDS   = "https://api.the-odds-api.com/v4/sports/{sport}/events/{event_id}/odds"

DEFAULT_PLAYER_MARKETS = ",".join([
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

DEFAULTS = dict(
    sport="americanfootball_nfl",
    regions="us",
    markets=DEFAULT_PLAYER_MARKETS,
    odds_format="american",
    date_format="iso",
    books="draftkings,fanduel,betmgm,caesars,espnbet,bet365,pointsbet,betway",
    sleep_secs=0.15,  # gentle between per-event calls
)

FIELDS = [
    "event_id","sport_key","commence_time_utc","home_team","away_team",
    "book","book_title","market","runner","price_american","point","last_update"
]

# ---------- helpers ----------
def now_stamp() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def env(k: str, default: str) -> str:
    v = os.getenv(k)
    return v if v not in (None, "") else default

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})

def http_json(url: str, params: Dict[str,str]) -> Any:
    r = requests.get(url, params=params, timeout=30)
    try:
        r.raise_for_status()
    except Exception:
        # helpful debug
        print("Error:  HTTP", getattr(r, "status_code", "?"), "from Odds API", file=sys.stderr)
        print("Error:  URL:", r.url, file=sys.stderr)
        try:
            print("Error:  Response:", r.text[:400], file=sys.stderr)
        except Exception:
            pass
        raise
    return r.json()

def fetch_events(api_key: str, sport: str, date_format: str) -> List[Dict[str,Any]]:
    params = {"apiKey": api_key, "dateFormat": date_format}
    url = EVENTS_URL.format(sport=sport)
    return http_json(url, params)

def fetch_event_player_props(api_key: str, sport: str, event_id: str,
                             regions: str, markets: str, odds_format: str,
                             date_format: str, books: str) -> Dict[str,Any]:
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
        "bookmakers": books,
    }
    url = EVENT_ODDS.format(sport=sport, event_id=event_id)
    return http_json(url, params)

def normalize_event_blob(ev_blob: Dict[str,Any]) -> List[Dict[str,Any]]:
    """
    /events/{id}/odds returns a single-event structure with bookmakers/markets/outcomes.
    """
    rows: List[Dict[str,Any]] = []
    eid  = ev_blob.get("id","")
    skey = ev_blob.get("sport_key","")
    t0   = ev_blob.get("commence_time","")
    home = ev_blob.get("home_team","")
    away = ev_blob.get("away_team","")
    for bm in ev_blob.get("bookmakers", []):
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
                    "price_american": out.get("price",""),
                    "point": out.get("point",""),
                    "last_update": upd,
                })
    return rows

# ---------- main ----------
def main() -> int:
    if requests is None:
        print("ERROR: requests not available.", file=sys.stderr)
        return 3

    api_key = env("ODDS_API_KEY", "")
    if not api_key:
        print("ERROR: Missing ODDS_API_KEY", file=sys.stderr)
        return 2

    sport       = env("ODDS_SPORT",       DEFAULTS["sport"])
    regions     = env("ODDS_REGIONS",     DEFAULTS["regions"])
    markets     = env("ODDS_PLAYER_MARKETS", DEFAULTS["markets"])
    odds_format = env("ODDS_FORMAT",      DEFAULTS["odds_format"])
    date_format = env("ODDS_DATE_FORMAT", DEFAULTS["date_format"])
    books       = env("ODDS_BOOKS",       DEFAULTS["books"])
    sleep_secs  = float(env("ODDS_SLEEP_SECS", str(DEFAULTS["sleep_secs"])))

    # Fetch the slate of events
    events = fetch_events(api_key, sport, date_format)
    if not isinstance(events, list) or len(events) == 0:
        print("[player-props] No events returned.", file=sys.stderr)
        # still exit 0; just writes empty outputs
        events = []

    ts = now_stamp()
    raw_bundle: Dict[str,Any] = {"events": []}
    all_rows: List[Dict[str,Any]] = []

    for i, ev in enumerate(events, 1):
        eid = ev.get("id")
        if not eid:
            continue
        try:
            blob = fetch_event_player_props(
                api_key, sport, eid, regions, markets, odds_format, date_format, books
            )
            raw_bundle["events"].append(blob)
            rows = normalize_event_blob(blob)
            all_rows.extend(rows)
        except Exception:
            # keep going to next event
            print(f"[player-props] WARN: failed on event {eid}", file=sys.stderr)
        time.sleep(sleep_secs)

    # Write outputs
    raw_path = RAW_DIR / f"player_props_raw_{ts}.json"
    csv_path = PROC_DIR / f"player_props_{ts}.csv"
    write_json(raw_path, raw_bundle)
    write_csv(csv_path, all_rows)

    print(f"[player-props] Wrote raw -> {raw_path}")
    print(f"[player-props] Wrote csv -> {csv_path}  rows={len(all_rows)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
