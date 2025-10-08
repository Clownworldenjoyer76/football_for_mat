#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch NFL odds from The Odds API.

This script does TWO passes:
  1) Featured markets (h2h/spreads/totals) via /sports/{sport}/odds
  2) Player props via per-event endpoint /sports/{sport}/events and then
     /sports/{sport}/events/{eventId}/odds?markets=...

Writes:
  - data/odds/raw/featured_raw_<UTCSTAMP>.json
  - data/odds/processed/featured_<UTCSTAMP>.csv
  - data/odds/raw/player_raw_<UTCSTAMP>.json       (one big JSON of all events)
  - data/odds/processed/player_<UTCSTAMP>.csv
  - data/props/props_current.csv                   (normalized superset)

Environment variables (set from workflow):
  ODDS_API_KEY           (required for live calls)
  ODDS_FEATURED_MARKETS  (default: "h2h,spreads,totals")
  ODDS_PLAYER_MARKETS    (comma list of player_* keys)
  ODDS_REGIONS           (default: "us")
  ODDS_BOOKMAKERS        (optional; comma list)
  ODDS_ODDS_FORMAT       (default: "american")
  ODDS_DATE_FORMAT       (default: "iso")
  TARGET_SEASON          (default: current UTC year; used in props_current.csv)
  DRY_RUN                ("1" to simulate without API)

Notes:
- We normalize player prop market keys to your internal market names used by
  scripts/02_build_features.py (e.g., player_pass_yds -> qb_passing_yards).
- We only record Over/Under prices when present; others become NaN.
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

# -------------------- filesystem --------------------
REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "odds" / "raw"
PROC = REPO / "data" / "odds" / "processed"
PROPS_DIR = REPO / "data" / "props"
for p in (RAW, PROC, PROPS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# -------------------- API base ----------------------
API_BASE = "https://api.the-odds-api.com/v4"

DEFAULT_SPORT = "americanfootball_nfl"

def utc_stamp() -> str:
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

def getenv(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name)
    return default if v is None else str(v)

# -------------------- HTTP -------------------------
def _requests() :
    try:
        import requests  # lazy import so dry-run works offline
        return requests
    except Exception as e:
        print("ERROR: requests not installed:", e, file=sys.stderr)
        sys.exit(3)

def http_get(url: str, params: Dict[str, Any], timeout: int = 30) -> Any:
    r = _requests().get(url, params=params, timeout=timeout)
    try:
        r.raise_for_status()
    except Exception:
        # surface provider message if present
        try:
            print("Error:", r.json(), file=sys.stderr)
        except Exception:
            pass
        raise
    return r.json()

# -------------------- Normalization helpers --------------------
# Map Odds API player markets -> internal market names used downstream
PLAYER_MARKET_MAP = {
    "player_pass_yds":        "qb_passing_yards",
    "player_rush_yds":        "rb_rushing_yards",
    "player_reception_yds":   "wr_rec_yards",
    "player_receptions":      "wrte_receptions",
    "player_pass_tds":        "qb_passing_tds",
    "player_interceptions":   "qb_interceptions",
    "player_anytime_td":      "player_tds",
    "player_sacks":           "player_sacks",
    "player_tackles":         "player_tackles",
    "player_tackles_assists": "player_tackles_assists",
    "player_field_goals_made":"player_field_goals_made",
    "player_kicking_points":  "player_kicking_points",
    "player_pass_rush_yds":   "player_pass_rush_yds",
    "player_rush_attempts":   "player_rush_attempts",
}

PROP_FIELDS = [
    "season","week","game_id","event_id",
    "commence_time_utc","home_team","away_team",
    "market","runner","team","opponent",
    "book","book_title","line","odds_over","odds_under","last_update"
]

def normalize_featured(featured: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ev in featured:
        event_id = ev.get("id")
        home = ev.get("home_team")
        away = ev.get("away_team")
        commence = ev.get("commence_time")
        for bm in ev.get("bookmakers", []) or []:
            book = bm.get("key")
            book_title = bm.get("title")
            last_update = bm.get("last_update")
            for m in bm.get("markets", []) or []:
                mkey = m.get("key")
                # h2h/spreads/totals may have different outcome structures
                for out in m.get("outcomes", []) or []:
                    rows.append({
                        "season": None, "week": None,
                        "game_id": event_id, "event_id": event_id,
                        "commence_time_utc": commence,
                        "home_team": home, "away_team": away,
                        "market": mkey,
                        "runner": out.get("name") or out.get("description"),
                        "team": None, "opponent": None,
                        "book": book, "book_title": book_title,
                        "line": out.get("point"),
                        "odds_over": out.get("price") if (out.get("name") == "Over") else None,
                        "odds_under": out.get("price") if (out.get("name") == "Under") else None,
                        "last_update": last_update,
                    })
    return rows

def normalize_player(event: Dict[str, Any], event_odds: Dict[str, Any], season_val: int) -> List[Dict[str, Any]]:
    """
    event: one object from /sports/{sport}/events
    event_odds: response from /events/{eventId}/odds
    """
    rows: List[Dict[str, Any]] = []
    event_id = event.get("id")
    home = event.get("home_team")
    away = event.get("away_team")
    commence = event.get("commence_time")

    for bm in event_odds.get("bookmakers", []) or []:
        book = bm.get("key")
        book_title = bm.get("title")
        last_update = bm.get("last_update")
        for m in bm.get("markets", []) or []:
            raw_key = m.get("key")
            internal_key = PLAYER_MARKET_MAP.get(raw_key)
            if not internal_key:
                continue
            # Outcomes for player props: each outcome describes a player, sometimes with name/description
            # and often separate Over/Under rows. We try to capture line/over/under if present.
            # The Odds API usually presents Over/Under pairs per player; some books present separate entries.
            # We aggregate by player+line within this bookmaker.
            grouped: Dict[str, Dict[str, Any]] = {}
            for out in m.get("outcomes", []) or []:
                player_name = out.get("description") or out.get("name")
                if not player_name:
                    continue
                key = f"{player_name}|{out.get('point')}"
                rec = grouped.setdefault(key, {
                    "runner": player_name,
                    "line": out.get("point"),
                    "odds_over": None,
                    "odds_under": None,
                })
                nm = (out.get("name") or "").lower()
                if nm == "over":
                    rec["odds_over"] = out.get("price")
                elif nm == "under":
                    rec["odds_under"] = out.get("price")
                # some books provide single price only → store in odds_over as generic
                if rec["odds_over"] is None and rec["odds_under"] is None and out.get("price") is not None:
                    rec["odds_over"] = out.get("price")

            for rec in grouped.values():
                rows.append({
                    "season": season_val, "week": None,
                    "game_id": event_id, "event_id": event_id,
                    "commence_time_utc": commence,
                    "home_team": home, "away_team": away,
                    "market": internal_key,
                    "runner": rec["runner"],
                    "team": None, "opponent": None,
                    "book": book, "book_title": book_title,
                    "line": rec["line"],
                    "odds_over": rec["odds_over"],
                    "odds_under": rec["odds_under"],
                    "last_update": last_update,
                })
    return rows

# -------------------- Main --------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", default=DEFAULT_SPORT)
    ap.add_argument("--sleep", type=float, default=0.0, help="sleep between API calls (seconds)")
    args = ap.parse_args(argv)

    api_key   = getenv("ODDS_API_KEY", "").strip()
    featured  = getenv("ODDS_FEATURED_MARKETS", "h2h,spreads,totals")
    pmarkets  = getenv("ODDS_PLAYER_MARKETS", "")
    regions   = getenv("ODDS_REGIONS", "us")
    books     = getenv("ODDS_BOOKMAKERS", "")
    odds_fmt  = getenv("ODDS_ODDS_FORMAT", "american")
    date_fmt  = getenv("ODDS_DATE_FORMAT", "iso")
    dry_run   = getenv("DRY_RUN", "0") == "1"
    season    = int(getenv("TARGET_SEASON", str(dt.datetime.utcnow().year)))

    ts = utc_stamp()

    if dry_run or not api_key:
        print("[DRY-RUN] Writing small synthetic outputs (no API calls).")
        sample_featured = []
        sample_player = []
        write_json(RAW / f"featured_raw_{ts}.json", sample_featured)
        write_json(RAW / f"player_raw_{ts}.json", sample_player)
        write_csv(PROC / f"featured_{ts}.csv", [], ["game_id"])
        write_csv(PROC / f"player_{ts}.csv", [], PROP_FIELDS)
        # keep/refresh props_current.csv (no-op here)
        return 0

    # ---------- 1) Featured ----------
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": featured,
        "oddsFormat": odds_fmt,
        "dateFormat": date_fmt,
    }
    if books:
        params["bookmakers"] = books

    url_feat = f"{API_BASE}/sports/{args.sport}/odds"
    feat_json = http_get(url_feat, params)
    write_json(RAW / f"featured_raw_{ts}.json", feat_json)
    feat_rows = normalize_featured(feat_json)
    write_csv(PROC / f"featured_{ts}.csv", feat_rows, list({
        "game_id","event_id","commence_time_utc","home_team","away_team",
        "market","runner","book","book_title","line","odds_over","odds_under","last_update"
    }))

    # List events we’ll query for player props
    # If the provider supports /events, use it to ensure we have *all* upcoming events
    url_events = f"{API_BASE}/sports/{args.sport}/events"
    events = http_get(url_events, {"apiKey": api_key, "dateFormat": date_fmt})
    event_by_id = {e.get("id"): e for e in events}

    # ---------- 2) Player props per event ----------
    player_rows: List[Dict[str, Any]] = []
    player_raw_agg: Dict[str, Any] = {"events": []}

    # sanitize player markets to valid list
    pmarkets_list = [s.strip() for s in pmarkets.split(",") if s.strip()]
    # keep only keys we know how to map downstream
    pmarkets_list = [k for k in pmarkets_list if k in PLAYER_MARKET_MAP]

    if not pmarkets_list:
        print("[INFO] No ODDS_PLAYER_MARKETS supplied; skipping player props.")
    else:
        for eid, ev in event_by_id.items():
            params_ev = {
                "apiKey": api_key,
                "regions": regions,
                "markets": ",".join(pmarkets_list),
                "oddsFormat": odds_fmt,
                "dateFormat": date_fmt,
            }
            if books:
                params_ev["bookmakers"] = books
            url_ev = f"{API_BASE}/sports/{args.sport}/events/{eid}/odds"
            try:
                ev_odds = http_get(url_ev, params_ev)
            except Exception as e:
                print(f"[WARN] event {eid} player props fetch failed: {e}", file=sys.stderr)
                continue
            player_raw_agg["events"].append({"event": ev, "odds": ev_odds})
            player_rows.extend(normalize_player(ev, ev_odds, season))
            if args.sleep:
                time.sleep(args.sleep)

    write_json(RAW / f"player_raw_{ts}.json", player_raw_agg)
    write_csv(PROC / f"player_{ts}.csv", player_rows, PROP_FIELDS)

    # ---------- 3) Build/refresh props_current.csv ----------
    # We write a unified props_current.csv from player_rows ONLY (that’s what downstream needs).
    props_path = PROPS_DIR / "props_current.csv"
    write_csv(props_path, player_rows, PROP_FIELDS)
    print(f"[OK] props_current.csv -> {props_path} rows={len(player_rows)}")

    print(f"[OK] Featured rows={len(feat_rows)}  Player rows={len(player_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
