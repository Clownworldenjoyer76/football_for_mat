#!/usr/bin/env python3
# Rebuild data/props/props_current.csv from cached raw JSON (no API calls).

from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv
import datetime as dt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "odds" / "raw"
PROPS_DIR = REPO / "data" / "props"
PROPS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = PROPS_DIR / "props_current.csv"

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

FIELDS = [
    "season","week","game_id","event_id",
    "commence_time_utc","home_team","away_team",
    "market","runner","team","opponent",
    "book","book_title","line","odds_over","odds_under","last_update"
]

def latest(path_glob: str) -> Optional[Path]:
    cands = sorted(RAW.glob(path_glob), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})

def normalize_player_blob(blob: Dict[str, Any], season: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for E in blob.get("events", []):
        ev = E.get("event", {})
        odds = E.get("odds", {})
        eid = ev.get("id")
        home, away = ev.get("home_team"), ev.get("away_team")
        commence = ev.get("commence_time")
        for bm in odds.get("bookmakers", []) or []:
            book, book_title = bm.get("key"), bm.get("title")
            last_update = bm.get("last_update")
            for m in bm.get("markets", []) or []:
                raw_key = m.get("key")
                market = PLAYER_MARKET_MAP.get(raw_key)
                if not market:
                    continue
                grouped: Dict[str, Dict[str, Any]] = {}
                for out in m.get("outcomes", []) or []:
                    player_name = out.get("description") or out.get("name")
                    if not player_name:
                        continue
                    key = f"{player_name}|{out.get('point')}"
                    rec = grouped.setdefault(key, {"runner": player_name, "line": out.get("point"),
                                                   "odds_over": None, "odds_under": None})
                    nm = (out.get("name") or "").lower()
                    if nm == "over":
                        rec["odds_over"] = out.get("price")
                    elif nm == "under":
                        rec["odds_under"] = out.get("price")
                    if rec["odds_over"] is None and rec["odds_under"] is None and out.get("price") is not None:
                        rec["odds_over"] = out.get("price")
                for rec in grouped.values():
                    rows.append({
                        "season": season, "week": None,
                        "game_id": eid, "event_id": eid,
                        "commence_time_utc": commence,
                        "home_team": home, "away_team": away,
                        "market": market, "runner": rec["runner"],
                        "team": None, "opponent": None,
                        "book": book, "book_title": book_title,
                        "line": rec["line"],
                        "odds_over": rec["odds_over"], "odds_under": rec["odds_under"],
                        "last_update": last_update,
                    })
    return rows

def guess_season() -> int:
    # use newest file mtime year as season fallback
    pts = list(RAW.glob("player_raw_*.json"))
    if pts:
        y = dt.datetime.utcfromtimestamp(max(p.stat().st_mtime for p in pts)).year
        return y
    return dt.datetime.utcnow().year

def main() -> int:
    season_env = (os.getenv("TARGET_SEASON") or "").strip()
    season = int(season_env) if season_env.isdigit() else guess_season()

    player_file = latest("player_raw_*.json")
    if not player_file:
        print("[cache] No player_raw_*.json found. Nothing to build.", file=sys.stderr)
        # still write an empty file with headers so downstream steps don't explode
        write_csv(OUT_CSV, [])
        print(f"[cache] Wrote empty {OUT_CSV}")
        return 0

    with player_file.open("r", encoding="utf-8") as f:
        blob = json.load(f)

    rows = normalize_player_blob(blob, season)
    write_csv(OUT_CSV, rows)
    print(f"[cache] Built {OUT_CSV} from {player_file.name} rows={len(rows)}")
    return 0

if __name__ == "__main__":
    import os
    raise SystemExit(main())
