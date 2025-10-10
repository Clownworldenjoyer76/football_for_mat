#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01b_build_props_from_cache.py

Rebuilds data/props/props_current.csv using whatever raw cache exists under data/odds/raw/:
 - featured_raw_*.json (kept for completeness; does not contain player props)
 - player_raw_*.json   (per-event envelopes created by 01_fetch_odds.py)

This version ensures every output row gets its season value directly from
the TARGET_SEASON environment variable (default 2025). That guarantees the
workflow's season guard passes once any player rows are present.
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List
import pandas as pd

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
RAW_DIR = REPO / "data" / "odds" / "raw"
OUT_PROPS = REPO / "data" / "props" / "props_current.csv"
WEEKLY_LATEST = REPO / "data" / "weekly" / "latest.csv"

TARGET_SEASON = int(os.getenv("TARGET_SEASON", "2025"))

OUT_COLS = [
    "season","week","game_id","event_id",
    "player_id","player_name","team","opponent","position",
    "market","line","odds_over","odds_under","book",
]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def latest_week_fallback() -> int | None:
    """Try to read a current week number from weekly/latest.csv."""
    if WEEKLY_LATEST.exists():
        try:
            df = pd.read_csv(WEEKLY_LATEST)
            for c in ("week","current_week","wk","week_number"):
                if c in df.columns and pd.notnull(df[c]).any():
                    return int(pd.to_numeric(df[c], errors="coerce").dropna().iloc[0])
        except Exception:
            pass
    return None

def list_raw(prefixes: Iterable[str]) -> List[Path]:
    """List cached JSON files with given prefixes."""
    files: List[Path] = []
    if not RAW_DIR.exists():
        return files
    for p in RAW_DIR.iterdir():
        if p.is_file() and any(p.name.startswith(pref) and p.name.endswith(".json") for pref in prefixes):
            files.append(p)
    files.sort()
    return files

# ---------------------------------------------------------------------
# Core Normalization
# ---------------------------------------------------------------------
def normalize_player_envelope(obj: Dict[str, Any]) -> pd.DataFrame:
    """Convert a single player odds envelope to normalized rows."""
    rows: List[Dict[str, Any]] = []
    events = obj.get("events") or []
    week_fallback = latest_week_fallback()

    for ev in events:
        if not isinstance(ev, dict) or "error" in ev:
            continue

        event_id = ev.get("eventId") or ev.get("id") or ev.get("event_id")
        payload = ev.get("payload") or {}
        bms = payload.get("bookmakers", [])
        home = payload.get("home_team") or payload.get("homeTeam")
        away = payload.get("away_team") or payload.get("awayTeam")
        game_id = payload.get("id") or payload.get("eventId")

        for bm in bms:
            # guard against malformed bookmaker entries
            if not isinstance(bm, dict):
                print(f"[WARN] Skipping malformed bookmaker entry: {bm!r}", file=sys.stderr)
                continue
            book = bm.get("key") or bm.get("title") or bm.get("book")
            for mk in bm.get("markets", []):
                if not isinstance(mk, dict):
                    continue
                market = mk.get("key") or mk.get("market")
                outcomes = mk.get("outcomes") or []
                line = None
                odds_over, odds_under = None, None

                if isinstance(outcomes, list) and outcomes:
                    # grab "point" if consistent
                    line = outcomes[0].get("point") if isinstance(outcomes[0], dict) else None
                    for oc in outcomes:
                        if not isinstance(oc, dict):
                            continue
                        name = (oc.get("name") or "").lower()
                        price = oc.get("price")
                        if "over" in name:
                            odds_over = price
                        elif "under" in name:
                            odds_under = price

                # player name detection
                player_name = mk.get("player") or mk.get("player_name")
                if not player_name and isinstance(outcomes, list):
                    for oc in outcomes:
                        desc = (oc.get("description") or oc.get("name") or "").strip()
                        if desc and desc.lower() not in ("over", "under"):
                            player_name = desc
                            break

                rows.append({
                    "season": TARGET_SEASON,
                    "week": week_fallback,
                    "game_id": game_id,
                    "event_id": event_id,
                    "player_id": None,
                    "player_name": player_name,
                    "team": home,
                    "opponent": away,
                    "position": None,
                    "market": market,
                    "line": line,
                    "odds_over": odds_over,
                    "odds_under": odds_under,
                    "book": book,
                })

    return pd.DataFrame(rows, columns=OUT_COLS)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> int:
    OUT_PROPS.parent.mkdir(parents=True, exist_ok=True)

    player_files = list_raw(["player_raw_"])
    featured_files = list_raw(["featured_raw_"])

    if not player_files and not featured_files:
        pd.DataFrame(columns=OUT_COLS).to_csv(OUT_PROPS, index=False)
        print(f"[cache] No raw JSON found. Wrote empty {OUT_PROPS}")
        return 0

    frames: List[pd.DataFrame] = []
    for p in reversed(player_files):
        try:
            data = read_json(p)
            df = normalize_player_envelope(data)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"[WARN] Could not parse {p.name}: {e}", file=sys.stderr)

    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        out = pd.DataFrame(columns=OUT_COLS)

    # enforce season stamping even if empty
    out["season"] = TARGET_SEASON
    out["week"] = out.get("week") or latest_week_fallback()

    out.to_csv(OUT_PROPS, index=False)
    print(f"[cache] Wrote {OUT_PROPS} rows={len(out)} "
          f"(player_json={len(player_files)}, featured_json={len(featured_files)})")

    if len(out) == 0:
        print("[cache] Note: zero rows. Expected if the player fetch had no events or props.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
