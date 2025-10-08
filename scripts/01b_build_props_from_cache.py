#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01b_build_props_from_cache.py

Rebuilds data/props/props_current.csv using whatever raw cache exists under data/odds/raw/:
 - featured_raw_*.json (kept for completeness; does not contain player props)
 - player_raw_*.json   (per-event envelopes created by 01_fetch_odds.py)

This script is intentionally conservative and schema-light. It extracts a minimal, generic set
of columns that downstream steps can work with (season, market, book, line, odds_over/under, player_name, team/opponent).

If no usable player data is found, it still writes props_current.csv with headers (0 rows) so
the workflow can continue without burning more API calls.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW_DIR = REPO / "data" / "odds" / "raw"
OUT_PROPS = REPO / "data" / "props" / "props_current.csv"
WEEKLY_LATEST = REPO / "data" / "weekly" / "latest.csv"

TARGET_SEASON = int(os.getenv("TARGET_SEASON", "2025"))

# Minimal output schema used later by 02_build_features.py
OUT_COLS = [
    "season","week","game_id","event_id",
    "player_id","player_name","team","opponent","position",
    "market","line","odds_over","odds_under","book",
]

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def latest_week_fallback() -> int | None:
    if WEEKLY_LATEST.exists():
        try:
            df = pd.read_csv(WEEKLY_LATEST)
            for c in ("week","current_week","wk","week_number"):
                if c in df.columns and pd.notnull(df[c]).any():
                    return int(pd.to_numeric(df[c], errors="coerce").dropna().iloc[0])
        except Exception:
            pass
    return None

def list_raw(patterns: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    if not RAW_DIR.exists():
        return files
    for p in RAW_DIR.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if any(name.startswith(prefix) and name.endswith(".json") for prefix in patterns):
            files.append(p)
    files.sort()
    return files

def normalize_player_envelope(obj: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert a single "player" envelope {meta:{...}, events:[...]} to rows.
    We don't rely on an exact Odds API shape; we look for common keys.
    """
    rows: List[Dict[str, Any]] = []
    events = obj.get("events") or []
    for ev in events:
        if "error" in ev:
            # keep record of the error but skip rows
            continue
        event_id = ev.get("eventId") or ev.get("id") or ev.get("event_id") or None
        payload = ev.get("payload") or {}
        # Different integrations may put markets under "bookmakers" or similar structure.
        # We try to walk: payload["bookmakers"][i]["markets"][j]["outcomes"]
        bms = payload.get("bookmakers", [])
        # Minimal matchup info if present
        home = payload.get("home_team") or payload.get("homeTeam") or None
        away = payload.get("away_team") or payload.get("awayTeam") or None
        game_id = payload.get("id") or payload.get("eventId") or None

        for bm in bms:
            book = bm.get("key") or bm.get("title") or bm.get("book") or None
            for mk in bm.get("markets", []):
                market = mk.get("key") or mk.get("market") or None
                # Player props are usually Over/Under with a "point"
                line = mk.get("outcomes", [{}])[0].get("point") if mk.get("outcomes") else None
                odds_over, odds_under = None, None
                for oc in mk.get("outcomes", []):
                    name = (oc.get("name") or "").lower()
                    price = oc.get("price")
                    # normalize odds
                    if "over" in name:
                        odds_over = price
                    elif "under" in name:
                        odds_under = price
                # Try to pull player name (some feeds put it on the market, others in outcome labels)
                player_name = mk.get("player") or mk.get("player_name") or None
                if not player_name:
                    # Fall back: outcomes sometimes have 'description' like "J. Jefferson"
                    for oc in mk.get("outcomes", []):
                        desc = (oc.get("description") or oc.get("name") or "").strip()
                        if desc and desc.lower() not in ("over","under"):
                            player_name = desc
                            break

                rows.append({
                    "season": TARGET_SEASON,
                    "week": latest_week_fallback(),
                    "game_id": game_id,
                    "event_id": event_id,
                    "player_id": None,              # not always available
                    "player_name": player_name,
                    "team": home,                    # best effort; refined later
                    "opponent": away,
                    "position": None,
                    "market": market,
                    "line": line,
                    "odds_over": odds_over,
                    "odds_under": odds_under,
                    "book": book,
                })
    return pd.DataFrame(rows, columns=OUT_COLS)

def main() -> int:
    OUT_PROPS.parent.mkdir(parents=True, exist_ok=True)

    player_files = list_raw(["player_raw_"])
    featured_files = list_raw(["featured_raw_"])  # currently unused for players, but kept for future joins

    if not player_files and not featured_files:
        # Nothing cached at all: write empty with headers (0 rows)
        pd.DataFrame(columns=OUT_COLS).to_csv(OUT_PROPS, index=False)
        print(f"[cache] No raw JSON found. Wrote empty {OUT_PROPS}")
        return 0

    frames: List[pd.DataFrame] = []

    # Prefer the newest player cache first
    for p in reversed(player_files):
        try:
            env = read_json(p)
            dfp = normalize_player_envelope(env)
            if not dfp.empty:
                frames.append(dfp)
        except Exception as e:
            print(f"[WARN] Could not parse {p.name}: {e}", file=sys.stderr)

    # (Optional) place-holder: featured data isn’t turned into player rows yet.
    # This keeps the door open for deriving simple team-level signals later if needed.

    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        out = pd.DataFrame(columns=OUT_COLS)

    # Final tidy
    # Ensure types for season/week when present
    if "season" in out.columns:
        out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    if "week" in out.columns:
        out["week"] = pd.to_numeric(out["week"], errors="coerce").astype("Int64")

    out.to_csv(OUT_PROPS, index=False)
    print(f"[cache] Wrote {OUT_PROPS} rows={len(out)} "
          f"(player_json={len(player_files)}, featured_json={len(featured_files)})")
    if len(out) == 0:
        print("[cache] Note: zero rows. This is expected if the player fetch ran with no event IDs "
              "or the provider returned no player markets for the selected events.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
