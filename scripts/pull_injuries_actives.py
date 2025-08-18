#!/usr/bin/env python3
"""
Pull NFL injuries/actives (day-of freshness).

Primary source (structured JSON):
  DraftKings sportsbook (semi-public) – try multiple known league ids.
  We walk the JSON rather than assuming a fixed schema.

Fallback source (HTML table; optional if libs available):
  FantasyPros injury report.

Outputs:
  data/raw/injuries/injury_reports_latest.csv
  data/raw/injuries/injury_reports_{season}.csv  (when --season given)

NOTE:
- Filename is LOCKED to scripts/pull_injuries_actives.py per user request.
- No external keys required. Works in GitHub Actions runner.

Usage:
  python scripts/pull_injuries_actives.py
  python scripts/pull_injuries_actives.py --season 2025
"""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

# ---------------- paths ----------------
RAW_DIR = Path("data/raw")
INJ_DIR = RAW_DIR / "injuries"
INJ_DIR.mkdir(parents=True, exist_ok=True)
NFLVERSE_DIR = RAW_DIR / "nflverse"

OUT_LATEST = INJ_DIR / "injury_reports_latest.csv"

# ---------------- constants ----------------
# DK sometimes uses different league ids; try several candidates.
DK_URL_CANDIDATES = [
    "https://sportsbook.draftkings.com/api/sportscontent/v1/leagues/889/injuries",
    "https://sportsbook.draftkings.com/api/sportscontent/v1/leagues/888/injuries",
]
FANTASYPROS_URL = "https://www.fantasypros.com/nfl/injuries/"

UA_HDRS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

TEAM_ABBR = {
    # AFC East
    "buffalo bills": "BUF", "bills": "BUF",
    "miami dolphins": "MIA", "dolphins": "MIA",
    "new england patriots": "NE", "patriots": "NE", "ne patriots": "NE",
    "new york jets": "NYJ", "jets": "NYJ",
    # AFC North
    "baltimore ravens": "BAL", "ravens": "BAL",
    "cincinnati bengals": "CIN", "bengals": "CIN",
    "cleveland browns": "CLE", "browns": "CLE",
    "pittsburgh steelers": "PIT", "steelers": "PIT",
    # AFC South
    "houston texans": "HOU", "texans": "HOU",
    "indianapolis colts": "IND", "colts": "IND",
    "jacksonville jaguars": "JAX", "jaguars": "JAX", "jax jaguars": "JAX",
    "tennessee titans": "TEN", "titans": "TEN",
    # AFC West
    "denver broncos": "DEN", "broncos": "DEN",
    "kansas city chiefs": "KC", "chiefs": "KC",
    "las vegas raiders": "LV", "raiders": "LV",
    "los angeles chargers": "LAC", "chargers": "LAC",
    # NFC East
    "dallas cowboys": "DAL", "cowboys": "DAL",
    "new york giants": "NYG", "giants": "NYG",
    "philadelphia eagles": "PHI", "eagles": "PHI",
    "washington commanders": "WAS", "commanders": "WAS", "washington football team": "WAS",
    # NFC North
    "chicago bears": "CHI", "bears": "CHI",
    "detroit lions": "DET", "lions": "DET",
    "green bay packers": "GB", "packers": "GB",
    "minnesota vikings": "MIN", "vikings": "MIN",
    # NFC South
    "atlanta falcons": "ATL", "falcons": "ATL",
    "carolina panthers": "CAR", "panthers": "CAR",
    "new orleans saints": "NO", "saints": "NO",
    "tampa bay buccaneers": "TB", "buccaneers": "TB", "bucs": "TB",
    # NFC West
    "arizona cardinals": "ARI", "cardinals": "ARI",
    "los angeles rams": "LAR", "rams": "LAR",
    "san francisco 49ers": "SF", "49ers": "SF", "niners": "SF",
    "seattle seahawks": "SEA", "seahawks": "SEA",
}

OUT_STATUSES = {
    "out", "injured reserve", "ir", "pup", "nfi", "suspended", "covid", "dnp", "did not practice",
}
QUESTIONABLE_STATUSES = {"questionable", "limited", "lp", "dtd", "day to day"}
PROBABLE_STATUSES = {"probable", "fp", "full"}

# ---------------- helpers ----------------
def norm(s: Any) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip()
    return re.sub(r"\s+", " ", s)

def lower_clean(s: str) -> str:
    s = norm(s).lower()
    s = s.replace("-", " ").replace("_", " ")
    return re.sub(r"[^a-z0-9\s]", "", s)

def map_team(team: str) -> Optional[str]:
    if not team:
        return None
    key = lower_clean(team)
    # sometimes DK returns abbreviations already
    if key.upper() in {v for v in TEAM_ABBR.values()}:
        return key.upper()
    return TEAM_ABBR.get(key)

def status_flags(status: str, practice: str = "") -> dict:
    st = lower_clean(status)
    pr = lower_clean(practice)
    is_out = (st in OUT_STATUSES) or (pr in {"did not practice", "dnp"})
    is_q   = st in QUESTIONABLE_STATUSES
    is_p   = st in PROBABLE_STATUSES
    return {"is_out": int(is_out), "is_questionable": int(is_q), "is_probable": int(is_p)}

def http_get_json(url: str, retries: int = 3, backoff: float = 0.6) -> dict:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=UA_HDRS, timeout=20)
            if r.status_code == 200:
                return r.json()
            last = f"{r.status_code} {r.text[:160]}"
        except Exception as e:
            last = str(e)
        time.sleep(backoff * (2 ** i))
    raise RuntimeError(f"GET {url} failed: {last}")

# ---------------- DK parser (schema-agnostic) ----------------
def dk_extract_players(payload: Any) -> List[Dict[str, Any]]:
    players: List[Dict[str, Any]] = []

    NAME_KEYS = {"name", "displayName", "playerName", "fullName"}
    TEAM_KEYS = {"team", "teamName", "teamAbbreviation", "teamAbbrev"}
    POS_KEYS  = {"position", "pos"}
    STAT_KEYS = {"status", "injuryStatus", "gameStatus", "designation"}
    NOTE_KEYS = {"note", "injury", "details", "comment", "description"}

    def walk(x: Any):
        if isinstance(x, dict):
            keys = set(x.keys())
            if (keys & NAME_KEYS) and (keys & TEAM_KEYS) and ((keys & STAT_KEYS) or (keys & NOTE_KEYS)):
                rec = {
                    "player_name": norm(next((x[k] for k in NAME_KEYS if k in x), "")),
                    "team_raw": norm(next((x[k] for k in TEAM_KEYS if k in x), "")),
                    "position": norm(next((x[k] for k in POS_KEYS if k in x), "")),
                    "status": norm(next((x[k] for k in STAT_KEYS if k in x), "")),
                    "details": norm(next((x[k] for k in NOTE_KEYS if k in x), "")),
                }
                players.append(rec)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(payload)
    if not players:
        return []
    df = pd.DataFrame(players)
    df = df.drop_duplicates(subset=["player_name", "team_raw", "position", "status", "details"])
    return df.to_dict("records")

def fetch_from_dk() -> pd.DataFrame:
    last_err = None
    for url in DK_URL_CANDIDATES:
        try:
            data = http_get_json(url)
            recs = dk_extract_players(data)
            if recs:
                df = pd.DataFrame(recs)
                df["team"] = df["team_raw"].apply(map_team)
                return df
            last_err = f"No recognizable player records at {url}"
        except Exception as e:
            last_err = f"{url} -> {e}"
    raise RuntimeError(last_err or "DraftKings fetch failed")

# ---------------- FantasyPros fallback ----------------
def fetch_from_fantasypros() -> pd.DataFrame:
    # pandas.read_html needs lxml or html5lib; handle import errors cleanly.
    try:
        tables = pd.read_html(FANTASYPROS_URL)  # type: ignore
    except Exception as e:
        raise RuntimeError(f"FantasyPros read_html failed (needs lxml/html5lib): {e}")

    frames = []
    for t in tables:
        # Normalize columns
        t.columns = [lower_clean(c) for c in t.columns]
        cols = set(t.columns)
        if "player" in cols and ("team" in cols or "pos" in cols):
            frames.append(t)

    if not frames:
        raise RuntimeError("FantasyPros: no injury tables recognized")

    df = pd.concat(frames, ignore_index=True)

    colmap = {
        "player": "player_name",
        "team": "team_raw",
        "pos": "position",
        "injury": "details",
        "practice status": "practice_status",
        "game status": "status",
        "practice": "practice_status",
        "status": "status",
        "notes": "details",
    }
    df = df.rename(columns={c: colmap.get(c, c) for c in df.columns})
    for col in ["player_name", "team_raw", "position", "status", "practice_status", "details"]:
        if col not in df.columns:
            df[col] = ""
    df = df[["player_name", "team_raw", "position", "status", "practice_status", "details"]]
    df["team"] = df["team_raw"].apply(map_team)
    return df

# ---------------- Optional: attach player_id via rosters_latest ----------------
def attach_player_ids(df: pd.DataFrame) -> pd.DataFrame:
    roster_path = NFLVERSE_DIR / "rosters_latest.csv.gz"
    if not roster_path.exists():
        return df
    try:
        rost = pd.read_csv(roster_path)
    except Exception:
        return df

    def n(s: str) -> str:
        s = lower_clean(s)
        return s.replace(".", "").replace(" jr", "").replace(" sr", "").strip()

    rost.columns = [lower_clean(c) for c in rost.columns]
    name_col = None
    for c in ["full_name", "player_name"]:
        if c in rost.columns:
            name_col = c
            break
    if name_col is None:
        return df

    if "team_abbr" in rost.columns and "team" not in rost.columns:
        rost = rost.rename(columns={"team_abbr": "team"})

    df = df.copy()
    df["name_key"] = df["player_name"].map(n)
    rost["name_key"] = rost[name_col].map(n)

    if "team" in rost.columns:
        merged = df.merge(
            rost[["player_id", "team", "name_key"]].drop_duplicates(),
            on=["team", "name_key"], how="left"
        )
    else:
        merged = df
        merged["player_id"] = pd.NA
    return merged

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None, help="If provided, also write a seasonal copy.")
    args = ap.parse_args()

    # Prefer DK JSON; fall back to FantasyPros if DK unavailable
    try:
        df = fetch_from_dk()
        source = "draftkings"
    except Exception as e:
        sys.stderr.write(f"[injuries] DraftKings failed: {e}\nFalling back to FantasyPros...\n")
        try:
            df = fetch_from_fantasypros()
            source = "fantasypros"
        except Exception as e2:
            sys.stderr.write(f"[injuries] FantasyPros failed: {e2}\n")
            sys.exit(1)

    if "practice_status" not in df.columns:
        df["practice_status"] = ""

    flags = df.apply(
        lambda r: status_flags(r.get("status", ""), r.get("practice_status", "")),
        axis=1, result_type="expand"
    )
    for c in ["is_out", "is_questionable", "is_probable"]:
        df[c] = flags[c].astype(int)

    if "team" not in df.columns:
        df["team"] = df["team_raw"].apply(map_team)

    df = attach_player_ids(df)

    keep = [
        "player_id", "player_name", "team", "position",
        "status", "practice_status", "details",
        "is_out", "is_questionable", "is_probable"
    ]
    for k in keep:
        if k not in df.columns:
            df[k] = ""
    out = df[keep].copy()
    out.insert(0, "source", source)
    out.insert(1, "fetched_at_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))

    OUT_LATEST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_LATEST, index=False)
    print(f"Wrote: {OUT_LATEST} (source={source}, rows={len(out)})")

    if args.season:
        seasonal = INJ_DIR / f"injury_reports_{args.season}.csv"
        out.to_csv(seasonal, index=False)
        print(f"Wrote: {seasonal}")

if __name__ == "__main__":
    main()
