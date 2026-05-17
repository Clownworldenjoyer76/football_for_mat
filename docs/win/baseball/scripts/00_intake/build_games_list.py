#!/usr/bin/env python3
# docs/win/baseball/scripts/00_intake/build_games_list.py
#
# Runs after scrape_mlb_raw.py and odds_parse.py.
# Joins mlb_raw to sportsbook on team + game hour to produce
# an authoritative {date}_games.csv for each date.
# Handles doubleheaders by using game hour as a tiebreaker.

import csv
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

MLB_RAW_DIR  = Path("docs/win/baseball/00_intake/mlb_raw")
BOOK_DIR     = Path("docs/win/baseball/00_intake/sportsbook")
MAPS_DIR     = Path("docs/win/baseball/maps")
OUT_DIR      = Path("docs/win/baseball/00_intake/games")
ERROR_DIR    = Path("docs/win/baseball/errors/00_intake")

OUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "build_games_list.txt"

OUTPUT_HEADER = [
    "gamePk", "game_id", "game_date", "game_time",
    "home_team", "away_team", "home_team_id", "away_team_id",
    "venue_id", "doubleheader", "gameNumber",
    "home_pitcher_id", "away_pitcher_id", "day_night",
]

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

def _now():
    return datetime.now(timezone.utc).isoformat()


def log(msg: str, level: str = "INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {msg.rstrip()}\n")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def norm(s):
    """Lowercase, strip punctuation, collapse spaces for team name matching."""
    import re
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def load_csv(path: Path) -> list:
    if not path.exists():
        log(f"MISSING: {path}", "WARN")
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_team_map() -> dict:
    """Returns dict: norm(team_name) -> team_id"""
    rows = load_csv(MAPS_DIR / "mlb_team_ids.csv")
    m = {}
    for r in rows:
        tid = r.get("team_id", "").strip()
        if not tid:
            continue
        for col in ["name", "team_name", "short_name", "club_name", "franchise_name"]:
            val = norm(r.get(col, ""))
            if val:
                m[val] = tid
    return m


def build_id_to_name_map(rows: list) -> dict:
    """Returns dict: team_id -> full name"""
    return {r.get("team_id", "").strip(): r.get("name", "").strip() for r in rows}


def utc_to_local_hour(utc_str: str, tz_id: str = "America/New_York") -> int:
    """Convert UTC time string to local hour integer."""
    try:
        dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
        local = dt.astimezone(ZoneInfo(tz_id))
        return local.hour
    except Exception:
        return -1


def parse_book_hour(time_str: str) -> int:
    """Parse sportsbook game_time (HH:MM:SS 24hr) to hour integer."""
    try:
        return int(time_str.strip().split(":")[0])
    except Exception:
        return -1


# ─────────────────────────────────────────────
# PROCESS ONE DATE
# ─────────────────────────────────────────────

def process_date(date_str: str, team_map: dict, id_to_name: dict, summary: dict) -> None:
    raw_path  = MLB_RAW_DIR / f"{date_str}_mlb_raw.csv"
    book_path = BOOK_DIR    / f"{date_str}_MLB.csv"

    raw_rows  = load_csv(raw_path)
    book_rows = load_csv(book_path)

    if not raw_rows:
        log(f"{date_str} | no mlb_raw — skipping", "WARN")
        summary["skipped"] += 1
        return

    if not book_rows:
        log(f"{date_str} | no sportsbook — skipping", "WARN")
        summary["skipped"] += 1
        return

    # Build sportsbook index: (norm_home, norm_away, hour) -> row
    book_idx = {}
    for b in book_rows:
        hour = parse_book_hour(b.get("game_time", ""))
        key  = (norm(b["home_team"]), norm(b["away_team"]), hour)
        if key in book_idx:
            log(f"{date_str} | DUPLICATE sportsbook key: {key}", "WARN")
        book_idx[key] = b

    output_rows = []
    matched     = 0
    unmatched   = 0

    for r in raw_rows:
        game_pk    = r.get("gamePk", "")
        game_time  = r.get("game_time", "")
        venue_id   = r.get("venue_id", "")
        home_tid   = r.get("home_team_id", "")
        away_tid   = r.get("away_team_id", "")

        # Look up team names from id_to_name map
        home_name = id_to_name.get(home_tid, "")
        away_name = id_to_name.get(away_tid, "")

        # Get local hour from UTC game_time
        local_hour = utc_to_local_hour(game_time)

        key = (norm(home_name), norm(away_name), local_hour)
        b   = book_idx.get(key)

        if not b:
            # Try without hour match as fallback (non-doubleheader safety net)
            fallback_matches = [
                v for k, v in book_idx.items()
                if k[0] == norm(home_name) and k[1] == norm(away_name)
            ]
            if len(fallback_matches) == 1:
                b = fallback_matches[0]
                log(f"{date_str} | {home_name} vs {away_name} — hour mismatch, used fallback match", "WARN")
            elif len(fallback_matches) > 1:
                log(f"{date_str} | {home_name} vs {away_name} — DOUBLEHEADER hour match failed, no fallback", "WARN")
                unmatched += 1
                continue
            else:
                log(f"{date_str} | UNMATCHED: {home_name} vs {away_name} hour={local_hour}", "WARN")
                unmatched += 1
                continue

        matched += 1
        output_rows.append({
            "gamePk":          game_pk,
            "game_id":         b["game_id"],
            "game_date":       r.get("game_date", ""),
            "game_time":       b["game_time"],  # use sportsbook 24hr time
            "home_team":       b["home_team"],  # use sportsbook team names (consistent downstream)
            "away_team":       b["away_team"],
            "home_team_id":    home_tid,
            "away_team_id":    away_tid,
            "venue_id":        venue_id,
            "doubleheader":    r.get("doubleheader", "N"),
            "gameNumber":      r.get("gameNumber", "1"),
            "home_pitcher_id": r.get("home_pitcher_id", ""),
            "away_pitcher_id": r.get("away_pitcher_id", ""),
            "day_night":       r.get("day_night", ""),
        })

    log(f"{date_str} | matched={matched} unmatched={unmatched}")
    summary["total_matched"]   += matched
    summary["total_unmatched"] += unmatched

    if output_rows:
        out_path = OUT_DIR / f"{date_str}_games.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADER)
            writer.writeheader()
            writer.writerows(output_rows)
        log(f"{date_str} | WROTE: {out_path.name} ({len(output_rows)} games)")
        summary["files_written"] += 1
    else:
        log(f"{date_str} | no matched games — file not written", "WARN")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== build_games_list RUN {_now()} ===\n")

    summary = {
        "files_written":   0,
        "total_matched":   0,
        "total_unmatched": 0,
        "skipped":         0,
        "errors":          0,
    }

    try:
        team_map = load_team_map()
        team_rows = load_csv(MAPS_DIR / "mlb_team_ids.csv")
        id_to_name = build_id_to_name_map(team_rows)
        log(f"Team map loaded: {len(team_map)} entries | id_to_name: {len(id_to_name)} entries")

        # Process all dates that have an mlb_raw file
        raw_files = sorted(MLB_RAW_DIR.glob("*_mlb_raw.csv"))
        log(f"mlb_raw files found: {len(raw_files)}")

        for rf in raw_files:
            date_str = rf.stem.replace("_mlb_raw", "")
            try:
                process_date(date_str, team_map, id_to_name, summary)
            except Exception as e:
                log(f"{date_str} FAILED: {e}\n{traceback.format_exc()}", "ERROR")
                summary["errors"] += 1

    except Exception as e:
        log(f"FATAL: {e}\n{traceback.format_exc()}", "ERROR")
        summary["errors"] += 1

    status = "SUCCESS" if summary["errors"] == 0 else "COMPLETED WITH ERRORS"
    lines = [
        "",
        "=" * 60,
        f"SUMMARY  {_now()}",
        "=" * 60,
        f"  files_written   : {summary['files_written']}",
        f"  total_matched   : {summary['total_matched']}",
        f"  total_unmatched : {summary['total_unmatched']}",
        f"  skipped         : {summary['skipped']}",
        f"  errors          : {summary['errors']}",
        "",
        f"STATUS: {status}",
        "=" * 60,
    ]
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"build_games_list complete. {summary['files_written']} files written. Status: {status}")


if __name__ == "__main__":
    main()
