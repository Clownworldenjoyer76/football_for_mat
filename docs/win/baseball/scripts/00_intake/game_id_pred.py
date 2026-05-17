#!/usr/bin/env python3
# docs/win/baseball/scripts/00_intake/game_id_pred.py
#
# Injects game_id into prediction files using:
#   docs/win/baseball/00_intake/games/{date}_games.csv
#
# Input:
#   docs/win/baseball/00_intake/predictions/{date}_MLB.csv
#   docs/win/baseball/00_intake/games/{date}_games.csv
#
# Output:
#   docs/win/baseball/00_intake/predictions/pred_with_game_id/{date}_MLB.csv
#
# Matching logic:
#   1. Match prediction row to games row by:
#        normalized home_team + normalized away_team + parsed local hour
#   2. Prediction times may be 12-hour format, e.g. "03:05 PM"
#   3. Games times may be 24-hour format, e.g. "15:05:00"
#   4. No team-only fallback is used here, because that can be unsafe for doubleheaders.

import csv
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path


PRED_DIR = Path("docs/win/baseball/00_intake/predictions")
GAMES_DIR = Path("docs/win/baseball/00_intake/games")
OUT_DIR = PRED_DIR / "pred_with_game_id"
ERROR_DIR = Path("docs/win/baseball/errors/00_intake")

OUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "game_id_pred.txt"

OUTPUT_HEADER = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "home_pitcher",
    "away_pitcher",
    "home_prob",
    "away_prob",
    "away_projected_runs",
    "home_projected_runs",
    "total_projected_runs",
]


def _now():
    return datetime.now(timezone.utc).isoformat()


def log(msg: str, level: str = "INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {msg.rstrip()}\n")


def norm(s: str) -> str:
    """
    Lowercase, strip punctuation, collapse spaces.
    Used for team name matching.
    """
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        log(f"MISSING: {path}", "WARN")
        return []

    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_hour(time_str: str) -> int:
    """
    Parses common game_time formats into a 0-23 hour.

    Supported examples:
      "03:05 PM"  -> 15
      "3:05 PM"   -> 15
      "04:10 PM"  -> 16
      "15:05:00"  -> 15
      "15:05"     -> 15

    Returns -1 if invalid.
    """
    s = (time_str or "").strip()
    if not s:
        return -1

    # Normalize spacing around AM/PM.
    s = re.sub(r"\s+", " ", s).strip().upper()

    formats = [
        "%I:%M %p",
        "%I:%M:%S %p",
        "%H:%M:%S",
        "%H:%M",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).hour
        except ValueError:
            continue

    return -1


def build_games_index(games_rows: list[dict], date_str: str) -> dict:
    """
    Builds index:
        (home_team_norm, away_team_norm, hour) -> game_id

    This preserves the doubleheader-safe intent:
        same teams + game hour
    """
    idx = {}

    for g in games_rows:
        home = norm(g.get("home_team", ""))
        away = norm(g.get("away_team", ""))
        hour = parse_hour(g.get("game_time", ""))
        game_id = (g.get("game_id") or "").strip()

        if not home or not away or hour < 0 or not game_id:
            log(
                f"{date_str} | bad games row skipped: "
                f"home={g.get('home_team', '')} "
                f"away={g.get('away_team', '')} "
                f"time={g.get('game_time', '')} "
                f"parsed_hour={hour} "
                f"game_id={game_id}",
                "WARN",
            )
            continue

        key = (home, away, hour)

        if key in idx:
            log(
                f"{date_str} | duplicate games key; later row overwrote earlier row: "
                f"key={key} old_game_id={idx[key]} new_game_id={game_id}",
                "WARN",
            )

        idx[key] = game_id

    return idx


def process_date(date_str: str, pred_path: Path, summary: dict) -> None:
    games_path = GAMES_DIR / f"{date_str}_games.csv"
    out_path = OUT_DIR / f"{date_str}_MLB.csv"

    pred_rows = load_csv(pred_path)
    games_rows = load_csv(games_path)

    if not pred_rows:
        log(f"{date_str} | no prediction rows — skipping", "WARN")
        summary["skipped"] += 1
        return

    if not games_rows:
        log(f"{date_str} | no games file/rows — skipping", "WARN")
        summary["skipped"] += 1
        return

    games_idx = build_games_index(games_rows, date_str)

    output_rows = []
    matched = 0
    unmatched = 0

    for p in pred_rows:
        home_raw = p.get("home_team", "")
        away_raw = p.get("away_team", "")
        time_raw = p.get("game_time", "")

        home = norm(home_raw)
        away = norm(away_raw)
        hour = parse_hour(time_raw)

        key = (home, away, hour)
        game_id = games_idx.get(key, "")

        if game_id:
            matched += 1
        else:
            unmatched += 1
            log(
                f"{date_str} | unmatched prediction: "
                f"away={away_raw} home={home_raw} "
                f"time={time_raw} parsed_hour={hour} key={key}",
                "WARN",
            )

        output_rows.append({
            "game_id": game_id,
            "sport": p.get("sport", ""),
            "league": p.get("league", ""),
            "game_date": p.get("game_date", ""),
            "game_time": p.get("game_time", ""),
            "home_team": p.get("home_team", ""),
            "away_team": p.get("away_team", ""),
            "home_pitcher": p.get("home_pitcher", ""),
            "away_pitcher": p.get("away_pitcher", ""),
            "home_prob": p.get("home_prob", ""),
            "away_prob": p.get("away_prob", ""),
            "away_projected_runs": p.get("away_projected_runs", ""),
            "home_projected_runs": p.get("home_projected_runs", ""),
            "total_projected_runs": p.get("total_projected_runs", ""),
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADER)
        writer.writeheader()
        writer.writerows(output_rows)

    log(
        f"{date_str} | WROTE: {out_path} | rows={len(output_rows)} "
        f"matched={matched} unmatched={unmatched}"
    )

    summary["files_written"] += 1
    summary["total_rows"] += len(output_rows)
    summary["matched"] += matched
    summary["unmatched"] += unmatched


def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== game_id_pred RUN {_now()} ===\n")

    summary = {
        "files_written": 0,
        "total_rows": 0,
        "matched": 0,
        "unmatched": 0,
        "skipped": 0,
        "errors": 0,
    }

    try:
        pred_files = sorted(PRED_DIR.glob("*_MLB.csv"))

        # Avoid accidentally processing output files if script is rerun.
        pred_files = [
            p for p in pred_files
            if "pred_with_game_id" not in str(p)
        ]

        log(f"prediction files found: {len(pred_files)}")

        for pred_path in pred_files:
            date_str = pred_path.stem.replace("_MLB", "")

            try:
                process_date(date_str, pred_path, summary)
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
        f"  files_written : {summary['files_written']}",
        f"  total_rows    : {summary['total_rows']}",
        f"  matched       : {summary['matched']}",
        f"  unmatched     : {summary['unmatched']}",
        f"  skipped       : {summary['skipped']}",
        f"  errors        : {summary['errors']}",
        "",
        f"STATUS: {status}",
        "=" * 60,
    ]

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(
        f"game_id_pred complete. "
        f"{summary['files_written']} files written. "
        f"matched={summary['matched']} unmatched={summary['unmatched']} "
        f"Status: {status}"
    )


if __name__ == "__main__":
    main()
