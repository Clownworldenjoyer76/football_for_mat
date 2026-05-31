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
# Matching rules:
#   1. Same home/away teams.
#   2. Never reuse a games row / game_id.
#   3. If same matchup has exactly one prediction row and exactly one games row:
#      match them even if source times differ, and log the time difference.
#   4. If same matchup has multiple prediction rows and the same number of games rows:
#      pair by chronological order. This handles doubleheaders.
#   5. Otherwise, match to the closest unused games row time within threshold.
#   6. Predictions without a safe games match are written with blank game_id.

import csv
import math
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

MAX_TIME_DIFF_MINUTES = 90

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

def norm(s: str) -> str:
    """Lowercase, strip punctuation, collapse spaces for team name matching."""
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


def parse_int(value, default=0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def clean_date(value: str) -> str:
    return str(value or "").strip().replace("_", "-")


def parse_prediction_datetime(date_str: str, time_str: str):
    date_clean = clean_date(date_str)
    time_clean = re.sub(r"\s+", " ", str(time_str or "").strip()).upper()

    if not date_clean or not time_clean:
        return None

    formats = [
        "%Y-%m-%d %I:%M %p",
        "%Y-%m-%d %I:%M:%S %p",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ]

    value = f"{date_clean} {time_clean}"

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    return None


def parse_games_datetime(date_str: str, time_str: str):
    date_clean = clean_date(date_str)
    time_clean = str(time_str or "").strip()

    if not date_clean or not time_clean:
        return None

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %I:%M %p",
        "%Y-%m-%d %I:%M:%S %p",
    ]

    value = f"{date_clean} {time_clean}"

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    return None


def minutes_between(a, b):
    if a is None or b is None:
        return None
    return abs((a - b).total_seconds()) / 60.0


def dt_sort_value(dt):
    if dt is None:
        return math.inf
    return dt.timestamp()


def make_output_row(pred_entry: dict, game_id: str) -> dict:
    p = pred_entry["row"]

    return {
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
    }


def matchup_label(entry: dict) -> str:
    return f"{entry.get('away_team', '')} @ {entry.get('home_team', '')}"


# ─────────────────────────────────────────────
# GROUP BUILDERS
# ─────────────────────────────────────────────

def build_prediction_groups(pred_rows: list[dict]) -> tuple[dict, list]:
    groups = {}
    key_order = []

    for idx, p in enumerate(pred_rows):
        key = (norm(p.get("home_team", "")), norm(p.get("away_team", "")))

        if key not in groups:
            groups[key] = []
            key_order.append(key)

        groups[key].append({
            "row": p,
            "key": key,
            "index": idx,
            "dt": parse_prediction_datetime(
                p.get("game_date", ""),
                p.get("game_time", ""),
            ),
        })

    return groups, key_order


def build_games_groups(games_rows: list[dict]) -> dict:
    groups = {}

    for idx, g in enumerate(games_rows):
        key = (norm(g.get("home_team", "")), norm(g.get("away_team", "")))

        if key not in groups:
            groups[key] = []

        groups[key].append({
            "row": g,
            "key": key,
            "index": idx,
            "dt": parse_games_datetime(
                g.get("game_date", ""),
                g.get("game_time", ""),
            ),
            "game_number": parse_int(g.get("gameNumber", "1"), 1),
            "used": False,
        })

    return groups


# ─────────────────────────────────────────────
# PROCESS ONE DATE
# ─────────────────────────────────────────────

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

    pred_groups, pred_key_order = build_prediction_groups(pred_rows)
    games_groups = build_games_groups(games_rows)

    output_by_pred_index = {}
    matched = 0
    unmatched = 0

    for key in pred_key_order:
        preds = pred_groups.get(key, [])
        games = games_groups.get(key, [])

        label = matchup_label(preds[0]["row"]) if preds else str(key)

        if not games:
            for pred_entry in preds:
                output_by_pred_index[pred_entry["index"]] = make_output_row(pred_entry, "")
                unmatched += 1
                log(
                    f"{date_str} | unmatched prediction no games row: "
                    f"{label} pred_time={pred_entry['row'].get('game_time', '')}",
                    "WARN",
                )
            continue

        unused_games = [g for g in games if not g["used"]]

        if len(preds) == 1 and len(unused_games) == 1:
            pred_entry = preds[0]
            game_entry = unused_games[0]
            game_entry["used"] = True

            game_id = (game_entry["row"].get("game_id") or "").strip()
            output_by_pred_index[pred_entry["index"]] = make_output_row(pred_entry, game_id)
            matched += 1

            diff = minutes_between(pred_entry.get("dt"), game_entry.get("dt"))
            diff_text = "" if diff is None else f" diff_minutes={round(diff, 1)}"

            level = "INFO"
            label_text = "MATCHED one-to-one"

            if diff is not None and diff > MAX_TIME_DIFF_MINUTES:
                level = "WARN"
                label_text = "MATCHED one-to-one with time mismatch"

            log(
                f"{date_str} | {label_text}: {label} "
                f"pred_time={pred_entry['row'].get('game_time', '')} "
                f"games_time={game_entry['row'].get('game_time', '')} "
                f"game_id={game_id}"
                f"{diff_text}",
                level,
            )

            continue

        if len(preds) > 1 and len(unused_games) > 1 and len(preds) == len(unused_games):
            sorted_preds = sorted(
                preds,
                key=lambda x: (
                    dt_sort_value(x.get("dt")),
                    x["index"],
                ),
            )

            sorted_games = sorted(
                unused_games,
                key=lambda x: (
                    dt_sort_value(x.get("dt")),
                    x["game_number"],
                    x["index"],
                ),
            )

            log(
                f"{date_str} | ORDER MATCH duplicate matchup: "
                f"{label} pred_count={len(sorted_preds)} games_count={len(sorted_games)}"
            )

            for pred_entry, game_entry in zip(sorted_preds, sorted_games):
                game_entry["used"] = True

                game_id = (game_entry["row"].get("game_id") or "").strip()
                output_by_pred_index[pred_entry["index"]] = make_output_row(pred_entry, game_id)
                matched += 1

                diff = minutes_between(pred_entry.get("dt"), game_entry.get("dt"))
                diff_text = "" if diff is None else f" diff_minutes={round(diff, 1)}"

                level = "INFO"
                if diff is not None and diff > MAX_TIME_DIFF_MINUTES:
                    level = "WARN"

                log(
                    f"{date_str} | MATCHED order: {label} "
                    f"pred_time={pred_entry['row'].get('game_time', '')} "
                    f"games_time={game_entry['row'].get('game_time', '')} "
                    f"gameNumber={game_entry['row'].get('gameNumber', '')} "
                    f"game_id={game_id}"
                    f"{diff_text}",
                    level,
                )

            continue

        sorted_preds = sorted(
            preds,
            key=lambda x: (
                dt_sort_value(x.get("dt")),
                x["index"],
            ),
        )

        for pred_entry in sorted_preds:
            available_games = [g for g in games if not g["used"]]

            if not available_games:
                output_by_pred_index[pred_entry["index"]] = make_output_row(pred_entry, "")
                unmatched += 1
                log(
                    f"{date_str} | unmatched prediction no unused games row: "
                    f"{label} pred_time={pred_entry['row'].get('game_time', '')}",
                    "WARN",
                )
                continue

            scored = []
            for game_entry in available_games:
                diff = minutes_between(pred_entry.get("dt"), game_entry.get("dt"))
                scored.append((diff, game_entry))

            scored_valid = [x for x in scored if x[0] is not None]

            selected = None
            selected_diff = None

            if scored_valid:
                selected_diff, selected = min(scored_valid, key=lambda x: x[0])

                if selected_diff > MAX_TIME_DIFF_MINUTES:
                    selected = None

            if selected is None:
                diffs_text = ", ".join(
                    [
                        f"{x[1]['row'].get('game_time', '')}:"
                        f"{'NA' if x[0] is None else round(x[0], 1)}"
                        for x in scored
                    ]
                )

                output_by_pred_index[pred_entry["index"]] = make_output_row(pred_entry, "")
                unmatched += 1

                log(
                    f"{date_str} | unmatched prediction time threshold: "
                    f"{label} pred_time={pred_entry['row'].get('game_time', '')} "
                    f"candidate_diffs={diffs_text}",
                    "WARN",
                )
                continue

            selected["used"] = True

            game_id = (selected["row"].get("game_id") or "").strip()
            output_by_pred_index[pred_entry["index"]] = make_output_row(pred_entry, game_id)
            matched += 1

            diff_text = "" if selected_diff is None else f" diff_minutes={round(selected_diff, 1)}"
            log(
                f"{date_str} | MATCHED closest: {label} "
                f"pred_time={pred_entry['row'].get('game_time', '')} "
                f"games_time={selected['row'].get('game_time', '')} "
                f"game_id={game_id}"
                f"{diff_text}"
            )

    for key, games in games_groups.items():
        unused = [g for g in games if not g["used"]]
        for game_entry in unused:
            g = game_entry["row"]
            log(
                f"{date_str} | UNUSED games row: "
                f"{g.get('away_team', '')} @ {g.get('home_team', '')} "
                f"game_id={g.get('game_id', '')} game_time={g.get('game_time', '')}",
                "WARN",
            )

    output_rows = []

    for idx in range(len(pred_rows)):
        if idx in output_by_pred_index:
            output_rows.append(output_by_pred_index[idx])
        else:
            pred_entry = {
                "row": pred_rows[idx],
                "index": idx,
            }
            output_rows.append(make_output_row(pred_entry, ""))
            unmatched += 1
            log(
                f"{date_str} | prediction row not processed: "
                f"index={idx} away={pred_rows[idx].get('away_team', '')} "
                f"home={pred_rows[idx].get('home_team', '')}",
                "ERROR",
            )
            summary["errors"] += 1

    seen_game_ids = {}
    duplicate_output_game_ids = 0

    for row in output_rows:
        game_id = (row.get("game_id") or "").strip()
        if not game_id:
            continue

        if game_id in seen_game_ids:
            duplicate_output_game_ids += 1
            log(
                f"{date_str} | DUPLICATE OUTPUT game_id={game_id} "
                f"first={seen_game_ids[game_id]} "
                f"second={row.get('away_team', '')} @ {row.get('home_team', '')} "
                f"time={row.get('game_time', '')}",
                "ERROR",
            )
        else:
            seen_game_ids[game_id] = (
                f"{row.get('away_team', '')} @ {row.get('home_team', '')} "
                f"time={row.get('game_time', '')}"
            )

    if duplicate_output_game_ids:
        summary["errors"] += duplicate_output_game_ids

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


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

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
