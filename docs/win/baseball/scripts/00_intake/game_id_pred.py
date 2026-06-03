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
#   6. Predictions without a safe games match are written to rejection CSV and trigger hard failure.
#
# Step 2 hardening:
#   - Never write blank game_id to pred_with_game_id output.
#   - Hard-fail if any prediction row cannot be assigned a game_id.
#   - Hard-fail if duplicate game_id appears in output.
#   - Hard-fail if matched output row count differs from input prediction row count.
#   - Write rejected prediction rows to:
#       docs/win/baseball/00_intake/predictions/pred_with_game_id/rejections/{date}_unmatched_predictions.csv

import csv
import math
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


PRED_DIR = Path("docs/win/baseball/00_intake/predictions")
GAMES_DIR = Path("docs/win/baseball/00_intake/games")
OUT_DIR = PRED_DIR / "pred_with_game_id"
REJECTION_DIR = OUT_DIR / "rejections"
ERROR_DIR = Path("docs/win/baseball/errors/00_intake")

OUT_DIR.mkdir(parents=True, exist_ok=True)
REJECTION_DIR.mkdir(parents=True, exist_ok=True)
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

REJECTION_HEADER = OUTPUT_HEADER + [
    "reject_reason",
    "candidate_games",
]

REQUIRED_PRED_COLS = [
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

REQUIRED_GAMES_COLS = [
    "gamePk",
    "game_id",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "home_team_id",
    "away_team_id",
    "gameNumber",
]


# ─────────────────────────────────────────────
# LOGGING / FAILURE
# ─────────────────────────────────────────────

def _now():
    return datetime.now(timezone.utc).isoformat()


def log(msg: str, level: str = "INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {msg.rstrip()}\n")


def fail(msg: str):
    log(f"FATAL VALIDATION ERROR: {msg}", "ERROR")
    raise RuntimeError(msg)


# ─────────────────────────────────────────────
# CSV HELPERS
# ─────────────────────────────────────────────

def duplicate_columns(header):
    seen = set()
    dupes = []

    for col in header:
        if col in seen and col not in dupes:
            dupes.append(col)
        seen.add(col)

    return dupes


def assert_no_duplicate_columns(header, label):
    dupes = duplicate_columns(header)

    if dupes:
        fail(f"{label} has duplicate columns: {dupes}")


def assert_required_columns(path: Path, header: list[str], required_cols: list[str], label: str):
    missing = [col for col in required_cols if col not in header]

    if missing:
        fail(f"{label} missing required columns in {path}: {missing}")


def load_csv(path: Path, required_cols=None, label=None) -> list[dict]:
    if not path.exists():
        log(f"MISSING: {path}", "WARN")
        return []

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []

        assert_no_duplicate_columns(header, f"{label or path} input")

        if required_cols:
            assert_required_columns(path, header, required_cols, label or str(path))

        return list(reader)


def write_csv(path: Path, header: list[str], rows: list[dict]):
    assert_no_duplicate_columns(header, f"{path} output")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    log(f"WROTE: {path} | rows={len(rows)}")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def norm(s: str) -> str:
    """Lowercase, strip punctuation, collapse spaces for team name matching."""
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


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


def make_rejection_row(pred_entry: dict, reason: str, candidate_games: str = "") -> dict:
    row = make_output_row(pred_entry, "")
    row["reject_reason"] = reason
    row["candidate_games"] = candidate_games
    return row


def matchup_label(row: dict) -> str:
    return f"{row.get('away_team', '')} @ {row.get('home_team', '')}"


def describe_game_entry(game_entry: dict) -> str:
    g = game_entry["row"]
    diff_fields = [
        f"game_id={g.get('game_id', '')}",
        f"gamePk={g.get('gamePk', '')}",
        f"time={g.get('game_time', '')}",
        f"gameNumber={g.get('gameNumber', '')}",
        f"away={g.get('away_team', '')}",
        f"home={g.get('home_team', '')}",
    ]
    return "|".join(diff_fields)


def describe_candidates(scored: list[tuple]) -> str:
    parts = []

    for diff, game_entry in scored:
        diff_text = "NA" if diff is None else str(round(diff, 1))
        parts.append(f"{describe_game_entry(game_entry)}|diff_minutes={diff_text}")

    return "; ".join(parts)


# ─────────────────────────────────────────────
# GROUP BUILDERS
# ─────────────────────────────────────────────

def build_prediction_groups(pred_rows: list[dict]) -> tuple[dict, list]:
    groups = {}
    key_order = []

    for idx, p in enumerate(pred_rows):
        key = (norm(p.get("home_team", "")), norm(p.get("away_team", "")))

        if not key[0] or not key[1]:
            fail(
                f"prediction row has blank team at csv_row={idx + 2}: "
                f"away={p.get('away_team', '')} home={p.get('home_team', '')}"
            )

        if key not in groups:
            groups[key] = []
            key_order.append(key)

        groups[key].append({
            "row": p,
            "key": key,
            "index": idx,
            "csv_row": idx + 2,
            "dt": parse_prediction_datetime(
                p.get("game_date", ""),
                p.get("game_time", ""),
            ),
        })

    return groups, key_order


def build_games_groups(games_rows: list[dict], date_str: str) -> dict:
    groups = {}
    seen_game_ids = {}
    seen_gamepks = {}

    for idx, g in enumerate(games_rows):
        csv_row = idx + 2

        game_id = (g.get("game_id") or "").strip()
        game_pk = (g.get("gamePk") or "").strip()

        if not game_id:
            fail(f"{date_str} | games row has blank game_id at csv_row={csv_row}: {g}")

        if not game_pk:
            fail(f"{date_str} | games row has blank gamePk at csv_row={csv_row}: {g}")

        if game_id in seen_game_ids:
            fail(
                f"{date_str} | duplicate games game_id={game_id} "
                f"first_csv_row={seen_game_ids[game_id]} second_csv_row={csv_row}"
            )

        if game_pk in seen_gamepks:
            fail(
                f"{date_str} | duplicate games gamePk={game_pk} "
                f"first_csv_row={seen_gamepks[game_pk]} second_csv_row={csv_row}"
            )

        seen_game_ids[game_id] = csv_row
        seen_gamepks[game_pk] = csv_row

        key = (norm(g.get("home_team", "")), norm(g.get("away_team", "")))

        if not key[0] or not key[1]:
            fail(
                f"{date_str} | games row has blank team at csv_row={csv_row}: "
                f"away={g.get('away_team', '')} home={g.get('home_team', '')}"
            )

        if key not in groups:
            groups[key] = []

        groups[key].append({
            "row": g,
            "key": key,
            "index": idx,
            "csv_row": csv_row,
            "dt": parse_games_datetime(
                g.get("game_date", ""),
                g.get("game_time", ""),
            ),
            "game_number": parse_int(g.get("gameNumber", "1"), 1),
            "used": False,
        })

    return groups


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

def validate_output_rows(date_str: str, pred_rows: list[dict], output_rows: list[dict]):
    if len(output_rows) != len(pred_rows):
        fail(
            f"{date_str} | output row count mismatch: "
            f"pred_rows={len(pred_rows)} output_rows={len(output_rows)}"
        )

    seen_game_ids = {}

    for idx, row in enumerate(output_rows):
        csv_row = idx + 2
        game_id = (row.get("game_id") or "").strip()

        if not game_id:
            fail(f"{date_str} | output row has blank game_id at csv_row={csv_row}: {row}")

        if game_id in seen_game_ids:
            fail(
                f"{date_str} | duplicate output game_id={game_id} "
                f"first_csv_row={seen_game_ids[game_id]} second_csv_row={csv_row}"
            )

        seen_game_ids[game_id] = csv_row


# ─────────────────────────────────────────────
# PROCESS ONE DATE
# ─────────────────────────────────────────────

def process_date(date_str: str, pred_path: Path, summary: dict) -> None:
    games_path = GAMES_DIR / f"{date_str}_games.csv"
    out_path = OUT_DIR / f"{date_str}_MLB.csv"
    rejection_path = REJECTION_DIR / f"{date_str}_unmatched_predictions.csv"

    pred_rows = load_csv(pred_path, REQUIRED_PRED_COLS, "prediction input")
    games_rows = load_csv(games_path, REQUIRED_GAMES_COLS, "games input")

    if not pred_rows:
        log(f"{date_str} | no prediction rows — skipping", "WARN")
        summary["skipped"] += 1
        return

    if not games_rows:
        fail(f"{date_str} | games file missing or has zero rows: {games_path}")

    pred_groups, pred_key_order = build_prediction_groups(pred_rows)
    games_groups = build_games_groups(games_rows, date_str)

    output_by_pred_index = {}
    rejection_rows = []
    matched = 0

    for key in pred_key_order:
        preds = pred_groups.get(key, [])
        games = games_groups.get(key, [])

        label = matchup_label(preds[0]["row"]) if preds else str(key)

        if not games:
            for pred_entry in preds:
                rejection_rows.append(make_rejection_row(
                    pred_entry,
                    "no_games_row_for_same_home_away",
                    "",
                ))
                log(
                    f"{date_str} | unmatched prediction no games row: "
                    f"{label} pred_time={pred_entry['row'].get('game_time', '')}",
                    "ERROR",
                )
            continue

        unused_games = [g for g in games if not g["used"]]

        if len(preds) == 1 and len(unused_games) == 1:
            pred_entry = preds[0]
            game_entry = unused_games[0]
            game_entry["used"] = True

            game_id = (game_entry["row"].get("game_id") or "").strip()

            if not game_id:
                rejection_rows.append(make_rejection_row(
                    pred_entry,
                    "matched_games_row_has_blank_game_id",
                    describe_game_entry(game_entry),
                ))
                log(
                    f"{date_str} | matched games row has blank game_id: "
                    f"{label} games_csv_row={game_entry['csv_row']}",
                    "ERROR",
                )
                continue

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

                if not game_id:
                    rejection_rows.append(make_rejection_row(
                        pred_entry,
                        "matched_games_row_has_blank_game_id",
                        describe_game_entry(game_entry),
                    ))
                    log(
                        f"{date_str} | matched games row has blank game_id: "
                        f"{label} games_csv_row={game_entry['csv_row']}",
                        "ERROR",
                    )
                    continue

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
                rejection_rows.append(make_rejection_row(
                    pred_entry,
                    "no_unused_games_row_for_same_home_away",
                    "",
                ))
                log(
                    f"{date_str} | unmatched prediction no unused games row: "
                    f"{label} pred_time={pred_entry['row'].get('game_time', '')}",
                    "ERROR",
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
                candidates = describe_candidates(scored)

                rejection_rows.append(make_rejection_row(
                    pred_entry,
                    "no_games_row_within_time_threshold",
                    candidates,
                ))

                log(
                    f"{date_str} | unmatched prediction time threshold: "
                    f"{label} pred_time={pred_entry['row'].get('game_time', '')} "
                    f"candidate_games={candidates}",
                    "ERROR",
                )
                continue

            selected["used"] = True

            game_id = (selected["row"].get("game_id") or "").strip()

            if not game_id:
                rejection_rows.append(make_rejection_row(
                    pred_entry,
                    "matched_games_row_has_blank_game_id",
                    describe_game_entry(selected),
                ))
                log(
                    f"{date_str} | matched games row has blank game_id: "
                    f"{label} games_csv_row={selected['csv_row']}",
                    "ERROR",
                )
                continue

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
            already_rejected = any(
                rejection_row.get("game_date", "") == pred_rows[idx].get("game_date", "")
                and rejection_row.get("game_time", "") == pred_rows[idx].get("game_time", "")
                and rejection_row.get("home_team", "") == pred_rows[idx].get("home_team", "")
                and rejection_row.get("away_team", "") == pred_rows[idx].get("away_team", "")
                for rejection_row in rejection_rows
            )

            if not already_rejected:
                pred_entry = {
                    "row": pred_rows[idx],
                    "index": idx,
                    "csv_row": idx + 2,
                }
                rejection_rows.append(make_rejection_row(
                    pred_entry,
                    "prediction_row_not_processed",
                    "",
                ))

                log(
                    f"{date_str} | prediction row not processed: "
                    f"csv_row={idx + 2} away={pred_rows[idx].get('away_team', '')} "
                    f"home={pred_rows[idx].get('home_team', '')}",
                    "ERROR",
                )

    if rejection_rows:
        write_csv(rejection_path, REJECTION_HEADER, rejection_rows)
        summary["rejected"] += len(rejection_rows)
        summary["errors"] += len(rejection_rows)

        fail(
            f"{date_str} | unmatched prediction rows found: "
            f"count={len(rejection_rows)} rejection_csv={rejection_path}"
        )

    validate_output_rows(date_str, pred_rows, output_rows)

    write_csv(out_path, OUTPUT_HEADER, output_rows)

    log(
        f"{date_str} | WROTE: {out_path} | rows={len(output_rows)} "
        f"matched={matched} unmatched=0"
    )

    summary["files_written"] += 1
    summary["total_rows"] += len(output_rows)
    summary["matched"] += matched


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
        "rejected": 0,
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
            process_date(date_str, pred_path, summary)

    except Exception as e:
        log(f"FATAL: {e}\n{traceback.format_exc()}", "ERROR")
        summary["errors"] += 1

    status = "SUCCESS" if summary["errors"] == 0 else "FAILED"

    lines = [
        "",
        "=" * 60,
        f"SUMMARY  {_now()}",
        "=" * 60,
        f"  files_written : {summary['files_written']}",
        f"  total_rows    : {summary['total_rows']}",
        f"  matched       : {summary['matched']}",
        f"  rejected      : {summary['rejected']}",
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
        f"matched={summary['matched']} rejected={summary['rejected']} "
        f"Status: {status}"
    )

    if summary["errors"] != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
