#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/01_merge/merge_intake.py

import csv
import traceback
from pathlib import Path
from datetime import datetime, UTC


BASE_DIR = Path("docs/win/hockey/nhl")

GAMES_DIR = BASE_DIR / "00_intake" / "games"
SPORTSBOOK_DIR = BASE_DIR / "00_intake" / "sportsbook"
PREDICTIONS_DIR = BASE_DIR / "00_intake" / "predictions"

MERGE_DIR = BASE_DIR / "01_merge"
AUDIT_DIR = MERGE_DIR / "audit"

ERROR_DIR = BASE_DIR / "errors" / "01_merge"
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "merge_intake.txt"

MERGE_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


MERGED_COLUMNS = [
    "sport",
    "league",
    "game_date",
    "game_time",
    "game_id",
    "away_team",
    "home_team",
    "away_prob_moneyline",
    "home_prob_moneyline",
    "away_projected_goals",
    "home_projected_goals",
    "total_projected_goals",
    "away_puck_line",
    "home_puck_line",
    "total",
    "away_dk_moneyline_american",
    "home_dk_moneyline_american",
    "away_dk_moneyline_decimal",
    "home_dk_moneyline_decimal",
    "away_dk_puck_line_american",
    "home_dk_puck_line_american",
    "away_dk_puck_line_decimal",
    "home_dk_puck_line_decimal",
    "dk_total_over_american",
    "dk_total_under_american",
    "dk_total_over_decimal",
    "dk_total_under_decimal",
]

AUDIT_COLUMNS = [
    "game_date",
    "game_id",
    "away_team",
    "home_team",
    "source_present_games",
    "source_present_sportsbook",
    "source_present_predictions",
    "status",
]

REJECTION_COLUMNS = [
    "reason",
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "away_team",
    "home_team",
]


REQUIRED_GAMES_COLUMNS = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
]

REQUIRED_SPORTSBOOK_COLUMNS = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "home_dk_moneyline_american",
    "away_dk_moneyline_american",
    "home_puck_line",
    "away_puck_line",
    "total",
    "home_dk_puck_line_american",
    "away_dk_puck_line_american",
    "dk_total_over_american",
    "dk_total_under_american",
    "home_dk_moneyline_decimal",
    "away_dk_moneyline_decimal",
    "home_dk_puck_line_decimal",
    "away_dk_puck_line_decimal",
    "dk_total_over_decimal",
    "dk_total_under_decimal",
]

REQUIRED_PREDICTION_COLUMNS = [
    "sport",
    "league",
    "game_id",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "home_prob_moneyline",
    "away_prob_moneyline",
    "away_projected_goals",
    "home_projected_goals",
    "total_projected_goals",
]


with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== merge_intake RUN {datetime.now(UTC).isoformat()} ===\n")


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(UTC).isoformat()} | {msg}\n")


def fail(message: str) -> None:
    log(f"FATAL: {message}")
    log("STATUS: FAILED")
    raise SystemExit(message)


def wipe_merge_outputs() -> None:
    removed = 0

    for path in MERGE_DIR.glob("*.csv"):
        path.unlink()
        removed += 1

    log(f"Wiped merge CSV outputs: {removed}")


def load_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    log(f"WROTE {path} ({len(rows)} rows)")


def validate_required_columns(path: Path, rows: list[dict[str, str]], required_columns: list[str]) -> None:
    if not rows:
        return

    missing = [col for col in required_columns if col not in rows[0]]

    if missing:
        fail(f"{path} missing required columns: {missing}")


def validate_game_ids(path: Path, rows: list[dict[str, str]], source_name: str) -> None:
    seen = {}
    missing_rows = []
    duplicate_ids = set()

    for idx, row in enumerate(rows, start=2):
        game_id = str(row.get("game_id", "")).strip()

        if not game_id:
            missing_rows.append(idx)
            continue

        if game_id in seen:
            duplicate_ids.add(game_id)
        else:
            seen[game_id] = idx

    if missing_rows:
        fail(f"{source_name} file has missing game_id values: {path} rows={missing_rows}")

    if duplicate_ids:
        fail(f"{source_name} file has duplicate game_id values: {path} duplicate_game_ids={sorted(duplicate_ids)}")


def rows_by_game_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("game_id", "")).strip(): row for row in rows}


def get_dates() -> list[str]:
    dates = set()

    for path in PREDICTIONS_DIR.glob("hockey_*.csv"):
        dates.add(path.name.replace("hockey_", "").replace(".csv", ""))

    return sorted(dates)


def rejection_from_row(reason: str, row: dict[str, str]) -> dict[str, str]:
    return {
        "reason": reason,
        "game_id": str(row.get("game_id", "")).strip(),
        "sport": row.get("sport", ""),
        "league": row.get("league", ""),
        "game_date": row.get("game_date", ""),
        "game_time": row.get("game_time", ""),
        "away_team": row.get("away_team", ""),
        "home_team": row.get("home_team", ""),
    }


def process_date(date_val: str) -> tuple[int, int, int, bool]:
    games_path = GAMES_DIR / f"{date_val}_nhl_games.csv"
    sportsbook_path = SPORTSBOOK_DIR / f"NHL_{date_val}.csv"
    predictions_path = PREDICTIONS_DIR / f"hockey_{date_val}.csv"

    merged_path = MERGE_DIR / f"{date_val}_NHL_merged.csv"
    audit_path = AUDIT_DIR / f"{date_val}_NHL_merge_audit.csv"
    rejected_sportsbook_path = AUDIT_DIR / f"{date_val}_NHL_rejected_sportsbook.csv"
    rejected_predictions_path = AUDIT_DIR / f"{date_val}_NHL_rejected_predictions.csv"

    log(f"Processing date: {date_val}")
    log(f"Games input: {games_path}")
    log(f"Sportsbook input: {sportsbook_path}")
    log(f"Predictions input: {predictions_path}")

    date_has_failure = False

    if not games_path.exists():
        date_has_failure = True
        log(f"HARD CHECK FAILURE: missing games file: {games_path}")
        games_rows = []
    else:
        games_rows = load_csv(games_path)
        validate_required_columns(games_path, games_rows, REQUIRED_GAMES_COLUMNS)
        validate_game_ids(games_path, games_rows, "games")

    if not sportsbook_path.exists():
        date_has_failure = True
        log(f"HARD CHECK FAILURE: missing sportsbook file: {sportsbook_path}")
        sportsbook_rows = []
    else:
        sportsbook_rows = load_csv(sportsbook_path)
        validate_required_columns(sportsbook_path, sportsbook_rows, REQUIRED_SPORTSBOOK_COLUMNS)
        validate_game_ids(sportsbook_path, sportsbook_rows, "sportsbook")

    if not predictions_path.exists():
        date_has_failure = True
        log(f"HARD CHECK FAILURE: missing predictions file: {predictions_path}")
        prediction_rows = []
    else:
        prediction_rows = load_csv(predictions_path)
        validate_required_columns(predictions_path, prediction_rows, REQUIRED_PREDICTION_COLUMNS)
        validate_game_ids(predictions_path, prediction_rows, "predictions")

    games_map = rows_by_game_id(games_rows)
    sportsbook_map = rows_by_game_id(sportsbook_rows)
    predictions_map = rows_by_game_id(prediction_rows)

    audit_rows = []
    rejected_sportsbook = []
    rejected_predictions = []
    merged_rows = []

    for game_id, row in sportsbook_map.items():
        if game_id not in games_map:
            date_has_failure = True
            rejected_sportsbook.append(rejection_from_row("sportsbook_row_not_found_in_games", row))

    for game_id, row in predictions_map.items():
        if game_id not in games_map:
            date_has_failure = True
            rejected_predictions.append(rejection_from_row("prediction_row_not_found_in_games", row))

    for game_id, game in games_map.items():
        has_sportsbook = game_id in sportsbook_map
        has_prediction = game_id in predictions_map

        if has_sportsbook and has_prediction:
            status = "matched"
        elif not has_sportsbook and not has_prediction:
            status = "missing_sportsbook_and_prediction"
            date_has_failure = True
        elif not has_sportsbook:
            status = "missing_sportsbook"
            date_has_failure = True
        else:
            status = "missing_prediction"
            date_has_failure = True

        audit_rows.append(
            {
                "game_date": game.get("game_date", date_val),
                "game_id": game_id,
                "away_team": game.get("away_team", ""),
                "home_team": game.get("home_team", ""),
                "source_present_games": "1",
                "source_present_sportsbook": "1" if has_sportsbook else "0",
                "source_present_predictions": "1" if has_prediction else "0",
                "status": status,
            }
        )

        if status != "matched":
            continue

        sportsbook = sportsbook_map[game_id]
        prediction = predictions_map[game_id]

        merged_rows.append(
            {
                "sport": game.get("sport", "hockey"),
                "league": game.get("league", "nhl"),
                "game_date": game.get("game_date", date_val),
                "game_time": game.get("game_time", ""),
                "game_id": game_id,
                "away_team": game.get("away_team", ""),
                "home_team": game.get("home_team", ""),
                "away_prob_moneyline": prediction.get("away_prob_moneyline", ""),
                "home_prob_moneyline": prediction.get("home_prob_moneyline", ""),
                "away_projected_goals": prediction.get("away_projected_goals", ""),
                "home_projected_goals": prediction.get("home_projected_goals", ""),
                "total_projected_goals": prediction.get("total_projected_goals", ""),
                "away_puck_line": sportsbook.get("away_puck_line", ""),
                "home_puck_line": sportsbook.get("home_puck_line", ""),
                "total": sportsbook.get("total", ""),
                "away_dk_moneyline_american": sportsbook.get("away_dk_moneyline_american", ""),
                "home_dk_moneyline_american": sportsbook.get("home_dk_moneyline_american", ""),
                "away_dk_moneyline_decimal": sportsbook.get("away_dk_moneyline_decimal", ""),
                "home_dk_moneyline_decimal": sportsbook.get("home_dk_moneyline_decimal", ""),
                "away_dk_puck_line_american": sportsbook.get("away_dk_puck_line_american", ""),
                "home_dk_puck_line_american": sportsbook.get("home_dk_puck_line_american", ""),
                "away_dk_puck_line_decimal": sportsbook.get("away_dk_puck_line_decimal", ""),
                "home_dk_puck_line_decimal": sportsbook.get("home_dk_puck_line_decimal", ""),
                "dk_total_over_american": sportsbook.get("dk_total_over_american", ""),
                "dk_total_under_american": sportsbook.get("dk_total_under_american", ""),
                "dk_total_over_decimal": sportsbook.get("dk_total_over_decimal", ""),
                "dk_total_under_decimal": sportsbook.get("dk_total_under_decimal", ""),
            }
        )

    write_csv(audit_path, AUDIT_COLUMNS, audit_rows)
    write_csv(rejected_sportsbook_path, REJECTION_COLUMNS, rejected_sportsbook)
    write_csv(rejected_predictions_path, REJECTION_COLUMNS, rejected_predictions)

    if merged_rows:
        write_csv(merged_path, MERGED_COLUMNS, merged_rows)
    else:
        log(f"No merged rows written for {date_val}")

    log(
        f"Date summary {date_val}: "
        f"games={len(games_rows)} sportsbook={len(sportsbook_rows)} predictions={len(prediction_rows)} "
        f"merged={len(merged_rows)} rejected_sportsbook={len(rejected_sportsbook)} "
        f"rejected_predictions={len(rejected_predictions)} audit_failures="
        f"{len([r for r in audit_rows if r['status'] != 'matched'])}"
    )

    return len(merged_rows), len(rejected_sportsbook), len(rejected_predictions), date_has_failure


def main() -> None:
    total_merged = 0
    total_rejected_sportsbook = 0
    total_rejected_predictions = 0
    dates_failed = 0

    try:
        wipe_merge_outputs()

        dates = get_dates()
        log(f"Dates found from prediction files: {len(dates)}")

        if not dates:
            fail("No Stage 01 prediction input files found.")

        for date_val in dates:
            merged_count, rejected_sportsbook_count, rejected_predictions_count, date_has_failure = process_date(date_val)

            total_merged += merged_count
            total_rejected_sportsbook += rejected_sportsbook_count
            total_rejected_predictions += rejected_predictions_count

            if date_has_failure:
                dates_failed += 1

        log("--- SUMMARY ---")
        log(f"Dates processed: {len(dates)}")
        log(f"Dates with failures: {dates_failed}")
        log(f"Rows merged: {total_merged}")
        log(f"Rejected sportsbook rows: {total_rejected_sportsbook}")
        log(f"Rejected prediction rows: {total_rejected_predictions}")

        if dates_failed > 0:
            fail(f"Stage 01 merge audit failed for {dates_failed} date(s). See audit/rejection CSVs.")

        log("STATUS: SUCCESS")

    except SystemExit:
        raise
    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        raise


if __name__ == "__main__":
    main()
