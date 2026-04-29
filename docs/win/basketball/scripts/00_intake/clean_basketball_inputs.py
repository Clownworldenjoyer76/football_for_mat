#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/clean_basketball_inputs.py

import csv
import math
import traceback
from pathlib import Path
from datetime import datetime

# =========================
# PATHS
# =========================

PREDICTION_DIRS = {
    "NBA": Path("docs/win/basketball/00_intake/predictions/nba"),
    "NCAAM": Path("docs/win/basketball/00_intake/predictions/ncaam"),
    "WNBA": Path("docs/win/basketball/00_intake/predictions/wnba"),
}

SPORTSBOOK_DIRS = {
    "NBA": Path("docs/win/basketball/00_intake/sportsbook/nba"),
    "NCAAM": Path("docs/win/basketball/00_intake/sportsbook/ncaam"),
    "WNBA": Path("docs/win/basketball/00_intake/sportsbook/wnba"),
}

ERROR_DIR = Path("docs/win/basketball/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "clean_basketball_inputs.txt"

# =========================
# SETTINGS
# =========================

MONEYLINE_HOLD_MIN = 1.01
MONEYLINE_HOLD_MAX = 1.08

SPREAD_OUTLIER_MAX = 25.0
TOTAL_OUTLIER_MAX = 40.0

MARGIN_BIAS = {
    "NBA": 0.4,
    "NCAAM": 0.6,
    "WNBA": 0.5,
}

TOTAL_BIAS = {
    "NBA": 0.4,
    "NCAAM": 1.2,
    "WNBA": 0.0,
}

# Sentinel column written into prediction rows after biases are applied.
# Rows containing this flag are skipped on subsequent runs to prevent
# bias stacking.
BIAS_FLAG_COLUMN = "bias_applied"
BIAS_FLAG_VALUE = "1"

# =========================
# LOGGING
# =========================

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== clean_basketball_inputs RUN {datetime.now().isoformat()} ===\n")

def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")

# =========================
# HELPERS
# =========================

def to_float(value):
    try:
        if value is None:
            return None
        value = str(value).strip()
        if value == "":
            return None
        return float(value)
    except Exception:
        return None

def row_key(row):
    game_id = str(row.get("game_id", "")).strip()
    if game_id:
        return game_id

    return "|".join([
        str(row.get("game_date", "")).strip(),
        str(row.get("home_team", "")).strip(),
        str(row.get("away_team", "")).strip(),
    ])

def read_csv(path: Path):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return fieldnames, rows

def write_csv(path: Path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def csv_files(folder: Path):
    if not folder.exists():
        log(f"WARN | Missing folder: {folder}")
        return []
    return sorted(folder.rglob("*.csv"))

# =========================
# STEP 1: DROP BAD NCAAM MONEYLINE ODDS
# =========================
# A row is dropped if EITHER:
#   - the implied book hold falls outside [MIN, MAX], OR
#   - the decimal odds are missing / unparseable / <= 1
#     (these can no longer pass through silently)

def clean_bad_ncaam_moneyline_rows():
    removed = 0

    folder = SPORTSBOOK_DIRS["NCAAM"]

    for path in csv_files(folder):
        fieldnames, rows = read_csv(path)

        if not {
            "home_dk_moneyline_decimal",
            "away_dk_moneyline_decimal",
        }.issubset(set(fieldnames)):
            continue

        kept = []

        for row in rows:
            home_dec = to_float(row.get("home_dk_moneyline_decimal"))
            away_dec = to_float(row.get("away_dk_moneyline_decimal"))

            # Reject malformed decimals outright.
            if home_dec is None or away_dec is None or home_dec <= 1 or away_dec <= 1:
                removed += 1
                log(
                    f"DROP_BAD_ML_MALFORMED | {path} | {row_key(row)} | "
                    f"home_dec={row.get('home_dk_moneyline_decimal')!r} "
                    f"away_dec={row.get('away_dk_moneyline_decimal')!r}"
                )
                continue

            hold = (1.0 / home_dec) + (1.0 / away_dec)

            if hold < MONEYLINE_HOLD_MIN or hold > MONEYLINE_HOLD_MAX:
                removed += 1
                log(
                    f"DROP_BAD_ML_HOLD | {path} | {row_key(row)} | "
                    f"home_dec={home_dec} away_dec={away_dec} hold={round(hold, 6)}"
                )
                continue

            kept.append(row)

        if len(kept) != len(rows):
            write_csv(path, fieldnames, kept)
            log(f"UPDATED | {path} | removed_bad_ml={len(rows) - len(kept)}")

    return removed

# =========================
# STEP 2: BUILD ROW INDEXES
# =========================

def load_prediction_index():
    index = {}
    file_rows = {}

    for league, folder in PREDICTION_DIRS.items():
        for path in csv_files(folder):
            fieldnames, rows = read_csv(path)
            file_rows[path] = [fieldnames, rows]

            for row in rows:
                key = row_key(row)
                if key:
                    index[key] = {
                        "league": league,
                        "path": path,
                        "row": row,
                    }

    return index, file_rows

def load_sportsbook_index():
    index = {}
    file_rows = {}

    for league, folder in SPORTSBOOK_DIRS.items():
        for path in csv_files(folder):
            fieldnames, rows = read_csv(path)
            file_rows[path] = [fieldnames, rows]

            for row in rows:
                key = row_key(row)
                if key:
                    index.setdefault(key, []).append({
                        "league": league,
                        "path": path,
                        "row": row,
                    })

    return index, file_rows

# =========================
# STEP 3: FIND MODEL VS BOOK OUTLIERS
# =========================

def find_outlier_keys(pred_index, book_index):
    drop_keys = set()

    for key, pred_item in pred_index.items():
        if key not in book_index:
            continue

        pred = pred_item["row"]

        home_proj = to_float(pred.get("home_projected_points"))
        away_proj = to_float(pred.get("away_projected_points"))
        model_total = to_float(pred.get("total_projected_points"))

        for book_item in book_index[key]:
            book = book_item["row"]

            book_spread = to_float(book.get("home_spread"))
            book_total = to_float(book.get("total"))

            if home_proj is not None and away_proj is not None and book_spread is not None:
                model_spread = home_proj - away_proj
                spread_diff = abs(model_spread - book_spread)

                if spread_diff > SPREAD_OUTLIER_MAX:
                    drop_keys.add(key)
                    log(
                        f"DROP_OUTLIER_SPREAD | {key} | "
                        f"model_spread={round(model_spread, 4)} "
                        f"book_spread={book_spread} diff={round(spread_diff, 4)}"
                    )

            if model_total is not None and book_total is not None:
                total_diff = abs(model_total - book_total)

                if total_diff > TOTAL_OUTLIER_MAX:
                    drop_keys.add(key)
                    log(
                        f"DROP_OUTLIER_TOTAL | {key} | "
                        f"model_total={model_total} book_total={book_total} "
                        f"diff={round(total_diff, 4)}"
                    )

    return drop_keys

# =========================
# STEP 4: DROP OUTLIERS FROM BOTH
# =========================

def drop_keys_from_files(file_rows, drop_keys, label):
    removed = 0

    for path, data in file_rows.items():
        fieldnames, rows = data

        kept = []
        file_removed = 0

        for row in rows:
            key = row_key(row)

            if key in drop_keys:
                removed += 1
                file_removed += 1
                continue

            kept.append(row)

        if file_removed:
            write_csv(path, fieldnames, kept)
            log(f"UPDATED | {label} | {path} | removed_outliers={file_removed}")

    return removed

# =========================
# STEP 5: APPLY PREDICTION BIASES
# =========================
# Idempotent: rows already flagged with BIAS_FLAG_COLUMN=BIAS_FLAG_VALUE
# are skipped. Total bias is split evenly across home and away so that
# (new_home + new_away) == new_total stays internally consistent.

def apply_prediction_biases():
    updated_files = 0
    adjusted_rows = 0
    skipped_already_applied = 0

    for league, folder in PREDICTION_DIRS.items():
        margin_bias = MARGIN_BIAS[league]
        total_bias = TOTAL_BIAS[league]
        margin_half = margin_bias / 2.0
        total_half = total_bias / 2.0

        for path in csv_files(folder):
            fieldnames, rows = read_csv(path)

            required = {
                "home_projected_points",
                "away_projected_points",
                "total_projected_points",
            }

            if not required.issubset(set(fieldnames)):
                log(f"WARN | Missing prediction columns, skipped: {path}")
                continue

            # Ensure sentinel column exists in the output schema.
            if BIAS_FLAG_COLUMN not in fieldnames:
                fieldnames = list(fieldnames) + [BIAS_FLAG_COLUMN]

            changed = False
            file_adjusted = 0
            file_skipped = 0

            for row in rows:
                # Skip rows already biased on a prior run.
                if str(row.get(BIAS_FLAG_COLUMN, "")).strip() == BIAS_FLAG_VALUE:
                    file_skipped += 1
                    skipped_already_applied += 1
                    continue

                home = to_float(row.get("home_projected_points"))
                away = to_float(row.get("away_projected_points"))
                total = to_float(row.get("total_projected_points"))

                if home is None or away is None or total is None:
                    continue

                new_home = round(home - margin_half - total_half, 2)
                new_away = round(away + margin_half - total_half, 2)
                new_total = round(total - total_bias, 2)

                row["home_projected_points"] = f"{new_home:.2f}"
                row["away_projected_points"] = f"{new_away:.2f}"
                row["total_projected_points"] = f"{new_total:.2f}"
                row[BIAS_FLAG_COLUMN] = BIAS_FLAG_VALUE

                changed = True
                file_adjusted += 1
                adjusted_rows += 1

            if changed:
                write_csv(path, fieldnames, rows)
                updated_files += 1
                log(
                    f"UPDATED | BIAS | {path} | league={league} "
                    f"margin_bias={margin_bias} total_bias={total_bias} "
                    f"adjusted={file_adjusted} skipped_already_applied={file_skipped}"
                )
            elif file_skipped:
                log(
                    f"SKIPPED | BIAS | {path} | league={league} "
                    f"all_rows_already_biased={file_skipped}"
                )

    return updated_files, adjusted_rows, skipped_already_applied

# =========================
# MAIN
# =========================

def main():
    log("INFO | Starting basketball input cleanup")

    bad_ml_removed = clean_bad_ncaam_moneyline_rows()

    pred_index, pred_files = load_prediction_index()
    book_index, book_files = load_sportsbook_index()

    outlier_keys = find_outlier_keys(pred_index, book_index)

    pred_outliers_removed = drop_keys_from_files(pred_files, outlier_keys, "PREDICTIONS")
    book_outliers_removed = drop_keys_from_files(book_files, outlier_keys, "SPORTSBOOK")

    bias_files_updated, bias_rows_adjusted, bias_rows_skipped = apply_prediction_biases()

    log("")
    log("============================================================")
    log("SUMMARY")
    log("============================================================")
    log(f"bad_ncaam_moneyline_rows_removed : {bad_ml_removed}")
    log(f"outlier_game_keys_found          : {len(outlier_keys)}")
    log(f"prediction_outlier_rows_removed  : {pred_outliers_removed}")
    log(f"sportsbook_outlier_rows_removed  : {book_outliers_removed}")
    log(f"bias_files_updated               : {bias_files_updated}")
    log(f"bias_rows_adjusted               : {bias_rows_adjusted}")
    log(f"bias_rows_skipped_already_applied: {bias_rows_skipped}")
    log("STATUS: SUCCESS")
    log("============================================================")

    print("STATUS: SUCCESS")
    print(f"bad_ncaam_moneyline_rows_removed : {bad_ml_removed}")
    print(f"outlier_game_keys_found          : {len(outlier_keys)}")
    print(f"prediction_outlier_rows_removed  : {pred_outliers_removed}")
    print(f"sportsbook_outlier_rows_removed  : {book_outliers_removed}")
    print(f"bias_files_updated               : {bias_files_updated}")
    print(f"bias_rows_adjusted               : {bias_rows_adjusted}")
    print(f"bias_rows_skipped_already_applied: {bias_rows_skipped}")
    print(f"log_file                         : {LOG_FILE}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("STATUS: FAILED")
        log(traceback.format_exc())
        print("STATUS: FAILED")
        print(f"See log: {LOG_FILE}")
        raise
