#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/clean_basketball_inputs.py
#
# Reads originals from:
#   docs/win/basketball/00_intake/predictions/{league}/
#   docs/win/basketball/00_intake/sportsbook/{league}/
#
# Writes cleaned copies to:
#   docs/win/basketball/00_intake/predictions/predictions_cleaned/{league}/
#   docs/win/basketball/00_intake/sportsbook/sportsbook_cleaned/{league}/
#
# Originals are never mutated. Filenames are preserved.

import csv
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

CLEANED_PREDICTION_DIRS = {
    "NBA": Path("docs/win/basketball/00_intake/predictions/predictions_cleaned/nba"),
    "NCAAM": Path("docs/win/basketball/00_intake/predictions/predictions_cleaned/ncaam"),
    "WNBA": Path("docs/win/basketball/00_intake/predictions/predictions_cleaned/wnba"),
}

CLEANED_SPORTSBOOK_DIRS = {
    "NBA": Path("docs/win/basketball/00_intake/sportsbook/sportsbook_cleaned/nba"),
    "NCAAM": Path("docs/win/basketball/00_intake/sportsbook/sportsbook_cleaned/ncaam"),
    "WNBA": Path("docs/win/basketball/00_intake/sportsbook/sportsbook_cleaned/wnba"),
}

ERROR_DIR = Path("docs/win/basketball/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "clean_basketball_inputs.txt"

# =========================
# SETTINGS
# =========================

MONEYLINE_HOLD_MIN = 1.01
MONEYLINE_HOLD_MAX = 1.08

# Same hold range applies to totals and spreads. Aliased for clarity at call sites.
ODDS_HOLD_MIN = MONEYLINE_HOLD_MIN
ODDS_HOLD_MAX = MONEYLINE_HOLD_MAX

# Markets to validate. (decimal_col_a, decimal_col_b, market_label)
ODDS_CHECKS = [
    ("home_dk_moneyline_decimal", "away_dk_moneyline_decimal", "ML"),
    ("dk_total_over_decimal",     "dk_total_under_decimal",    "TOTAL"),
    ("home_dk_spread_decimal",    "away_dk_spread_decimal",    "SPREAD"),
]

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
# Acts as a marker that the row has been processed by this cleaner.
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
    path.parent.mkdir(parents=True, exist_ok=True)
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
# LOAD ALL ORIGINALS INTO MEMORY
# =========================
# Returns: { league: { path: [fieldnames, rows] } }

def load_all(dir_map):
    loaded = {}
    for league, folder in dir_map.items():
        loaded[league] = {}
        for path in csv_files(folder):
            fieldnames, rows = read_csv(path)
            loaded[league][path] = [fieldnames, rows]
    return loaded

# =========================
# STEP 1: DROP BAD ODDS ROWS (IN MEMORY)
# =========================
# Validates moneyline, totals, and spreads decimal odds across ALL leagues.
# Drops a row if any market on that row has:
#   - decimals missing / unparseable / <= 1, OR
#   - implied book hold outside [ODDS_HOLD_MIN, ODDS_HOLD_MAX]
# A bad row is dropped from the snapshot entirely (one bad market = the
# whole row is suspect at that timestamp).

def drop_bad_odds_rows(book_files):
    removed_by_market = {market: 0 for _, _, market in ODDS_CHECKS}

    for league, files in book_files.items():
        for path, data in files.items():
            fieldnames, rows = data
            fieldset = set(fieldnames)

            kept = []
            file_removed = 0

            for row in rows:
                drop_reason = None

                for col_a, col_b, market in ODDS_CHECKS:
                    # Skip the check if this file's schema doesn't include
                    # the market's decimal columns.
                    if col_a not in fieldset or col_b not in fieldset:
                        continue

                    dec_a = to_float(row.get(col_a))
                    dec_b = to_float(row.get(col_b))

                    if dec_a is None or dec_b is None or dec_a <= 1 or dec_b <= 1:
                        drop_reason = (market, "MALFORMED",
                                       row.get(col_a), row.get(col_b), None)
                        break

                    hold = (1.0 / dec_a) + (1.0 / dec_b)
                    if hold < ODDS_HOLD_MIN or hold > ODDS_HOLD_MAX:
                        drop_reason = (market, "HOLD", dec_a, dec_b, hold)
                        break

                if drop_reason is not None:
                    market, kind, val_a, val_b, hold = drop_reason
                    removed_by_market[market] += 1
                    file_removed += 1
                    if kind == "MALFORMED":
                        log(
                            f"DROP_BAD_{market}_MALFORMED | {path} | {row_key(row)} | "
                            f"a={val_a!r} b={val_b!r}"
                        )
                    else:
                        log(
                            f"DROP_BAD_{market}_HOLD | {path} | {row_key(row)} | "
                            f"a={val_a} b={val_b} hold={round(hold, 6)}"
                        )
                    continue

                kept.append(row)

            if file_removed:
                data[1] = kept
                log(f"FILTERED | {path} | removed_bad_odds={file_removed}")

    return removed_by_market

# =========================
# STEP 2: BUILD INDEXES FROM IN-MEMORY DATA
# =========================

def build_pred_index(pred_files):
    index = {}
    for league, files in pred_files.items():
        for path, (fieldnames, rows) in files.items():
            for row in rows:
                key = row_key(row)
                if key:
                    index[key] = {"league": league, "path": path, "row": row}
    return index

def build_book_index(book_files):
    index = {}
    for league, files in book_files.items():
        for path, (fieldnames, rows) in files.items():
            for row in rows:
                key = row_key(row)
                if key:
                    index.setdefault(key, []).append(
                        {"league": league, "path": path, "row": row}
                    )
    return index

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
# STEP 4: DROP OUTLIER KEYS FROM IN-MEMORY DATA
# =========================

def drop_outlier_keys(loaded_files, drop_keys, label):
    removed = 0

    for league, files in loaded_files.items():
        for path, data in files.items():
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
                data[1] = kept
                log(f"FILTERED | {label} | {path} | removed_outliers={file_removed}")

    return removed

# =========================
# STEP 5: APPLY PREDICTION BIASES (IN MEMORY)
# =========================
# Total bias is split evenly across home and away so that
# (new_home + new_away) == new_total stays internally consistent.

def apply_prediction_biases(pred_files):
    files_with_biased_rows = 0
    rows_adjusted = 0
    rows_skipped_already_flagged = 0

    required = {
        "home_projected_points",
        "away_projected_points",
        "total_projected_points",
    }

    for league, files in pred_files.items():
        margin_bias = MARGIN_BIAS[league]
        total_bias = TOTAL_BIAS[league]
        margin_half = margin_bias / 2.0
        total_half = total_bias / 2.0

        for path, data in files.items():
            fieldnames, rows = data

            if not required.issubset(set(fieldnames)):
                log(f"WARN | Missing prediction columns, skipped: {path}")
                continue

            # Ensure sentinel column exists in the output schema. This is
            # written to disk later regardless of whether any rows were
            # biased, so downstream schema checks always find it.
            if BIAS_FLAG_COLUMN not in fieldnames:
                fieldnames = list(fieldnames) + [BIAS_FLAG_COLUMN]
                data[0] = fieldnames

            file_adjusted = 0
            file_skipped = 0

            for row in rows:
                # Defensive: skip rows that already carry the flag (originals
                # should never have this column, but if a manual copy slipped
                # through, don't double-bias it).
                if str(row.get(BIAS_FLAG_COLUMN, "")).strip() == BIAS_FLAG_VALUE:
                    file_skipped += 1
                    rows_skipped_already_flagged += 1
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

                file_adjusted += 1
                rows_adjusted += 1

            if file_adjusted:
                files_with_biased_rows += 1
                log(
                    f"BIASED | {path} | league={league} "
                    f"margin_bias={margin_bias} total_bias={total_bias} "
                    f"adjusted={file_adjusted} skipped_already_flagged={file_skipped}"
                )

    return files_with_biased_rows, rows_adjusted, rows_skipped_already_flagged

# =========================
# STEP 6: WRITE EVERYTHING TO CLEANED FOLDERS
# =========================

def write_cleaned(loaded_files, cleaned_dirs, label):
    files_written = 0
    rows_written = 0

    for league, files in loaded_files.items():
        cleaned_root = cleaned_dirs[league]

        for path, (fieldnames, rows) in files.items():
            new_path = cleaned_root / path.name
            write_csv(new_path, fieldnames, rows)
            files_written += 1
            rows_written += len(rows)
            log(f"WROTE | {label} | {new_path} | rows={len(rows)}")

    return files_written, rows_written

# =========================
# MAIN
# =========================

def main():
    log("INFO | Starting basketball input cleanup (cleaned-folder mode)")

    # Load originals into memory
    pred_files = load_all(PREDICTION_DIRS)
    book_files = load_all(SPORTSBOOK_DIRS)

    pred_loaded = sum(len(v) for v in pred_files.values())
    book_loaded = sum(len(v) for v in book_files.values())
    log(f"LOADED | predictions_files={pred_loaded} sportsbook_files={book_loaded}")

    # Step 1: drop bad odds rows across all leagues / all markets (in memory)
    bad_odds_removed = drop_bad_odds_rows(book_files)
    bad_ml_removed     = bad_odds_removed.get("ML", 0)
    bad_total_removed  = bad_odds_removed.get("TOTAL", 0)
    bad_spread_removed = bad_odds_removed.get("SPREAD", 0)
    bad_total_removed_total = bad_ml_removed + bad_total_removed + bad_spread_removed

    # Step 2: build indexes for outlier detection
    pred_index = build_pred_index(pred_files)
    book_index = build_book_index(book_files)

    # Step 3: find outlier game keys
    outlier_keys = find_outlier_keys(pred_index, book_index)

    # Step 4: drop outliers from both predictions and sportsbook (in memory)
    pred_outliers_removed = drop_outlier_keys(pred_files, outlier_keys, "PREDICTIONS")
    book_outliers_removed = drop_outlier_keys(book_files, outlier_keys, "SPORTSBOOK")

    # Step 5: apply biases to predictions (in memory)
    bias_files, bias_rows, bias_skipped = apply_prediction_biases(pred_files)

    # Step 6: write everything to cleaned folders
    pred_files_written, pred_rows_written = write_cleaned(
        pred_files, CLEANED_PREDICTION_DIRS, "PREDICTIONS"
    )
    book_files_written, book_rows_written = write_cleaned(
        book_files, CLEANED_SPORTSBOOK_DIRS, "SPORTSBOOK"
    )

    log("")
    log("============================================================")
    log("SUMMARY")
    log("============================================================")
    log(f"bad_ml_rows_removed              : {bad_ml_removed}")
    log(f"bad_total_rows_removed           : {bad_total_removed}")
    log(f"bad_spread_rows_removed          : {bad_spread_removed}")
    log(f"bad_odds_rows_removed_total      : {bad_total_removed_total}")
    log(f"outlier_game_keys_found          : {len(outlier_keys)}")
    log(f"prediction_outlier_rows_removed  : {pred_outliers_removed}")
    log(f"sportsbook_outlier_rows_removed  : {book_outliers_removed}")
    log(f"bias_files_with_adjusted_rows    : {bias_files}")
    log(f"bias_rows_adjusted               : {bias_rows}")
    log(f"bias_rows_skipped_already_flagged: {bias_skipped}")
    log(f"prediction_files_written         : {pred_files_written}")
    log(f"prediction_rows_written          : {pred_rows_written}")
    log(f"sportsbook_files_written         : {book_files_written}")
    log(f"sportsbook_rows_written          : {book_rows_written}")
    log("STATUS: SUCCESS")
    log("============================================================")

    print("STATUS: SUCCESS")
    print(f"bad_ml_rows_removed              : {bad_ml_removed}")
    print(f"bad_total_rows_removed           : {bad_total_removed}")
    print(f"bad_spread_rows_removed          : {bad_spread_removed}")
    print(f"bad_odds_rows_removed_total      : {bad_total_removed_total}")
    print(f"outlier_game_keys_found          : {len(outlier_keys)}")
    print(f"prediction_outlier_rows_removed  : {pred_outliers_removed}")
    print(f"sportsbook_outlier_rows_removed  : {book_outliers_removed}")
    print(f"bias_files_with_adjusted_rows    : {bias_files}")
    print(f"bias_rows_adjusted               : {bias_rows}")
    print(f"bias_rows_skipped_already_flagged: {bias_skipped}")
    print(f"prediction_files_written         : {pred_files_written}")
    print(f"prediction_rows_written          : {pred_rows_written}")
    print(f"sportsbook_files_written         : {book_files_written}")
    print(f"sportsbook_rows_written          : {book_rows_written}")
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
