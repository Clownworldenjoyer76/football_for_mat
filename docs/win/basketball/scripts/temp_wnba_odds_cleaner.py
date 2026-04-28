#!/usr/bin/env python3
# docs/win/basketball/scripts/temp_wnba_odds_cleaner.py

import csv
import traceback
from pathlib import Path
from datetime import datetime

# =========================
# PATHS
# =========================

INPUT_DIR  = Path("docs/win/basketball/00_intake/temp")
OUTPUT_DIR = Path("docs/win/basketball/00_intake/sportsbook/wnba")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERROR_DIR = Path("docs/win/basketball/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "temp_wnba_odds_cleaner.txt"

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== temp_wnba_odds_cleaner RUN {datetime.now().isoformat()} ===\n")


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")


# =========================
# AMERICAN TO DECIMAL
# =========================

def american_to_decimal(american):
    if american is None:
        return ""
    try:
        val = int(str(american).strip())
    except (ValueError, TypeError):
        return ""
    if val == 0:
        return ""
    if val > 0:
        return round((val / 100) + 1, 10)
    else:
        return round((100 / abs(val)) + 1, 10)


# =========================
# OUTPUT FIELDNAMES
# =========================

OUTPUT_FIELDNAMES = [
    "sport",
    "league",
    "game_date",
    "game_id",
    "odds_last_update",
    "game_time",
    "home_team",
    "away_team",
    "home_spread",
    "away_spread",
    "total",
    "home_dk_moneyline_american",
    "away_dk_moneyline_american",
    "home_dk_spread_american",
    "away_dk_spread_american",
    "dk_total_over_american",
    "dk_total_under_american",
    "home_dk_moneyline_decimal",
    "away_dk_moneyline_decimal",
    "home_dk_spread_decimal",
    "away_dk_spread_decimal",
    "dk_total_over_decimal",
    "dk_total_under_decimal",
]


# =========================
# PARSE DATE FROM FILENAME
# =========================

def parse_date_from_filename(filename: str) -> str:
    # e.g. basketball_WNBA_2026_05_08.csv -> 2026_05_08
    stem = Path(filename).stem
    parts = stem.split("_")
    # find the date portion: YYYY_MM_DD
    for i in range(len(parts) - 2):
        if len(parts[i]) == 4 and parts[i].isdigit():
            return f"{parts[i]}_{parts[i+1]}_{parts[i+2]}"
    return stem


# =========================
# MAIN
# =========================

def main():
    files_written = []
    files_processed = 0
    rows_written = 0

    try:
        if not INPUT_DIR.exists():
            log(f"INPUT DIR NOT FOUND: {INPUT_DIR}")
            log("STATUS: SUCCESS (nothing to do)")
            print("temp_wnba_odds_cleaner: no input directory found.")
            return

        csv_files = sorted(INPUT_DIR.glob("*.csv"))

        if not csv_files:
            log(f"NO INPUT FILES: {INPUT_DIR}")
            log("STATUS: SUCCESS (nothing to do)")
            print("temp_wnba_odds_cleaner: no input files found.")
            return

        for csv_path in csv_files:
            log(f"PROCESSING: {csv_path.name}")
            try:
                game_date = parse_date_from_filename(csv_path.name)
                out_path = OUTPUT_DIR / f"{game_date}_WNBA_odds.csv"

                with open(csv_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    input_rows = list(reader)

                output_rows = []
                for row in input_rows:
                    # Step 2: rename 'league' -> 'sport', then step 3: rename 'market' -> 'league'
                    sport = row.get("league", "").strip()
                    league = row.get("market", "").strip()

                    home_ml_am  = row.get("home_dk_moneyline_american", "").strip()
                    away_ml_am  = row.get("away_dk_moneyline_american", "").strip()
                    home_sp_am  = row.get("home_dk_spread_american", "").strip()
                    away_sp_am  = row.get("away_dk_spread_american", "").strip()
                    over_am     = row.get("dk_total_over_american", "").strip()
                    under_am    = row.get("dk_total_under_american", "").strip()

                    output_rows.append({
                        "sport":                      sport,
                        "league":                     league,
                        "game_date":                  row.get("game_date", "").strip(),
                        "game_id":                    "",
                        "odds_last_update":           "",
                        "game_time":                  row.get("game_time", "").strip(),
                        "home_team":                  row.get("home_team", "").strip(),
                        "away_team":                  row.get("away_team", "").strip(),
                        "home_spread":                row.get("home_spread", "").strip(),
                        "away_spread":                row.get("away_spread", "").strip(),
                        "total":                      row.get("total", "").strip(),
                        "home_dk_moneyline_american": home_ml_am,
                        "away_dk_moneyline_american": away_ml_am,
                        "home_dk_spread_american":    home_sp_am,
                        "away_dk_spread_american":    away_sp_am,
                        "dk_total_over_american":     over_am,
                        "dk_total_under_american":    under_am,
                        "home_dk_moneyline_decimal":  american_to_decimal(home_ml_am),
                        "away_dk_moneyline_decimal":  american_to_decimal(away_ml_am),
                        "home_dk_spread_decimal":     american_to_decimal(home_sp_am),
                        "away_dk_spread_decimal":     american_to_decimal(away_sp_am),
                        "dk_total_over_decimal":      american_to_decimal(over_am),
                        "dk_total_under_decimal":     american_to_decimal(under_am),
                    })

                with open(out_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
                    writer.writeheader()
                    writer.writerows(output_rows)

                files_written.append((str(out_path), len(output_rows)))
                rows_written += len(output_rows)
                files_processed += 1
                log(f"WROTE: {out_path} ({len(output_rows)} rows)")

            except Exception as e:
                log(f"ERROR processing {csv_path.name}: {e}\n{traceback.format_exc()}")

        log("--- SUMMARY ---")
        log(f"Files processed: {files_processed}")
        log(f"Rows written: {rows_written}")
        log(f"Files written: {len(files_written)}")
        for path, count in files_written:
            log(f"  FILE: {path} ({count} rows)")
        log("STATUS: SUCCESS")

        print("temp_wnba_odds_cleaner complete.")

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        raise


if __name__ == "__main__":
    main()
