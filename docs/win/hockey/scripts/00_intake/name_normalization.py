# docs/win/hockey/scripts/00_intake/name_normalization.py
#!/usr/bin/env python3

import csv
import traceback
from pathlib import Path
from datetime import datetime

SPORTSBOOK_DIR  = Path("docs/win/hockey/00_intake/sportsbook")
PREDICTIONS_DIR = Path("docs/win/hockey/00_intake/predictions")

MAP_FILE = Path("mappings/hockey/team_map_hockey.csv")

NO_MAP_DIR = Path("mappings/hockey/no_map")
NO_MAP_DIR.mkdir(parents=True, exist_ok=True)
NO_MAP_FILE = NO_MAP_DIR / "no_map_hockey.csv"

ERROR_DIR = Path("docs/win/hockey/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "name_normalization.txt"

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== name_normalization RUN {datetime.utcnow().isoformat()} ===\n")

def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()} | {msg}\n")

# =========================
# LOAD TEAM MAP
# =========================

team_map = {}

if MAP_FILE.exists():
    with open(MAP_FILE, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row.get("market", "").strip().lower(),
                row.get("alias", "").strip().lower(),
            )
            canonical = row.get("canonical_team", "").strip()
            if key[0] and key[1] and canonical:
                team_map[key] = canonical
    log(f"Team map loaded: {len(team_map)} entries")
else:
    log("WARNING: team_map_hockey.csv not found")

# =========================
# TARGET FILES
# =========================

target_files = []

for f in SPORTSBOOK_DIR.glob("hockey_*.csv"):
    target_files.append(f)

for f in PREDICTIONS_DIR.glob("hockey_*.csv"):
    target_files.append(f)

log(f"Files to process: {len(target_files)}")

# =========================
# PROCESS FILES
# =========================

unmapped        = set()
files_processed = 0
rows_processed  = 0
rows_updated    = 0

try:
    for csv_file in target_files:
        try:
            files_processed += 1
            updated_rows = []
            modified     = False

            with open(csv_file, newline="", encoding="utf-8") as f:
                reader     = csv.DictReader(f)
                fieldnames = reader.fieldnames or []

                for row in reader:
                    rows_processed += 1
                    market = row.get("market", "").strip().lower()

                    for side in ["home_team", "away_team"]:
                        team = row.get(side, "").strip()
                        if not team:
                            continue

                        key = (market, team.lower())

                        if key in team_map:
                            canonical = team_map[key]
                            if row.get(side) != canonical:
                                row[side] = canonical
                                modified  = True
                                rows_updated += 1
                        else:
                            unmapped.add((market, team.lower()))

                    updated_rows.append(row)

            if modified and fieldnames:
                with open(csv_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(updated_rows)
                log(f"UPDATED: {csv_file}")

        except Exception as e:
            log(f"ERROR processing {csv_file}: {e}\n{traceback.format_exc()}")

    # =========================
    # WRITE UNMAPPED (overwrite, not append)
    # =========================
    with open(NO_MAP_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["market", "team"])
        for market, team in sorted(unmapped):
            writer.writerow([market, team])

    log("--- SUMMARY ---")
    log(f"Files processed: {files_processed}")
    log(f"Rows processed: {rows_processed}")
    log(f"Names normalized: {rows_updated}")
    log(f"Unmapped teams: {len(unmapped)}")
    log("STATUS: SUCCESS")

except Exception as e:
    log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
    log("STATUS: FAILED")
    raise

print("Hockey name normalization complete.")
