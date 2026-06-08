#!/usr/bin/env python3
# docs/win/baseball/scripts/00_intake/name_normalization.py

import csv
from pathlib import Path
from datetime import datetime, timezone

# =========================
# PATHS
# =========================

SPORTSBOOK_DIR = Path("docs/win/baseball/00_intake/sportsbook")
PREDICTIONS_DIR = Path("docs/win/baseball/00_intake/predictions")

MAP_FILE = Path("mappings/baseball/team_map_mlb.csv")

NO_MAP_DIR = Path("mappings/baseball/no_map")
NO_MAP_DIR.mkdir(parents=True, exist_ok=True)
NO_MAP_FILE = NO_MAP_DIR / "no_map_mlb.csv"

ERROR_DIR = Path("docs/win/baseball/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "name_normalization_log.txt"

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("")

def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} | {msg}\n")

# =========================
# LOAD TEAM MAP
# =========================

team_map = {}

if MAP_FILE.exists():
    with open(MAP_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            league = row.get("league", "").strip().lower()
            alias = row.get("alias", "").strip().lower()
            canonical = row.get("canonical_team", "").strip()

            if league and alias and canonical:
                team_map[(league, alias)] = canonical
else:
    log("WARNING: team_map_mlb.csv not found")

# =========================
# TARGET FILES ONLY
# =========================

target_files = []

for f in SPORTSBOOK_DIR.glob("*_MLB.csv"):
    target_files.append(f)

for f in PREDICTIONS_DIR.glob("*_MLB.csv"):
    target_files.append(f)

# =========================
# PROCESS FILES
# =========================

unmapped = set()
files_processed = 0
rows_processed = 0
rows_updated = 0

for csv_file in target_files:
    files_processed += 1
    updated_rows = []
    modified = False

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        for row in reader:
            rows_processed += 1
            league = row.get("league", "").strip().lower()

            for side in ["home_team", "away_team"]:
                team = row.get(side, "").strip()
                if not team:
                    continue

                key = (league, team.lower())

                if key in team_map:
                    canonical = team_map[key]
                    if row.get(side) != canonical:
                        row[side] = canonical
                        modified = True
                        rows_updated += 1
                else:
                    unmapped.add((league, team))

            updated_rows.append(row)

    if modified and fieldnames:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)

# =========================
# WRITE UNMAPPED
# =========================

existing = set()

if NO_MAP_FILE.exists():
    with open(NO_MAP_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "league" in reader.fieldnames and "team" in reader.fieldnames:
            for row in reader:
                l = (row.get("league") or "").strip().lower()
                t = (row.get("team") or "").strip()
                if l and t:
                    existing.add((l, t))

new_only = unmapped - existing
combined = existing | unmapped

with open(NO_MAP_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["league", "team"])
    for league, team in sorted(combined):
        writer.writerow([league, team])

# =========================
# LOG SUMMARY
# =========================

log(
    f"SUMMARY: files_processed={files_processed}, "
    f"rows_processed={rows_processed}, "
    f"rows_updated={rows_updated}, "
    f"unmapped_found={len(unmapped)}, "
    f"unmapped_new_added={len(new_only)}"
)

print("MLB name normalization complete.")
