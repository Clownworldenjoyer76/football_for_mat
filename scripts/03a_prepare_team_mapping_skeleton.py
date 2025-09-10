#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full script path: /mnt/data/football_for_mat-main/scripts/03a_prepare_team_mapping_skeleton.py

Purpose:
  Inspect staged schedules and produce:
    - output/reports/unique_teams.csv  (distinct team strings + counts)
    - mappings/team_aliases.csv        (append-only skeleton rows for each team string if not present)

Inputs:
  - data/processed/schedules/_staging/schedules_staged.csv
  - mappings/team_aliases.csv  (created if missing)

Outputs:
  - output/reports/unique_teams.csv
  - mappings/team_aliases.csv (ensures at least one row per unique team string with empty team_abbr/team_full)
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGED = ROOT / "data" / "processed" / "schedules" / "_staging" / "schedules_staged.csv"
ALIAS_CSV = ROOT / "mappings" / "team_aliases.csv"
REPORTS = ROOT / "output" / "reports"
UNIQUE_REPORT = REPORTS / "unique_teams.csv"

def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not STAGED.exists():
        print(f"Missing staged file: {STAGED}")
        return 2

    df = pd.read_csv(STAGED, dtype=str).fillna("")
    # Collect unique team strings from both columns
    teams = pd.concat([df["home_team"], df["away_team"]], ignore_index=True)
    teams = teams[teams.astype(str).str.strip() != ""]
    s = teams.value_counts().reset_index()
    s.columns = ["alias", "count"]
    s["alias_norm"] = s["alias"].str.strip()

    # Write report of what must be mapped
    s[["alias", "count"]].to_csv(UNIQUE_REPORT, index=False)

    # Ensure aliases CSV exists with header
    ALIAS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not ALIAS_CSV.exists():
        pd.DataFrame(columns=["alias","team_abbr","team_full"]).to_csv(ALIAS_CSV, index=False)

    # Load existing aliases to avoid duplicates
    existing = pd.read_csv(ALIAS_CSV, dtype=str).fillna("")
    if not set(["alias","team_abbr","team_full"]).issubset(existing.columns):
        existing = pd.DataFrame(columns=["alias","team_abbr","team_full"])

    # Determine which aliases are new
    existing_aliases = set(existing["alias"].astype(str).str.strip().str.casefold())
    to_add = s[~s["alias_norm"].str.casefold().isin(existing_aliases)].copy()
    if not to_add.empty:
        skel = to_add[["alias_norm"]].rename(columns={"alias_norm":"alias"})
        skel["team_abbr"] = ""   # fill these manually with canonical abbreviations
        skel["team_full"] = ""   # optional
        updated = pd.concat([existing, skel], ignore_index=True)
        updated = updated.drop_duplicates(subset=["alias"], keep="first")
        updated.to_csv(ALIAS_CSV, index=False)

    print(f"Wrote: {UNIQUE_REPORT}")
    print(f"Updated skeleton: {ALIAS_CSV}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
