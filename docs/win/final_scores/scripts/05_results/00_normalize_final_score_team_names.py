#!/usr/bin/env python3
# docs/win/final_scores/scripts/00_normalize_final_score_team_names.py

from pathlib import Path
from datetime import datetime, UTC
import pandas as pd

BASE = Path("docs/win/final_scores/results")

LEAGUE_CONFIG = {
    "mlb": {
        "input_dir": BASE / "mlb" / "final_scores",
        "pattern": "*_final_scores_MLB.csv",
        "map_file": Path("mappings/baseball/team_map_mlb.csv"),
        "map_filter_col": "league",
        "map_filter_val": "mlb",
    },
    "nba": {
        "input_dir": BASE / "nba" / "final_scores",
        "pattern": "*_final_scores_NBA.csv",
        "map_file": Path("mappings/basketball/team_map_nba.csv"),
        "map_filter_col": "league",
        "map_filter_val": "NBA",
    },
    "ncaab": {
        "input_dir": BASE / "ncaab" / "final_scores",
        "pattern": "*_final_scores_NCAAB.csv",
        "map_file": Path("mappings/basketball/team_map_ncaab.csv"),
        "map_filter_col": "league",
        "map_filter_val": "NCAAB",
    },
    "nhl": {
        "input_dir": BASE / "nhl" / "final_scores",
        "pattern": "*_final_scores_NHL.csv",
        "map_file": Path("mappings/hockey/team_map_hockey.csv"),
        "map_filter_col": "market",
        "map_filter_val": "nhl",
    },
}

ERROR_DIR = Path("docs/win/final_scores/errors")
ERROR_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "team_normalization_log.txt"
NO_MAP_FILE = ERROR_DIR / "team_normalization_no_map.csv"


def reset_outputs():
    LOG_FILE.write_text("", encoding="utf-8")
    if NO_MAP_FILE.exists():
        NO_MAP_FILE.unlink()


def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(UTC).isoformat()}] {msg}\n")


def norm_key(val):
    return str(val).strip().lower()


def load_map(map_file: Path, filter_col: str, filter_val: str):
    if not map_file.exists():
        raise FileNotFoundError(f"Missing map file: {map_file}")

    df = pd.read_csv(map_file)

    required = {"alias", "canonical_team"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{map_file} missing columns: {sorted(missing)}")

    if filter_col in df.columns:
        df = df[df[filter_col].astype(str).str.strip().str.lower() == str(filter_val).strip().lower()].copy()

    df["alias_key"] = df["alias"].astype(str).map(norm_key)
    df["canonical_team"] = df["canonical_team"].astype(str).str.strip()

    mapping = dict(zip(df["alias_key"], df["canonical_team"]))
    return mapping


def normalize_file(file_path: Path, league_name: str, mapping: dict):
    try:
        df = pd.read_csv(file_path)

        needed = {"home_team", "away_team"}
        missing = needed - set(df.columns)
        if missing:
            log(f"SKIP MISSING COLS | {file_path} | missing={sorted(missing)}")
            return []

        no_map_rows = []

        for col in ["away_team", "home_team"]:
            original_col = f"{col}_original"
            key_col = f"{col}_key"

            df[original_col] = df[col].astype(str).str.strip()
            df[key_col] = df[original_col].map(norm_key)

            mapped = df[key_col].map(mapping)
            missing_mask = mapped.isna()

            if missing_mask.any():
                for team_val in df.loc[missing_mask, original_col].dropna().unique():
                    no_map_rows.append({
                        "league": league_name,
                        "file_name": file_path.name,
                        "team_col": col,
                        "unmapped_value": team_val,
                    })

            df[col] = mapped.fillna(df[original_col])

        df = df.drop(columns=[
            "away_team_original", "home_team_original",
            "away_team_key", "home_team_key"
        ], errors="ignore")

        df.to_csv(file_path, index=False)
        log(f"NORMALIZED | {file_path} | rows={len(df)}")

        return no_map_rows

    except Exception as e:
        log(f"ERROR | {file_path} | {e}")
        return []


def main():
    reset_outputs()
    all_no_map = []

    for league_name, cfg in LEAGUE_CONFIG.items():
        try:
            mapping = load_map(
                cfg["map_file"],
                cfg["map_filter_col"],
                cfg["map_filter_val"],
            )
            log(f"MAP LOADED | {league_name} | aliases={len(mapping)}")
        except Exception as e:
            log(f"MAP LOAD ERROR | {league_name} | {e}")
            continue

        files = sorted(cfg["input_dir"].glob(cfg["pattern"]))
        if not files:
            log(f"NO FILES | {league_name} | {cfg['input_dir']}")
            continue

        for file_path in files:
            all_no_map.extend(normalize_file(file_path, league_name, mapping))

    if all_no_map:
        no_map_df = pd.DataFrame(all_no_map).drop_duplicates()
        no_map_df = no_map_df.sort_values(
            ["league", "file_name", "team_col", "unmapped_value"],
            kind="mergesort"
        )
        no_map_df.to_csv(NO_MAP_FILE, index=False)
        log(f"NO MAP CSV WRITTEN | {NO_MAP_FILE} | rows={len(no_map_df)}")
    else:
        pd.DataFrame(columns=["league", "file_name", "team_col", "unmapped_value"]).to_csv(NO_MAP_FILE, index=False)
        log(f"NO MAP CSV WRITTEN EMPTY | {NO_MAP_FILE}")

    print("Final score team normalization complete.")


if __name__ == "__main__":
    main()
