#!/usr/bin/env python3
# docs/win/basketball/scripts/master_scripts/00_intake_basketball.py

import csv
import subprocess
import sys
import traceback
from pathlib import Path
from datetime import datetime

# =========================
# PATHS
# =========================

ROOT = Path(".")

TODAY = datetime.now().strftime("%Y_%m_%d")

JOBS = [
    {
        "script": Path("docs/win/basketball/scripts/00_intake/basketball_odds_parse_wnba.py"),
        "skip": False,
        "checks": [
            {
                "type": "glob_count",
                "path": Path("docs/win/basketball/00_intake/sportsbook/wnba"),
                "pattern": "*_WNBA_odds.csv",
                "min_count": 1,
            },
        ],
    },
    {
        "script": Path("docs/win/basketball/scripts/00_intake/basketball_odds_parse_nba.py"),
        "skip": False,
        "checks": [
            {
                "type": "glob_count",
                "path": Path("docs/win/basketball/00_intake/sportsbook/nba"),
                "pattern": "*_NBA_odds.csv",
                "min_count": 1,
            },
        ],
    },
    {
        "script": Path("docs/win/basketball/scripts/00_intake/basketball_drat_scraper.py"),
        "skip": False,
        "checks": [
            {
                "type": "file_exists",
                "path": Path(f"docs/win/basketball/00_intake/drat_raw/nba/{TODAY}_nba_raw.json"),
            },
            {
                "type": "file_exists",
                "path": Path(f"docs/win/basketball/00_intake/drat_raw/ncaam/{TODAY}_ncaam_raw.json"),
            },
            {
                "type": "file_exists",
                "path": Path(f"docs/win/basketball/00_intake/drat_raw/wnba/{TODAY}_wnba_raw.json"),
            },
            {
                "type": "json_list_has_rows",
                "path": Path(f"docs/win/basketball/00_intake/drat_raw/nba/{TODAY}_nba_raw.json"),
                "min_rows": 1,
            },
            {
                "type": "json_list_has_rows",
                "path": Path(f"docs/win/basketball/00_intake/drat_raw/ncaam/{TODAY}_ncaam_raw.json"),
                "min_rows": 1,
            },
            {
                "type": "json_list_has_rows",
                "path": Path(f"docs/win/basketball/00_intake/drat_raw/wnba/{TODAY}_wnba_raw.json"),
                "min_rows": 1,
            },
        ],
    },
    {
        "script": Path("docs/win/basketball/scripts/00_intake/transform_basketball.py"),
        "skip": False,
        "checks": [
            {
                "type": "glob_count",
                "path": Path("docs/win/basketball/00_intake/predictions/nba"),
                "pattern": "*_NBA_predictions.csv",
                "min_count": 1,
            },
            {
                "type": "glob_count",
                "path": Path("docs/win/basketball/00_intake/predictions/ncaam"),
                "pattern": "*_NCAAM_predictions.csv",
                "min_count": 1,
            },
            {
                "type": "glob_count",
                "path": Path("docs/win/basketball/00_intake/predictions/wnba"),
                "pattern": "*_WNBA_predictions.csv",
                "min_count": 1,
            },
        ],
    },
    {
        "script": Path("docs/win/basketball/scripts/00_intake/basketball_name_normalization.py"),
        "skip": False,
        "checks": [
            {
                "type": "file_exists",
                "path": Path("mappings/basketball/no_map/no_map_basketball.csv"),
            },
            {
                "type": "file_exists",
                "path": Path("docs/win/basketball/errors/00_intake/name_normalization.txt"),
            },
            {
                "type": "file_exists",
                "path": Path("docs/win/basketball/errors/00_intake/condensed_summary.txt"),
            },
        ],
    },
    {
        "script": Path("docs/win/basketball/scripts/00_intake/basketball_daily_games.py"),
        "skip": False,
        "checks": [
            {
                "type": "glob_count",
                "path": Path("docs/win/basketball/daily_games/nba"),
                "pattern": "*_NBA.csv",
                "min_count": 1,
            },
            {
                "type": "glob_count",
                "path": Path("docs/win/basketball/daily_games/ncaam"),
                "pattern": "*_NCAAM.csv",
                "min_count": 1,
            },
            {
                "type": "glob_count",
                "path": Path("docs/win/basketball/daily_games/wnba"),
                "pattern": "*_WNBA.csv",
                "min_count": 1,
            },
        ],
    },
    {
        "script": Path("docs/win/basketball/scripts/00_intake/basketball_game_id.py"),
        "skip": False,
        "checks": [
            {
                "type": "csv_has_columns",
                "path": Path("docs/win/basketball/00_intake/predictions/nba"),
                "pattern": "*.csv",
                "required_columns": ["game_id", "game_time"],
                "at_least_one_non_blank": ["game_id", "game_time"],
            },
            {
                "type": "csv_has_columns",
                "path": Path("docs/win/basketball/00_intake/predictions/ncaam"),
                "pattern": "*.csv",
                "required_columns": ["game_id", "game_time"],
                "at_least_one_non_blank": ["game_id", "game_time"],
            },
            {
                "type": "csv_has_columns",
                "path": Path("docs/win/basketball/00_intake/predictions/wnba"),
                "pattern": "*.csv",
                "required_columns": ["game_id", "game_time"],
                "at_least_one_non_blank": ["game_id", "game_time"],
            },
            {
                "type": "csv_has_columns",
                "path": Path("docs/win/basketball/05_final_scores/results/nba"),
                "pattern": "*_final_scores_*.csv",
                "required_columns": ["game_id"],
                "at_least_one_non_blank": ["game_id"],
            },
            {
                "type": "csv_has_columns",
                "path": Path("docs/win/basketball/05_final_scores/results/ncaam"),
                "pattern": "*_final_scores_*.csv",
                "required_columns": ["game_id"],
                "at_least_one_non_blank": ["game_id"],
            },
            {
                "type": "csv_has_columns",
                "path": Path("docs/win/basketball/05_final_scores/results/wnba"),
                "pattern": "*_final_scores_*.csv",
                "required_columns": ["game_id"],
                "at_least_one_non_blank": ["game_id"],
            },
        ],
    },
    {
        "script": Path("docs/win/basketball/scripts/00_intake/clean_basketball_inputs.py"),
        "skip": False,
        "checks": [
            {
                "type": "file_exists",
                "path": Path("docs/win/basketball/errors/00_intake/clean_basketball_inputs.txt"),
            },
            {
                "type": "csv_has_columns",
                "path": Path("docs/win/basketball/00_intake/predictions/predictions_cleaned/nba"),
                "pattern": "*.csv",
                "required_columns": ["bias_applied"],
            },
            {
                "type": "csv_has_columns",
                "path": Path("docs/win/basketball/00_intake/predictions/predictions_cleaned/ncaam"),
                "pattern": "*.csv",
                "required_columns": ["bias_applied"],
            },
            {
                "type": "csv_has_columns",
                "path": Path("docs/win/basketball/00_intake/predictions/predictions_cleaned/wnba"),
                "pattern": "*.csv",
                "required_columns": ["bias_applied"],
            },
            {
                "type": "glob_count",
                "path": Path("docs/win/basketball/00_intake/sportsbook/sportsbook_cleaned/nba"),
                "pattern": "*.csv",
                "min_count": 1,
            },
            {
                "type": "glob_count",
                "path": Path("docs/win/basketball/00_intake/sportsbook/sportsbook_cleaned/ncaam"),
                "pattern": "*.csv",
                "min_count": 1,
            },
            {
                "type": "glob_count",
                "path": Path("docs/win/basketball/00_intake/sportsbook/sportsbook_cleaned/wnba"),
                "pattern": "*.csv",
                "min_count": 1,
            },
        ],
    },
]

OUTPUT_DIRS = [
    Path("docs/win/basketball/errors/00_intake"),
    Path("docs/win/basketball/odds/ncaam"),
    Path("docs/win/basketball/odds/wnba"),
    Path("docs/win/basketball/00_intake/predictions/nba"),
    Path("docs/win/basketball/00_intake/predictions/ncaam"),
    Path("docs/win/basketball/00_intake/predictions/wnba"),
    Path("docs/win/basketball/00_intake/sportsbook/wnba"),
    Path("docs/win/basketball/00_intake/sportsbook/ncaam"),
    Path("docs/win/basketball/00_intake/sportsbook/nba"),
    Path("docs/win/basketball/05_final_scores/results/nba"),
    Path("docs/win/basketball/05_final_scores/results/ncaam"),
    Path("docs/win/basketball/05_final_scores/results/wnba"),
    Path("docs/win/basketball/00_intake/drat_raw/nba"),
    Path("docs/win/basketball/00_intake/drat_raw/ncaam"),
    Path("docs/win/basketball/00_intake/drat_raw/wnba"),
    Path("docs/win/basketball/daily_games/wnba"),
    Path("docs/win/basketball/daily_games/nba"),
    Path("docs/win/basketball/daily_games/ncaam"),
    Path("docs/win/basketball/00_intake/predictions/predictions_cleaned/nba"),
    Path("docs/win/basketball/00_intake/predictions/predictions_cleaned/ncaam"),
    Path("docs/win/basketball/00_intake/predictions/predictions_cleaned/wnba"),
    Path("docs/win/basketball/00_intake/sportsbook/sportsbook_cleaned/nba"),
    Path("docs/win/basketball/00_intake/sportsbook/sportsbook_cleaned/ncaam"),
    Path("docs/win/basketball/00_intake/sportsbook/sportsbook_cleaned/wnba"),
]

MASTER_LOG = Path("docs/win/basketball/errors/00_intake/00_intake_basketball.txt")


# =========================
# LOGGING
# =========================

def init_log() -> None:
    MASTER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(MASTER_LOG, "w", encoding="utf-8") as f:
        f.write(f"=== 00_intake_basketball RUN {datetime.now().isoformat()} ===\n")


def log(msg: str) -> None:
    line = f"{datetime.now().isoformat()} | {msg}"
    print(line, flush=True)
    with open(MASTER_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# =========================
# HELPERS
# =========================

def ensure_output_dirs() -> None:
    for path in OUTPUT_DIRS:
        path.mkdir(parents=True, exist_ok=True)
        log(f"ENSURED DIR: {path}")


def validate_scripts() -> None:
    missing = [
        str(job["script"])
        for job in JOBS
        if not job.get("skip") and not job["script"].exists()
    ]
    if missing:
        for path in missing:
            log(f"MISSING SCRIPT: {path}")
        raise FileNotFoundError("One or more required scripts do not exist.")


def run_script(script_path: Path) -> None:
    log(f"START SCRIPT: {script_path}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )

    if result.stdout:
        for line in result.stdout.splitlines():
            log(f"[STDOUT] {line}")

    if result.stderr:
        for line in result.stderr.splitlines():
            log(f"[STDERR] {line}")

    if result.returncode != 0:
        raise RuntimeError(
            f"Script failed with exit code {result.returncode}: {script_path}"
        )

    log(f"SUCCESS SCRIPT: {script_path}")


def read_csv_fieldnames(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or []


def csv_row_count(path: Path) -> int:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    return max(len(rows) - 1, 0)


def json_list_row_count(path: Path) -> int:
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise RuntimeError(f"JSON file is not a list: {path}")

    return len(data)


def check_file_exists(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"Check failed: file does not exist: {path}")
    log(f"CHECK OK file_exists: {path}")


def check_glob_count(folder: Path, pattern: str, min_count: int) -> None:
    if not folder.exists():
        raise RuntimeError(f"Check failed: folder does not exist: {folder}")

    matches = list(folder.glob(pattern))
    if len(matches) < min_count:
        raise RuntimeError(
            f"Check failed: {folder} pattern={pattern} found {len(matches)} files; expected at least {min_count}"
        )

    log(f"CHECK OK glob_count: {folder} pattern={pattern} count={len(matches)}")


def check_csv_has_rows(path: Path, min_rows: int) -> None:
    if not path.exists():
        raise RuntimeError(f"Check failed: CSV does not exist: {path}")

    rows = csv_row_count(path)
    if rows < min_rows:
        raise RuntimeError(
            f"Check failed: CSV has {rows} data rows; expected at least {min_rows}: {path}"
        )

    log(f"CHECK OK csv_has_rows: {path} rows={rows}")


def check_json_list_has_rows(path: Path, min_rows: int) -> None:
    if not path.exists():
        raise RuntimeError(f"Check failed: JSON does not exist: {path}")

    rows = json_list_row_count(path)
    if rows < min_rows:
        raise RuntimeError(
            f"Check failed: JSON list has {rows} rows; expected at least {min_rows}: {path}"
        )

    log(f"CHECK OK json_list_has_rows: {path} rows={rows}")


def check_csv_has_columns(folder: Path, pattern: str, required_columns, at_least_one_non_blank=None) -> None:
    if not folder.exists():
        raise RuntimeError(f"Check failed: folder does not exist: {folder}")

    matches = sorted(folder.glob(pattern))
    if not matches:
        raise RuntimeError(f"Check failed: no files found in {folder} matching {pattern}")

    checked_files = 0
    non_blank_hits = {col: 0 for col in (at_least_one_non_blank or [])}

    for path in matches:
        fieldnames = read_csv_fieldnames(path)
        missing = [col for col in required_columns if col not in fieldnames]
        if missing:
            raise RuntimeError(f"Check failed: {path} missing columns: {missing}")

        if at_least_one_non_blank:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for col in at_least_one_non_blank:
                        if str(row.get(col, "")).strip():
                            non_blank_hits[col] += 1

        checked_files += 1

    for col, count in non_blank_hits.items():
        if count < 1:
            raise RuntimeError(
                f"Check failed: column {col} is blank across all matched files in {folder} pattern={pattern}"
            )

    log(
        f"CHECK OK csv_has_columns: {folder} pattern={pattern} files={checked_files} "
        f"required_columns={required_columns}"
    )


def run_checks(checks) -> None:
    for check in checks:
        check_type = check["type"]

        if check_type == "file_exists":
            check_file_exists(check["path"])

        elif check_type == "glob_count":
            check_glob_count(
                folder=check["path"],
                pattern=check["pattern"],
                min_count=check.get("min_count", 1),
            )

        elif check_type == "csv_has_rows":
            check_csv_has_rows(
                path=check["path"],
                min_rows=check.get("min_rows", 1),
            )

        elif check_type == "json_list_has_rows":
            check_json_list_has_rows(
                path=check["path"],
                min_rows=check.get("min_rows", 1),
            )

        elif check_type == "csv_has_columns":
            check_csv_has_columns(
                folder=check["path"],
                pattern=check["pattern"],
                required_columns=check.get("required_columns", []),
                at_least_one_non_blank=check.get("at_least_one_non_blank"),
            )

        else:
            raise RuntimeError(f"Unknown check type: {check_type}")


# =========================
# MAIN
# =========================

def main():
    jobs_completed = 0

    try:
        init_log()
        log("MASTER SCRIPT START")
        log(f"TODAY = {TODAY}")

        ensure_output_dirs()
        validate_scripts()

        for job in JOBS:
            if job.get("skip"):
                log(f"SKIPPED: {job['script']}")
                jobs_completed += 1
                continue

            script_path = job["script"]
            checks = job.get("checks", [])

            run_script(script_path)
            run_checks(checks)
            jobs_completed += 1

        log("--- SUMMARY ---")
        log(f"Jobs requested: {len(JOBS)}")
        log(f"Jobs completed: {jobs_completed}")
        log("STATUS: SUCCESS")

        print("Basketball 00 intake master complete.", flush=True)

    except Exception as e:
        log(f"FATAL ERROR: {e}")
        log(traceback.format_exc())
        log("--- SUMMARY ---")
        log(f"Jobs requested: {len(JOBS)}")
        log(f"Jobs completed: {jobs_completed}")
        log("STATUS: FAILED")
        raise


if __name__ == "__main__":
    main()
