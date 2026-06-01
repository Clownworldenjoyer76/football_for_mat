#!/usr/bin/env python3
# docs/win/baseball/scripts/01_merge/build_juice_files.py

import glob
import math
import shutil
import sys
import traceback
from pathlib import Path
from datetime import datetime, UTC

import pandas as pd
from scipy.stats import poisson, skellam

INPUT_DIR = Path("docs/win/baseball/01_merge")
OUTPUT_DIR = INPUT_DIR / "01_merguiced"

ERROR_DIR = Path("docs/win/baseball/errors/01_merge")
ERROR_DIR.mkdir(parents=True, exist_ok=True)

<<<<<<< HEAD
RUN_TS = datetime.now(UTC).isoformat()
=======
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186
LOG_FILE = ERROR_DIR / "build_juice_files.txt"

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== build_juice_files RUN {RUN_TS} ===\n")


CONTEXT_COLS = [
    "gamePk",
    "home_team_id", "away_team_id", "venue_id",
    "roof_type", "turf_type",
    "home_pitcher_id", "away_pitcher_id",
    "home_pitcher_hand", "away_pitcher_hand",
    "home_sp_xwoba", "away_sp_xwoba",
    "home_sp_k_pct", "away_sp_k_pct",
    "home_sp_bb_pct", "away_sp_bb_pct",
    "home_sp_barrel_pct", "away_sp_barrel_pct",
    "home_sp_whiff_pct", "away_sp_whiff_pct",
    "home_sp_sample_flag", "away_sp_sample_flag",
    "home_lineup_xwoba", "home_lineup_barrel_pct", "home_lineup_hard_hit_pct",
    "home_lineup_k_pct", "home_lineup_bb_pct", "home_lineup_exit_velo",
    "home_lineup_frv", "home_lineup_brv", "home_catcher_framing",
    "home_low_sample_count", "home_n_left", "home_n_right", "home_n_switch",
    "away_lineup_xwoba", "away_lineup_barrel_pct", "away_lineup_hard_hit_pct",
    "away_lineup_k_pct", "away_lineup_bb_pct", "away_lineup_exit_velo",
    "away_lineup_frv", "away_lineup_brv", "away_catcher_framing",
    "away_low_sample_count", "away_n_left", "away_n_right", "away_n_switch",
    "park_factor", "park_wOBAcon", "park_xwOBAcon", "park_HR", "park_R",
    "park_factor_B", "park_wOBAcon_B", "park_xwOBAcon_B", "park_HR_B", "park_R_B",
    "weather_applicable", "weather_time",
    "temp_f", "wind_mph", "wind_dir",
    "precip_in", "humidity", "will_it_rain", "wind_blowing_out",
    "air_pressure_at_sea_level", "dew_point_f", "symbol_code",
    "sp_data_available", "lineup_data_available",
]

MONEYLINE_REQUIRED_COLUMNS = [
<<<<<<< HEAD
    "last_run",
    "game_id", "sport", "league", "game_date", "game_time", "home_team", "away_team",
    "away_run_line", "home_run_line", "total",
    "away_dk_moneyline_american", "home_dk_moneyline_american",
    "away_dk_moneyline_decimal", "home_dk_moneyline_decimal",
    "home_pitcher", "away_pitcher", "home_prob", "away_prob",
    "away_projected_runs", "home_projected_runs", "total_projected_runs",
] + CONTEXT_COLS

RUN_LINE_REQUIRED_COLUMNS = [
    "last_run",
    "game_id", "sport", "league", "game_date", "game_time", "home_team", "away_team",
    "away_run_line", "home_run_line", "total",
    "away_dk_run_line_american", "home_dk_run_line_american",
    "away_dk_run_line_decimal", "home_dk_run_line_decimal",
    "home_pitcher", "away_pitcher", "home_prob", "away_prob",
    "away_projected_runs", "home_projected_runs", "total_projected_runs",
    "home_run_line_prob", "away_run_line_prob",
] + CONTEXT_COLS

TOTAL_REQUIRED_COLUMNS = [
    "last_run",
    "game_id", "sport", "league", "game_date", "game_time", "home_team", "away_team",
    "away_run_line", "home_run_line", "total",
    "dk_total_over_american", "dk_total_under_american",
    "dk_total_over_decimal", "dk_total_under_decimal",
    "home_pitcher", "away_pitcher", "home_prob", "away_prob",
    "away_projected_runs", "home_projected_runs", "total_projected_runs",
    "total_runs_over_prob", "total_runs_under_prob",
=======
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "away_run_line",
    "home_run_line",
    "total",
    "away_dk_moneyline_american",
    "home_dk_moneyline_american",
    "away_dk_moneyline_decimal",
    "home_dk_moneyline_decimal",
    "home_pitcher",
    "away_pitcher",
    "home_prob",
    "away_prob",
    "away_projected_runs",
    "home_projected_runs",
    "total_projected_runs",
] + CONTEXT_COLS

RUN_LINE_REQUIRED_COLUMNS = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "away_run_line",
    "home_run_line",
    "total",
    "away_dk_run_line_american",
    "home_dk_run_line_american",
    "away_dk_run_line_decimal",
    "home_dk_run_line_decimal",
    "home_pitcher",
    "away_pitcher",
    "home_prob",
    "away_prob",
    "away_projected_runs",
    "home_projected_runs",
    "total_projected_runs",
] + CONTEXT_COLS

TOTAL_REQUIRED_COLUMNS = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "away_run_line",
    "home_run_line",
    "total",
    "dk_total_over_american",
    "dk_total_under_american",
    "dk_total_over_decimal",
    "dk_total_under_decimal",
    "home_pitcher",
    "away_pitcher",
    "home_prob",
    "away_prob",
    "away_projected_runs",
    "home_projected_runs",
    "total_projected_runs",
    "total_runs_over_prob",
    "total_runs_under_prob",
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186
] + CONTEXT_COLS

FORBIDDEN_RUN_LINE_COLUMNS = [
    "home_run_line_prob",
    "away_run_line_prob",
]


def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(UTC).isoformat()} | {msg}\n")


<<<<<<< HEAD
def reset_output_dir():
    """
    Permanently deletes docs/win/baseball/01_merge/01_merguiced before rebuilding.

    If the directory or any CSV file remains after deletion, the script fails immediately.
    """
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        log(f"DELETED OUTPUT DIRECTORY: {OUTPUT_DIR}")

    if OUTPUT_DIR.exists():
        raise RuntimeError(f"FAILED TO DELETE OUTPUT DIRECTORY: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    remaining_csvs = sorted([p for p in OUTPUT_DIR.glob("*.csv") if p.is_file()])

    if remaining_csvs:
        remaining_text = ", ".join(str(p) for p in remaining_csvs)
        raise RuntimeError(
            f"FAILED TO START WITH EMPTY 01_MERGUICED DIRECTORY. Remaining CSV files: {remaining_text}"
        )

    log(f"CREATED EMPTY OUTPUT DIRECTORY: {OUTPUT_DIR}")
    log("CONFIRMED: docs/win/baseball/01_merge/01_merguiced has zero CSV files before rebuild")
=======
def duplicate_columns(columns):
    seen = set()
    duplicates = []

    for col in columns:
        if col in seen and col not in duplicates:
            duplicates.append(col)
        seen.add(col)

    return duplicates


def validate_no_duplicate_columns(df, label):
    dupes = duplicate_columns(list(df.columns))

    if dupes:
        raise ValueError(f"{label} has duplicate columns: {dupes}")


def validate_required_columns(df, required_columns, label):
    missing_cols = [c for c in required_columns if c not in df.columns]

    if missing_cols:
        raise ValueError(f"{label} missing required columns: {missing_cols}")


def validate_forbidden_columns(df, forbidden_columns, label):
    present = [c for c in forbidden_columns if c in df.columns]

    if present:
        raise ValueError(
            f"{label} contains obsolete forbidden columns: {present}. "
            f"Use home_prob_run_line / away_prob_run_line only."
        )


def read_csv_validated(file_path, required_columns, label, forbidden_columns=None):
    df = pd.read_csv(file_path)

    validate_no_duplicate_columns(df, f"{label} input")
    validate_required_columns(df, required_columns, f"{label} input")

    if forbidden_columns:
        validate_forbidden_columns(df, forbidden_columns, f"{label} input")

    return df


def write_csv_validated(df, out_path, label):
    validate_no_duplicate_columns(df, f"{label} output")
    df.to_csv(out_path, index=False)
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186


def american_to_decimal(odds):
    try:
        if pd.isna(odds):
            return None

        odds = float(odds)

        if odds == 0:
            return None

        if odds > 0:
            return 1 + (odds / 100)

        return 1 + (100 / abs(odds))

    except Exception:
        return None


def parse_slate_date_and_market(file_path: str):
    stem = Path(file_path).stem

    if stem.endswith("_mlb_moneyline"):
        return stem.replace("_mlb_moneyline", ""), "moneyline"

    if stem.endswith("_mlb_run_line"):
        return stem.replace("_mlb_run_line", ""), "run_line"

    if stem.endswith("_mlb_total"):
        return stem.replace("_mlb_total", ""), "total"

    return None, None


<<<<<<< HEAD
def validate_schema(df, required_columns, file_path):
    missing_cols = [c for c in required_columns if c not in df.columns]

    if missing_cols:
        log(f"SCHEMA ERROR: {file_path} missing columns: {missing_cols}")
        return False

    return True


=======
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186
def coerce_numeric(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def stamp_last_run(df):
    df["last_run"] = RUN_TS
    return df


def process_moneyline(file_path, summary):
    try:
<<<<<<< HEAD
        df = pd.read_csv(file_path)
=======
        df = read_csv_validated(
            file_path,
            MONEYLINE_REQUIRED_COLUMNS,
            f"moneyline {file_path}",
        )
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186

        if df.empty:
            log(f"EMPTY: {file_path} — skipping")
            summary["empty"] += 1
            return
<<<<<<< HEAD

        if not validate_schema(df, MONEYLINE_REQUIRED_COLUMNS, file_path):
            summary["schema_errors"] += 1
            return

        coerce_numeric(df, [
            "home_prob", "away_prob",
            "home_projected_runs", "away_projected_runs", "total_projected_runs",
            "away_run_line", "home_run_line", "total",
            "away_dk_moneyline_american", "home_dk_moneyline_american",
            "away_dk_moneyline_decimal", "home_dk_moneyline_decimal",
=======

        coerce_numeric(df, [
            "home_prob",
            "away_prob",
            "home_projected_runs",
            "away_projected_runs",
            "total_projected_runs",
            "away_run_line",
            "home_run_line",
            "total",
            "away_dk_moneyline_american",
            "home_dk_moneyline_american",
            "away_dk_moneyline_decimal",
            "home_dk_moneyline_decimal",
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186
        ])

        for i, r in df.iterrows():
            if pd.isna(r["home_prob"]) or pd.isna(r["away_prob"]):
                log(f"ROW ISSUE: {file_path} idx={i} bad probs")
                summary["row_issues"] += 1

        ml = df.copy()
<<<<<<< HEAD
        ml = stamp_last_run(ml)

        ml["away_dk_decimal_moneyline"] = ml["away_dk_moneyline_american"].apply(american_to_decimal)
        ml["home_dk_decimal_moneyline"] = ml["home_dk_moneyline_american"].apply(american_to_decimal)

=======
        ml["away_dk_decimal_moneyline"] = ml["away_dk_moneyline_american"].apply(american_to_decimal)
        ml["home_dk_decimal_moneyline"] = ml["home_dk_moneyline_american"].apply(american_to_decimal)
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186
        ml["away_fair_decimal_moneyline"] = ml["away_prob"].apply(
            lambda x: 1 / x if pd.notna(x) and x > 0 else None
        )
        ml["home_fair_decimal_moneyline"] = ml["home_prob"].apply(
            lambda x: 1 / x if pd.notna(x) and x > 0 else None
        )

        slate_date, market = parse_slate_date_and_market(file_path)

        if not slate_date or market != "moneyline":
            log(f"FILENAME ERROR: {file_path}")
            summary["errors"] += 1
            return

        out = OUTPUT_DIR / f"{slate_date}_mlb_moneyline.csv"
<<<<<<< HEAD
        ml.to_csv(out, index=False)

        log(f"WROTE {out} ({len(ml)} rows) last_run={RUN_TS}")
        summary["files_written"] += 1
        summary["rows_written"] += len(ml)
=======
        write_csv_validated(ml, out, f"moneyline {out}")

        log(f"WROTE {out} ({len(ml)} rows)")
        summary["files_written"] += 1
        summary["rows_written"] += len(ml)

    except ValueError as e:
        log(f"SCHEMA ERROR moneyline {file_path}: {e}\n{traceback.format_exc()}")
        summary["schema_errors"] += 1
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186

    except Exception as e:
        log(f"ERROR moneyline {file_path}: {e}\n{traceback.format_exc()}")
        summary["errors"] += 1


def process_total(file_path, summary):
    try:
<<<<<<< HEAD
        df = pd.read_csv(file_path)
=======
        df = read_csv_validated(
            file_path,
            TOTAL_REQUIRED_COLUMNS,
            f"total {file_path}",
        )
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186

        if df.empty:
            log(f"EMPTY: {file_path} — skipping")
            summary["empty"] += 1
            return
<<<<<<< HEAD

        if not validate_schema(df, TOTAL_REQUIRED_COLUMNS, file_path):
            summary["schema_errors"] += 1
            return

        coerce_numeric(df, [
            "home_prob", "away_prob",
            "home_projected_runs", "away_projected_runs", "total_projected_runs",
            "away_run_line", "home_run_line", "total",
            "dk_total_over_american", "dk_total_under_american",
            "dk_total_over_decimal", "dk_total_under_decimal",
            "total_runs_over_prob", "total_runs_under_prob",
        ])

        tot = df.copy()
        tot = stamp_last_run(tot)

        tot["dk_total_over_decimal"] = tot["dk_total_over_american"].apply(american_to_decimal)
        tot["dk_total_under_decimal"] = tot["dk_total_under_american"].apply(american_to_decimal)

        fair_over_decimals = []
        fair_under_decimals = []

        model_over_probs = []
        model_under_probs = []
        model_push_probs = []
=======

        coerce_numeric(df, [
            "home_prob",
            "away_prob",
            "home_projected_runs",
            "away_projected_runs",
            "total_projected_runs",
            "away_run_line",
            "home_run_line",
            "total",
            "dk_total_over_american",
            "dk_total_under_american",
            "dk_total_over_decimal",
            "dk_total_under_decimal",
            "total_runs_over_prob",
            "total_runs_under_prob",
        ])

        tot = df.copy()
        tot["dk_total_over_decimal"] = tot["dk_total_over_american"].apply(american_to_decimal)
        tot["dk_total_under_decimal"] = tot["dk_total_under_american"].apply(american_to_decimal)

        over = []
        under = []
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186

        for i, r in tot.iterrows():
            lam = r["total_projected_runs"]
            total_line = r["total"]

            if pd.isna(lam) or pd.isna(total_line) or lam <= 0:
                log(f"ROW ISSUE: {file_path} idx={i} bad total inputs")
                summary["row_issues"] += 1

                fair_over_decimals.append(None)
                fair_under_decimals.append(None)

                model_over_probs.append(None)
                model_under_probs.append(None)
                model_push_probs.append(None)
                continue

            if total_line % 1 == 0:
                k = int(total_line)
<<<<<<< HEAD

=======
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186
                p_over = 1 - poisson.cdf(k, lam)
                p_under = poisson.cdf(k - 1, lam)
                p_push = poisson.pmf(k, lam)

                log(
<<<<<<< HEAD
                    f"WHOLE NUMBER TOTAL: {file_path} idx={i} "
                    f"total={total_line} lam={lam:.3f} p_push={p_push:.4f} — modelled with push"
                )

                model_over_probs.append(p_over)
                model_under_probs.append(p_under)
                model_push_probs.append(p_push)

                fair_over_decimals.append(1 / p_over if p_over > 0 else None)
                fair_under_decimals.append(1 / p_under if p_under > 0 else None)
                continue

            k = math.floor(total_line)

            p_under = poisson.cdf(k, lam)
            p_over = 1 - p_under
            p_push = 0.0

            model_over_probs.append(p_over)
            model_under_probs.append(p_under)
            model_push_probs.append(p_push)

            fair_over_decimals.append(1 / p_over if p_over > 0 else None)
            fair_under_decimals.append(1 / p_under if p_under > 0 else None)

        tot["model_total_over_prob"] = model_over_probs
        tot["model_total_under_prob"] = model_under_probs
        tot["model_total_push_prob"] = model_push_probs

        tot["fair_total_over_decimal"] = fair_over_decimals
        tot["fair_total_under_decimal"] = fair_under_decimals
=======
                    f"WHOLE NUMBER TOTAL: {file_path} idx={i} total={total_line} "
                    f"lam={lam:.3f} p_push={p_push:.4f} — modelled with push"
                )

                under.append(1 / p_under if p_under > 0 else None)
                over.append(1 / p_over if p_over > 0 else None)
                continue

            k = math.floor(total_line)
            p_under = poisson.cdf(k, lam)
            p_over = 1 - p_under

            under.append(1 / p_under if p_under > 0 else None)
            over.append(1 / p_over if p_over > 0 else None)

        tot["fair_total_over_decimal"] = over
        tot["fair_total_under_decimal"] = under
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186

        slate_date, market = parse_slate_date_and_market(file_path)

        if not slate_date or market != "total":
            log(f"FILENAME ERROR: {file_path}")
            summary["errors"] += 1
            return

        out = OUTPUT_DIR / f"{slate_date}_mlb_total.csv"
<<<<<<< HEAD
        tot.to_csv(out, index=False)

        log(f"WROTE {out} ({len(tot)} rows) last_run={RUN_TS}")
        summary["files_written"] += 1
        summary["rows_written"] += len(tot)
=======
        write_csv_validated(tot, out, f"total {out}")

        log(f"WROTE {out} ({len(tot)} rows)")
        summary["files_written"] += 1
        summary["rows_written"] += len(tot)

    except ValueError as e:
        log(f"SCHEMA ERROR total {file_path}: {e}\n{traceback.format_exc()}")
        summary["schema_errors"] += 1
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186

    except Exception as e:
        log(f"ERROR total {file_path}: {e}\n{traceback.format_exc()}")
        summary["errors"] += 1


def process_run_line(file_path, summary):
    try:
<<<<<<< HEAD
        df = pd.read_csv(file_path)
=======
        df = read_csv_validated(
            file_path,
            RUN_LINE_REQUIRED_COLUMNS,
            f"run_line {file_path}",
            FORBIDDEN_RUN_LINE_COLUMNS,
        )
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186

        if df.empty:
            log(f"EMPTY: {file_path} — skipping")
            summary["empty"] += 1
            return
<<<<<<< HEAD

        if not validate_schema(df, RUN_LINE_REQUIRED_COLUMNS, file_path):
            summary["schema_errors"] += 1
            return

        coerce_numeric(df, [
            "home_prob", "away_prob",
            "home_projected_runs", "away_projected_runs", "total_projected_runs",
            "away_run_line", "home_run_line", "total",
            "away_dk_run_line_american", "home_dk_run_line_american",
            "away_dk_run_line_decimal", "home_dk_run_line_decimal",
            "home_run_line_prob", "away_run_line_prob",
=======

        coerce_numeric(df, [
            "home_prob",
            "away_prob",
            "home_projected_runs",
            "away_projected_runs",
            "total_projected_runs",
            "away_run_line",
            "home_run_line",
            "total",
            "away_dk_run_line_american",
            "home_dk_run_line_american",
            "away_dk_run_line_decimal",
            "home_dk_run_line_decimal",
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186
        ])

        rl = df.copy()
        rl = stamp_last_run(rl)

        rl["home_dk_run_line_decimal"] = rl["home_dk_run_line_american"].apply(american_to_decimal)
        rl["away_dk_run_line_decimal"] = rl["away_dk_run_line_american"].apply(american_to_decimal)

        home_vals = []
        away_vals = []
        home_probs = []
        away_probs = []

        for i, r in rl.iterrows():
            lambda_home = r["home_projected_runs"]
            lambda_away = r["away_projected_runs"]

            if pd.isna(lambda_home) or pd.isna(lambda_away) or lambda_home <= 0 or lambda_away <= 0:
                log(f"ROW ISSUE: {file_path} idx={i} run line invalid lambdas")
                summary["row_issues"] += 1
<<<<<<< HEAD

=======
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186
                home_vals.append(None)
                away_vals.append(None)
                home_probs.append(None)
                away_probs.append(None)
                continue

            home_line = r["home_run_line"]
            away_line = r["away_run_line"]

            if pd.isna(home_line) or pd.isna(away_line):
                log(f"ROW ISSUE: {file_path} idx={i} missing run lines")
                summary["row_issues"] += 1
<<<<<<< HEAD

=======
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186
                home_vals.append(None)
                away_vals.append(None)
                home_probs.append(None)
                away_probs.append(None)
                continue

            if home_line == -1.5:
                p_home = 1 - skellam.cdf(1, lambda_home, lambda_away)
                p_away = skellam.cdf(1, lambda_home, lambda_away)
            elif away_line == -1.5:
                p_away = 1 - skellam.cdf(1, lambda_away, lambda_home)
                p_home = skellam.cdf(1, lambda_away, lambda_home)
            else:
                log(
                    f"ROW ISSUE: {file_path} idx={i} unexpected run lines: "
                    f"home={home_line} away={away_line}"
                )
                summary["row_issues"] += 1
<<<<<<< HEAD

=======
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186
                home_vals.append(None)
                away_vals.append(None)
                home_probs.append(None)
                away_probs.append(None)
                continue

            p_home = min(max(p_home, 0.01), 0.99)
            p_away = min(max(p_away, 0.01), 0.99)

            home_probs.append(p_home)
            away_probs.append(p_away)
<<<<<<< HEAD

=======
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186
            home_vals.append(1 / p_home)
            away_vals.append(1 / p_away)

        rl["home_fair_run_line_decimal"] = home_vals
        rl["away_fair_run_line_decimal"] = away_vals
<<<<<<< HEAD

        rl["home_prob_run_line"] = home_probs
        rl["away_prob_run_line"] = away_probs
=======
        rl["home_prob_run_line"] = home_probs
        rl["away_prob_run_line"] = away_probs

        validate_forbidden_columns(
            rl,
            FORBIDDEN_RUN_LINE_COLUMNS,
            f"run_line output {file_path}",
        )
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186

        slate_date, market = parse_slate_date_and_market(file_path)

        if not slate_date or market != "run_line":
            log(f"FILENAME ERROR: {file_path}")
            summary["errors"] += 1
            return

        out = OUTPUT_DIR / f"{slate_date}_mlb_run_line.csv"
<<<<<<< HEAD
        rl.to_csv(out, index=False)

        log(f"WROTE {out} ({len(rl)} rows) last_run={RUN_TS}")
        summary["files_written"] += 1
        summary["rows_written"] += len(rl)
=======
        write_csv_validated(rl, out, f"run_line {out}")

        log(f"WROTE {out} ({len(rl)} rows)")
        summary["files_written"] += 1
        summary["rows_written"] += len(rl)

    except ValueError as e:
        log(f"SCHEMA ERROR run_line {file_path}: {e}\n{traceback.format_exc()}")
        summary["schema_errors"] += 1
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186

    except Exception as e:
        log(f"ERROR run_line {file_path}: {e}\n{traceback.format_exc()}")
        summary["errors"] += 1


def main():
    summary = {
        "files_written": 0,
        "rows_written": 0,
        "empty": 0,
        "schema_errors": 0,
        "row_issues": 0,
        "errors": 0,
    }

<<<<<<< HEAD
=======
    for f in OUTPUT_DIR.glob("*.csv"):
        f.unlink()

>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186
    try:
        reset_output_dir()

        moneyline_files = sorted(glob.glob(str(INPUT_DIR / "*_mlb_moneyline.csv")))
        run_line_files = sorted(glob.glob(str(INPUT_DIR / "*_mlb_run_line.csv")))
        total_files = sorted(glob.glob(str(INPUT_DIR / "*_mlb_total.csv")))

        log(f"Moneyline files: {len(moneyline_files)}")
        log(f"Run line files: {len(run_line_files)}")
        log(f"Total files: {len(total_files)}")

        for f in moneyline_files:
            process_moneyline(f, summary)

        for f in run_line_files:
            process_run_line(f, summary)

        for f in total_files:
            process_total(f, summary)

        log("--- SUMMARY ---")
        log(f"Last run timestamp: {RUN_TS}")
        log(f"Files written: {summary['files_written']}")
        log(f"Rows written: {summary['rows_written']}")
        log(f"Empty files: {summary['empty']}")
        log(f"Schema errors: {summary['schema_errors']}")
        log(f"Row issues: {summary['row_issues']}")
        log(f"Errors: {summary['errors']}")

<<<<<<< HEAD
        status = "SUCCESS" if summary["errors"] == 0 else "COMPLETED WITH ERRORS"
        log(f"STATUS: {status}")

        print(
            f"build_juice_files complete. "
            f"last_run={RUN_TS} "
            f"files_written={summary['files_written']} "
            f"rows_written={summary['rows_written']} "
            f"errors={summary['errors']} "
            f"Status: {status}"
        )

        if summary["errors"] > 0:
            sys.exit(1)
=======
        if summary["schema_errors"] > 0 or summary["errors"] > 0:
            log("STATUS: FAILED")
            print(
                f"build_juice_files failed. "
                f"schema_errors={summary['schema_errors']} errors={summary['errors']}"
            )
            sys.exit(1)

        log("STATUS: SUCCESS")
        print(
            f"build_juice_files complete. "
            f"files_written={summary['files_written']} "
            f"rows_written={summary['rows_written']} "
            f"schema_errors={summary['schema_errors']} "
            f"errors={summary['errors']}"
        )
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        print(f"build_juice_files failed. {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
