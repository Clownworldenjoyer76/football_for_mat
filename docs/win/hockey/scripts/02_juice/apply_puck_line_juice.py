#!/usr/bin/env python3
# docs/win/hockey/scripts/02_juice/apply_puck_line_juice.py

import glob
import math
import traceback
from datetime import datetime, UTC
from pathlib import Path
import sys

import pandas as pd

INPUT_DIR  = Path("docs/win/hockey/01_merge/01_merguiced")
OUTPUT_DIR = Path("docs/win/hockey/02_juice")
JUICE_FILE = Path("config/hockey/nhl/nhl_puck_line_juice.csv")

ERROR_DIR = Path("docs/win/hockey/errors/02_juice")
LOG_FILE  = ERROR_DIR / "apply_puck_line_juice.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# LOGGING
# =========================

def _now():
    return datetime.now(UTC).isoformat()


def _log(msg: str, level: str = "INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {msg.rstrip()}\n")


def _write_summary(summary: dict, per_file: list) -> None:
    lines = [
        "",
        "=" * 60,
        f"SUMMARY  {_now()}",
        "=" * 60,
        f"  files_found    : {summary['files_found']}",
        f"  files_written  : {summary['files_written']}",
        f"  total_rows     : {summary['total_rows']}",
        f"  applied        : {summary['applied']}",
        f"  skipped_bad    : {summary['skipped_bad']}",
        f"  skipped_noband : {summary['skipped_noband']}",
        f"  errors         : {summary['errors']}",
        "",
        f"  {'file':<45} {'rows':>5} {'applied':>8} {'skipped_bad':>12} {'skipped_noband':>15}",
    ]
    for pf in per_file:
        lines.append(
            f"  {pf['name']:<45} {pf['rows']:>5} {pf['applied']:>8} "
            f"{pf['skipped_bad']:>12} {pf['skipped_noband']:>15}"
        )
    status = "SUCCESS" if summary["errors"] == 0 else "COMPLETED WITH ERRORS"
    lines += ["", f"STATUS: {status}", "=" * 60]
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =========================
# JUICE LOOKUP
# =========================

def find_band_row(juice_df, puck_line, venue, fav_ud):
    band = juice_df[
        (abs(juice_df["band_min"] - puck_line) < 0.01) &
        (juice_df["venue"] == venue) &
        (juice_df["fav_ud"] == fav_ud)
    ]
    if band.empty:
        return None
    if len(band) > 1:
        _log(f"Multiple band matches puck_line={puck_line} venue={venue} fav_ud={fav_ud}", "WARN")
    return float(band.iloc[0]["extra_juice"])


# =========================
# ROW PROCESSOR
# =========================

def process_rows(df, juice_df):
    for c in ["home_juiced_decimal_puck_line", "away_juiced_decimal_puck_line",
              "home_juiced_prob_puck_line",    "away_juiced_prob_puck_line",
              "home_normalized_prob_puck_line","away_normalized_prob_puck_line"]:
        df[c] = pd.NA

    applied = skipped_bad = skipped_noband = 0

    for idx, row in df.iterrows():
        try:
            home_line = float(row["home_puck_line"])
            away_line = float(row["away_puck_line"])
            home_dec  = float(row["home_fair_puck_line_decimal"])
            away_dec  = float(row["away_fair_puck_line_decimal"])
        except Exception:
            _log(f"row={idx} reason=cast_error", "SKIP")
            skipped_bad += 1
            continue

        if (not math.isfinite(home_dec) or home_dec <= 1 or
                not math.isfinite(away_dec) or away_dec <= 1):
            _log(f"row={idx} reason=bad_fair_decimal home={home_dec} away={away_dec}", "SKIP")
            skipped_bad += 1
            continue

        home_fav_ud = "favorite" if home_line < 0 else "underdog"
        away_fav_ud = "favorite" if away_line < 0 else "underdog"

        home_extra = find_band_row(juice_df, home_line, "home", home_fav_ud)
        away_extra = find_band_row(juice_df, away_line, "away", away_fav_ud)

        if home_extra is None or away_extra is None:
            _log(f"row={idx} reason=no_band home_line={home_line} away_line={away_line}", "SKIP")
            skipped_noband += 1
            continue

        try:
            home_adj = max((1 / home_dec) * (1 - home_extra), 1e-6)
            away_adj = max((1 / away_dec) * (1 - away_extra), 1e-6)
            total    = home_adj + away_adj

            if total <= 0 or not math.isfinite(total):
                _log(f"row={idx} reason=bad_total_after_adjustment val={total}", "SKIP")
                skipped_bad += 1
                continue

            home_final = home_adj / total
            away_final = away_adj / total

            df.at[idx, "home_juiced_prob_puck_line"]      = home_final
            df.at[idx, "away_juiced_prob_puck_line"]      = away_final
            df.at[idx, "home_juiced_decimal_puck_line"]   = 1 / home_final
            df.at[idx, "away_juiced_decimal_puck_line"]   = 1 / away_final
            df.at[idx, "home_normalized_prob_puck_line"]  = home_final
            df.at[idx, "away_normalized_prob_puck_line"]  = away_final
            applied += 1

        except Exception:
            _log(f"row={idx} reason=calc_error", "SKIP")
            skipped_bad += 1

    return df, applied, skipped_noband, skipped_bad


# =========================
# MAIN
# =========================

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== apply_puck_line_juice RUN {_now()} ===\n")

    summary = {
        "files_found": 0, "files_written": 0, "total_rows": 0,
        "applied": 0, "skipped_bad": 0, "skipped_noband": 0, "errors": 0,
    }
    per_file = []

    for f in OUTPUT_DIR.glob("*puck_line.csv"):
        f.unlink()

    try:
        _log(f"INPUT_DIR : {INPUT_DIR}")
        _log(f"JUICE_FILE: {JUICE_FILE}")

        juice_df = pd.read_csv(JUICE_FILE)
        juice_df["band_min"]    = juice_df["band_min"].astype(float)
        juice_df["venue"]       = juice_df["venue"].astype(str).str.strip()
        juice_df["fav_ud"]      = juice_df["fav_ud"].astype(str).str.strip()
        juice_df["extra_juice"] = juice_df["extra_juice"].astype(float)

        files = sorted(glob.glob(str(INPUT_DIR / "*_NHL_puck_line.csv")))
        summary["files_found"] = len(files)
        _log(f"Files found: {len(files)}")

        for file_path in files:
            in_path  = Path(file_path)
            out_path = OUTPUT_DIR / in_path.name
            pf = {"name": in_path.name, "rows": 0, "applied": 0,
                  "skipped_bad": 0, "skipped_noband": 0}

            _log(f"--- FILE: {in_path.name}")

            try:
                df = pd.read_csv(in_path)

                if df.empty:
                    _log(f"{in_path.name} empty — skipping")
                    per_file.append(pf)
                    continue

                pf["rows"] = len(df)
                summary["total_rows"] += len(df)

                df, applied, no_band, bad = process_rows(df, juice_df)
                pf["applied"]        = applied
                pf["skipped_noband"] = no_band
                pf["skipped_bad"]    = bad

                df.to_csv(out_path, index=False)
                summary["files_written"] += 1
                summary["applied"]        += applied
                summary["skipped_bad"]    += bad
                summary["skipped_noband"] += no_band

                _log(f"{in_path.name} | rows={pf['rows']} applied={applied} "
                     f"skipped_bad={bad} skipped_noband={no_band}")
                _log(f"WROTE: {out_path}")

            except Exception as e:
                _log(f"{in_path.name} FAILED: {e}\n{traceback.format_exc()}", "ERROR")
                summary["errors"] += 1

            per_file.append(pf)

    except Exception as e:
        _log(f"FATAL: {e}\n{traceback.format_exc()}", "ERROR")
        _write_summary(summary, per_file)
        sys.exit(1)

    _write_summary(summary, per_file)
    print("apply_puck_line_juice complete.")


if __name__ == "__main__":
    main()
