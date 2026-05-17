#!/usr/bin/env python3
# docs/win/baseball/scripts/01_merge/merge_intake.py

import csv
import traceback
from pathlib import Path
from datetime import datetime, timezone

PRED_DIR    = Path("docs/win/baseball/00_intake/predictions")
BOOK_DIR    = Path("docs/win/baseball/00_intake/sportsbook")
GAMES_DIR   = Path("docs/win/baseball/00_intake/games")
CONTEXT_DIR = Path("docs/win/baseball/00_intake/mlb_raw")
OUT_DIR     = Path("docs/win/baseball/01_merge")
LOG_DIR     = Path("docs/win/baseball/errors/01_merge")

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "merge_intake.txt"

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== merge_intake RUN {datetime.now(timezone.utc).isoformat()} ===\n")


def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} | {msg}\n")


def norm(s):
    return (s or "").strip().lower()


def load_csv(path):
    rows = []
    if not path.exists():
        log(f"MISSING FILE: {path}")
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def build_team_index(rows):
    idx = {}
    for r in rows:
        key = (norm(r["home_team"]), norm(r["away_team"]))
        idx[key] = r
    return idx


def american_to_prob(odds):
    try:
        odds = float(odds)
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return -odds / (-odds + 100)
    except Exception:
        return None


def normalize_probs(p1, p2):
    if p1 is None or p2 is None:
        return "", ""
    total = p1 + p2
    if total == 0:
        return "", ""
    return str(p1 / total), str(p2 / total)


# ─────────────────────────────────────────────
# GAME CONTEXT LOADING
# ─────────────────────────────────────────────

# All columns from game_context that travel downstream
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

NULL_CONTEXT = {col: "" for col in CONTEXT_COLS}


def load_games_index(date: str) -> dict:
    """
    Returns dict: game_id (sportsbook odds hash) -> gamePk (MLB ID).
    Reads from 00_intake/games/{date}_games.csv.
    """
    path = GAMES_DIR / f"{date}_games.csv"
    if not path.exists():
        log(f"MISSING games file: {path}")
        return {}
    idx = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            game_id = r.get("game_id", "").strip()
            game_pk = r.get("gamePk", "").strip()
            if game_id and game_pk:
                idx[game_id] = game_pk
    return idx


def load_context_index(date: str) -> dict:
    """
    Returns dict: gamePk -> context row dict (only CONTEXT_COLS).
    Reads from 00_intake/mlb_raw/{date}_game_context.csv.
    """
    path = CONTEXT_DIR / f"{date}_game_context.csv"
    if not path.exists():
        log(f"MISSING game_context file: {path} — context columns will be null")
        return {}
    idx = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pk = r.get("gamePk", "").strip()
            if pk:
                idx[pk] = {col: r.get(col, "") for col in CONTEXT_COLS}
    return idx


def get_context(game_id: str, games_idx: dict, context_idx: dict) -> dict:
    """
    Resolve game_id -> gamePk -> context row.
    Returns null context with warning if either lookup fails.
    """
    game_pk = games_idx.get(game_id, "")
    if not game_pk:
        log(f"  WARNING: game_id={game_id} not found in games file")
        return {**NULL_CONTEXT}

    ctx = context_idx.get(game_pk)
    if ctx is None:
        log(f"  WARNING: gamePk={game_pk} (game_id={game_id}) not found in game_context")
        return {**NULL_CONTEXT, "gamePk": game_pk}

    return ctx


# ─────────────────────────────────────────────
# PROCESS ONE DATE
# ─────────────────────────────────────────────

def process_date(date, summary):
    pred_path = PRED_DIR / f"{date}_MLB.csv"
    book_path = BOOK_DIR / f"{date}_MLB.csv"

    preds = load_csv(pred_path)
    books = load_csv(book_path)

    if not preds:
        log(f"SKIP {date}: no predictions")
        summary["skipped"] += 1
        return

    if not books:
        log(f"SKIP {date}: no sportsbook")
        summary["skipped"] += 1
        return

    # Load game context lookup tables
    games_idx   = load_games_index(date)
    context_idx = load_context_index(date)
    log(f"{date} | games_idx={len(games_idx)} context_idx={len(context_idx)}")

    pred_idx  = build_team_index(preds)
    matched   = 0
    unmatched = 0

    ml_rows  = []
    rl_rows  = []
    tot_rows = []

    for b in books:
        key = (norm(b["home_team"]), norm(b["away_team"]))
        p   = pred_idx.get(key)

        if not p:
            unmatched += 1
            log(f"UNMATCHED: {key}")
            continue

        matched += 1

        game_id = b["game_id"]
        ctx     = get_context(game_id, games_idx, context_idx)
        ctx_vals = [ctx.get(col, "") for col in CONTEXT_COLS]

        ml_rows.append([
            game_id, b["sport"], b["league"], b["game_date"], b["game_time"],
            b["home_team"], b["away_team"],
            b["away_run_line"], b["home_run_line"], b["total"],
            b["away_dk_moneyline_american"], b["home_dk_moneyline_american"],
            b["away_dk_moneyline_decimal"], b["home_dk_moneyline_decimal"],
            p["home_pitcher"], p["away_pitcher"],
            p["home_prob"], p["away_prob"],
            p["away_projected_runs"], p["home_projected_runs"], p["total_projected_runs"],
        ] + ctx_vals)

        try:
            home_prob_ml = float(p["home_prob"])
            away_prob_ml = float(p["away_prob"])
            home_rl_prob = home_prob_ml * 0.75
            away_rl_prob = away_prob_ml * 0.75
        except Exception:
            home_rl_prob, away_rl_prob = "", ""

        rl_rows.append([
            game_id, b["sport"], b["league"], b["game_date"], b["game_time"],
            b["home_team"], b["away_team"],
            b["away_run_line"], b["home_run_line"], b["total"],
            b["away_dk_run_line_american"], b["home_dk_run_line_american"],
            b["away_dk_run_line_decimal"], b["home_dk_run_line_decimal"],
            p["home_pitcher"], p["away_pitcher"],
            p["home_prob"], p["away_prob"],
            p["away_projected_runs"], p["home_projected_runs"], p["total_projected_runs"],
            home_rl_prob, away_rl_prob,
        ] + ctx_vals)

        over_raw  = american_to_prob(b["dk_total_over_american"])
        under_raw = american_to_prob(b["dk_total_under_american"])
        over_prob, under_prob = normalize_probs(over_raw, under_raw)

        tot_rows.append([
            game_id, b["sport"], b["league"], b["game_date"], b["game_time"],
            b["home_team"], b["away_team"],
            b["away_run_line"], b["home_run_line"], b["total"],
            b["dk_total_over_american"], b["dk_total_under_american"],
            b["dk_total_over_decimal"], b["dk_total_under_decimal"],
            p["home_pitcher"], p["away_pitcher"],
            p["home_prob"], p["away_prob"],
            p["away_projected_runs"], p["home_projected_runs"], p["total_projected_runs"],
            over_prob, under_prob,
        ] + ctx_vals)

    log(f"{date} | matched={matched} | unmatched={unmatched}")
    summary["total_matched"]   += matched
    summary["total_unmatched"] += unmatched

    def write(path, header, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        log(f"WROTE {path} ({len(rows)} rows)")
        summary["files_written"] += 1

    base_ml_header = [
        "game_id", "sport", "league", "game_date", "game_time",
        "home_team", "away_team",
        "away_run_line", "home_run_line", "total",
        "away_dk_moneyline_american", "home_dk_moneyline_american",
        "away_dk_moneyline_decimal", "home_dk_moneyline_decimal",
        "home_pitcher", "away_pitcher", "home_prob", "away_prob",
        "away_projected_runs", "home_projected_runs", "total_projected_runs",
    ]

    base_rl_header = [
        "game_id", "sport", "league", "game_date", "game_time",
        "home_team", "away_team",
        "away_run_line", "home_run_line", "total",
        "away_dk_run_line_american", "home_dk_run_line_american",
        "away_dk_run_line_decimal", "home_dk_run_line_decimal",
        "home_pitcher", "away_pitcher", "home_prob", "away_prob",
        "away_projected_runs", "home_projected_runs", "total_projected_runs",
        "home_run_line_prob", "away_run_line_prob",
    ]

    base_tot_header = [
        "game_id", "sport", "league", "game_date", "game_time",
        "home_team", "away_team",
        "away_run_line", "home_run_line", "total",
        "dk_total_over_american", "dk_total_under_american",
        "dk_total_over_decimal", "dk_total_under_decimal",
        "home_pitcher", "away_pitcher", "home_prob", "away_prob",
        "away_projected_runs", "home_projected_runs", "total_projected_runs",
        "total_runs_over_prob", "total_runs_under_prob",
    ]

    write(OUT_DIR / f"{date}_mlb_moneyline.csv", base_ml_header  + CONTEXT_COLS, ml_rows)
    write(OUT_DIR / f"{date}_mlb_run_line.csv",  base_rl_header  + CONTEXT_COLS, rl_rows)
    write(OUT_DIR / f"{date}_mlb_total.csv",     base_tot_header + CONTEXT_COLS, tot_rows)

    summary["slates_written"] += 1


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    summary = {
        "slates_processed": 0,
        "slates_written":   0,
        "skipped":          0,
        "files_written":    0,
        "total_matched":    0,
        "total_unmatched":  0,
    }

    try:
        pred_files = sorted(PRED_DIR.glob("*_MLB.csv"))
        log(f"Prediction files found: {len(pred_files)}")

        for file in pred_files:
            date = file.stem.replace("_MLB", "")
            summary["slates_processed"] += 1
            process_date(date, summary)

        log("--- SUMMARY ---")
        log(f"Slates processed: {summary['slates_processed']}")
        log(f"Slates written: {summary['slates_written']}")
        log(f"Slates skipped: {summary['skipped']}")
        log(f"Files written: {summary['files_written']}")
        log(f"Total matched: {summary['total_matched']}")
        log(f"Total unmatched: {summary['total_unmatched']}")
        log("STATUS: SUCCESS")

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        raise
