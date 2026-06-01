#!/usr/bin/env python3
# docs/win/baseball/scripts/01_merge/merge_intake.py

import csv
import math
import traceback
from pathlib import Path
from datetime import datetime, timezone

PRED_DIR = Path("docs/win/baseball/00_intake/predictions/pred_with_game_id")
BOOK_DIR = Path("docs/win/baseball/00_intake/sportsbook")
GAMES_DIR = Path("docs/win/baseball/00_intake/games")
CONTEXT_DIR = Path("docs/win/baseball/00_intake/mlb_raw")
OUT_DIR = Path("docs/win/baseball/01_merge")
LOG_DIR = Path("docs/win/baseball/errors/01_merge")

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

RUN_TS = datetime.now(timezone.utc).isoformat()
LOG_FILE = LOG_DIR / "merge_intake.txt"

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== merge_intake RUN {RUN_TS} ===\n")


def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} | {msg}\n")


<<<<<<< HEAD
def clear_old_outputs():
    """
    Permanently deletes every root-level CSV in docs/win/baseball/01_merge before rebuilding.

    This intentionally does NOT delete files inside subfolders such as:
      docs/win/baseball/01_merge/01_merguiced/

    If any root-level CSV remains after deletion, the script fails immediately.
    """
    old_files = sorted([p for p in OUT_DIR.glob("*.csv") if p.is_file()])
    deleted = 0

    for old_file in old_files:
        old_file.unlink()
        deleted += 1
        log(f"DELETED OLD ROOT MERGE OUTPUT: {old_file}")

    remaining = sorted([p for p in OUT_DIR.glob("*.csv") if p.is_file()])

    if remaining:
        remaining_text = ", ".join(str(p) for p in remaining)
        raise RuntimeError(
            f"FAILED TO DELETE ALL ROOT MERGE CSV OUTPUTS. Remaining files: {remaining_text}"
        )

    log(f"OLD ROOT MERGE CSV OUTPUTS PERMANENTLY DELETED: {deleted}")
    log("CONFIRMED: docs/win/baseball/01_merge has zero root-level CSV files before rebuild")
=======
def fail(msg):
    log(f"FATAL VALIDATION ERROR: {msg}")
    raise RuntimeError(msg)
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186


def clear_old_outputs():
    """
    Permanently deletes every root-level CSV in docs/win/baseball/01_merge before rebuilding.

    This intentionally does NOT delete files inside subfolders such as:
      docs/win/baseball/01_merge/01_merguiced/

    If any root-level CSV remains after deletion, the script fails immediately.
    """
    old_files = sorted([p for p in OUT_DIR.glob("*.csv") if p.is_file()])
    deleted = 0

    for old_file in old_files:
        old_file.unlink()
        deleted += 1
        log(f"DELETED OLD ROOT MERGE OUTPUT: {old_file}")

    remaining = sorted([p for p in OUT_DIR.glob("*.csv") if p.is_file()])

    if remaining:
        remaining_text = ", ".join(str(p) for p in remaining)
        raise RuntimeError(
            f"FAILED TO DELETE ALL ROOT MERGE CSV OUTPUTS. Remaining files: {remaining_text}"
        )

    log(f"OLD ROOT MERGE CSV OUTPUTS PERMANENTLY DELETED: {deleted}")
    log("CONFIRMED: docs/win/baseball/01_merge has zero root-level CSV files before rebuild")


def duplicate_columns(header):
    seen = set()
    dupes = []

    for col in header:
        if col in seen and col not in dupes:
            dupes.append(col)
        seen.add(col)

    return dupes


def assert_no_duplicate_columns(header, label):
    dupes = duplicate_columns(header)

    if dupes:
        fail(f"{label} has duplicate columns: {dupes}")


def assert_required_columns(path, header, required_cols, label):
    missing = [col for col in required_cols if col not in header]

    if missing:
        fail(f"{label} missing required columns in {path}: {missing}")


def load_csv(path, required_cols=None, label=None):
    rows = []

    if not path.exists():
        log(f"MISSING FILE: {path}")
        return rows

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []

        assert_no_duplicate_columns(header, f"{label or path} input")

        if required_cols:
            assert_required_columns(path, header, required_cols, label or str(path))

        for r in reader:
            rows.append(r)

    return rows


def build_game_id_index(rows, date):
    idx = {}
    duplicates = 0
    blanks = 0

    for r in rows:
        game_id = (r.get("game_id") or "").strip()

        if not game_id:
            blanks += 1
            log(
                f"{date} | prediction row missing game_id: "
                f"away={r.get('away_team', '')} "
                f"home={r.get('home_team', '')} "
                f"time={r.get('game_time', '')}"
            )
            continue

        if game_id in idx:
            duplicates += 1
            log(
                f"{date} | DUPLICATE prediction game_id={game_id}; "
                f"later row overwrote earlier row"
            )

        idx[game_id] = r

    log(
        f"{date} | prediction game_id index built: "
        f"indexed={len(idx)} blanks={blanks} duplicates={duplicates}"
    )

    return idx


def american_to_prob(odds):
    try:
        odds = float(odds)

        if odds > 0:
            return 100 / (odds + 100)

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


def fv(value):
    try:
        if value is None:
            return None

        s = str(value).strip()

        if not s:
            return None

        x = float(s)

        if not math.isfinite(x):
            return None

        return x

    except Exception:
        return None


def poisson_probs(lam, max_runs=30):
    """
    Returns P(X = k) for k=0..max_runs using a Poisson model.

    This avoids scipy dependency. Tail mass beyond max_runs is normally tiny
    for MLB projected run means. The distribution is normalized after truncation.
    """
    if lam is None or lam < 0:
        return None

    probs = [0.0] * (max_runs + 1)

    try:
        probs[0] = math.exp(-lam)

        for k in range(1, max_runs + 1):
            probs[k] = probs[k - 1] * lam / k

        total = sum(probs)

        if total <= 0:
            return None

        return [p / total for p in probs]

    except Exception:
        return None


def run_line_cover_probability(home_projected_runs, away_projected_runs, home_run_line, away_run_line):
    """
    Calculates run-line cover probabilities from projected runs and actual sportsbook run lines.

    Home cover condition:
      home_score + home_run_line > away_score

    Away cover condition:
      away_score + away_run_line > home_score

    For standard MLB half-run lines, no push is possible.
    """
    home_lambda = fv(home_projected_runs)
    away_lambda = fv(away_projected_runs)
    home_line = fv(home_run_line)
    away_line = fv(away_run_line)

    if home_lambda is None or away_lambda is None:
        return "", ""

    if home_line is None or away_line is None:
        return "", ""

    home_dist = poisson_probs(home_lambda)
    away_dist = poisson_probs(away_lambda)

    if home_dist is None or away_dist is None:
        return "", ""

    home_cover = 0.0
    away_cover = 0.0

    for home_score, hp in enumerate(home_dist):
        for away_score, ap in enumerate(away_dist):
            prob = hp * ap

            if home_score + home_line > away_score:
                home_cover += prob

            if away_score + away_line > home_score:
                away_cover += prob

    return str(home_cover), str(away_cover)


# ─────────────────────────────────────────────
# REQUIRED INPUT SCHEMAS
# ─────────────────────────────────────────────

REQUIRED_PRED_COLS = [
    "game_id",
    "home_team",
    "away_team",
    "game_time",
    "home_pitcher",
    "away_pitcher",
    "home_prob",
    "away_prob",
    "away_projected_runs",
    "home_projected_runs",
    "total_projected_runs",
]

REQUIRED_BOOK_COLS = [
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
    "away_dk_run_line_american",
    "home_dk_run_line_american",
    "away_dk_run_line_decimal",
    "home_dk_run_line_decimal",
    "dk_total_over_american",
    "dk_total_under_american",
    "dk_total_over_decimal",
    "dk_total_under_decimal",
]


# ─────────────────────────────────────────────
# GAME CONTEXT LOADING
# ─────────────────────────────────────────────

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
    Returns dict: game_id -> gamePk.
    Reads docs/win/baseball/00_intake/games/{date}_games.csv.
    """
    path = GAMES_DIR / f"{date}_games.csv"

    if not path.exists():
        log(f"MISSING games file: {path}")
        return {}

    idx = {}

    with open(path, newline="", encoding="utf-8-sig") as f:
<<<<<<< HEAD
        for r in csv.DictReader(f):
=======
        reader = csv.DictReader(f)
        header = reader.fieldnames or []

        assert_no_duplicate_columns(header, f"{path} input")
        assert_required_columns(path, header, ["game_id", "gamePk"], "games input")

        for r in reader:
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186
            game_id = (r.get("game_id") or "").strip()
            game_pk = (r.get("gamePk") or "").strip()

            if game_id and game_pk:
                idx[game_id] = game_pk

    return idx


def load_context_index(date: str) -> dict:
    """
    Returns dict: gamePk -> context row dict.
    Reads docs/win/baseball/00_intake/mlb_raw/{date}_game_context.csv.
    """
    path = CONTEXT_DIR / f"{date}_game_context.csv"

    if not path.exists():
        log(f"MISSING game_context file: {path} — context columns will be null")
        return {}

    idx = {}

    with open(path, newline="", encoding="utf-8-sig") as f:
<<<<<<< HEAD
        for r in csv.DictReader(f):
=======
        reader = csv.DictReader(f)
        header = reader.fieldnames or []

        assert_no_duplicate_columns(header, f"{path} input")
        assert_required_columns(path, header, ["gamePk"], "game_context input")

        for r in reader:
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186
            pk = (r.get("gamePk") or "").strip()

            if pk:
                idx[pk] = {col: r.get(col, "") for col in CONTEXT_COLS}

    return idx


def get_context(game_id: str, games_idx: dict, context_idx: dict) -> dict:
    """
    Resolves game_id -> gamePk -> context row.
    """
    game_pk = games_idx.get(game_id, "")

    if not game_pk:
        log(f"  WARNING: game_id={game_id} not found in games file")
        return {**NULL_CONTEXT}

    ctx = context_idx.get(game_pk)

    if ctx is None:
        log(f"  WARNING: gamePk={game_pk} game_id={game_id} not found in game_context")
        return {**NULL_CONTEXT, "gamePk": game_pk}

    return ctx


# ─────────────────────────────────────────────
# PROCESS ONE DATE
# ─────────────────────────────────────────────

def process_date(date, summary):
    pred_path = PRED_DIR / f"{date}_MLB.csv"
    book_path = BOOK_DIR / f"{date}_MLB.csv"

    preds = load_csv(pred_path, REQUIRED_PRED_COLS, "prediction input")
    books = load_csv(book_path, REQUIRED_BOOK_COLS, "sportsbook input")

    if not preds:
        log(f"SKIP {date}: no predictions")
        summary["skipped"] += 1
        return

    if not books:
        log(f"SKIP {date}: no sportsbook")
        summary["skipped"] += 1
        return

    games_idx = load_games_index(date)
    context_idx = load_context_index(date)

    log(f"{date} | games_idx={len(games_idx)} context_idx={len(context_idx)}")

    pred_idx = build_game_id_index(preds, date)
<<<<<<< HEAD

    matched = 0
    unmatched = 0

=======

    matched = 0
    unmatched = 0
    run_line_prob_missing = 0

>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186
    ml_rows = []
    rl_rows = []
    tot_rows = []

    for b in books:
        game_id = (b.get("game_id") or "").strip()

        if not game_id:
            unmatched += 1
            log(
                f"{date} | UNMATCHED sportsbook row missing game_id: "
                f"away={b.get('away_team', '')} "
                f"home={b.get('home_team', '')} "
                f"time={b.get('game_time', '')}"
            )
            continue

        p = pred_idx.get(game_id)

        if not p:
            unmatched += 1
            log(
                f"{date} | UNMATCHED sportsbook game_id not found in predictions: "
                f"game_id={game_id} "
                f"away={b.get('away_team', '')} "
                f"home={b.get('home_team', '')} "
                f"time={b.get('game_time', '')}"
            )
            continue

        matched += 1

        ctx = get_context(game_id, games_idx, context_idx)
        ctx_vals = [ctx.get(col, "") for col in CONTEXT_COLS]

        ml_rows.append([
            RUN_TS,
            game_id,
            b.get("sport", ""),
            b.get("league", ""),
            b.get("game_date", ""),
            b.get("game_time", ""),
            b.get("home_team", ""),
            b.get("away_team", ""),
            b.get("away_run_line", ""),
            b.get("home_run_line", ""),
            b.get("total", ""),
            b.get("away_dk_moneyline_american", ""),
            b.get("home_dk_moneyline_american", ""),
            b.get("away_dk_moneyline_decimal", ""),
            b.get("home_dk_moneyline_decimal", ""),
            p.get("home_pitcher", ""),
            p.get("away_pitcher", ""),
            p.get("home_prob", ""),
            p.get("away_prob", ""),
            p.get("away_projected_runs", ""),
            p.get("home_projected_runs", ""),
            p.get("total_projected_runs", ""),
        ] + ctx_vals)

<<<<<<< HEAD
        try:
            home_prob_ml = float(p.get("home_prob", ""))
            away_prob_ml = float(p.get("away_prob", ""))
            home_rl_prob = home_prob_ml * 0.75
            away_rl_prob = away_prob_ml * 0.75
        except Exception:
            home_rl_prob = ""
            away_rl_prob = ""
=======
        home_prob_run_line, away_prob_run_line = run_line_cover_probability(
            home_projected_runs=p.get("home_projected_runs", ""),
            away_projected_runs=p.get("away_projected_runs", ""),
            home_run_line=b.get("home_run_line", ""),
            away_run_line=b.get("away_run_line", ""),
        )

        if home_prob_run_line == "" or away_prob_run_line == "":
            run_line_prob_missing += 1
            log(
                f"{date} | run-line probability unavailable: "
                f"game_id={game_id} "
                f"home_projected_runs={p.get('home_projected_runs', '')} "
                f"away_projected_runs={p.get('away_projected_runs', '')} "
                f"home_run_line={b.get('home_run_line', '')} "
                f"away_run_line={b.get('away_run_line', '')}"
            )
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186

        rl_rows.append([
            RUN_TS,
            game_id,
            b.get("sport", ""),
            b.get("league", ""),
            b.get("game_date", ""),
            b.get("game_time", ""),
            b.get("home_team", ""),
            b.get("away_team", ""),
            b.get("away_run_line", ""),
            b.get("home_run_line", ""),
            b.get("total", ""),
            b.get("away_dk_run_line_american", ""),
            b.get("home_dk_run_line_american", ""),
            b.get("away_dk_run_line_decimal", ""),
            b.get("home_dk_run_line_decimal", ""),
            p.get("home_pitcher", ""),
            p.get("away_pitcher", ""),
            p.get("home_prob", ""),
            p.get("away_prob", ""),
            p.get("away_projected_runs", ""),
            p.get("home_projected_runs", ""),
            p.get("total_projected_runs", ""),
<<<<<<< HEAD
            home_rl_prob,
            away_rl_prob,
=======
            home_prob_run_line,
            away_prob_run_line,
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186
        ] + ctx_vals)

        over_raw = american_to_prob(b.get("dk_total_over_american", ""))
        under_raw = american_to_prob(b.get("dk_total_under_american", ""))
        over_prob, under_prob = normalize_probs(over_raw, under_raw)

        tot_rows.append([
            RUN_TS,
            game_id,
            b.get("sport", ""),
            b.get("league", ""),
            b.get("game_date", ""),
            b.get("game_time", ""),
            b.get("home_team", ""),
            b.get("away_team", ""),
            b.get("away_run_line", ""),
            b.get("home_run_line", ""),
            b.get("total", ""),
            b.get("dk_total_over_american", ""),
            b.get("dk_total_under_american", ""),
            b.get("dk_total_over_decimal", ""),
            b.get("dk_total_under_decimal", ""),
            p.get("home_pitcher", ""),
            p.get("away_pitcher", ""),
            p.get("home_prob", ""),
            p.get("away_prob", ""),
            p.get("away_projected_runs", ""),
            p.get("home_projected_runs", ""),
            p.get("total_projected_runs", ""),
            over_prob,
            under_prob,
        ] + ctx_vals)

<<<<<<< HEAD
    log(f"{date} | matched={matched} | unmatched={unmatched}")
=======
    log(
        f"{date} | matched={matched} | unmatched={unmatched} "
        f"| run_line_prob_missing={run_line_prob_missing}"
    )
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186

    summary["total_matched"] += matched
    summary["total_unmatched"] += unmatched
    summary["run_line_prob_missing"] += run_line_prob_missing

    def write(path, header, rows):
        assert_no_duplicate_columns(header, f"{path} output")

        expected_width = len(header)
        bad_rows = []

        for i, row in enumerate(rows, start=1):
            if len(row) != expected_width:
                bad_rows.append((i, len(row), expected_width))

        if bad_rows:
            fail(f"{path} has row/header width mismatch: {bad_rows[:10]}")

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        log(f"WROTE {path} ({len(rows)} rows)")
        summary["files_written"] += 1

    base_ml_header = [
        "last_run",
        "game_id", "sport", "league", "game_date", "game_time",
        "home_team", "away_team",
        "away_run_line", "home_run_line", "total",
        "away_dk_moneyline_american", "home_dk_moneyline_american",
        "away_dk_moneyline_decimal", "home_dk_moneyline_decimal",
        "home_pitcher", "away_pitcher", "home_prob", "away_prob",
        "away_projected_runs", "home_projected_runs", "total_projected_runs",
    ]

    base_rl_header = [
        "last_run",
        "game_id", "sport", "league", "game_date", "game_time",
        "home_team", "away_team",
        "away_run_line", "home_run_line", "total",
        "away_dk_run_line_american", "home_dk_run_line_american",
        "away_dk_run_line_decimal", "home_dk_run_line_decimal",
        "home_pitcher", "away_pitcher", "home_prob", "away_prob",
        "away_projected_runs", "home_projected_runs", "total_projected_runs",
        "home_prob_run_line", "away_prob_run_line",
    ]

    base_tot_header = [
        "last_run",
        "game_id", "sport", "league", "game_date", "game_time",
        "home_team", "away_team",
        "away_run_line", "home_run_line", "total",
        "dk_total_over_american", "dk_total_under_american",
        "dk_total_over_decimal", "dk_total_under_decimal",
        "home_pitcher", "away_pitcher", "home_prob", "away_prob",
        "away_projected_runs", "home_projected_runs", "total_projected_runs",
        "total_runs_over_prob", "total_runs_under_prob",
    ]

    write(OUT_DIR / f"{date}_mlb_moneyline.csv", base_ml_header + CONTEXT_COLS, ml_rows)
    write(OUT_DIR / f"{date}_mlb_run_line.csv", base_rl_header + CONTEXT_COLS, rl_rows)
    write(OUT_DIR / f"{date}_mlb_total.csv", base_tot_header + CONTEXT_COLS, tot_rows)

    summary["slates_written"] += 1


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    summary = {
        "slates_processed": 0,
        "slates_written": 0,
        "skipped": 0,
        "files_written": 0,
        "total_matched": 0,
        "total_unmatched": 0,
<<<<<<< HEAD
=======
        "run_line_prob_missing": 0,
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186
    }

    try:
        clear_old_outputs()

        pred_files = sorted(PRED_DIR.glob("*_MLB.csv"))
        log(f"Prediction files found: {len(pred_files)}")

        for file in pred_files:
            date = file.stem.replace("_MLB", "")
            summary["slates_processed"] += 1
            process_date(date, summary)

        log("--- SUMMARY ---")
        log(f"Last run timestamp: {RUN_TS}")
        log(f"Slates processed: {summary['slates_processed']}")
        log(f"Slates written: {summary['slates_written']}")
        log(f"Slates skipped: {summary['skipped']}")
        log(f"Files written: {summary['files_written']}")
        log(f"Total matched: {summary['total_matched']}")
        log(f"Total unmatched: {summary['total_unmatched']}")
        log(f"Run-line probability missing: {summary['run_line_prob_missing']}")
        log("STATUS: SUCCESS")

        print(
            f"merge_intake complete. "
            f"last_run={RUN_TS} "
            f"files_written={summary['files_written']} "
            f"matched={summary['total_matched']} "
            f"unmatched={summary['total_unmatched']} "
<<<<<<< HEAD
=======
            f"run_line_prob_missing={summary['run_line_prob_missing']} "
>>>>>>> 67f12db62f4d2ba562c5ffa0eebb8972e2235186
            f"Status: SUCCESS"
        )

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        raise
