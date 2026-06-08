#!/usr/bin/env python3
# docs/win/final_scores/scripts/05_results/soccer/01_soccer_results_grade.py

from datetime import datetime
from pathlib import Path
import pandas as pd

# =========================
# PATHS
# =========================

SELECT_DIR       = Path("docs/win/soccer/04_select")
FINAL_SCORES_DIR = Path("docs/win/final_scores/results/soccer/final_scores")

OUTPUT_DIR = Path("docs/win/final_scores/results/soccer/graded")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERROR_DIR = Path("docs/win/final_scores/errors")
ERROR_DIR.mkdir(parents=True, exist_ok=True)

ERROR_LOG   = ERROR_DIR / "soccer_results_grade_errors.txt"
SUMMARY_LOG = ERROR_DIR / "soccer_results_grade_summary.txt"

MASTER_FILE = OUTPUT_DIR / "SOCCER_final.csv"

LEAGUE_MAP = {
    "epl":        "EPL",
    "bundesliga": "BUNDESLIGA",
    "laliga":     "LALIGA",
    "ligue1":     "LIGUE1",
    "seriea":     "SERIEA",
    "mls":        "MLS",
}


# =========================
# LOGGING
# =========================

def reset_logs():
    ERROR_LOG.write_text("", encoding="utf-8")
    SUMMARY_LOG.write_text("", encoding="utf-8")


def log_error(msg):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


def log_summary(msg):
    with open(SUMMARY_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


# =========================
# HELPERS
# =========================

def safe_read(path):
    try:
        path = Path(path)
        if not path.exists():
            log_error(f"MISSING FILE | {path}")
            return pd.DataFrame()
        df = pd.read_csv(path)
        if df.empty:
            log_error(f"EMPTY FILE | {path}")
            return pd.DataFrame()
        return df
    except Exception as e:
        log_error(f"READ ERROR | {path} | {e}")
        return pd.DataFrame()


def derive_take_bet(market: str, side: str) -> str:
    market = str(market).lower().strip()
    side   = str(side).lower().strip()

    if market == "match_odds":
        return side
    if market == "total25":
        return f"{side}25"
    if market == "total35":
        return f"{side}35"
    if market == "btts":
        return f"btts_{side}"
    return market


def find_score_file(game_date: str, league_raw: str) -> Path | None:
    league_upper = LEAGUE_MAP.get(league_raw.lower(), league_raw.upper())
    league_lower = league_upper.lower()

    candidates = [
        FINAL_SCORES_DIR / league_upper / f"{game_date}_{league_upper}.csv",
        FINAL_SCORES_DIR / league_lower / f"{game_date}_{league_lower}.csv",
        FINAL_SCORES_DIR / league_upper / f"{game_date}_{league_lower}.csv",
        FINAL_SCORES_DIR / league_lower / f"{game_date}_{league_upper}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_scores_for_league_date(game_date: str, league_raw: str) -> pd.DataFrame:
    path = find_score_file(game_date, league_raw)
    if path is None:
        log_error(f"NO SCORE FILE | league={league_raw} date={game_date}")
        return pd.DataFrame()

    df = safe_read(path)
    if df.empty:
        return pd.DataFrame()

    needed = ["sport", "league", "game_date", "match_time",
              "home_team", "away_team", "home_score", "away_score"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        log_error(f"MISSING COLUMNS {missing} | {path}")
        return pd.DataFrame()

    df = df[needed].copy()
    for col in ["home_team", "away_team", "game_date"]:
        df[col] = df[col].astype(str).str.strip()
    df["league_score"] = df["league"].astype(str).str.lower().str.strip()

    return df.drop_duplicates(subset=["league_score", "home_team", "away_team", "game_date"])


# =========================
# GRADING
# =========================

def grade_row(row) -> str:
    try:
        take_bet = str(row.get("take_bet", "")).lower().strip()
        home     = pd.to_numeric(row.get("home_score"), errors="coerce")
        away     = pd.to_numeric(row.get("away_score"), errors="coerce")

        if pd.isna(home) or pd.isna(away):
            return "Push"

        goals = home + away

        if take_bet == "home":
            return "Win" if home > away else "Loss"
        if take_bet == "away":
            return "Win" if away > home else "Loss"
        if take_bet == "draw":
            return "Win" if home == away else "Loss"

        if take_bet == "over25":
            return "Win" if goals > 2.5 else "Loss"
        if take_bet == "under25":
            return "Win" if goals < 2.5 else "Loss"

        if take_bet == "over35":
            return "Win" if goals > 3.5 else "Loss"
        if take_bet == "under35":
            return "Win" if goals < 3.5 else "Loss"

        if take_bet == "btts_yes":
            return "Win" if home > 0 and away > 0 else "Loss"
        if take_bet == "btts_no":
            return "Win" if home == 0 or away == 0 else "Loss"

    except Exception as e:
        log_error(f"GRADE ERROR | game_id={row.get('game_id', '')} take_bet={row.get('take_bet', '')} | {e}")

    return "Push"


# =========================
# PROCESS
# =========================

def process():
    select_files = sorted(SELECT_DIR.glob("*_soccer_bets.csv"))
    log_summary(f"Select files found: {len(select_files)}")

    if not select_files:
        log_error("NO SELECT FILES FOUND")
        return

    all_rows = []

    for file in select_files:
        game_date = file.stem.replace("_soccer_bets", "")
        log_summary(f"Processing: {file.name} | date={game_date}")

        bets_df = safe_read(file)
        if bets_df.empty:
            continue

        # Normalise key columns
        for col in ["market", "side", "league", "home_team", "away_team"]:
            if col in bets_df.columns:
                bets_df[col] = bets_df[col].astype(str).str.strip()

        # ── FIX: stamp game_date onto bets so merge key exists ──
        bets_df["game_date"]     = game_date
        bets_df["league_lower"]  = bets_df["league"].str.lower()
        bets_df["market_type"]   = bets_df["market"].str.lower()
        bets_df["take_bet"]      = bets_df.apply(
            lambda r: derive_take_bet(r["market"], r["side"]), axis=1
        )
        bets_df["odds_american"] = pd.to_numeric(bets_df.get("odds"), errors="coerce")
        bets_df["edge_pct"]      = pd.to_numeric(bets_df.get("ev"),   errors="coerce")

        merged_frames = []
        for league_raw, league_bets in bets_df.groupby("league_lower"):
            scores = load_scores_for_league_date(game_date, league_raw)
            if scores.empty:
                log_error(f"NO SCORES MERGED | league={league_raw} date={game_date}")
                league_bets = league_bets.copy()
                league_bets["home_score"]       = None
                league_bets["away_score"]       = None
                league_bets["league_score"]     = league_raw
                league_bets["market_scorefile"] = league_raw
                merged_frames.append(league_bets)
                continue

            scores["league_lower"] = scores["league_score"]

            merged = league_bets.merge(
                scores[["league_lower", "home_team", "away_team",
                         "game_date", "home_score", "away_score", "league_score"]],
                on=["league_lower", "home_team", "away_team", "game_date"],
                how="left"
            )
            merged["market_scorefile"] = merged["league_score"].fillna(league_raw)
            merged_frames.append(merged)

        if not merged_frames:
            continue

        day_df = pd.concat(merged_frames, ignore_index=True)
        day_df["bet_result"] = day_df.apply(grade_row, axis=1)

        wins   = (day_df["bet_result"] == "Win").sum()
        losses = (day_df["bet_result"] == "Loss").sum()
        pushes = (day_df["bet_result"] == "Push").sum()
        log_summary(f"  {file.name} | rows={len(day_df)} W={wins} L={losses} P={pushes}")

        out_file = OUTPUT_DIR / f"{game_date}_results_SOCCER.csv"
        day_df.to_csv(out_file, index=False)
        log_summary(f"  WROTE: {out_file}")

        all_rows.append(day_df)

    if all_rows:
        final = pd.concat(all_rows, ignore_index=True)
        final.to_csv(MASTER_FILE, index=False)
        log_summary(f"MASTER FILE WRITTEN | rows={len(final)} | {MASTER_FILE}")
    else:
        log_summary("NO ROWS TO WRITE — master file not updated")


# =========================
# MAIN
# =========================

def main():
    reset_logs()
    log_summary(f"=== START 01_soccer_results_grade.py {datetime.now().isoformat()} ===")
    process()
    log_summary(f"=== END 01_soccer_results_grade.py {datetime.now().isoformat()} ===")
    print("Soccer grading complete.")


if __name__ == "__main__":
    main()
