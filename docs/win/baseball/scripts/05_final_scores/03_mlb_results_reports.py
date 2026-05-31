#!/usr/bin/env python3
# docs/win/baseball/scripts/05_final_scores/03_mlb_results_reports.py

import shutil
import pandas as pd
from pathlib import Path

###############################################################
######################## PATH CONFIG ##########################
###############################################################

INPUT_FILE = Path("docs/win/baseball/05_final_scores/intermediate/work_mlb.csv")

SUMMARY_DIR  = Path("docs/win/baseball/05_final_scores")
REPORTS_DIR  = Path("docs/win/baseball/05_final_scores/reports")
OVERVIEW_DIR = Path("docs/win/baseball/05_final_scores/reports/overview")
ML_DIR       = Path("docs/win/baseball/05_final_scores/reports/moneyline")
RL_DIR       = Path("docs/win/baseball/05_final_scores/reports/run_line")
TOT_DIR      = Path("docs/win/baseball/05_final_scores/reports/totals")

for d in [SUMMARY_DIR, OVERVIEW_DIR, ML_DIR, RL_DIR, TOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LEAGUE = "MLB"

###############################################################
######################## HELPERS ##############################
###############################################################

def clear_report_outputs():
    if REPORTS_DIR.exists():
        shutil.rmtree(REPORTS_DIR)

    for d in [OVERVIEW_DIR, ML_DIR, RL_DIR, TOT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def to_float(value):
    try:
        v = float(value)
        import math
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def write_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def units_won(odds, result):
    result = str(result).strip().title()
    if result == "Push":
        return 0.0
    if result == "Loss":
        return -1.0
    odds = to_float(odds)
    if odds is None:
        return None
    return odds / 100.0 if odds >= 0 else 100.0 / abs(odds)


###############################################################
######################## ENRICH ###############################
###############################################################

def enrich(df):
    df = df.copy()

    df["market_type"] = df["market_type"].astype(str).str.strip().str.lower()
    df["bet_side"]    = df["bet_side"].astype(str).str.strip().str.lower()
    df["bet_result"]  = df["bet_result"].astype(str).str.strip().str.title()

    # Ensure side_group present
    if "side_group" not in df.columns:
        def _sg(row):
            mt = row["market_type"]
            bs = row["bet_side"]
            if mt in {"moneyline", "run_line"}:
                return "HOME" if bs == "home" else ("AWAY" if bs == "away" else "")
            if mt == "total":
                return "OVER" if bs == "over" else ("UNDER" if bs == "under" else "")
            return ""
        df["side_group"] = df.apply(_sg, axis=1)

    # Units won per bet
    df["bet_units"] = df.apply(
        lambda r: units_won(r.get("dk_odds_american"), r["bet_result"]), axis=1
    )

    return df


###############################################################
######################## AGGREGATE ############################
###############################################################

def agg(df, group_cols, variable_label=None):
    """
    Returns one row per group with:
    league, market_type, [variable,] Win, Loss, Push, Total, Win_Pct, units, roi, avg_ev, avg_odds
    variable_label: if provided, renames the last group_col to 'variable'
    """
    if df.empty:
        cols = ["league", "market_type"] + (["variable"] if variable_label else []) + \
               ["Win", "Loss", "Push", "Total", "Win_Pct", "units", "roi", "avg_ev", "avg_odds"]
        return pd.DataFrame(columns=cols)

    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        wins   = int((sub["bet_result"] == "Win").sum())
        losses = int((sub["bet_result"] == "Loss").sum())
        pushes = int((sub["bet_result"] == "Push").sum())
        total  = wins + losses + pushes

        win_pct = round(wins / (wins + losses), 4) if (wins + losses) > 0 else 0.0

        unit_vals = sub["bet_units"].dropna()
        total_units = round(float(unit_vals.sum()), 4) if not unit_vals.empty else 0.0
        roi = round(total_units / total, 4) if total > 0 else 0.0

        ev_vals = pd.to_numeric(sub["ev"], errors="coerce").dropna()
        avg_ev = round(float(ev_vals.mean()), 4) if not ev_vals.empty else None

        od_vals = pd.to_numeric(sub["dk_odds_american"], errors="coerce").dropna()
        avg_odds = round(float(od_vals.mean()), 1) if not od_vals.empty else None

        row = {}
        for i, col in enumerate(group_cols):
            label = "variable" if (variable_label and i == len(group_cols) - 1) else col
            row[label] = keys[i]

        row.update({
            "Win": wins,
            "Loss": losses,
            "Push": pushes,
            "Total": total,
            "Win_Pct": win_pct,
            "units": total_units,
            "roi": roi,
            "avg_ev": avg_ev,
            "avg_odds": avg_odds,
        })

        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(list(out.columns[:len(group_cols)])).reset_index(drop=True)


def agg_simple(df, group_cols):
    """league + market_type tally — used for top-level summary."""
    if df.empty:
        return pd.DataFrame(columns=["league", "market_type", "Win", "Loss", "Push", "Total", "Win_Pct"])

    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        wins   = int((sub["bet_result"] == "Win").sum())
        losses = int((sub["bet_result"] == "Loss").sum())
        pushes = int((sub["bet_result"] == "Push").sum())
        total  = wins + losses + pushes

        win_pct = round(wins / (wins + losses), 4) if (wins + losses) > 0 else 0.0

        row = {col: keys[i] for i, col in enumerate(group_cols)}
        row.update({
            "Win": wins,
            "Loss": losses,
            "Push": pushes,
            "Total": total,
            "Win_Pct": win_pct,
        })

        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


###############################################################
###################### REPORT BUILDERS ########################
###############################################################

# ── TOP-LEVEL SUMMARY ─────────────────────────────────────────

def build_top_summary(df):
    df["league"] = LEAGUE
    out = agg_simple(df, ["league", "market_type"])
    write_csv(out, SUMMARY_DIR / "mlb_summary_overall.csv")


# ── OVERVIEW ──────────────────────────────────────────────────

def build_overview(df):
    df["league"] = LEAGUE

    # Overall
    write_csv(agg(df, ["league"]), OVERVIEW_DIR / "mlb_summary_overall.csv")

    # By market
    write_csv(
        agg(df, ["league", "market_type"], variable_label="market_type"),
        OVERVIEW_DIR / "mlb_summary_by_market.csv"
    )

    # By side group
    write_csv(
        agg(df, ["league", "side_group"], variable_label="side_group"),
        OVERVIEW_DIR / "mlb_summary_by_side_group.csv"
    )

    # By date with cumulative units
    date_df = agg(df, ["league", "game_date"], variable_label="game_date")
    date_df = date_df.sort_values("variable")
    date_df["cumulative_units"] = date_df["units"].cumsum().round(4)
    write_csv(date_df, OVERVIEW_DIR / "mlb_summary_by_date.csv")

    # By day/night
    if "day_night" in df.columns:
        write_csv(
            agg(df, ["league", "day_night"], variable_label="day_night"),
            OVERVIEW_DIR / "mlb_summary_by_day_night.csv"
        )

    # By low_confidence
    if "low_confidence" in df.columns:
        write_csv(
            agg(df, ["league", "low_confidence"], variable_label="low_confidence"),
            OVERVIEW_DIR / "mlb_summary_by_low_confidence.csv"
        )

    # Bet log
    log_cols = [
        "game_date",
        "game_id",
        "away_team",
        "home_team",
        "market_type",
        "bet_side",
        "line",
        "dk_odds_american",
        "ev",
        "kelly",
        "model_prob",
        "low_confidence",
        "day_night",
        "bet_result",
        "bet_units",
    ]

    available = [c for c in log_cols if c in df.columns]
    write_csv(df[available].copy(), OVERVIEW_DIR / "mlb_bet_log.csv")


# ── MONEYLINE ─────────────────────────────────────────────────

def build_moneyline(df):
    ml = df[df["market_type"] == "moneyline"].copy()
    if ml.empty:
        return

    ml["league"] = LEAGUE

    def _write(src, bucket_col, fname, home_away=False):
        if src.empty:
            return

        if bucket_col not in src.columns:
            return

        src = src[src[bucket_col] != "UNBUCKETED"].copy()

        if home_away:
            result = agg(
                src,
                ["league", "market_type", "side_group", bucket_col],
                variable_label=bucket_col
            )
        else:
            result = agg(
                src,
                ["league", "market_type", bucket_col],
                variable_label=bucket_col
            )

        write_csv(result, ML_DIR / fname)

    _write(ml, "ev_bucket",       "mlb_moneyline_by_ev.csv")
    _write(ml, "odds_bucket",     "mlb_moneyline_by_odds.csv")
    _write(ml, "kelly_bucket",    "mlb_moneyline_by_kelly.csv")
    _write(ml, "win_prob_bucket", "mlb_moneyline_by_win_prob.csv")

    ha = ml[ml["side_group"].isin(["HOME", "AWAY"])]

    _write(ha, "ev_bucket",       "mlb_moneyline_by_ev_home_away_summary.csv",       home_away=True)
    _write(ha, "odds_bucket",     "mlb_moneyline_by_odds_home_away_summary.csv",     home_away=True)
    _write(ha, "kelly_bucket",    "mlb_moneyline_by_kelly_home_away_summary.csv",    home_away=True)
    _write(ha, "win_prob_bucket", "mlb_moneyline_by_win_prob_home_away_summary.csv", home_away=True)


# ── RUN LINE ──────────────────────────────────────────────────

def build_run_line(df):
    rl = df[df["market_type"] == "run_line"].copy()
    if rl.empty:
        return

    rl["league"] = LEAGUE

    def _write(src, bucket_col, fname, home_away=False):
        if src.empty:
            return

        if bucket_col not in src.columns:
            return

        src = src[src[bucket_col] != "UNBUCKETED"].copy()

        if home_away:
            result = agg(
                src,
                ["league", "market_type", "side_group", bucket_col],
                variable_label=bucket_col
            )
        else:
            result = agg(
                src,
                ["league", "market_type", bucket_col],
                variable_label=bucket_col
            )

        write_csv(result, RL_DIR / fname)

    _write(rl, "ev_bucket",       "mlb_run_line_by_ev.csv")
    _write(rl, "odds_bucket",     "mlb_run_line_by_odds.csv")
    _write(rl, "kelly_bucket",    "mlb_run_line_by_kelly.csv")
    _write(rl, "win_prob_bucket", "mlb_run_line_by_win_prob.csv")
    _write(rl, "run_line_side",   "mlb_run_line_by_side.csv")

    ha = rl[rl["side_group"].isin(["HOME", "AWAY"])]

    _write(ha, "ev_bucket",       "mlb_run_line_by_ev_home_away_summary.csv",       home_away=True)
    _write(ha, "odds_bucket",     "mlb_run_line_by_odds_home_away_summary.csv",     home_away=True)
    _write(ha, "kelly_bucket",    "mlb_run_line_by_kelly_home_away_summary.csv",    home_away=True)
    _write(ha, "win_prob_bucket", "mlb_run_line_by_win_prob_home_away_summary.csv", home_away=True)
    _write(ha, "run_line_side",   "mlb_run_line_by_side_home_away_summary.csv",     home_away=True)


# ── TOTALS ────────────────────────────────────────────────────

def build_totals(df):
    tot = df[df["market_type"] == "total"].copy()
    if tot.empty:
        return

    tot["league"] = LEAGUE

    def _write(src, bucket_col, fname, home_away=False):
        if src.empty:
            return

        if bucket_col not in src.columns:
            return

        src = src[src[bucket_col] != "UNBUCKETED"].copy()

        if home_away:
            result = agg(
                src,
                ["league", "market_type", "side_group", bucket_col],
                variable_label=bucket_col
            )
        else:
            result = agg(
                src,
                ["league", "market_type", bucket_col],
                variable_label=bucket_col
            )

        write_csv(result, TOT_DIR / fname)

    _write(tot, "ev_bucket",          "mlb_total_by_ev.csv")
    _write(tot, "odds_bucket",        "mlb_total_by_odds.csv")
    _write(tot, "kelly_bucket",       "mlb_total_by_kelly.csv")
    _write(tot, "win_prob_bucket",    "mlb_total_by_win_prob.csv")
    _write(tot, "total_range_bucket", "mlb_total_by_total_range.csv")
    _write(tot, "side_group",         "mlb_total_by_side.csv")

    ou = tot[tot["side_group"].isin(["OVER", "UNDER"])]

    _write(ou, "ev_bucket",          "mlb_total_by_ev_home_away_summary.csv",          home_away=True)
    _write(ou, "odds_bucket",        "mlb_total_by_odds_home_away_summary.csv",        home_away=True)
    _write(ou, "kelly_bucket",       "mlb_total_by_kelly_home_away_summary.csv",       home_away=True)
    _write(ou, "win_prob_bucket",    "mlb_total_by_win_prob_home_away_summary.csv",    home_away=True)
    _write(ou, "total_range_bucket", "mlb_total_by_total_range_home_away_summary.csv", home_away=True)
    _write(ou, "side_group",         "mlb_total_by_side_home_away_summary.csv",        home_away=True)


###############################################################
######################## MAIN #################################
###############################################################

def run():
    clear_report_outputs()

    if not INPUT_FILE.exists():
        print(f"ERROR: input file not found: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE, dtype=str)
    df = enrich(df)

    build_top_summary(df)
    build_overview(df)
    build_moneyline(df)
    build_run_line(df)
    build_totals(df)

    print("MLB reports complete.")


if __name__ == "__main__":
    run()
