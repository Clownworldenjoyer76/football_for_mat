#!/usr/bin/env python3
# docs/win/final_scores/scripts/05_results/soccer/03_soccer_results_reports.py

from datetime import datetime
from pathlib import Path
import pandas as pd

INTERMEDIATE  = Path("docs/win/final_scores/intermediate/work_soccer.csv")
SUMMARY_DIR   = Path("docs/win/final_scores/deeper_summaries/soccer")
ERROR_DIR     = Path("docs/win/final_scores/errors")
MARKET_TALLY  = Path("docs/win/final_scores/soccer_market_tally.csv")
ERROR_LOG     = ERROR_DIR / "soccer_results_reports_errors.txt"

SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

# market_type values from the updated select file
MARKET_TYPES = ["match_odds", "total25", "total35", "btts"]


# =========================
# LOGGING
# =========================

def log_error(msg):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


def log_summary(msg):
    summary_log = ERROR_DIR / "soccer_results_reports_summary.txt"
    with open(summary_log, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


# =========================
# HELPERS
# =========================

def summarize(df):
    wins   = int((df["bet_result"] == "Win").sum())
    losses = int((df["bet_result"] == "Loss").sum())
    pushes = int((df["bet_result"] == "Push").sum())
    total  = wins + losses + pushes
    pct    = wins / (wins + losses) if (wins + losses) > 0 else 0
    return wins, losses, pushes, total, round(pct, 4)


def aggregate(df, cols):
    rows = []
    for keys, sub in df.groupby(cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        w, l, p, t, pct = summarize(sub)
        r = {c: keys[i] for i, c in enumerate(cols)}
        r.update({"Win": w, "Loss": l, "Push": p, "Total": t, "Win_Pct": pct})
        rows.append(r)
    return pd.DataFrame(rows)


def write(df, path):
    df.to_csv(path, index=False)
    log_summary(f"WROTE {path} ({len(df)} rows)")


# =========================
# REPORTS
# =========================

def build_reports(df):
    # Per market_type (total25, total35, btts, match_odds)
    for market_type in MARKET_TYPES:
        sub = df[df["market_type"] == market_type]
        if sub.empty:
            continue

        write(aggregate(sub, ["market", "market_type", "edge_bucket"]),
              SUMMARY_DIR / f"{market_type}_edge_bucket_summary.csv")

        write(aggregate(sub, ["market", "market_type", "odds_bucket"]),
              SUMMARY_DIR / f"{market_type}_odds_bucket_summary.csv")

        write(aggregate(sub, ["league_market", "market_type", "edge_bucket"]),
              SUMMARY_DIR / f"{market_type}_by_league_edge_bucket_summary.csv")

        write(aggregate(sub, ["league_market", "market_type", "odds_bucket"]),
              SUMMARY_DIR / f"{market_type}_by_league_odds_bucket_summary.csv")

    # Overall across all market types
    write(aggregate(df, ["market", "edge_bucket"]),
          SUMMARY_DIR / "by_market_edge_bucket_summary.csv")

    write(aggregate(df, ["market", "odds_bucket"]),
          SUMMARY_DIR / "by_market_odds_bucket_summary.csv")

    write(aggregate(df, ["league_market", "edge_bucket"]),
          SUMMARY_DIR / "by_league_edge_bucket_summary.csv")

    write(aggregate(df, ["league_market", "odds_bucket"]),
          SUMMARY_DIR / "by_league_odds_bucket_summary.csv")


# =========================
# MARKET TALLY
# =========================

def build_market_tally(df):
    rows = []

    # Overall SOCCER per market_type
    for m in MARKET_TYPES:
        sub = df[df["market_type"] == m]
        if sub.empty:
            continue
        w, l, p, t, pct = summarize(sub)
        rows.append({"market": "SOCCER", "market_type": m,
                     "Win": w, "Loss": l, "Push": p, "Total": t, "Win_Pct": pct})

    # Per league (all market types combined)
    for league, sub in df.groupby("league_market"):
        w, l, p, t, pct = summarize(sub)
        rows.append({"market": league, "market_type": "ALL",
                     "Win": w, "Loss": l, "Push": p, "Total": t, "Win_Pct": pct})

    # Per league × market_type
    for (league, mtype), sub in df.groupby(["league_market", "market_type"]):
        w, l, p, t, pct = summarize(sub)
        rows.append({"market": league, "market_type": mtype,
                     "Win": w, "Loss": l, "Push": p, "Total": t, "Win_Pct": pct})

    tally = pd.DataFrame(rows)
    tally.to_csv(MARKET_TALLY, index=False)
    log_summary(f"MARKET TALLY WRITTEN | {MARKET_TALLY} ({len(tally)} rows)")


# =========================
# MAIN
# =========================

def main():
    # reset summary log for this script
    summary_log = ERROR_DIR / "soccer_results_reports_summary.txt"
    summary_log.write_text("", encoding="utf-8")
    log_summary(f"=== START 03_soccer_results_reports.py {datetime.now().isoformat()} ===")

    if not INTERMEDIATE.exists():
        log_error("INTERMEDIATE FILE MISSING — run 02 first")
        print("ERROR: intermediate file missing.")
        return

    df = pd.read_csv(INTERMEDIATE)

    if df.empty:
        log_error("INTERMEDIATE FILE EMPTY")
        print("ERROR: intermediate file empty.")
        return

    # league_market comes from market_scorefile set in grade step
    df["league_market"] = df["market_scorefile"].astype(str).str.lower().str.strip()

    log_summary(f"Rows loaded: {len(df)}")
    log_summary(f"market_types: {df['market_type'].value_counts().to_dict()}")
    log_summary(f"leagues: {df['league_market'].value_counts().to_dict()}")

    build_reports(df)
    build_market_tally(df)

    log_summary(f"=== END 03_soccer_results_reports.py {datetime.now().isoformat()} ===")
    print("Soccer reports generated.")


if __name__ == "__main__":
    main()
