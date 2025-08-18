#!/usr/bin/env python3
"""
Pull injury reports / practice status and derive OUT/QUESTIONABLE/PROBABLE flags.

Fix for CI error:
- Some nfl_data_py versions expose different function names.
  We detect and call whichever exists:
    - import_injuries(years)
    - import_injury_reports(years)
    - import_injury(years)
    - import_player_injuries(years)
Outputs:
  - data/raw/injuries/injury_reports_latest.csv
  - data/raw/injuries/injury_reports_{season}.csv  (when --season provided)
"""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import pandas as pd

# ----------------- Config -----------------
RAW_DIR = Path("data/raw")
OUT_DIR = RAW_DIR / "injuries"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_LATEST = OUT_DIR / "injury_reports_latest.csv"
SCHEDULES_GLOB = RAW_DIR / "nflverse" / "schedules_*.csv.gz"

# ----------------- Utils -----------------
def infer_seasons_from_schedules() -> list[int]:
    paths = sorted(SCHEDULES_GLOB.parent.glob(SCHEDULES_GLOB.name))
    if not paths:
        return []
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            pass
    if not frames:
        return []
    sched = pd.concat(frames, ignore_index=True)
    if "season" not in sched.columns:
        return []
    return sorted(pd.to_numeric(sched["season"], errors="coerce").dropna().astype(int).unique())

def normalize_status(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    return s.replace("_", " ").replace("-", " ")

OUT_STATUSES = {
    "out", "injured reserve", "ir", "pup", "non football injury",
    "nfi", "suspended", "did not practice", "dnp", "covid 19",
}
QUESTIONABLE_STATUSES = {"questionable", "limited practice", "lp", "limited"}
PROBABLE_STATUSES = {"probable", "full practice", "fp"}

# ----------------- Main -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None, help="e.g., 2025")
    args = parser.parse_args()

    try:
        import nfl_data_py as nfl
    except Exception as e:
        sys.stderr.write(f"ERROR: cannot import nfl_data_py: {e}\n")
        sys.exit(1)

    # Determine seasons
    if args.season:
        seasons = [args.season]
    else:
        seasons = infer_seasons_from_schedules()
        if not seasons:
            # fallback to recent 3 years
            nowy = datetime.now().year
            seasons = [nowy - 2, nowy - 1, nowy]

    # Pick an available injuries function & parameter name
    candidates = [
        "import_injuries",
        "import_injury_reports",
        "import_injury",
        "import_player_injuries",
    ]
    inj_func = None
    for name in candidates:
        if hasattr(nfl, name):
            inj_func = getattr(nfl, name)
            break
    if inj_func is None:
        sys.stderr.write(
            "ERROR: nfl_data_py does not expose an injuries import function "
            f"(tried {', '.join(candidates)}). Please update nfl_data_py.\n"
        )
        sys.exit(1)

    # Some versions expect 'years=' vs positional; support both
    try:
        inj = inj_func(seasons)  # positional
    except TypeError:
        try:
            inj = inj_func(years=seasons)  # keyword
        except TypeError:
            inj = inj_func(seasons=seasons)  # alternate keyword

    # -------- Normalize schema --------
    inj = inj.copy()
    inj.columns = [c.strip().lower() for c in inj.columns]

    # Unify player_id
    if "player_id" not in inj.columns:
        for alt in ["gsis_id", "player_gsis_id", "gsis_player_id", "nfl_id"]:
            if alt in inj.columns:
                inj = inj.rename(columns={alt: "player_id"})
                break

    # Keep useful columns when present
    keep_candidates = [
        "season", "week", "game_id",
        "team", "opponent",
        "player_id", "player_name", "position",
        "report_status", "game_status", "status",
        "practice_status", "practice_participation",
        "entry_date", "report_primary_injury",
    ]
    keep = [c for c in keep_candidates if c in inj.columns]
    if keep:
        inj = inj[keep].copy()

    # Build normalized status fields from whatever exists
    base_status = None
    for c in ["report_status", "game_status", "status"]:
        if c in inj.columns:
            base_status = c; break
    if base_status is None:
        inj["report_status_norm"] = ""
    else:
        inj["report_status_norm"] = inj[base_status].apply(normalize_status)

    practice_col = None
    for c in ["practice_status", "practice_participation"]:
        if c in inj.columns:
            practice_col = c; break
    inj["practice_status_norm"] = inj[practice_col].apply(normalize_status) if practice_col else ""

    is_out = inj["report_status_norm"].isin(OUT_STATUSES) | inj["practice_status_norm"].isin({"did not practice", "dnp"})
    is_q = inj["report_status_norm"].isin(QUESTIONABLE_STATUSES)
    is_prob = inj["report_status_norm"].isin(PROBABLE_STATUSES)

    inj["is_out"] = is_out.astype(int)
    inj["is_questionable"] = is_q.astype(int)
    inj["is_probable"] = is_prob.astype(int)

    # Deduplicate to most recent per player/game if we have a timestamp
    if "entry_date" in inj.columns:
        inj["entry_dt"] = pd.to_datetime(inj["entry_date"], errors="coerce", utc=True)
        subset = [c for c in ["season", "week", "game_id", "player_id"] if c in inj.columns]
        sort_cols = subset + (["entry_dt"] if "entry_dt" in inj.columns else [])
        inj = inj.sort_values(sort_cols).drop_duplicates(subset=subset, keep="last")
        inj = inj.drop(columns=["entry_dt"], errors="ignore")

    inj["fetched_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Write outputs
    OUT_LATEST.parent.mkdir(parents=True, exist_ok=True)
    inj.to_csv(OUT_LATEST, index=False)
    print(f"Wrote: {OUT_LATEST}")

    if args.season:
        seasonal = OUT_DIR / f"injury_reports_{args.season}.csv"
        inj.to_csv(seasonal, index=False)
        print(f"Wrote: {seasonal}")

if __name__ == "__main__":
    main()
