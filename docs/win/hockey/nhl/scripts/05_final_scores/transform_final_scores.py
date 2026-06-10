#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/05_final_scores/transform_final_scores.py

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import pandas as pd


SPORT = "hockey"
LEAGUE = "nhl"
LEAGUE_OUT = "NHL"

RAW_DIR = Path("docs/win/hockey/nhl/00_intake/drat_raw")
GAMES_DIR = Path("docs/win/hockey/nhl/00_intake/games")

OUT_DIR = Path("docs/win/hockey/nhl/05_final_scores/final_scores")
ERROR_DIR = Path("docs/win/hockey/nhl/errors/05_final_scores")
LOG_FILE = ERROR_DIR / "transform_final_scores.txt"

RAW_PATTERN = "*_nhl_raw.json"
DATE_RE = re.compile(r"(\d{4}_\d{2}_\d{2})_nhl_raw\.json$", re.IGNORECASE)

OUTPUT_COLUMNS = [
    "sport",
    "league",
    "game_date",
    "game_id",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
    "total_score",
    "away_puck_line_result",
    "home_puck_line_result",
]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_DIR.mkdir(parents=True, exist_ok=True)


def reset_log() -> None:
    LOG_FILE.write_text("", encoding="utf-8")


def log(msg: str) -> None:
    stamp = datetime.now(UTC).isoformat()
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{stamp} | {msg}\n")


def fail(msg: str) -> None:
    log(f"ERROR: {msg}")
    raise RuntimeError(msg)


def wipe_outputs() -> None:
    for path in OUT_DIR.glob("*_NHL_final_scores.csv"):
        path.unlink()
        log(f"Deleted old output: {path}")


def extract_date_from_path(path: Path) -> str:
    m = DATE_RE.search(path.name)
    if not m:
        fail(f"Could not extract date from raw filename: {path}")
    return m.group(1)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        fail(f"Failed reading JSON {path}: {e}")


def flatten_raw_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        for key in ["rows", "data", "games", "events", "raw_rows"]:
            if key in payload and isinstance(payload[key], list):
                rows = payload[key]
                break
        else:
            rows = [payload]
    else:
        fail(f"Unsupported raw JSON payload type: {type(payload).__name__}")

    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            out.append(row)

    return out


def norm_key(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def first_present(row: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in row and row[name] not in [None, ""]:
            return row[name]

    lower_map = {str(k).lower(): k for k in row.keys()}
    for name in names:
        key = lower_map.get(name.lower())
        if key is not None and row[key] not in [None, ""]:
            return row[key]

    return None


def parse_int_score(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if pd.isna(value):
            return None
        if value.is_integer():
            return int(value)
        return None

    s = str(value).strip()
    if not s:
        return None

    m = re.search(r"-?\d+", s)
    if not m:
        return None

    return int(m.group(0))


def is_completed(row: dict[str, Any]) -> bool:
    status = first_present(
        row,
        [
            "game_status",
            "status",
            "status_type",
            "event_status",
            "state",
            "game_state",
        ],
    )

    status_norm = norm_key(status)

    completed_values = {
        "completed",
        "complete",
        "final",
        "final ot",
        "final so",
        "ended",
        "closed",
    }

    return status_norm in completed_values


def get_team(row: dict[str, Any], side: str) -> str:
    if side == "away":
        names = [
            "away_team",
            "away",
            "visitor_team",
            "visitor",
            "road_team",
            "team_away",
            "away_name",
        ]
    elif side == "home":
        names = [
            "home_team",
            "home",
            "team_home",
            "home_name",
        ]
    else:
        fail(f"Invalid side: {side}")

    value = first_present(row, names)
    if value is None:
        return ""

    return str(value).strip()


def get_score(row: dict[str, Any], side: str) -> int | None:
    if side == "away":
        names = [
            "away_score",
            "away_team_score",
            "visitor_score",
            "road_score",
            "away_points",
            "score_away",
        ]
    elif side == "home":
        names = [
            "home_score",
            "home_team_score",
            "home_points",
            "score_home",
        ]
    else:
        fail(f"Invalid side: {side}")

    return parse_int_score(first_present(row, names))


def load_games_for_date(game_date: str) -> pd.DataFrame:
    path = GAMES_DIR / f"{game_date}_nhl_games.csv"

    if not path.exists():
        log(f"Stage 00 games file not found for {game_date}: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, dtype=str)

    required = {"game_id", "game_date", "away_team", "home_team"}
    missing = sorted(required - set(df.columns))
    if missing:
        fail(f"Stage 00 games file missing required columns {missing}: {path}")

    df = df.copy()
    df["game_date"] = df["game_date"].astype(str).str.strip()
    df["away_team_key"] = df["away_team"].map(norm_key)
    df["home_team_key"] = df["home_team"].map(norm_key)

    return df


def attach_game_id(row: dict[str, Any], games_df: pd.DataFrame, game_date: str) -> str:
    raw_game_id = first_present(row, ["game_id", "id", "event_id", "gameId", "eventId"])

    if raw_game_id not in [None, ""]:
        return str(raw_game_id).strip()

    if games_df.empty:
        return ""

    away_key = norm_key(row.get("away_team"))
    home_key = norm_key(row.get("home_team"))

    matches = games_df[
        (games_df["game_date"] == game_date)
        & (games_df["away_team_key"] == away_key)
        & (games_df["home_team_key"] == home_key)
    ]

    if len(matches) == 1:
        return str(matches.iloc[0]["game_id"]).strip()

    if len(matches) > 1:
        log(
            f"Multiple Stage 00 game_id matches for {game_date}: "
            f"{row.get('away_team')} at {row.get('home_team')}"
        )
        return ""

    log(
        f"No Stage 00 game_id match for {game_date}: "
        f"{row.get('away_team')} at {row.get('home_team')}"
    )
    return ""


def build_final_score_rows(raw_rows: list[dict[str, Any]], game_date: str) -> pd.DataFrame:
    games_df = load_games_for_date(game_date)

    rows: list[dict[str, Any]] = []
    raw_count = 0
    completed_count = 0
    skipped_missing_team = 0
    skipped_missing_score = 0

    for raw in raw_rows:
        raw_count += 1

        if not is_completed(raw):
            continue

        completed_count += 1

        away_team = get_team(raw, "away")
        home_team = get_team(raw, "home")
        away_score = get_score(raw, "away")
        home_score = get_score(raw, "home")

        if not away_team or not home_team:
            skipped_missing_team += 1
            log(f"Skipped completed row with missing team name: {raw}")
            continue

        if away_score is None or home_score is None:
            skipped_missing_score += 1
            log(f"Skipped completed row with missing score: {raw}")
            continue

        shaped = {
            "sport": SPORT,
            "league": LEAGUE,
            "game_date": game_date,
            "game_id": "",
            "away_team": away_team,
            "home_team": home_team,
            "away_score": int(away_score),
            "home_score": int(home_score),
            "total_score": int(away_score) + int(home_score),
            "away_puck_line_result": "",
            "home_puck_line_result": "",
        }

        shaped["game_id"] = attach_game_id(shaped | raw, games_df, game_date)

        rows.append(shaped)

    log(
        f"{game_date}: raw_rows={raw_count}, completed_rows={completed_count}, "
        f"written_rows={len(rows)}, skipped_missing_team={skipped_missing_team}, "
        f"skipped_missing_score={skipped_missing_score}"
    )

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    if df.empty:
        return df

    missing_game_id = df["game_id"].astype(str).str.strip().eq("")
    if missing_game_id.any():
        bad = df.loc[missing_game_id, ["game_date", "away_team", "home_team"]]
        for _, r in bad.iterrows():
            log(
                f"Missing game_id after transform: "
                f"{r['game_date']} | {r['away_team']} at {r['home_team']}"
            )

    dupes = df[df["game_id"].astype(str).str.strip().ne("")].duplicated(
        subset=["game_id"],
        keep=False,
    )
    if dupes.any():
        duped_ids = sorted(df.loc[dupes, "game_id"].astype(str).unique().tolist())
        fail(f"Duplicate game_id values in transformed final scores for {game_date}: {duped_ids}")

    return df


def write_final_scores(df: pd.DataFrame, game_date: str) -> Path:
    out_path = OUT_DIR / f"{game_date}_{LEAGUE_OUT}_final_scores.csv"
    df.to_csv(out_path, index=False)
    log(f"Wrote {len(df)} rows: {out_path}")
    return out_path


def main() -> int:
    ensure_dirs()
    reset_log()

    log("=== transform_final_scores START ===")
    log(f"RAW_DIR={RAW_DIR}")
    log(f"GAMES_DIR={GAMES_DIR}")
    log(f"OUT_DIR={OUT_DIR}")

    if not RAW_DIR.exists():
        fail(f"Raw final-score source folder does not exist: {RAW_DIR}")

    wipe_outputs()

    raw_files = sorted(RAW_DIR.glob(RAW_PATTERN))
    if not raw_files:
        log(f"No raw files found matching {RAW_PATTERN} in {RAW_DIR}")
        log("=== transform_final_scores END ===")
        return 0

    total_files = 0
    total_rows = 0

    for raw_path in raw_files:
        total_files += 1
        game_date = extract_date_from_path(raw_path)

        log(f"Processing raw file: {raw_path}")

        payload = load_json(raw_path)
        raw_rows = flatten_raw_payload(payload)
        df = build_final_score_rows(raw_rows, game_date)

        write_final_scores(df, game_date)
        total_rows += len(df)

    log(f"Files processed: {total_files}")
    log(f"Total final-score rows written: {total_rows}")
    log("=== transform_final_scores END ===")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        ensure_dirs()
        log(f"FATAL: {e}")
        print(f"transform_final_scores failed: {e}", file=sys.stderr)
        raise
