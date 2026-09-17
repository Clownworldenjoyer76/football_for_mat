#!/usr/bin/env python3
"""
Build weekly NFL quarterback statistics from one season of local nflverse PBP.

Input:
    docs/win/football/nfl/00_intake/pbp/{season}_pbp.csv.gz

Output:
    docs/win/football/nfl/00_intake/qb/{season}_qb_stats.csv

Standard report:
    docs/win/football/nfl/errors/00_intake/pull_qb_stats.json

This script is run by the Tuesday NFL pipeline and may also be run manually.
The established QB-stat formulas are preserved:
- dropbacks = count(qb_dropback == 1)
- epa_per_play = mean(qb_epa) over dropbacks
- cpoe = mean(cpoe) over pass attempts
- air_yards = mean(air_yards) over pass attempts
- sack_rate = sacks / dropbacks
- interception_rate = interceptions / dropbacks
- fumble_rate = passer fumbles / dropbacks

starts, adjusted_completion_pct, pressure_to_sack_rate, and
turnover_worthy_play_rate remain intentionally blank.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter


PBP_DIR = NFL_ROOT / "00_intake" / "pbp"
OUTPUT_DIR = NFL_ROOT / "00_intake" / "qb"
RESULTS_DIR = NFL_ROOT / "04_final_results" / "results"
REPORT_ROOT = NFL_ROOT / "errors"

GROUP_COLS = [
    "season",
    "week",
    "posteam",
    "passer_player_id",
    "passer_player_name",
]

OUTPUT_HEADERS = [
    "sport",
    "league",
    "season",
    "week",
    "team",
    "player_id",
    "qb_name",
    "starts",
    "dropbacks",
    "epa_per_play",
    "cpoe",
    "air_yards",
    "adjusted_completion_pct",
    "sack_rate",
    "pressure_to_sack_rate",
    "turnover_worthy_play_rate",
    "interception_rate",
    "fumble_rate",
]

PBP_REQUIRED_COLUMNS = [
    "season",
    "week",
    "game_id",
    "play_id",
    "posteam",
    "passer_player_id",
    "passer_player_name",
    "qb_dropback",
    "pass_attempt",
    "qb_epa",
    "cpoe",
    "air_yards",
    "sack",
    "interception",
    "fumbled_1_player_id",
]

NUMERIC_PBP_COLUMNS = [
    "season",
    "week",
    "play_id",
    "qb_dropback",
    "pass_attempt",
    "qb_epa",
    "cpoe",
    "air_yards",
    "sack",
    "interception",
]

NUMERIC_OUTPUT_COLUMNS = [
    "dropbacks",
    "epa_per_play",
    "cpoe",
    "air_yards",
    "sack_rate",
    "interception_rate",
    "fumble_rate",
]

RATE_COLUMNS = [
    "sack_rate",
    "interception_rate",
    "fumble_rate",
]

INTENTIONALLY_BLANK_COLUMNS = [
    "starts",
    "adjusted_completion_pct",
    "pressure_to_sack_rate",
    "turnover_worthy_play_rate",
]


class QBStatsError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build NFL weekly quarterback stats from one season of PBP."
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="NFL season to build.",
    )
    args = parser.parse_args()

    if args.season < 2000 or args.season > 2100:
        parser.error("--season must be between 2000 and 2100")

    return args


def _blank_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().eq("")


def read_pbp(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"PBP input file not found: {path}")

    if path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(
            path,
            compression="gzip",
            low_memory=False,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_existing_output(
    output_path: Path,
    reporter: PipelineReporter,
) -> tuple[pd.DataFrame | None, bool]:
    if not output_path.exists():
        return None, True

    reporter.add_input(output_path)

    try:
        return pd.read_csv(output_path, low_memory=False), True
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), True
    except Exception as exc:
        reporter.warning(
            "Existing QB-stat output could not be read for empty-input safety",
            path=str(output_path),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return None, False


def count_completed_games(
    season: int,
    reporter: PipelineReporter,
) -> tuple[int, bool, int]:
    paths = sorted(RESULTS_DIR.glob(f"{season}_*.csv"))
    completed: set[str] = set()
    inspection_uncertain = False

    for path in paths:
        reporter.add_input(path)

        try:
            frame = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            inspection_uncertain = True
            reporter.warning(
                "Could not inspect final-results file while evaluating empty QB-stat safety",
                path=str(path),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            continue

        if "status" not in frame.columns:
            inspection_uncertain = True
            reporter.warning(
                "Final-results file lacks status column while evaluating empty QB-stat safety",
                path=str(path),
            )
            continue

        statuses = (
            frame["status"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.casefold()
        )
        done = (
            statuses.str.contains("final", regex=False)
            | statuses.str.contains("completed", regex=False)
        )

        if not done.any():
            continue

        if "game_id" in frame.columns:
            ids = (
                frame.loc[done, "game_id"]
                .dropna()
                .astype(str)
                .str.strip()
                .tolist()
            )
            completed.update(value for value in ids if value)
        else:
            completed.update(
                f"{path.name}:{index}"
                for index in frame.index[done].tolist()
            )

    return len(completed), inspection_uncertain, len(paths)


def coerce_pbp_numeric(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    for column in NUMERIC_PBP_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    return result


def validate_pbp_input(
    pbp: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    if pbp.empty:
        raise QBStatsError("Nonempty PBP validation received an empty dataframe")

    missing = [
        column
        for column in PBP_REQUIRED_COLUMNS
        if column not in pbp.columns
    ]
    if missing:
        raise QBStatsError(
            "PBP input is missing required QB-stat columns: "
            + ",".join(missing)
        )

    df = coerce_pbp_numeric(pbp)

    season_values = pd.to_numeric(df["season"], errors="coerce")
    invalid_season = season_values.isna() | season_values.ne(season)
    if invalid_season.any():
        examples = (
            df.loc[
                invalid_season,
                ["season", "week", "game_id", "play_id"],
            ]
            .head(5)
            .to_dict("records")
        )
        raise QBStatsError(
            f"PBP input contains rows outside requested season {season}: {examples}"
        )

    week_values = pd.to_numeric(df["week"], errors="coerce")
    invalid_week = (
        week_values.isna()
        | week_values.le(0)
        | week_values.mod(1).ne(0)
    )
    if invalid_week.any():
        examples = (
            df.loc[
                invalid_week,
                ["season", "week", "game_id", "play_id"],
            ]
            .head(5)
            .to_dict("records")
        )
        raise QBStatsError(
            f"PBP input contains invalid week values: {examples}"
        )

    blank_game_id = _blank_mask(df["game_id"])
    blank_play_id = _blank_mask(df["play_id"])

    if blank_game_id.any() or blank_play_id.any():
        examples = (
            df.loc[
                blank_game_id | blank_play_id,
                ["game_id", "play_id"],
            ]
            .head(5)
            .to_dict("records")
        )
        raise QBStatsError(
            "PBP input contains blank game_id/play_id values: "
            f"{examples}"
        )

    duplicate_mask = df.duplicated(
        subset=["game_id", "play_id"],
        keep=False,
    )
    if duplicate_mask.any():
        examples = (
            df.loc[
                duplicate_mask,
                ["game_id", "play_id"],
            ]
            .head(5)
            .to_dict("records")
        )
        raise QBStatsError(
            "PBP input contains duplicate game_id/play_id rows: "
            f"{examples}"
        )

    relevant = df[
        df["qb_dropback"].eq(1)
        | df["pass_attempt"].eq(1)
    ]

    if not relevant.empty:
        identity_blank = (
            _blank_mask(relevant["posteam"])
            | _blank_mask(relevant["passer_player_id"])
            | _blank_mask(relevant["passer_player_name"])
        )
        if identity_blank.any():
            examples = (
                relevant.loc[
                    identity_blank,
                    [
                        "game_id",
                        "play_id",
                        "posteam",
                        "passer_player_id",
                        "passer_player_name",
                    ],
                ]
                .head(5)
                .to_dict("records")
            )
            raise QBStatsError(
                "QB dropback/pass-attempt rows contain blank grouping identity: "
                f"{examples}"
            )

    for flag_column in [
        "qb_dropback",
        "pass_attempt",
        "sack",
        "interception",
    ]:
        values = df[flag_column].dropna()
        invalid = ~values.isin([0, 1])
        if invalid.any():
            examples = values.loc[invalid].head(5).tolist()
            raise QBStatsError(
                f"PBP column {flag_column} contains values outside 0/1: {examples}"
            )

    return df


def build_qb_stats(
    pbp: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    dropback_df = pbp[pbp["qb_dropback"].eq(1)].copy()
    pass_df = pbp[pbp["pass_attempt"].eq(1)].copy()

    dropback_rows = int(len(dropback_df))
    pass_attempt_rows = int(len(pass_df))

    if dropback_df.empty:
        raise QBStatsError(
            "Nonempty PBP input produced zero QB dropback rows"
        )

    dropback_agg = (
        dropback_df.groupby(GROUP_COLS, dropna=False)
        .agg(
            dropbacks=("qb_dropback", "size"),
            epa_per_play=("qb_epa", "mean"),
            sacks=("sack", "sum"),
            interceptions=("interception", "sum"),
        )
        .reset_index()
    )

    fumble_rows = dropback_df[
        dropback_df["fumbled_1_player_id"].astype("string").str.strip()
        == dropback_df["passer_player_id"].astype("string").str.strip()
    ]

    fumble_agg = (
        fumble_rows.groupby(GROUP_COLS, dropna=False)
        .size()
        .reset_index(name="fumbles")
    )

    pass_agg = (
        pass_df.groupby(GROUP_COLS, dropna=False)
        .agg(
            cpoe=("cpoe", "mean"),
            air_yards=("air_yards", "mean"),
        )
        .reset_index()
    )

    merged = dropback_agg.merge(
        fumble_agg,
        on=GROUP_COLS,
        how="left",
    )
    merged = merged.merge(
        pass_agg,
        on=GROUP_COLS,
        how="left",
    )

    merged["fumbles"] = merged["fumbles"].fillna(0)

    merged["sack_rate"] = merged["sacks"] / merged["dropbacks"]
    merged["interception_rate"] = (
        merged["interceptions"] / merged["dropbacks"]
    )
    merged["fumble_rate"] = merged["fumbles"] / merged["dropbacks"]

    merged["sport"] = "football"
    merged["league"] = "nfl"
    merged["starts"] = ""
    merged["adjusted_completion_pct"] = ""
    merged["pressure_to_sack_rate"] = ""
    merged["turnover_worthy_play_rate"] = ""

    merged = merged.rename(
        columns={
            "posteam": "team",
            "passer_player_id": "player_id",
            "passer_player_name": "qb_name",
        }
    )

    result = merged[OUTPUT_HEADERS].copy()
    result = result.sort_values(
        ["season", "week", "team", "player_id"],
        kind="stable",
    ).reset_index(drop=True)

    if result.empty:
        raise QBStatsError(
            "Nonempty PBP input produced zero QB-stat output rows"
        )

    return result, dropback_rows, pass_attempt_rows


def _require_finite_numeric(
    df: pd.DataFrame,
    column: str,
    *,
    allow_blank: bool,
) -> pd.Series:
    raw = df[column]
    numeric = pd.to_numeric(raw, errors="coerce")
    blank = _blank_mask(raw)

    invalid_parse = numeric.isna() & ~blank
    if invalid_parse.any():
        examples = raw.loc[invalid_parse].head(5).tolist()
        raise QBStatsError(
            f"QB output column {column} contains nonnumeric values: {examples}"
        )

    finite = numeric.dropna().map(math.isfinite)
    if not finite.all():
        raise QBStatsError(
            f"QB output column {column} contains non-finite values"
        )

    if not allow_blank and numeric.isna().any():
        raise QBStatsError(
            f"QB output column {column} contains blank values"
        )

    return numeric


def validate_qb_candidate(
    df: pd.DataFrame,
    season: int,
    *,
    allow_empty: bool,
) -> None:
    if list(df.columns) != OUTPUT_HEADERS:
        raise QBStatsError(
            "QB output columns do not exactly match the required schema"
        )

    if df.empty:
        if allow_empty:
            return
        raise QBStatsError("QB output unexpectedly contains zero rows")

    for column in [
        "season",
        "week",
        "team",
        "player_id",
        "qb_name",
    ]:
        if _blank_mask(df[column]).any():
            raise QBStatsError(
                f"QB output contains blank key values in {column}"
            )

    sport_values = (
        df["sport"].astype(str).str.strip().str.casefold()
    )
    league_values = (
        df["league"].astype(str).str.strip().str.casefold()
    )

    if not sport_values.eq("football").all():
        raise QBStatsError("QB output contains sport values other than football")
    if not league_values.eq("nfl").all():
        raise QBStatsError("QB output contains league values other than nfl")

    seasons = pd.to_numeric(df["season"], errors="coerce")
    if seasons.isna().any() or not seasons.eq(season).all():
        raise QBStatsError(
            f"QB output contains rows outside requested season {season}"
        )

    weeks = pd.to_numeric(df["week"], errors="coerce")
    invalid_week = (
        weeks.isna()
        | weeks.le(0)
        | weeks.mod(1).ne(0)
    )
    if invalid_week.any():
        raise QBStatsError("QB output contains invalid week values")

    duplicate_mask = df.duplicated(
        subset=["season", "week", "team", "player_id"],
        keep=False,
    )
    if duplicate_mask.any():
        examples = (
            df.loc[
                duplicate_mask,
                ["season", "week", "team", "player_id"],
            ]
            .head(5)
            .to_dict("records")
        )
        raise QBStatsError(
            "QB output contains duplicate season/week/team/player_id rows: "
            f"{examples}"
        )

    dropbacks = _require_finite_numeric(
        df,
        "dropbacks",
        allow_blank=False,
    )
    invalid_dropbacks = (
        dropbacks.le(0)
        | dropbacks.mod(1).ne(0)
    )
    if invalid_dropbacks.any():
        examples = dropbacks.loc[invalid_dropbacks].head(5).tolist()
        raise QBStatsError(
            f"QB output contains invalid dropbacks values: {examples}"
        )

    for column in NUMERIC_OUTPUT_COLUMNS:
        if column == "dropbacks":
            continue
        _require_finite_numeric(
            df,
            column,
            allow_blank=True,
        )

    for column in RATE_COLUMNS:
        rates = pd.to_numeric(df[column], errors="coerce").dropna()
        invalid = rates.lt(0) | rates.gt(1)
        if invalid.any():
            examples = rates.loc[invalid].head(5).tolist()
            raise QBStatsError(
                f"QB output rate {column} is outside [0, 1]: {examples}"
            )

    for column in INTENTIONALLY_BLANK_COLUMNS:
        if (~_blank_mask(df[column])).any():
            raise QBStatsError(
                f"QB output column {column} must remain blank"
            )


def write_candidate_file(
    df: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    try:
        df.to_csv(
            temporary_path,
            index=False,
        )

        with temporary_path.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())

        return temporary_path
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def verify_candidate_file(
    temporary_path: Path,
    expected: pd.DataFrame,
    season: int,
    *,
    allow_empty: bool,
) -> None:
    try:
        reloaded = pd.read_csv(
            temporary_path,
            low_memory=False,
        )
    except pd.errors.EmptyDataError:
        reloaded = pd.DataFrame(columns=OUTPUT_HEADERS)

    if len(reloaded) != len(expected):
        raise QBStatsError(
            "Staged QB-stat row count changed during CSV round trip: "
            f"expected={len(expected)} actual={len(reloaded)}"
        )

    if list(reloaded.columns) != list(expected.columns):
        raise QBStatsError(
            "Staged QB-stat columns changed during CSV round trip"
        )

    validate_qb_candidate(
        reloaded,
        season,
        allow_empty=allow_empty,
    )


def publish_candidate(
    temporary_path: Path,
    output_path: Path,
) -> None:
    try:
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    season = int(args.season)

    pbp_path = PBP_DIR / f"{season}_pbp.csv.gz"
    output_path = OUTPUT_DIR / f"{season}_qb_stats.csv"

    try:
        with PipelineReporter(
            script=SCRIPT_PATH,
            stage="00_intake",
            report_root=REPORT_ROOT,
            pipeline="NFL",
            league="NFL",
            season=season,
            extra_context={
                "component": "quarterback statistics",
            },
        ) as reporter:
            reporter.add_input(pbp_path)

            pbp_rows = 0
            dropback_rows = 0
            pass_attempt_rows = 0
            output_rows = 0
            existing_rows: int | None = None
            existing_readable = True
            completed_games = 0
            result_files_checked = 0
            completed_game_inspection_uncertain = False
            empty_candidate_allowed = False
            staged_roundtrip_verified = False
            publication_completed = False
            weeks_represented: list[int] = []
            teams_represented: list[str] = []
            temporary_path: Path | None = None

            existing, existing_readable = read_existing_output(
                output_path,
                reporter,
            )
            if existing is not None:
                existing_rows = int(len(existing))

            try:
                pbp = read_pbp(pbp_path)
                pbp_rows = int(len(pbp))

                if pbp.empty:
                    existing_has_real_rows = (
                        existing is not None and not existing.empty
                    )
                    unreadable_existing_may_contain_data = (
                        output_path.exists()
                        and not existing_readable
                        and output_path.stat().st_size > 0
                    )

                    (
                        completed_games,
                        completed_game_inspection_uncertain,
                        result_files_checked,
                    ) = count_completed_games(
                        season,
                        reporter,
                    )

                    if (
                        existing_has_real_rows
                        or unreadable_existing_may_contain_data
                        or completed_games > 0
                        or completed_game_inspection_uncertain
                    ):
                        raise QBStatsError(
                            "PBP input contains zero rows after QB statistics may "
                            "already be required or available; existing canonical "
                            "QB statistics were preserved"
                        )

                    candidate = pd.DataFrame(
                        columns=OUTPUT_HEADERS
                    )
                    empty_candidate_allowed = True
                    reporter.warning(
                        "PBP contains zero rows before completed games and before "
                        "any readable nonempty canonical QB output; publishing "
                        "empty pre-game QB snapshot",
                        season=season,
                    )
                else:
                    validated_pbp = validate_pbp_input(
                        pbp,
                        season,
                    )
                    (
                        candidate,
                        dropback_rows,
                        pass_attempt_rows,
                    ) = build_qb_stats(validated_pbp)

                    validate_qb_candidate(
                        candidate,
                        season,
                        allow_empty=False,
                    )

                    output_rows = int(len(candidate))
                    weeks_represented = sorted(
                        {
                            int(value)
                            for value in pd.to_numeric(
                                candidate["week"],
                                errors="coerce",
                            ).dropna().tolist()
                        }
                    )
                    teams_represented = sorted(
                        {
                            str(value).strip()
                            for value in candidate["team"].tolist()
                            if str(value).strip()
                        }
                    )

                temporary_path = write_candidate_file(
                    candidate,
                    output_path,
                )

                verify_candidate_file(
                    temporary_path,
                    candidate,
                    season,
                    allow_empty=empty_candidate_allowed,
                )
                staged_roundtrip_verified = True

                publish_candidate(
                    temporary_path,
                    output_path,
                )
                temporary_path = None
                publication_completed = True
                reporter.add_output(output_path)

                print(
                    f"Wrote {len(candidate)} rows to {output_path}"
                )

            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

                reporter.set_rows(
                    rows_in=pbp_rows,
                    rows_out=(
                        output_rows
                        if publication_completed
                        else 0
                    ),
                )
                reporter.update_details(
                    {
                        "pbp_rows": pbp_rows,
                        "dropback_rows": dropback_rows,
                        "pass_attempt_rows": pass_attempt_rows,
                        "output_rows": output_rows,
                        "existing_rows": existing_rows,
                        "existing_readable": existing_readable,
                        "completed_games_detected_for_empty_check": completed_games,
                        "result_files_checked_for_empty_safety": result_files_checked,
                        "completed_game_inspection_uncertain": completed_game_inspection_uncertain,
                        "empty_candidate_allowed": empty_candidate_allowed,
                        "staged_roundtrip_verified": staged_roundtrip_verified,
                        "publication_completed": publication_completed,
                        "weeks_represented": weeks_represented,
                        "teams_represented": teams_represented,
                        "team_count": len(teams_represented),
                    }
                )

        return 0

    except Exception as exc:
        print("nfl pull_qb_stats failed", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
