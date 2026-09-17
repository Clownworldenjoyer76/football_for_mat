#!/usr/bin/env python3
# docs/win/football/nfl/scripts/00_intake/pull_pbp.py
#
# Pulls nflverse play-by-play data for one NFL season.
#
# Output:
#   docs/win/football/nfl/00_intake/pbp/{season}_pbp.csv.gz
#
# Standard report:
#   docs/win/football/nfl/errors/00_intake/pull_pbp.json
#
# Legacy log:
#   docs/win/football/nfl/errors/00_intake/pull_pbp.txt

from __future__ import annotations

import argparse
import gzip
import os
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter


PBP_DIR = NFL_ROOT / "00_intake" / "pbp"
RESULTS_DIR = NFL_ROOT / "04_final_results" / "results"
REPORT_ROOT = NFL_ROOT / "errors"
LEGACY_LOG_FILE = REPORT_ROOT / "00_intake" / "pull_pbp.txt"


FEATURE_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "old_game_id",
    "play_id",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "side_of_field",
    "yardline_100",
    "game_date",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "qtr",
    "down",
    "ydstogo",
    "ydsnet",
    "desc",
    "play_type",
    "yards_gained",
    "epa",
    "success",
    "wp",
    "wpa",
    "cp",
    "cpoe",
    "qb_epa",
    "pass",
    "rush",
    "sack",
    "interception",
    "fumble",
    "fumble_lost",
    "turnover",
    "touchdown",
    "pass_touchdown",
    "rush_touchdown",
    "first_down",
    "third_down_converted",
    "third_down_failed",
    "fourth_down_converted",
    "fourth_down_failed",
    "series",
    "series_success",
    "drive",
    "fixed_drive",
    "fixed_drive_result",
    "drive_real_start_time",
    "drive_play_count",
    "drive_time_of_possession",
    "drive_first_downs",
    "drive_inside20",
    "drive_ended_with_score",
    "drive_quarter_start",
    "drive_quarter_end",
    "drive_yards_penalized",
    "posteam_score",
    "defteam_score",
    "score_differential",
    "total_home_score",
    "total_away_score",
    "passer_player_id",
    "passer_player_name",
    "rusher_player_id",
    "rusher_player_name",
    "receiver_player_id",
    "receiver_player_name",
]

ADDITIONAL_DOWNSTREAM_COLUMNS = [
    "posteam_score_post",
    "qb_dropback",
    "pass_attempt",
    "air_yards",
    "fumbled_1_player_id",
]

REQUIRED_NONEMPTY_COLUMNS = list(
    dict.fromkeys(FEATURE_COLUMNS + ADDITIONAL_DOWNSTREAM_COLUMNS)
)


class PBPError(RuntimeError):
    pass


class RunLog:
    def __init__(self, reporter: PipelineReporter) -> None:
        self.reporter = reporter
        self.lines: list[str] = []

    @staticmethod
    def _stamp() -> str:
        return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    def info(self, message: str) -> None:
        line = f"[{self._stamp()}] {str(message).rstrip()}"
        self.lines.append(line)
        print(line)

    def warning(self, message: str, **details: Any) -> None:
        text = str(message).rstrip()
        line = f"[{self._stamp()}] WARNING: {text}"
        self.lines.append(line)
        self.reporter.warning(text, **details)
        print(line, file=sys.stderr)

    def error_text(self, message: str) -> None:
        line = f"[{self._stamp()}] ERROR: {str(message).rstrip()}"
        self.lines.append(line)
        print(line, file=sys.stderr)

    def write_legacy(self) -> None:
        LEGACY_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = LEGACY_LOG_FILE.with_name(
            f".{LEGACY_LOG_FILE.name}.tmp"
        )
        payload = "\n".join(self.lines).rstrip() + "\n"

        try:
            with temporary_path.open("w", encoding="utf-8", newline="") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temporary_path, LEGACY_LOG_FILE)
        finally:
            temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull nflverse play-by-play data for one NFL season."
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="NFL season to pull.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "nflreadpy", "nfl_data_py"],
        default="auto",
        help="Data loader to use. Default: auto.",
    )
    args = parser.parse_args()

    if args.season < 2000 or args.season > 2100:
        parser.error("--season must be between 2000 and 2100")

    return args


def _to_pandas(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()

    if hasattr(data, "to_pandas"):
        return data.to_pandas()

    return pd.DataFrame(data)


def load_with_nflreadpy(season: int) -> pd.DataFrame:
    import nflreadpy as nfl

    try:
        data = nfl.load_pbp(seasons=[season])
    except TypeError:
        data = nfl.load_pbp([season])

    return _to_pandas(data)


def load_with_nfl_data_py(season: int) -> pd.DataFrame:
    import nfl_data_py as nfl

    try:
        data = nfl.import_pbp_data(
            [season],
            columns=None,
            downcast=True,
            cache=False,
        )
    except TypeError:
        data = nfl.import_pbp_data(
            [season],
            downcast=True,
            cache=False,
        )

    return _to_pandas(data)


def load_pbp(
    season: int,
    source: str,
    log: RunLog,
) -> tuple[pd.DataFrame, str, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    loaders: list[tuple[str, Callable[[int], pd.DataFrame]]] = []

    if source in {"auto", "nflreadpy"}:
        loaders.append(("nflreadpy", load_with_nflreadpy))
    if source in {"auto", "nfl_data_py"}:
        loaders.append(("nfl_data_py", load_with_nfl_data_py))

    last_empty: tuple[pd.DataFrame, str] | None = None
    failures: list[str] = []

    for index, (source_name, loader) in enumerate(loaders):
        try:
            frame = loader(season)
        except Exception as exc:
            failure = {
                "source": source_name,
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            attempts.append(failure)
            failures.append(
                f"{source_name} failed: {type(exc).__name__}: {exc}"
            )

            if source == "auto":
                log.warning(
                    f"{source_name} PBP load failed; trying fallback",
                    source=source_name,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                continue

            raise PBPError(failures[-1]) from exc

        attempt = {
            "source": source_name,
            "status": "SUCCESS" if not frame.empty else "EMPTY",
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
        }
        attempts.append(attempt)

        if not frame.empty:
            return frame, source_name, attempts

        last_empty = (frame, source_name)

        if source == "auto" and index < len(loaders) - 1:
            log.warning(
                f"{source_name} returned zero PBP rows; trying fallback",
                source=source_name,
            )
            continue

        return frame, source_name, attempts

    if last_empty is not None:
        frame, source_name = last_empty
        return frame, source_name, attempts

    if failures:
        raise PBPError("; ".join(failures))

    raise PBPError("No PBP source attempted.")


def add_derived_spec_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    df = df.copy()

    if "turnover" not in df.columns:
        if {"interception", "fumble_lost"}.issubset(df.columns):
            interception = pd.to_numeric(
                df["interception"], errors="coerce"
            ).fillna(0)
            fumble_lost = pd.to_numeric(
                df["fumble_lost"], errors="coerce"
            ).fillna(0)
            df["turnover"] = (
                (interception == 1) | (fumble_lost == 1)
            ).astype(int)

    return df


def clean_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    df = add_derived_spec_columns(df)

    if df.empty:
        return df

    sort_cols = [
        column
        for column in ["game_id", "play_id"]
        if column in df.columns
    ]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="stable")

    ordered = [
        column
        for column in FEATURE_COLUMNS
        if column in df.columns
    ]
    remaining = [
        column
        for column in df.columns
        if column not in ordered
    ]

    return df[ordered + remaining]


def _blank_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().eq("")


def validate_nonempty_candidate(df: pd.DataFrame, season: int) -> None:
    if df.empty:
        raise PBPError("Nonempty PBP validation received an empty dataframe")

    missing = [
        column
        for column in REQUIRED_NONEMPTY_COLUMNS
        if column not in df.columns
    ]
    if missing:
        raise PBPError(
            "PBP candidate is missing required downstream columns: "
            + ",".join(missing)
        )

    season_values = pd.to_numeric(df["season"], errors="coerce")
    bad_season = season_values.isna() | season_values.ne(season)
    if bad_season.any():
        examples = (
            df.loc[bad_season, ["season", "game_id", "play_id"]]
            .head(5)
            .to_dict("records")
        )
        raise PBPError(
            f"PBP candidate contains rows outside requested season {season}: "
            f"{examples}"
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
        raise PBPError(
            "PBP candidate contains blank game_id/play_id values: "
            f"{examples}"
        )

    duplicate_mask = df.duplicated(
        subset=["game_id", "play_id"],
        keep=False,
    )
    if duplicate_mask.any():
        examples = (
            df.loc[duplicate_mask, ["game_id", "play_id"]]
            .head(5)
            .to_dict("records")
        )
        raise PBPError(
            "PBP candidate contains duplicate game_id/play_id rows: "
            f"{examples}"
        )


def game_ids(df: pd.DataFrame) -> set[str]:
    if df.empty or "game_id" not in df.columns:
        return set()

    return {
        value
        for value in df["game_id"].dropna().astype(str).str.strip().tolist()
        if value
    }


def read_existing_pbp(
    path: Path,
    log: RunLog,
) -> tuple[pd.DataFrame | None, bool]:
    if not path.exists():
        return None, True

    try:
        frame = pd.read_csv(
            path,
            compression="gzip",
            low_memory=False,
        )
        return frame, True
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), True
    except Exception as exc:
        log.warning(
            "Existing PBP file could not be read for regression comparison; "
            "a valid nonempty candidate may replace it",
            path=str(path),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return None, False


def count_completed_games(season: int, log: RunLog) -> tuple[int, bool, int]:
    paths = sorted(RESULTS_DIR.glob(f"{season}_*.csv"))
    completed: set[str] = set()
    inspection_uncertain = False

    for path in paths:
        try:
            frame = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            inspection_uncertain = True
            log.warning(
                "Could not inspect final-results file while evaluating empty PBP safety",
                path=str(path),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            continue

        if "status" not in frame.columns:
            inspection_uncertain = True
            log.warning(
                "Final-results file lacks status column while evaluating empty PBP safety",
                path=str(path),
            )
            continue

        statuses = frame["status"].fillna("").astype(str).str.strip().str.casefold()
        done = statuses.str.contains("final", regex=False) | statuses.str.contains(
            "completed", regex=False
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


def validate_regression(
    existing: pd.DataFrame | None,
    candidate: pd.DataFrame,
) -> tuple[int, int]:
    existing_games = game_ids(existing) if existing is not None else set()
    candidate_games = game_ids(candidate)

    missing_games = sorted(existing_games - candidate_games)
    if missing_games:
        raise PBPError(
            "PBP refresh would drop previously available games. "
            f"Missing count={len(missing_games)} examples={missing_games[:5]}"
        )

    return len(existing_games), len(candidate_games)


def write_candidate_file(df: pd.DataFrame, output_file: Path) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_file.stem}.",
        suffix=".tmp.gz",
        dir=output_file.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    try:
        df.to_csv(
            temporary_path,
            index=False,
            compression="gzip",
        )

        with temporary_path.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())

        return temporary_path
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def verify_candidate_file(
    path: Path,
    expected: pd.DataFrame,
    season: int,
) -> None:
    with gzip.open(path, "rb") as handle:
        handle.read(1)

    try:
        reloaded = pd.read_csv(
            path,
            compression="gzip",
            low_memory=False,
        )
    except pd.errors.EmptyDataError:
        reloaded = pd.DataFrame()

    if len(reloaded) != len(expected):
        raise PBPError(
            "Staged PBP row count changed during gzip round trip: "
            f"expected={len(expected)} actual={len(reloaded)}"
        )

    if list(reloaded.columns) != list(expected.columns):
        raise PBPError(
            "Staged PBP columns changed during gzip round trip"
        )

    if not reloaded.empty:
        validate_nonempty_candidate(reloaded, season)


def publish_candidate(temporary_path: Path, output_file: Path) -> None:
    try:
        os.replace(temporary_path, output_file)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    season = int(args.season)
    output_file = PBP_DIR / f"{season}_pbp.csv.gz"

    try:
        with PipelineReporter(
            script=SCRIPT_PATH,
            stage="00_intake",
            report_root=REPORT_ROOT,
            pipeline="NFL",
            league="NFL",
            season=season,
            extra_context={
                "component": "nflverse play-by-play intake",
                "source_requested": args.source,
            },
        ) as reporter:
            log = RunLog(reporter)
            source_attempts: list[dict[str, Any]] = []
            used_source: str | None = None
            candidate_rows = 0
            candidate_columns = 0
            existing_rows: int | None = None
            existing_games = 0
            candidate_games = 0
            existing_readable = True
            completed_games = 0
            result_files_checked = 0
            completed_game_inspection_uncertain = False
            empty_candidate_allowed = False
            publication_completed = False
            temporary_path: Path | None = None

            if output_file.exists():
                reporter.add_input(output_file)

            try:
                log.info("=" * 80)
                log.info(
                    f"pull_pbp.py started | season={season} | source={args.source}"
                )

                existing, existing_readable = read_existing_pbp(
                    output_file,
                    log,
                )
                if existing is not None:
                    existing_rows = int(len(existing))

                frame, used_source, source_attempts = load_pbp(
                    season=season,
                    source=args.source,
                    log=log,
                )
                frame = clean_for_csv(frame)
                candidate_rows = int(len(frame))
                candidate_columns = int(len(frame.columns))

                if frame.empty:
                    existing_has_real_rows = (
                        existing is not None and not existing.empty
                    )
                    unreadable_existing_may_contain_data = (
                        output_file.exists()
                        and not existing_readable
                        and output_file.stat().st_size > 0
                    )

                    (
                        completed_games,
                        completed_game_inspection_uncertain,
                        result_files_checked,
                    ) = count_completed_games(season, log)

                    if (
                        existing_has_real_rows
                        or unreadable_existing_may_contain_data
                        or completed_games > 0
                        or completed_game_inspection_uncertain
                    ):
                        raise PBPError(
                            "PBP source returned zero rows after PBP data may already be "
                            "required or available; existing canonical PBP was preserved"
                        )

                    empty_candidate_allowed = True
                    log.warning(
                        "PBP source returned zero rows before completed games and before "
                        "any readable nonempty canonical PBP; publishing empty pre-game snapshot",
                        season=season,
                    )
                else:
                    validate_nonempty_candidate(frame, season)
                    existing_games, candidate_games = validate_regression(
                        existing,
                        frame,
                    )

                temporary_path = write_candidate_file(
                    frame,
                    output_file,
                )
                verify_candidate_file(
                    temporary_path,
                    frame,
                    season,
                )
                publish_candidate(temporary_path, output_file)
                temporary_path = None
                publication_completed = True
                reporter.add_output(output_file)

                log.info(f"source_used={used_source}")
                log.info(
                    f"rows={candidate_rows} columns={candidate_columns}"
                )
                log.info(f"output={output_file}")
                log.info("missing_required_columns=none")
                log.info("publication_completed=true")
                log.info("pull_pbp.py completed")

                print("nfl pull_pbp completed")
                print(f"season: {season}")
                print(f"source_used: {used_source}")
                print(f"rows: {candidate_rows}")
                print(f"columns: {candidate_columns}")
                print(f"output: {output_file}")
                print("missing_required_columns: none")

            except Exception as exc:
                log.error_text(f"{type(exc).__name__}: {exc}")
                for line in traceback.format_exc().rstrip().splitlines():
                    log.error_text(line)
                raise
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

                reporter.set_rows(
                    rows_in=candidate_rows,
                    rows_out=candidate_rows if publication_completed else 0,
                )
                reporter.update_details(
                    {
                        "source_used": used_source,
                        "source_attempts": source_attempts,
                        "candidate_rows": candidate_rows,
                        "candidate_columns": candidate_columns,
                        "existing_rows": existing_rows,
                        "existing_readable": existing_readable,
                        "existing_game_count": existing_games,
                        "candidate_game_count": candidate_games,
                        "completed_games_detected_for_empty_check": completed_games,
                        "result_files_checked_for_empty_safety": result_files_checked,
                        "completed_game_inspection_uncertain": completed_game_inspection_uncertain,
                        "empty_candidate_allowed": empty_candidate_allowed,
                        "publication_completed": publication_completed,
                        "required_nonempty_column_count": len(REQUIRED_NONEMPTY_COLUMNS),
                    }
                )

                log.write_legacy()
                reporter.add_output(LEGACY_LOG_FILE)

        return 0

    except Exception as exc:
        print("nfl pull_pbp failed", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
