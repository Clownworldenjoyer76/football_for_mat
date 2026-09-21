#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import shutil
import sys
import tempfile
import urllib.request
import uuid
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]
REPORT_ROOT = NFL_ROOT / "errors"
HISTORIC_ROOT = NFL_ROOT / "data" / "historic_data"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter

NFLVERSE_BASE = (
    "https://github.com/nflverse/nflverse-data/releases/download"
)
USER_AGENT = "football_for_mat/1.0"
DOWNLOAD_TIMEOUT_SECONDS = 90

SNAP_REQUIRED_COLUMNS = {
    "season",
    "team",
    "week",
    "offense_pct",
    "defense_pct",
}
SNAP_ID_COLUMNS = (
    "pfr_player_id",
    "pfr_id",
    "player",
    "full_name",
    "player_name",
)

DEPTH_TEAM_COLUMNS = (
    "club_code",
    "team",
)
DEPTH_WEEK_RANK_COLUMNS = (
    "depth_team",
    "pos_rank",
)
DEPTH_TIMESTAMP_RANK_COLUMNS = (
    "pos_rank",
    "depth_team",
)
DEPTH_SLOT_COLUMNS = (
    "depth_position",
    "pos_slot",
    "position",
)
DEPTH_TIMESTAMP_COLUMNS = (
    "dt",
    "timestamp",
    "date_modified",
    "date",
)
DEPTH_ID_COLUMNS = (
    "gsis_id",
    "espn_id",
    "full_name",
    "player_name",
    "display_name",
)

pd = None


class ProjectionSourceRefreshError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise ProjectionSourceRefreshError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh required current-season nflverse sources "
            "for the NFL in-season projection."
        )
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
    )
    args = parser.parse_args()

    if args.season < 2000 or args.season > 2100:
        parser.error(
            "--season must be between 2000 and 2100"
        )

    return args


def import_pandas() -> None:
    global pd

    if pd is not None:
        return

    try:
        import pandas as pandas_module
    except ImportError as exc:
        fail(
            "This script requires pandas and a parquet engine. "
            "Install docs/win/football/nfl/requirements.txt."
        )

    pd = pandas_module


def clean(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd is not None and pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if text.casefold() in {
        "",
        "nan",
        "none",
        "null",
        "<na>",
        "nat",
    }:
        return ""

    return text


def first_present(
    columns: set[str],
    candidates: tuple[str, ...],
    *,
    label: str,
) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate

    fail(
        f"{label}: none of the required columns are present: "
        f"{list(candidates)}"
    )


def require_columns(
    df,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        fail(
            f"{label}: missing required columns: {missing}"
        )


def require_nonblank_series(
    series,
    *,
    label: str,
) -> None:
    blank = series.map(clean).eq("")

    if blank.any():
        examples = (
            series.index[blank]
            .tolist()[:10]
        )
        fail(
            f"{label}: blank values found "
            f"at row indexes {examples}"
        )


def numeric_series(
    series,
    *,
    label: str,
):
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    array = values.to_numpy(
        dtype=float,
        na_value=float("nan"),
    )

    if not all(
        math.isfinite(value)
        for value in array
    ):
        bad = (
            series.index[
                ~values.notna()
            ]
            .tolist()[:10]
        )
        fail(
            f"{label}: non-numeric/non-finite values "
            f"found at row indexes {bad}"
        )

    return values.astype(float)


def integer_series(
    series,
    *,
    label: str,
):
    values = numeric_series(
        series,
        label=label,
    )

    rounded = values.round()

    mismatch = (
        (values - rounded)
        .abs()
        .gt(1e-9)
    )

    if mismatch.any():
        examples = (
            series.loc[mismatch]
            .astype(str)
            .head(10)
            .tolist()
        )
        fail(
            f"{label}: non-integer values found: "
            f"{examples}"
        )

    return rounded.astype(int)


def identity_mask(
    df,
    *,
    candidates: tuple[str, ...],
):
    available = [
        column
        for column in candidates
        if column in df.columns
    ]

    if not available:
        fail(
            "No usable player-identity column found; "
            f"expected one of {list(candidates)}"
        )

    usable = None

    for column in available:
        current = (
            df[column]
            .map(clean)
            .ne("")
        )

        usable = (
            current
            if usable is None
            else usable | current
        )

    return usable, available


def validate_snaps(
    df,
    *,
    season: int,
) -> dict[str, Any]:
    if df.empty:
        fail(
            "snap counts: downloaded source is empty"
        )

    require_columns(
        df,
        SNAP_REQUIRED_COLUMNS,
        "snap counts",
    )

    season_values = integer_series(
        df["season"],
        label="snap counts season",
    )

    if not season_values.eq(season).all():
        actual = sorted(
            set(
                season_values
                .astype(int)
                .tolist()
            )
        )
        fail(
            "snap counts: season values do not match "
            f"requested season={season}; actual={actual}"
        )

    require_nonblank_series(
        df["team"],
        label="snap counts team",
    )

    weeks = integer_series(
        df["week"],
        label="snap counts week",
    )

    if (
        weeks.lt(1).any()
        or weeks.gt(25).any()
    ):
        actual_min = int(weeks.min())
        actual_max = int(weeks.max())
        fail(
            "snap counts: week values outside 1..25; "
            f"min={actual_min} max={actual_max}"
        )

    for column in (
        "offense_pct",
        "defense_pct",
    ):
        raw = (
            df[column]
            .map(clean)
            .str.replace(
                "%",
                "",
                regex=False,
            )
        )

        values = numeric_series(
            raw,
            label=f"snap counts {column}",
        )

        if (
            values.lt(0).any()
            or values.gt(100).any()
        ):
            fail(
                f"snap counts {column}: values "
                "must be within 0..100"
            )

    usable_identity, identity_columns = (
        identity_mask(
            df,
            candidates=SNAP_ID_COLUMNS,
        )
    )

    if not usable_identity.all():
        bad = (
            df.index[
                ~usable_identity
            ]
            .tolist()[:10]
        )
        fail(
            "snap counts: rows without a usable "
            f"player identity at indexes {bad}"
        )

    teams = sorted(
        {
            clean(value)
            for value in df["team"]
            if clean(value)
        }
    )

    if len(teams) < 2:
        fail(
            "snap counts: fewer than two teams "
            "were present after validation"
        )

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "min_week": int(weeks.min()),
        "max_week": int(weeks.max()),
        "team_count": len(teams),
        "identity_columns": identity_columns,
    }


def validate_depth(
    df,
    *,
    season: int,
) -> dict[str, Any]:
    if df.empty:
        fail(
            "depth charts: downloaded source is empty"
        )

    columns = set(df.columns)

    week_numeric = (
        pd.to_numeric(
            df["week"],
            errors="coerce",
        )
        if "week" in columns
        else None
    )

    has_week = bool(
        week_numeric is not None
        and week_numeric.notna().any()
    )

    team_column = first_present(
        columns,
        DEPTH_TEAM_COLUMNS,
        label="depth charts team",
    )

    require_nonblank_series(
        df[team_column],
        label="depth charts team",
    )

    slot_column = first_present(
        columns,
        DEPTH_SLOT_COLUMNS,
        label="depth charts slot",
    )

    require_nonblank_series(
        df[slot_column],
        label="depth charts slot",
    )

    usable_identity, identity_columns = (
        identity_mask(
            df,
            candidates=DEPTH_ID_COLUMNS,
        )
    )

    if not usable_identity.all():
        bad = (
            df.index[
                ~usable_identity
            ]
            .tolist()[:10]
        )
        fail(
            "depth charts: rows without a usable "
            f"player identity at indexes {bad}"
        )

    if "season" in columns:
        season_values = integer_series(
            df["season"],
            label="depth charts season",
        )

        if not season_values.eq(season).all():
            actual = sorted(
                set(
                    season_values
                    .astype(int)
                    .tolist()
                )
            )
            fail(
                "depth charts: season values do not "
                f"match requested season={season}; "
                f"actual={actual}"
            )

    teams = sorted(
        {
            clean(value)
            for value in df[team_column]
            if clean(value)
        }
    )

    if len(teams) < 2:
        fail(
            "depth charts: fewer than two teams "
            "were present after validation"
        )

    if has_week:
        rank_column = first_present(
            columns,
            DEPTH_WEEK_RANK_COLUMNS,
            label="depth charts rank",
        )

        weeks = integer_series(
            df["week"],
            label="depth charts week",
        )

        if (
            weeks.lt(1).any()
            or weeks.gt(25).any()
        ):
            fail(
                "depth charts: week values outside 1..25"
            )

        ranks = integer_series(
            df[rank_column],
            label="depth charts rank",
        )

        if ranks.lt(1).any():
            fail(
                "depth charts: rank values must be >= 1"
            )

        return {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "schema_mode": "weekly",
            "team_column": team_column,
            "rank_column": rank_column,
            "slot_column": slot_column,
            "identity_columns": identity_columns,
            "team_count": len(teams),
            "min_week": int(weeks.min()),
            "max_week": int(weeks.max()),
            "min_timestamp_utc": None,
            "max_timestamp_utc": None,
        }

    timestamp_column = first_present(
        columns,
        DEPTH_TIMESTAMP_COLUMNS,
        label="depth charts timestamp",
    )
    rank_column = first_present(
        columns,
        DEPTH_TIMESTAMP_RANK_COLUMNS,
        label="depth charts rank",
    )

    timestamps = pd.to_datetime(
        df[timestamp_column],
        utc=True,
        errors="coerce",
    )

    if timestamps.isna().any():
        bad = (
            df.index[
                timestamps.isna()
            ]
            .tolist()[:10]
        )
        fail(
            "depth charts: unparseable timestamps "
            f"at row indexes {bad}"
        )

    ranks = integer_series(
        df[rank_column],
        label="depth charts rank",
    )

    if ranks.lt(1).any():
        fail(
            "depth charts: rank values must be >= 1"
        )

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "schema_mode": "timestamped",
        "team_column": team_column,
        "rank_column": rank_column,
        "slot_column": slot_column,
        "identity_columns": identity_columns,
        "team_count": len(teams),
        "min_week": None,
        "max_week": None,
        "min_timestamp_utc": (
            timestamps.min().isoformat()
        ),
        "max_timestamp_utc": (
            timestamps.max().isoformat()
        ),
    }


def source_url(
    tag: str,
    filename: str,
) -> str:
    return (
        f"{NFLVERSE_BASE}/{tag}/{filename}"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def download_parquet(
    *,
    tag: str,
    filename: str,
) -> tuple[Any, dict[str, Any]]:
    url = source_url(
        tag,
        filename,
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
        },
    )

    temp_path: Path | None = None

    try:
        with urllib.request.urlopen(
            request,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        ) as response:
            handle = tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".projection_source_download_",
                suffix=".parquet",
                delete=False,
            )

            temp_path = Path(
                handle.name
            )

            with handle:
                shutil.copyfileobj(
                    response,
                    handle,
                )

            final_url = response.geturl()
            content_length = (
                response.headers.get(
                    "Content-Length"
                )
            )

        if not temp_path.exists():
            fail(
                f"Downloaded source disappeared: {url}"
            )

        byte_count = temp_path.stat().st_size

        if byte_count <= 0:
            fail(
                f"Downloaded source is zero bytes: {url}"
            )

        checksum = file_sha256(
            temp_path
        )

        try:
            df = pd.read_parquet(
                temp_path
            )
        except Exception as exc:
            fail(
                "Downloaded source is not readable parquet: "
                f"{url} | {type(exc).__name__}: {exc}"
            )

        if df.empty:
            fail(
                f"Downloaded source is empty: {url}"
            )

        return (
            df,
            {
                "url": url,
                "final_url": final_url,
                "download_bytes": int(
                    byte_count
                ),
                "content_length_header": (
                    clean(content_length)
                ),
                "sha256": checksum,
            },
        )

    finally:
        if (
            temp_path is not None
            and temp_path.exists()
        ):
            temp_path.unlink()


def write_stage_parquet(
    df,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        df.to_parquet(
            path,
            index=False,
        )
    except Exception as exc:
        fail(
            "Could not write staged parquet "
            f"{path}: {type(exc).__name__}: {exc}"
        )

    if (
        not path.exists()
        or path.stat().st_size <= 0
    ):
        fail(
            f"Staged parquet is missing/empty: {path}"
        )


def validate_roundtrip(
    *,
    original,
    staged_path: Path,
    validator,
    season: int,
    label: str,
) -> dict[str, Any]:
    try:
        staged = pd.read_parquet(
            staged_path
        )
    except Exception as exc:
        fail(
            f"Could not reread staged {label}: "
            f"{type(exc).__name__}: {exc}"
        )

    metadata = validator(
        staged,
        season=season,
    )

    try:
        pd.testing.assert_frame_equal(
            original.reset_index(
                drop=True
            ),
            staged.reset_index(
                drop=True
            ),
            check_dtype=True,
            check_like=False,
        )
    except AssertionError as exc:
        fail(
            f"Staged {label} differs from "
            f"downloaded source after parquet roundtrip: {exc}"
        )

    return metadata


def publish_pair_with_rollback(
    *,
    snap_stage: Path,
    depth_stage: Path,
    snap_target: Path,
    depth_target: Path,
    reporter: PipelineReporter,
) -> None:
    token = uuid.uuid4().hex

    backups = {
        snap_target: (
            snap_target.parent
            / f".{snap_target.name}.backup_{token}"
        ),
        depth_target: (
            depth_target.parent
            / f".{depth_target.name}.backup_{token}"
        ),
    }

    targets = (
        (snap_stage, snap_target),
        (depth_stage, depth_target),
    )

    backed_up: list[Path] = []
    published: list[Path] = []

    try:
        for _, target in targets:
            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if target.exists():
                os.replace(
                    target,
                    backups[target],
                )
                backed_up.append(
                    target
                )

        for staged, target in targets:
            os.replace(
                staged,
                target,
            )
            published.append(
                target
            )

    except Exception:
        for target in reversed(
            published
        ):
            try:
                if target.exists():
                    target.unlink()
            except Exception:
                pass

        restore_errors = []

        for target in reversed(
            backed_up
        ):
            backup = backups[target]

            try:
                if backup.exists():
                    os.replace(
                        backup,
                        target,
                    )
            except Exception as exc:
                restore_errors.append(
                    (
                        str(target),
                        type(exc).__name__,
                        str(exc),
                    )
                )

        if restore_errors:
            fail(
                "Projection-source publication failed and "
                "rollback was incomplete: "
                f"{restore_errors}"
            )

        raise

    for target in backed_up:
        backup = backups[target]

        if backup.exists():
            try:
                backup.unlink()
            except Exception as exc:
                reporter.warning(
                    "Projection sources published but "
                    "temporary backup cleanup failed",
                    backup_path=str(
                        backup
                    ),
                    error_type=type(
                        exc
                    ).__name__,
                    error=str(exc),
                )


def run(
    reporter: PipelineReporter,
    *,
    season: int,
) -> None:
    import_pandas()

    snap_filename = (
        f"snap_counts_{season}.parquet"
    )
    depth_filename = (
        f"depth_charts_{season}.parquet"
    )

    snap_url = source_url(
        "snap_counts",
        snap_filename,
    )
    depth_url = source_url(
        "depth_charts",
        depth_filename,
    )

    reporter.add_input(
        snap_url
    )
    reporter.add_input(
        depth_url
    )

    snaps, snap_download = (
        download_parquet(
            tag="snap_counts",
            filename=snap_filename,
        )
    )

    snap_metadata = validate_snaps(
        snaps,
        season=season,
    )

    depth, depth_download = (
        download_parquet(
            tag="depth_charts",
            filename=depth_filename,
        )
    )

    depth_metadata = validate_depth(
        depth,
        season=season,
    )

    snap_target = (
        HISTORIC_ROOT
        / "snap_counts"
        / snap_filename
    )
    depth_target = (
        HISTORIC_ROOT
        / "depth_charts"
        / depth_filename
    )

    HISTORIC_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    reporter.set_rows(
        rows_in=(
            len(snaps)
            + len(depth)
        ),
        rows_out=0,
    )

    reporter.update_details(
        {
            "season": season,
            "sources": {
                "snap_counts": {
                    **snap_download,
                    **snap_metadata,
                },
                "depth_charts": {
                    **depth_download,
                    **depth_metadata,
                },
            },
            "participation_required": False,
            "publication_mode": (
                "validated_two_file_transaction_with_rollback"
            ),
            "staged_roundtrip_verified": False,
            "publication_completed": False,
        }
    )

    stage_root = Path(
        tempfile.mkdtemp(
            prefix=(
                ".projection_sources_stage_"
            ),
            dir=str(
                HISTORIC_ROOT
            ),
        )
    )

    try:
        snap_stage = (
            stage_root
            / snap_filename
        )
        depth_stage = (
            stage_root
            / depth_filename
        )

        write_stage_parquet(
            snaps,
            snap_stage,
        )
        write_stage_parquet(
            depth,
            depth_stage,
        )

        staged_snap_metadata = (
            validate_roundtrip(
                original=snaps,
                staged_path=snap_stage,
                validator=validate_snaps,
                season=season,
                label="snap counts",
            )
        )

        staged_depth_metadata = (
            validate_roundtrip(
                original=depth,
                staged_path=depth_stage,
                validator=validate_depth,
                season=season,
                label="depth charts",
            )
        )

        if (
            staged_snap_metadata
            != snap_metadata
        ):
            fail(
                "Staged snap-count validation metadata "
                "differs from downloaded-source metadata"
            )

        if (
            staged_depth_metadata
            != depth_metadata
        ):
            fail(
                "Staged depth-chart validation metadata "
                "differs from downloaded-source metadata"
            )

        reporter.set_detail(
            "staged_roundtrip_verified",
            True,
        )

        publish_pair_with_rollback(
            snap_stage=snap_stage,
            depth_stage=depth_stage,
            snap_target=snap_target,
            depth_target=depth_target,
            reporter=reporter,
        )

    finally:
        shutil.rmtree(
            stage_root,
            ignore_errors=True,
        )

    for target, validator in (
        (
            snap_target,
            validate_snaps,
        ),
        (
            depth_target,
            validate_depth,
        ),
    ):
        try:
            published = pd.read_parquet(
                target
            )
        except Exception as exc:
            fail(
                f"Published parquet cannot be reread: "
                f"{target} | {type(exc).__name__}: {exc}"
            )

        validator(
            published,
            season=season,
        )

    reporter.add_output(
        snap_target
    )
    reporter.add_output(
        depth_target
    )

    reporter.set_rows(
        rows_in=(
            len(snaps)
            + len(depth)
        ),
        rows_out=(
            len(snaps)
            + len(depth)
        ),
    )

    reporter.update_details(
        {
            "publication_completed": True,
            "published_outputs": {
                "snap_counts": str(
                    snap_target
                ),
                "depth_charts": str(
                    depth_target
                ),
            },
        }
    )

    print(
        f"WROTE {snap_target} | "
        f"rows={len(snaps)}"
    )
    print(
        f"WROTE {depth_target} | "
        f"rows={len(depth)}"
    )
    print(
        "Participation is intentionally optional for "
        "current-season projection."
    )


def main() -> int:
    args = parse_args()

    try:
        with PipelineReporter(
            script=SCRIPT_PATH,
            stage="00_intake",
            report_root=REPORT_ROOT,
            pipeline="NFL",
            league="NFL",
            season=args.season,
            extra_context={
                "component": (
                    "projection source refresh"
                ),
                "refresh_scope": (
                    "current-season snap counts "
                    "and depth charts"
                ),
            },
        ) as reporter:
            run(
                reporter,
                season=args.season,
            )

        return 0

    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
