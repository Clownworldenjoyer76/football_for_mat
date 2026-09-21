#!/usr/bin/env python3
"""
Pull the configured NFL season depth charts from ESPN Core, preserve the
legacy flattened raw_depth.csv contract consumed by depth_cleanup.py, validate
the complete 32-team result, and publish atomically.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter

DEPTHCHART_URL_TEMPLATE = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/"
    "seasons/{season}/teams/{team_id}/depthcharts"
)

OUTPUT_PATH = NFL_ROOT / "data" / "raw" / "raw_depth.csv"
TEAM_MAP_PATH = NFL_ROOT / "config" / "mapping" / "team_map.csv"
REPORT_ROOT = NFL_ROOT / "errors"

ATHLETE_FIELDS = ["id", "displayName", "shortName", "guid", "uid"]
REQUIRED_ATHLETE_FIELDS = ["id", "displayName"]
POSITION_KEY_PATTERN = re.compile(r"^[a-z0-9]+$")
ATHLETE_ID_COLUMN_PATTERN = re.compile(
    r"^depthchart\.(\d+)\.positions\.([a-z0-9]+)\.athletes\.(\d+)\.id$"
)

REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/140.0 Safari/537.36"
    ),
}

RETRYABLE_HTTP_CODES = {
    408,
    425,
    429,
    500,
    502,
    503,
    504,
}

athlete_cache: dict[str, dict[str, Any]] = {}


class DepthChartPullError(RuntimeError):
    pass


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def fail(message: str) -> None:
    raise DepthChartPullError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True, type=int)
    args = parser.parse_args()

    if not 2000 <= args.season <= 2100:
        parser.error("--season must be between 2000 and 2100")

    return args


def normalize_ref(url: Any) -> str:
    text = clean(url)

    if text.startswith("http://"):
        return "https://" + text[len("http://"):]

    return text


def fetch_json(
    url: str,
    *,
    retries: int = 4,
    timeout: int = 30,
) -> dict[str, Any]:
    url = normalize_ref(url)

    if not url:
        fail("Cannot fetch blank ESPN URL")

    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            headers=REQUEST_HEADERS,
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout,
            ) as response:
                raw = response.read().decode("utf-8")

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise DepthChartPullError(
                    f"ESPN returned invalid JSON url={url}: {exc}"
                ) from exc

            if not isinstance(payload, dict):
                fail(
                    "ESPN response must be a JSON object "
                    f"url={url}"
                )

            return payload

        except urllib.error.HTTPError as exc:
            if (
                exc.code in RETRYABLE_HTTP_CODES
                and attempt < retries
            ):
                delay = min(2 ** (attempt - 1), 8)
                print(
                    f"retry_http status={exc.code} "
                    f"attempt={attempt}/{retries} "
                    f"delay={delay}s url={url}"
                )
                time.sleep(delay)
                continue

            raise DepthChartPullError(
                f"ESPN HTTP error status={exc.code} "
                f"url={url}"
            ) from exc

        except urllib.error.URLError as exc:
            if attempt < retries:
                delay = min(2 ** (attempt - 1), 8)
                print(
                    f"retry_url_error "
                    f"attempt={attempt}/{retries} "
                    f"delay={delay}s url={url} "
                    f"reason={exc.reason}"
                )
                time.sleep(delay)
                continue

            raise DepthChartPullError(
                f"ESPN request failed url={url} "
                f"reason={exc.reason}"
            ) from exc

        except TimeoutError as exc:
            if attempt < retries:
                delay = min(2 ** (attempt - 1), 8)
                print(
                    f"retry_timeout "
                    f"attempt={attempt}/{retries} "
                    f"delay={delay}s url={url}"
                )
                time.sleep(delay)
                continue

            raise DepthChartPullError(
                f"ESPN request timed out url={url}"
            ) from exc

    fail(f"ESPN request exhausted retries url={url}")


def load_canonical_teams() -> dict[str, str]:
    if not TEAM_MAP_PATH.is_file():
        fail(f"Missing team map: {TEAM_MAP_PATH}")

    if TEAM_MAP_PATH.stat().st_size == 0:
        fail(f"Zero-byte team map: {TEAM_MAP_PATH}")

    with TEAM_MAP_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    required = {
        "sport",
        "league",
        "team_id",
        "team_abbr",
    }
    missing = sorted(required - set(fieldnames))

    if missing:
        fail(
            f"Team map missing required columns: {missing}"
        )

    by_id: dict[str, str] = {}
    by_abbr: dict[str, str] = {}

    for row in rows:
        if clean(row.get("sport")).lower() != "football":
            continue
        if clean(row.get("league")).lower() != "nfl":
            continue

        team_id = clean(row.get("team_id"))
        team_abbr = clean(row.get("team_abbr")).upper()

        if not team_id or not team_abbr:
            fail(
                "NFL team map contains blank team_id/team_abbr"
            )

        previous_abbr = by_id.get(team_id)
        if previous_abbr and previous_abbr != team_abbr:
            fail(
                "NFL team map has conflicting abbreviations "
                f"for team_id={team_id}: "
                f"{previous_abbr!r} vs {team_abbr!r}"
            )

        previous_id = by_abbr.get(team_abbr)
        if previous_id and previous_id != team_id:
            fail(
                "NFL team map has conflicting team IDs "
                f"for team_abbr={team_abbr}: "
                f"{previous_id!r} vs {team_id!r}"
            )

        by_id[team_id] = team_abbr
        by_abbr[team_abbr] = team_id

    if len(by_id) != 32 or len(by_abbr) != 32:
        fail(
            "Canonical NFL team map must contain exactly "
            f"32 unique teams; ids={len(by_id)} "
            f"abbrs={len(by_abbr)}"
        )

    return by_id



def resolve_injuries(
    injuries_field: Any,
) -> list[dict[str, str]]:
    if not isinstance(injuries_field, list):
        return []

    injuries_out = []

    for item in injuries_field:
        if not isinstance(item, dict):
            continue

        status = clean(item.get("status"))
        if status:
            injuries_out.append(
                {"status": status}
            )

    return injuries_out


def normalize_athlete(
    data: dict[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    result = {
        field: data.get(field, "")
        for field in ATHLETE_FIELDS
    }
    result["injuries"] = resolve_injuries(
        data.get("injuries", {})
    )

    missing = [
        field
        for field in REQUIRED_ATHLETE_FIELDS
        if not clean(result.get(field))
    ]

    if missing:
        fail(
            f"Athlete data missing required fields "
            f"{missing} context={context}"
        )

    return result


def resolve_athlete_ref(
    ref_url: str,
    *,
    context: str,
) -> dict[str, Any]:
    ref_url = normalize_ref(ref_url)

    if not ref_url:
        fail(
            f"Blank athlete reference context={context}"
        )

    if ref_url in athlete_cache:
        return athlete_cache[ref_url]

    try:
        data = fetch_json(
            ref_url,
            timeout=30,
        )
    except Exception as exc:
        raise DepthChartPullError(
            "Failed to resolve athlete reference "
            f"context={context} "
            f"url={ref_url}: {exc}"
        ) from exc

    result = normalize_athlete(
        data,
        context=f"{context} url={ref_url}",
    )

    athlete_cache[ref_url] = result
    return result


def rank_value(
    athlete_entry: dict[str, Any],
) -> int:
    raw_rank = athlete_entry.get("rank", 999)

    try:
        return int(raw_rank)
    except (TypeError, ValueError):
        return 999


def build_old_shape(
    core_response: dict[str, Any],
    team_id: str,
    team_abbr: str,
    season: int,
) -> tuple[dict[str, Any], int]:
    items = core_response.get("items")

    if not isinstance(items, list) or not items:
        fail(
            "Depth-chart response has no items "
            f"team_id={team_id} "
            f"team_abbr={team_abbr} "
            f"season={season}"
        )

    depthchart = []
    athlete_entry_count = 0

    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            fail(
                "Depth-chart item is not an object "
                f"team_id={team_id} "
                f"item_index={item_index}"
            )

        chart_id = clean(item.get("id"))
        chart_name = clean(item.get("name"))

        if not chart_id or not chart_name:
            fail(
                "Depth-chart item missing id/name "
                f"team_id={team_id} "
                f"item_index={item_index}"
            )

        positions = item.get("positions")
        if not isinstance(positions, dict):
            fail(
                "Depth-chart item positions is not an object "
                f"team_id={team_id} "
                f"chart_id={chart_id}"
            )

        positions_out = {}

        for pos_key, pos_val in positions.items():
            pos_key = clean(pos_key)

            if not POSITION_KEY_PATTERN.fullmatch(pos_key):
                fail(
                    "Depth-chart position key is incompatible "
                    "with depth_cleanup.py "
                    f"team_id={team_id} "
                    f"chart_id={chart_id} "
                    f"position_key={pos_key!r}"
                )

            if not isinstance(pos_val, dict):
                fail(
                    "Depth-chart position entry is not an object "
                    f"team_id={team_id} "
                    f"chart_id={chart_id} "
                    f"position_key={pos_key}"
                )

            position_info = pos_val.get("position")
            if not isinstance(position_info, dict):
                fail(
                    "Depth-chart position metadata missing "
                    f"team_id={team_id} "
                    f"chart_id={chart_id} "
                    f"position_key={pos_key}"
                )

            position_abbr = clean(
                position_info.get("abbreviation")
            )
            position_name = clean(
                position_info.get("name")
            )
            position_display = clean(
                position_info.get("displayName")
            )

            if (
                not position_abbr
                or not position_display
            ):
                fail(
                    "Depth-chart position metadata incomplete "
                    f"team_id={team_id} "
                    f"chart_id={chart_id} "
                    f"position_key={pos_key}"
                )

            athletes_raw = pos_val.get("athletes", [])
            if not isinstance(athletes_raw, list):
                fail(
                    "Depth-chart athletes is not a list "
                    f"team_id={team_id} "
                    f"chart_id={chart_id} "
                    f"position_key={pos_key}"
                )

            ranked_entries: list[
                tuple[int, int, dict[str, Any]]
            ] = []

            for athlete_index, athlete_entry in enumerate(
                athletes_raw
            ):
                if not isinstance(athlete_entry, dict):
                    fail(
                        "Depth-chart athlete entry is not an object "
                        f"team_id={team_id} "
                        f"chart_id={chart_id} "
                        f"position_key={pos_key} "
                        f"athlete_index={athlete_index}"
                    )

                context = (
                    f"team_id={team_id} "
                    f"chart_id={chart_id} "
                    f"position_key={pos_key} "
                    f"athlete_index={athlete_index}"
                )
                rank = rank_value(
                    athlete_entry
                )

                ranked_entries.append(
                    (
                        rank,
                        athlete_index,
                        athlete_entry,
                    )
                )

            ranked_entries.sort(
                key=lambda entry: (
                    entry[0],
                    entry[1],
                )
            )

            athletes_out = []

            for (
                rank,
                athlete_index,
                athlete_entry,
            ) in ranked_entries:
                context = (
                    f"team_id={team_id} "
                    f"chart_id={chart_id} "
                    f"position_key={pos_key} "
                    f"rank={rank}"
                )

                athlete_ref = athlete_entry.get(
                    "athlete",
                    {},
                )

                if (
                    isinstance(athlete_ref, dict)
                    and "$ref" in athlete_ref
                ):
                    resolved = resolve_athlete_ref(
                        athlete_ref["$ref"],
                        context=context,
                    )
                elif isinstance(athlete_ref, dict):
                    resolved = normalize_athlete(
                        athlete_ref,
                        context=context,
                    )
                else:
                    fail(
                        "Depth-chart athlete payload is invalid "
                        f"context={context}"
                    )

                athletes_out.append(resolved)
                athlete_entry_count += 1

            positions_out[pos_key] = {
                "position": {
                    "abbreviation": position_abbr,
                    "name": position_name,
                    "displayName": position_display,
                },
                "athletes": athletes_out,
            }

        depthchart.append(
            {
                "id": chart_id,
                "name": chart_name,
                "positions": positions_out,
            }
        )

    if athlete_entry_count == 0:
        fail(
            "Depth-chart response contains no athletes "
            f"team_id={team_id} "
            f"team_abbr={team_abbr} "
            f"season={season}"
        )

    return (
        {
            "depthchart": depthchart,
            "team": {
                "id": team_id,
                "abbreviation": team_abbr,
            },
            "team_id": team_id,
            "season": {
                "year": season,
            },
        },
        athlete_entry_count,
    )


def flatten(
    obj: Any,
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    items: dict[str, Any] = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = (
                f"{parent_key}{sep}{key}"
                if parent_key
                else str(key)
            )
            items.update(
                flatten(
                    value,
                    new_key,
                    sep,
                )
            )
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            new_key = (
                f"{parent_key}{sep}{index}"
                if parent_key
                else str(index)
            )
            items.update(
                flatten(
                    value,
                    new_key,
                    sep,
                )
            )
    else:
        items[parent_key] = obj

    return items


def validate_flat_rows(
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    *,
    canonical_teams: dict[str, str],
    season: int,
) -> int:
    if len(rows) != 32:
        fail(
            "Raw depth chart must contain exactly 32 team rows; "
            f"received={len(rows)}"
        )

    if not fieldnames:
        fail("Raw depth chart has no columns")

    if len(fieldnames) != len(set(fieldnames)):
        fail("Raw depth chart contains duplicate columns")

    if fieldnames != sorted(fieldnames):
        fail(
            "Raw depth chart columns are not in the "
            "legacy sorted-column order"
        )

    required_columns = {
        "team.id",
        "team.abbreviation",
        "team_id",
        "season.year",
    }
    missing_columns = sorted(
        required_columns - set(fieldnames)
    )

    if missing_columns:
        fail(
            "Raw depth chart missing required columns: "
            f"{missing_columns}"
        )

    athlete_id_columns = [
        column
        for column in fieldnames
        if ATHLETE_ID_COLUMN_PATTERN.fullmatch(column)
    ]

    if not athlete_id_columns:
        fail(
            "Raw depth chart contains no compatible "
            "athlete ID columns"
        )

    seen_team_ids: set[str] = set()
    seen_team_abbrs: set[str] = set()
    athlete_entries = 0

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        team_id = clean(row.get("team_id"))
        team_field_id = clean(row.get("team.id"))
        team_abbr = clean(
            row.get("team.abbreviation")
        ).upper()
        row_season = clean(row.get("season.year"))

        if team_id in seen_team_ids:
            fail(
                "Duplicate team row in raw depth chart "
                f"team_id={team_id}"
            )

        if team_abbr in seen_team_abbrs:
            fail(
                "Duplicate team abbreviation row in raw depth chart "
                f"team_abbr={team_abbr}"
            )

        expected_abbr = canonical_teams.get(team_id)
        if expected_abbr is None:
            fail(
                "Raw depth chart contains unknown team "
                f"row={row_number} "
                f"team_id={team_id!r}"
            )

        if team_field_id != team_id:
            fail(
                "Raw depth chart team.id/team_id mismatch "
                f"row={row_number} "
                f"team.id={team_field_id!r} "
                f"team_id={team_id!r}"
            )

        if team_abbr != expected_abbr:
            fail(
                "Raw depth chart team abbreviation mismatch "
                f"row={row_number} "
                f"team_id={team_id} "
                f"expected={expected_abbr} "
                f"received={team_abbr}"
            )

        if row_season != str(season):
            fail(
                "Raw depth chart season mismatch "
                f"row={row_number} "
                f"expected={season} "
                f"received={row_season!r}"
            )

        team_athlete_entries = 0

        for id_column in athlete_id_columns:
            match = ATHLETE_ID_COLUMN_PATTERN.fullmatch(
                id_column
            )
            if match is None:
                continue

            player_id = clean(row.get(id_column))
            prefix = id_column[:-3]
            name_column = f"{prefix}.displayName"
            guid_column = f"{prefix}.guid"
            uid_column = f"{prefix}.uid"

            related_values = {
                "displayName": clean(
                    row.get(name_column)
                ),
                "guid": clean(
                    row.get(guid_column)
                ),
                "uid": clean(
                    row.get(uid_column)
                ),
            }

            if not player_id:
                if any(related_values.values()):
                    fail(
                        "Raw depth chart contains athlete metadata "
                        "without athlete ID "
                        f"row={row_number} "
                        f"column={id_column}"
                    )
                continue

            missing_athlete_columns = [
                column
                for column in (
                    name_column,
                    guid_column,
                    uid_column,
                )
                if column not in fieldnames
            ]
            if missing_athlete_columns:
                fail(
                    "Raw depth chart missing athlete companion "
                    f"columns={missing_athlete_columns}"
                )

            if not related_values["displayName"]:
                fail(
                    "Raw depth chart athlete missing displayName "
                    f"row={row_number} "
                    f"column={id_column} "
                    f"player_id={player_id}"
                )

            team_athlete_entries += 1
            athlete_entries += 1

        if team_athlete_entries == 0:
            fail(
                "Raw depth chart team row contains no athletes "
                f"team_id={team_id} "
                f"team_abbr={team_abbr}"
            )

        seen_team_ids.add(team_id)
        seen_team_abbrs.add(team_abbr)

    if seen_team_ids != set(canonical_teams):
        missing_ids = sorted(
            set(canonical_teams) - seen_team_ids
        )
        extra_ids = sorted(
            seen_team_ids - set(canonical_teams)
        )
        fail(
            "Raw depth chart team universe mismatch "
            f"missing={missing_ids} extra={extra_ids}"
        )

    if seen_team_abbrs != set(canonical_teams.values()):
        fail(
            "Raw depth chart abbreviation universe "
            "does not match canonical NFL teams"
        )

    return athlete_entries


def normalize_csv_rows(
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> list[dict[str, str]]:
    return [
        {
            field: (
                ""
                if row.get(field) is None
                else str(row.get(field))
            )
            for field in fieldnames
        }
        for row in rows
    ]


def write_staged_csv(
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> Path:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        dir=OUTPUT_PATH.parent,
        prefix=".raw_depth.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        staged_path = Path(handle.name)
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    return staged_path


def read_csv(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        fail(f"Staged raw depth file missing: {path}")

    if path.stat().st_size == 0:
        fail(f"Staged raw depth file is zero bytes: {path}")

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    return fieldnames, rows


def main() -> None:
    args = parse_args()

    with PipelineReporter(
        script=__file__,
        stage="00_intake",
        report_root=REPORT_ROOT,
        pipeline="NFL 01 Pipeline",
        league="nfl",
        season=args.season,
    ) as reporter:
        reporter.add_input(TEAM_MAP_PATH)
        reporter.add_output(OUTPUT_PATH)
        reporter.update_details(
            {
                "team_source": str(TEAM_MAP_PATH),
                "depthchart_url_template": DEPTHCHART_URL_TEMPLATE,
                "requested_season": args.season,
                "publication_mode": "staged_atomic_replace",
            }
        )

        canonical_teams = load_canonical_teams()
        teams = list(canonical_teams.items())

        reporter.set_detail(
            "canonical_team_count",
            len(canonical_teams),
        )
        reporter.set_detail(
            "team_count",
            len(teams),
        )

        all_rows: list[dict[str, Any]] = []
        all_columns: set[str] = set()
        fetched_athlete_entries = 0

        for team_id, team_abbr in teams:
            url = DEPTHCHART_URL_TEMPLATE.format(
                season=args.season,
                team_id=team_id,
            )

            try:
                core_data = fetch_json(
                    url,
                    timeout=30,
                )
            except Exception as exc:
                raise DepthChartPullError(
                    "Failed to pull depth chart "
                    f"season={args.season} "
                    f"team_id={team_id} "
                    f"team_abbr={team_abbr}: {exc}"
                ) from exc

            reshaped, team_athlete_entries = (
                build_old_shape(
                    core_data,
                    team_id,
                    team_abbr,
                    args.season,
                )
            )
            flat_row = flatten(reshaped)

            all_rows.append(flat_row)
            all_columns.update(flat_row.keys())
            fetched_athlete_entries += (
                team_athlete_entries
            )

            print(
                f"team={team_abbr} "
                f"team_id={team_id} "
                f"athlete_entries={team_athlete_entries} "
                "done"
            )

        fieldnames = sorted(all_columns)

        validated_athlete_entries = (
            validate_flat_rows(
                all_rows,
                fieldnames,
                canonical_teams=canonical_teams,
                season=args.season,
            )
        )

        if (
            validated_athlete_entries
            != fetched_athlete_entries
        ):
            fail(
                "Athlete-entry count changed during flattening "
                f"fetched={fetched_athlete_entries} "
                f"flattened={validated_athlete_entries}"
            )

        reporter.set_rows(
            rows_in=len(teams),
            rows_out=len(all_rows),
        )
        reporter.update_details(
            {
                "column_count": len(fieldnames),
                "athlete_entry_count": (
                    validated_athlete_entries
                ),
                "resolved_athlete_ref_count": (
                    len(athlete_cache)
                ),
            }
        )

        staged_path: Path | None = None

        try:
            staged_path = write_staged_csv(
                all_rows,
                fieldnames,
            )

            staged_fieldnames, staged_rows = read_csv(
                staged_path
            )

            staged_athlete_entries = (
                validate_flat_rows(
                    staged_rows,
                    staged_fieldnames,
                    canonical_teams=canonical_teams,
                    season=args.season,
                )
            )

            if staged_fieldnames != fieldnames:
                fail(
                    "Staged raw depth header differs from "
                    "validated in-memory header"
                )

            expected_rows = normalize_csv_rows(
                all_rows,
                fieldnames,
            )

            if staged_rows != expected_rows:
                fail(
                    "Staged raw depth rows differ from "
                    "validated in-memory rows"
                )

            if (
                staged_athlete_entries
                != validated_athlete_entries
            ):
                fail(
                    "Staged raw depth athlete count mismatch "
                    f"expected={validated_athlete_entries} "
                    f"received={staged_athlete_entries}"
                )

            os.replace(
                staged_path,
                OUTPUT_PATH,
            )
            staged_path = None

        finally:
            if (
                staged_path is not None
                and staged_path.exists()
            ):
                staged_path.unlink()

        reporter.set_detail(
            "published",
            True,
        )

        print(
            f"rows={len(all_rows)} "
            f"columns={len(fieldnames)} "
            f"athlete_entries={validated_athlete_entries} "
            f"output={OUTPUT_PATH}"
        )


if __name__ == "__main__":
    main()
