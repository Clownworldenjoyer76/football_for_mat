#!/usr/bin/env python3
"""
Build the NFL league master and long-format standings outputs from ESPN.

Source:
    https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/types/{season_type}/groups

Outputs:
    docs/win/football/nfl/data/master/league_master.csv
    docs/win/football/nfl/data/master/league_standings.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter


TEAM_MASTER_PATH = (
    NFL_ROOT
    / "data"
    / "master"
    / "team_master.csv"
)

LEAGUE_MASTER_PATH = (
    NFL_ROOT
    / "data"
    / "master"
    / "league_master.csv"
)

LEAGUE_STANDINGS_PATH = (
    NFL_ROOT
    / "data"
    / "master"
    / "league_standings.csv"
)

REPORT_ROOT = NFL_ROOT / "errors"

GROUPS_URL_TEMPLATE = (
    "https://sports.core.api.espn.com/v2/sports/"
    "football/leagues/nfl/seasons/{season}/"
    "types/{season_type}/groups"
)

TEAM_ID_PATTERN = re.compile(
    r"/teams/(\d+)(?:[/?]|$)"
)

MASTER_HEADERS = [
    "team_id",
    "team_abbr",
    "conference",
    "conference_abbr",
    "division",
    "division_abbr",
    "season",
]

STANDINGS_HEADERS = [
    "team_id",
    "team_abbr",
    "conference",
    "conference_abbr",
    "division",
    "division_abbr",
    "standings_type",
    "stat_name",
    "stat_value",
    "season",
]


class LeagueMasterError(RuntimeError):
    pass


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build NFL league master and standings "
            "data from ESPN."
        )
    )

    parser.add_argument(
        "--season",
        required=True,
        type=int,
        help="NFL season to build.",
    )

    parser.add_argument(
        "--season-type",
        required=True,
        type=int,
        choices=(1, 2, 3),
        help=(
            "ESPN season type: "
            "1=preseason, 2=regular season, "
            "3=postseason."
        ),
    )

    args = parser.parse_args()

    if (
        args.season < 2000
        or args.season > 2100
    ):
        parser.error(
            "--season must be between 2000 and 2100"
        )

    return args


def fetch_json(
    url: str,
    timeout: int = 10,
) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(
            url,
            timeout=timeout,
        ) as response:
            raw = response.read()

    except Exception as exc:
        raise LeagueMasterError(
            f"Failed ESPN request: {url}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        payload = json.loads(
            raw.decode("utf-8")
        )

    except Exception as exc:
        raise LeagueMasterError(
            f"Invalid JSON from ESPN request: "
            f"{url}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise LeagueMasterError(
            f"Unexpected ESPN response type "
            f"for {url}: "
            f"{type(payload).__name__}"
        )

    return payload


def extract_team_id(
    ref_url: str,
) -> str:
    match = TEAM_ID_PATTERN.search(
        clean_text(ref_url)
    )

    return (
        match.group(1)
        if match
        else ""
    )


def require_items(
    payload: dict[str, Any],
    context: str,
) -> list[dict[str, Any]]:
    items = payload.get("items")

    if (
        not isinstance(items, list)
        or not items
    ):
        raise LeagueMasterError(
            f"{context} returned no usable items"
        )

    invalid = [
        index
        for index, item
        in enumerate(items)
        if not isinstance(item, dict)
    ]

    if invalid:
        raise LeagueMasterError(
            f"{context} contains non-object "
            f"item(s) at indices "
            f"{invalid[:5]}"
        )

    return items


def require_item_ref(
    item: dict[str, Any],
    context: str,
) -> str:
    ref = clean_text(
        item.get("$ref")
    )

    if not ref:
        raise LeagueMasterError(
            f"{context} is missing "
            "required $ref"
        )

    return ref


def require_nested_ref(
    payload: dict[str, Any],
    key: str,
    context: str,
) -> str:
    value = payload.get(key)

    if not isinstance(value, dict):
        raise LeagueMasterError(
            f"{context} is missing "
            f"required {key} object"
        )

    ref = clean_text(
        value.get("$ref")
    )

    if not ref:
        raise LeagueMasterError(
            f"{context} is missing "
            f"required {key}.$ref"
        )

    return ref


def load_team_master(
) -> tuple[dict[str, str], int]:
    if not TEAM_MASTER_PATH.is_file():
        raise LeagueMasterError(
            f"Missing team master: "
            f"{TEAM_MASTER_PATH}"
        )

    with TEAM_MASTER_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = (
            reader.fieldnames
            or []
        )

        missing = [
            column
            for column
            in (
                "team_id",
                "team_abbr",
            )
            if column
            not in fieldnames
        ]

        if missing:
            raise LeagueMasterError(
                f"{TEAM_MASTER_PATH} "
                f"is missing columns: "
                f"{missing}"
            )

        rows = list(reader)

    if not rows:
        raise LeagueMasterError(
            f"{TEAM_MASTER_PATH} "
            "contains no data rows"
        )

    lookup: dict[str, str] = {}

    for line_number, row in enumerate(
        rows,
        start=2,
    ):
        team_id = clean_text(
            row.get("team_id")
        )
        team_abbr = clean_text(
            row.get("team_abbr")
        )

        if (
            not team_id
            or not team_abbr
        ):
            raise LeagueMasterError(
                f"{TEAM_MASTER_PATH} "
                f"line {line_number} "
                "has blank team_id "
                "or team_abbr"
            )

        previous = lookup.get(
            team_id
        )

        if (
            previous is not None
            and previous != team_abbr
        ):
            raise LeagueMasterError(
                f"{TEAM_MASTER_PATH} maps "
                f"team_id={team_id!r} "
                "to conflicting abbreviations: "
                f"{previous!r} and "
                f"{team_abbr!r}"
            )

        lookup[team_id] = (
            team_abbr
        )

    return lookup, len(rows)


def get_standings_rows(
    *,
    division: dict[str, Any],
    div_name: str,
    div_abbr: str,
    conf_name: str,
    conf_abbr: str,
    team_id_to_abbr: dict[
        str,
        str,
    ],
    season: int,
    standings_type_team_counts: dict[
        str,
        int,
    ],
) -> list[
    dict[str, str | int]
]:
    rows: list[
        dict[str, str | int]
    ] = []

    standings_ref = (
        require_nested_ref(
            division,
            "standings",
            f"division {div_name!r}",
        )
    )

    standings_types_list = (
        fetch_json(
            standings_ref
        )
    )

    type_items = require_items(
        standings_types_list,
        (
            "standings list for "
            f"division {div_name!r}"
        ),
    )

    for (
        type_index,
        type_item,
    ) in enumerate(type_items):
        type_ref = require_item_ref(
            type_item,
            (
                f"standings item "
                f"{type_index} for "
                f"division "
                f"{div_name!r}"
            ),
        )

        standings_type = (
            fetch_json(
                type_ref
            )
        )

        type_name = clean_text(
            standings_type.get(
                "name"
            )
        )

        if not type_name:
            raise LeagueMasterError(
                "Standings type for "
                f"division {div_name!r} "
                "has a blank name"
            )

        team_standings = (
            standings_type.get(
                "standings"
            )
        )

        if (
            not isinstance(
                team_standings,
                list,
            )
            or not team_standings
        ):
            raise LeagueMasterError(
                f"Standings type "
                f"{type_name!r} for "
                f"division {div_name!r} "
                "returned no team "
                "standings"
            )

        type_team_count = 0

        for (
            standing_index,
            team_standing,
        ) in enumerate(
            team_standings
        ):
            if not isinstance(
                team_standing,
                dict,
            ):
                raise LeagueMasterError(
                    f"Standings type "
                    f"{type_name!r} for "
                    f"division "
                    f"{div_name!r} "
                    "contains a "
                    "non-object team "
                    "standing at index "
                    f"{standing_index}"
                )

            team_ref = (
                require_nested_ref(
                    team_standing,
                    "team",
                    (
                        f"standings type "
                        f"{type_name!r}, "
                        f"division "
                        f"{div_name!r}, "
                        "team index "
                        f"{standing_index}"
                    ),
                )
            )

            team_id = (
                extract_team_id(
                    team_ref
                )
            )

            if not team_id:
                raise LeagueMasterError(
                    "Could not extract "
                    "team ID from "
                    "standings ref: "
                    f"{team_ref}"
                )

            team_abbr = (
                team_id_to_abbr.get(
                    team_id,
                    "",
                )
            )

            if not team_abbr:
                raise LeagueMasterError(
                    "Standings returned "
                    f"team_id={team_id!r} "
                    "that is absent from "
                    "team_master.csv"
                )

            records = (
                team_standing.get(
                    "records"
                )
            )

            if (
                not isinstance(
                    records,
                    list,
                )
                or not records
            ):
                raise LeagueMasterError(
                    f"Standings type "
                    f"{type_name!r} for "
                    f"team_id="
                    f"{team_id!r} "
                    "returned no records"
                )

            rows_before_team = (
                len(rows)
            )

            for (
                record_index,
                record,
            ) in enumerate(records):
                if not isinstance(
                    record,
                    dict,
                ):
                    raise LeagueMasterError(
                        f"Standings type "
                        f"{type_name!r} "
                        "for team_id="
                        f"{team_id!r} "
                        "contains a "
                        "non-object record "
                        "at index "
                        f"{record_index}"
                    )

                stats = (
                    record.get(
                        "stats"
                    )
                )

                if not isinstance(
                    stats,
                    list,
                ):
                    raise LeagueMasterError(
                        f"Standings type "
                        f"{type_name!r} "
                        "for team_id="
                        f"{team_id!r} "
                        "has invalid stats"
                    )

                for (
                    stat_index,
                    stat,
                ) in enumerate(stats):
                    if not isinstance(
                        stat,
                        dict,
                    ):
                        raise LeagueMasterError(
                            "Standings type "
                            f"{type_name!r} "
                            "for team_id="
                            f"{team_id!r} "
                            "contains a "
                            "non-object stat "
                            "at index "
                            f"{stat_index}"
                        )

                    stat_name = (
                        clean_text(
                            stat.get(
                                "name"
                            )
                        )
                    )

                    if not stat_name:
                        raise LeagueMasterError(
                            "Standings type "
                            f"{type_name!r} "
                            "for team_id="
                            f"{team_id!r} "
                            "contains a blank "
                            "stat name"
                        )

                    rows.append(
                        {
                            "team_id": (
                                team_id
                            ),
                            "team_abbr": (
                                team_abbr
                            ),
                            "conference": (
                                conf_name
                            ),
                            "conference_abbr": (
                                conf_abbr
                            ),
                            "division": (
                                div_name
                            ),
                            "division_abbr": (
                                div_abbr
                            ),
                            "standings_type": (
                                type_name
                            ),
                            "stat_name": (
                                stat_name
                            ),
                            "stat_value": (
                                clean_text(
                                    stat.get(
                                        "value"
                                    )
                                )
                            ),
                            "season": season,
                        }
                    )

            if (
                len(rows)
                == rows_before_team
            ):
                raise LeagueMasterError(
                    f"Standings type "
                    f"{type_name!r} "
                    "for team_id="
                    f"{team_id!r} "
                    "produced no "
                    "stat rows"
                )

            type_team_count += 1

        standings_type_team_counts[
            type_name
        ] = (
            standings_type_team_counts.get(
                type_name,
                0,
            )
            + type_team_count
        )

        print(
            f"  standings_type="
            f"{type_name} "
            f"teams="
            f"{type_team_count}"
        )

    return rows


def build_candidates(
    *,
    season: int,
    season_type: int,
    team_id_to_abbr: dict[
        str,
        str,
    ],
) -> tuple[
    list[
        dict[str, str | int]
    ],
    list[
        dict[str, str | int]
    ],
    int,
    int,
    dict[str, int],
]:
    groups_url = (
        GROUPS_URL_TEMPLATE.format(
            season=season,
            season_type=season_type,
        )
    )

    top_groups = fetch_json(
        groups_url
    )

    conference_items = (
        require_items(
            top_groups,
            "NFL conference groups",
        )
    )

    master_rows: list[
        dict[str, str | int]
    ] = []

    standings_rows: list[
        dict[str, str | int]
    ] = []

    standings_type_team_counts: dict[
        str,
        int,
    ] = {}

    conference_count = 0
    division_count = 0

    for (
        conf_index,
        conf_item,
    ) in enumerate(
        conference_items
    ):
        conf_ref = (
            require_item_ref(
                conf_item,
                (
                    "conference item "
                    f"{conf_index}"
                ),
            )
        )

        conf = fetch_json(
            conf_ref
        )

        conf_name = clean_text(
            conf.get("name")
        )
        conf_abbr = clean_text(
            conf.get(
                "abbreviation"
            )
        )

        if (
            not conf_name
            or not conf_abbr
        ):
            raise LeagueMasterError(
                "Conference resolved "
                f"from {conf_ref} "
                "has blank name or "
                "abbreviation"
            )

        conference_count += 1

        children_ref = (
            require_nested_ref(
                conf,
                "children",
                (
                    "conference "
                    f"{conf_name!r}"
                ),
            )
        )

        children_list = (
            fetch_json(
                children_ref
            )
        )

        division_items = (
            require_items(
                children_list,
                (
                    "divisions for "
                    "conference "
                    f"{conf_name!r}"
                ),
            )
        )

        for (
            div_index,
            div_item,
        ) in enumerate(
            division_items
        ):
            div_ref = (
                require_item_ref(
                    div_item,
                    (
                        "division item "
                        f"{div_index} "
                        "for conference "
                        f"{conf_name!r}"
                    ),
                )
            )

            division = (
                fetch_json(
                    div_ref
                )
            )

            div_name = clean_text(
                division.get(
                    "name"
                )
            )
            div_abbr = clean_text(
                division.get(
                    "abbreviation"
                )
            )

            if (
                not div_name
                or not div_abbr
            ):
                raise LeagueMasterError(
                    "Division resolved "
                    f"from {div_ref} "
                    "has blank name or "
                    "abbreviation"
                )

            division_count += 1

            teams_ref = (
                require_nested_ref(
                    division,
                    "teams",
                    (
                        "division "
                        f"{div_name!r}"
                    ),
                )
            )

            teams_list = (
                fetch_json(
                    teams_ref
                )
            )

            team_items = (
                require_items(
                    teams_list,
                    (
                        "teams for "
                        "division "
                        f"{div_name!r}"
                    ),
                )
            )

            for (
                team_index,
                team_item,
            ) in enumerate(
                team_items
            ):
                team_ref = (
                    require_item_ref(
                        team_item,
                        (
                            "team item "
                            f"{team_index} "
                            "for division "
                            f"{div_name!r}"
                        ),
                    )
                )

                team_id = (
                    extract_team_id(
                        team_ref
                    )
                )

                if not team_id:
                    raise LeagueMasterError(
                        "Could not extract "
                        "team ID from "
                        "team ref: "
                        f"{team_ref}"
                    )

                team_abbr = (
                    team_id_to_abbr.get(
                        team_id,
                        "",
                    )
                )

                if not team_abbr:
                    raise LeagueMasterError(
                        "ESPN returned "
                        f"team_id={team_id!r} "
                        "that is absent "
                        "from "
                        "team_master.csv"
                    )

                master_rows.append(
                    {
                        "team_id": (
                            team_id
                        ),
                        "team_abbr": (
                            team_abbr
                        ),
                        "conference": (
                            conf_name
                        ),
                        "conference_abbr": (
                            conf_abbr
                        ),
                        "division": (
                            div_name
                        ),
                        "division_abbr": (
                            div_abbr
                        ),
                        "season": season,
                    }
                )

            print(
                f"division="
                f"{div_name}"
            )

            standings_rows.extend(
                get_standings_rows(
                    division=division,
                    div_name=div_name,
                    div_abbr=div_abbr,
                    conf_name=conf_name,
                    conf_abbr=conf_abbr,
                    team_id_to_abbr=(
                        team_id_to_abbr
                    ),
                    season=season,
                    standings_type_team_counts=(
                        standings_type_team_counts
                    ),
                )
            )

    return (
        master_rows,
        standings_rows,
        conference_count,
        division_count,
        standings_type_team_counts,
    )


def validate_master_rows(
    rows: list[
        dict[str, Any]
    ],
    *,
    season: int,
) -> dict[str, str]:
    if len(rows) != 32:
        raise LeagueMasterError(
            "league_master candidate "
            "must contain exactly "
            f"32 rows; found "
            f"{len(rows)}"
        )

    team_ids: set[str] = set()
    team_abbrs: set[str] = set()

    team_id_to_abbr: dict[
        str,
        str,
    ] = {}

    for (
        row_number,
        row,
    ) in enumerate(
        rows,
        start=2,
    ):
        values = {
            field: clean_text(
                row.get(field)
            )
            for field
            in (
                "team_id",
                "team_abbr",
                "conference",
                "conference_abbr",
                "division",
                "division_abbr",
            )
        }

        blanks = [
            field
            for field, value
            in values.items()
            if not value
        ]

        if blanks:
            raise LeagueMasterError(
                "league_master "
                f"candidate row "
                f"{row_number} "
                "has blank required "
                f"fields: {blanks}"
            )

        row_season = clean_text(
            row.get("season")
        )

        if (
            row_season
            != str(season)
        ):
            raise LeagueMasterError(
                "league_master "
                f"candidate row "
                f"{row_number} "
                f"has season="
                f"{row_season!r}; "
                "expected "
                f"{str(season)!r}"
            )

        team_id = values[
            "team_id"
        ]
        team_abbr = values[
            "team_abbr"
        ]

        if team_id in team_ids:
            raise LeagueMasterError(
                "league_master "
                "candidate contains "
                "duplicate "
                f"team_id="
                f"{team_id!r}"
            )

        if team_abbr in team_abbrs:
            raise LeagueMasterError(
                "league_master "
                "candidate contains "
                "duplicate "
                f"team_abbr="
                f"{team_abbr!r}"
            )

        team_ids.add(
            team_id
        )
        team_abbrs.add(
            team_abbr
        )

        team_id_to_abbr[
            team_id
        ] = team_abbr

    if (
        len(team_ids) != 32
        or len(team_abbrs) != 32
    ):
        raise LeagueMasterError(
            "league_master candidate "
            "does not contain "
            "32 unique teams"
        )

    return team_id_to_abbr


def validate_standings_rows(
    rows: list[
        dict[str, Any]
    ],
    *,
    season: int,
    master_team_id_to_abbr: dict[
        str,
        str,
    ],
) -> set[str]:
    if not rows:
        raise LeagueMasterError(
            "league_standings "
            "candidate contains "
            "no rows"
        )

    standings_types: set[str] = set()
    observed_team_ids: set[str] = set()

    for (
        row_number,
        row,
    ) in enumerate(
        rows,
        start=2,
    ):
        required_values = {
            field: clean_text(
                row.get(field)
            )
            for field
            in (
                "team_id",
                "team_abbr",
                "conference",
                "conference_abbr",
                "division",
                "division_abbr",
                "standings_type",
                "stat_name",
            )
        }

        blanks = [
            field
            for field, value
            in required_values.items()
            if not value
        ]

        if blanks:
            raise LeagueMasterError(
                "league_standings "
                f"candidate row "
                f"{row_number} "
                "has blank required "
                f"fields: {blanks}"
            )

        row_season = clean_text(
            row.get("season")
        )

        if (
            row_season
            != str(season)
        ):
            raise LeagueMasterError(
                "league_standings "
                f"candidate row "
                f"{row_number} "
                f"has season="
                f"{row_season!r}; "
                "expected "
                f"{str(season)!r}"
            )

        team_id = (
            required_values[
                "team_id"
            ]
        )
        team_abbr = (
            required_values[
                "team_abbr"
            ]
        )

        expected_abbr = (
            master_team_id_to_abbr.get(
                team_id
            )
        )

        if expected_abbr is None:
            raise LeagueMasterError(
                "league_standings "
                "candidate contains "
                f"team_id={team_id!r} "
                "absent from "
                "league_master "
                "candidate"
            )

        if team_abbr != expected_abbr:
            raise LeagueMasterError(
                "league_standings "
                "candidate has team "
                "abbreviation mismatch "
                "for team_id="
                f"{team_id!r}: "
                f"{team_abbr!r} != "
                f"{expected_abbr!r}"
            )

        observed_team_ids.add(
            team_id
        )

        standings_types.add(
            required_values[
                "standings_type"
            ]
        )

    if not standings_types:
        raise LeagueMasterError(
            "league_standings "
            "candidate contains no "
            "standings types"
        )

    return standings_types


def write_csv(
    path: Path,
    *,
    headers: list[str],
    rows: list[
        dict[str, Any]
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=headers,
            extrasaction="ignore",
            lineterminator="\n",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    header: clean_text(
                        row.get(header)
                    )
                    for header
                    in headers
                }
            )

        handle.flush()
        os.fsync(
            handle.fileno()
        )


def read_csv_rows(
    path: Path,
    *,
    expected_headers: list[str],
) -> list[
    dict[str, str]
]:
    if not path.is_file():
        raise LeagueMasterError(
            "Staged output is "
            f"missing: {path}"
        )

    if path.stat().st_size == 0:
        raise LeagueMasterError(
            "Staged output is "
            f"zero bytes: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        actual_headers = (
            reader.fieldnames
            or []
        )

        if (
            actual_headers
            != expected_headers
        ):
            raise LeagueMasterError(
                f"Staged output "
                f"{path} has "
                "unexpected headers: "
                f"{actual_headers}"
            )

        return list(reader)


def restore_backups(
    *,
    backup_dir: Path,
    moved_outputs: list[Path],
) -> None:
    rollback_errors: list[str] = []

    for output_path in reversed(
        moved_outputs
    ):
        backup_path = (
            backup_dir
            / output_path.name
        )

        if not backup_path.exists():
            rollback_errors.append(
                "missing backup for "
                f"{output_path}"
            )
            continue

        try:
            os.replace(
                backup_path,
                output_path,
            )

        except Exception as exc:
            rollback_errors.append(
                f"{output_path}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    if rollback_errors:
        raise LeagueMasterError(
            "Rollback failed: "
            + "; ".join(
                rollback_errors
            )
        )


def publish_pair(
    *,
    staged_master: Path,
    staged_standings: Path,
) -> None:
    output_dir = (
        LEAGUE_MASTER_PATH.parent
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_pairs = [
        (
            staged_master,
            LEAGUE_MASTER_PATH,
        ),
        (
            staged_standings,
            LEAGUE_STANDINGS_PATH,
        ),
    ]

    backup_dir = Path(
        tempfile.mkdtemp(
            prefix=(
                ".league_master_"
                "backup_"
            ),
            dir=output_dir,
        )
    )

    moved_outputs: list[
        Path
    ] = []

    published_outputs: list[
        Path
    ] = []

    try:
        try:
            for (
                _,
                output_path,
            ) in output_pairs:
                if output_path.exists():
                    os.replace(
                        output_path,
                        (
                            backup_dir
                            / output_path.name
                        ),
                    )

                    moved_outputs.append(
                        output_path
                    )

        except Exception as exc:
            try:
                restore_backups(
                    backup_dir=(
                        backup_dir
                    ),
                    moved_outputs=(
                        moved_outputs
                    ),
                )

            except Exception as rollback_exc:
                raise LeagueMasterError(
                    "Could not stage "
                    "canonical outputs "
                    "for replacement and "
                    "rollback also failed: "
                    f"{rollback_exc}"
                ) from exc

            raise

        try:
            for (
                staged_path,
                output_path,
            ) in output_pairs:
                os.replace(
                    staged_path,
                    output_path,
                )

                published_outputs.append(
                    output_path
                )

        except Exception as exc:
            cleanup_errors: list[
                str
            ] = []

            for output_path in (
                published_outputs
            ):
                try:
                    output_path.unlink(
                        missing_ok=True
                    )

                except Exception as cleanup_exc:
                    cleanup_errors.append(
                        f"{output_path}: "
                        f"{type(cleanup_exc).__name__}: "
                        f"{cleanup_exc}"
                    )

            try:
                restore_backups(
                    backup_dir=(
                        backup_dir
                    ),
                    moved_outputs=(
                        moved_outputs
                    ),
                )

            except Exception as rollback_exc:
                cleanup_errors.append(
                    str(rollback_exc)
                )

            if cleanup_errors:
                raise LeagueMasterError(
                    "Publication failed "
                    "and rollback was "
                    "incomplete: "
                    + "; ".join(
                        cleanup_errors
                    )
                ) from exc

            raise

    finally:
        shutil.rmtree(
            backup_dir,
            ignore_errors=True,
        )


def run(
    *,
    season: int,
    season_type: int,
    reporter: PipelineReporter,
) -> None:
    groups_url = (
        GROUPS_URL_TEMPLATE.format(
            season=season,
            season_type=season_type,
        )
    )

    reporter.add_input(
        TEAM_MASTER_PATH
    )

    reporter.update_details(
        {
            "season_type": (
                season_type
            ),
            "groups_url": (
                groups_url
            ),
            "publication_completed": (
                False
            ),
            "staged_roundtrip_verified": (
                False
            ),
        }
    )

    (
        team_id_to_abbr,
        team_master_rows,
    ) = load_team_master()

    reporter.set_rows(
        rows_in=team_master_rows,
        rows_out=0,
    )

    reporter.update_details(
        {
            "team_master_rows": (
                team_master_rows
            ),
            "team_master_unique_team_ids": (
                len(
                    team_id_to_abbr
                )
            ),
        }
    )

    (
        master_rows,
        standings_rows,
        conference_count,
        division_count,
        standings_type_team_counts,
    ) = build_candidates(
        season=season,
        season_type=season_type,
        team_id_to_abbr=(
            team_id_to_abbr
        ),
    )

    master_team_id_to_abbr = (
        validate_master_rows(
            master_rows,
            season=season,
        )
    )

    standings_types = (
        validate_standings_rows(
            standings_rows,
            season=season,
            master_team_id_to_abbr=(
                master_team_id_to_abbr
            ),
        )
    )

    reporter.update_details(
        {
            "conferences_fetched": (
                conference_count
            ),
            "divisions_fetched": (
                division_count
            ),
            "master_candidate_rows": (
                len(master_rows)
            ),
            "standings_candidate_rows": (
                len(
                    standings_rows
                )
            ),
            "standings_unique_team_ids": (
                len(
                    {
                        clean_text(
                            row.get(
                                "team_id"
                            )
                        )
                        for row
                        in standings_rows
                    }
                )
            ),
            "standings_types": (
                sorted(
                    standings_types
                )
            ),
            "standings_type_team_counts": (
                dict(
                    sorted(
                        standings_type_team_counts.items()
                    )
                )
            ),
        }
    )

    output_dir = (
        LEAGUE_MASTER_PATH.parent
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix=".league_master_stage_",
        dir=output_dir,
    ) as staging_name:
        staging_dir = Path(
            staging_name
        )

        staged_master = (
            staging_dir
            / LEAGUE_MASTER_PATH.name
        )

        staged_standings = (
            staging_dir
            / LEAGUE_STANDINGS_PATH.name
        )

        write_csv(
            staged_master,
            headers=MASTER_HEADERS,
            rows=master_rows,
        )

        write_csv(
            staged_standings,
            headers=STANDINGS_HEADERS,
            rows=standings_rows,
        )

        staged_master_rows = (
            read_csv_rows(
                staged_master,
                expected_headers=(
                    MASTER_HEADERS
                ),
            )
        )

        staged_standings_rows = (
            read_csv_rows(
                staged_standings,
                expected_headers=(
                    STANDINGS_HEADERS
                ),
            )
        )

        staged_master_lookup = (
            validate_master_rows(
                staged_master_rows,
                season=season,
            )
        )

        staged_standings_types = (
            validate_standings_rows(
                staged_standings_rows,
                season=season,
                master_team_id_to_abbr=(
                    staged_master_lookup
                ),
            )
        )

        if (
            len(staged_master_rows)
            != len(master_rows)
        ):
            raise LeagueMasterError(
                "Staged league_master "
                "row count changed during "
                "round-trip verification"
            )

        if (
            len(
                staged_standings_rows
            )
            != len(
                standings_rows
            )
        ):
            raise LeagueMasterError(
                "Staged "
                "league_standings row "
                "count changed during "
                "round-trip verification"
            )

        if (
            staged_standings_types
            != standings_types
        ):
            raise LeagueMasterError(
                "Staged standings types "
                "changed during "
                "round-trip verification"
            )

        reporter.set_detail(
            "staged_roundtrip_verified",
            True,
        )

        publish_pair(
            staged_master=(
                staged_master
            ),
            staged_standings=(
                staged_standings
            ),
        )

    reporter.add_output(
        LEAGUE_MASTER_PATH
    )

    reporter.add_output(
        LEAGUE_STANDINGS_PATH
    )

    reporter.set_rows(
        rows_in=team_master_rows,
        rows_out=(
            len(master_rows)
            + len(
                standings_rows
            )
        ),
    )

    reporter.update_details(
        {
            "master_rows_published": (
                len(master_rows)
            ),
            "standings_rows_published": (
                len(
                    standings_rows
                )
            ),
            "publication_completed": (
                True
            ),
        }
    )

    print(
        f"rows={len(master_rows)} "
        f"output="
        f"{LEAGUE_MASTER_PATH}"
    )

    print(
        f"rows="
        f"{len(standings_rows)} "
        f"output="
        f"{LEAGUE_STANDINGS_PATH}"
    )


def main() -> int:
    args = parse_args()

    season = int(
        args.season
    )
    season_type = int(
        args.season_type
    )

    try:
        with PipelineReporter(
            script=SCRIPT_PATH,
            stage="00_intake",
            report_root=REPORT_ROOT,
            pipeline="NFL",
            league="NFL",
            season=season,
            extra_context={
                "component": (
                    "league master "
                    "and standings"
                ),
                "season_type": (
                    season_type
                ),
            },
        ) as reporter:
            run(
                season=season,
                season_type=season_type,
                reporter=reporter,
            )

        return 0

    except Exception as exc:
        print(
            f"ERROR: "
            f"{type(exc).__name__}: "
            f"{exc}",
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())
