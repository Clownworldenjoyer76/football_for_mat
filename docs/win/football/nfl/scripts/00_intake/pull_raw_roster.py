#!/usr/bin/env python3

"""
pull_raw_roster.py

Pulls the current NFL roster from ESPN's Core API and writes one combined
flattened CSV compatible with the existing roster_cleanup.py.

The old site.api.espn.com roster endpoints are no longer used.

Core endpoints used:
    NFL teams:
        https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/
        seasons/{season}/teams?limit=100

    Team athletes:
        https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/
        seasons/{season}/teams/{team_id}/athletes?limit=200

    Athlete:
        Athlete $ref returned by ESPN

    College:
        Athlete college.$ref

    Position parent:
        Athlete position.parent.$ref

    NFL season:
        https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/
        seasons/{season}

Output:
    docs/win/football/nfl/data/raw/raw_roster.csv

Environment variables:
    NFL_SEASON
        NFL season to pull. Defaults to 2026.

    NFL_ROSTER_WORKERS
        Number of concurrent ESPN requests. Defaults to 8.
"""

import csv
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse


SEASON = int(os.environ.get("NFL_SEASON", "2026"))
MAX_WORKERS = int(os.environ.get("NFL_ROSTER_WORKERS", "8"))

CORE_BASE = "https://sports.core.api.espn.com/v2"

TEAMS_URL = (
    f"{CORE_BASE}/sports/football/leagues/nfl/"
    f"seasons/{SEASON}/teams?limit=100"
)

TEAM_ATHLETES_URL_TEMPLATE = (
    f"{CORE_BASE}/sports/football/leagues/nfl/"
    f"seasons/{SEASON}/teams/{{team_id}}/athletes?limit=200"
)

SEASON_URL = (
    f"{CORE_BASE}/sports/football/leagues/nfl/"
    f"seasons/{SEASON}?lang=en&region=us"
)

OUTPUT_PATH = "docs/win/football/nfl/data/raw/raw_roster.csv"

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

# These columns are required by the existing roster_cleanup.py.
# Core does not currently expose the old hand fields, so those columns
# are retained and written blank rather than breaking the downstream schema.
COMPATIBILITY_COLUMNS = [
    "age",
    "alternateIds.sdr",
    "birthPlace.city",
    "birthPlace.country",
    "birthPlace.state",
    "college.abbrev",
    "college.guid",
    "college.id",
    "college.name",
    "college.shortName",
    "contract.active",
    "contract.bonus",
    "contract.optionType",
    "contract.salary",
    "contract.salaryRemaining",
    "contract.season.endDate",
    "contract.season.startDate",
    "contract.season.year",
    "contract.signedThrough",
    "dateOfBirth",
    "debutYear",
    "displayHeight",
    "displayName",
    "displayWeight",
    "experience.years",
    "firstName",
    "fullName",
    "guid",
    "hand.abbreviation",
    "hand.displayValue",
    "hand.type",
    "headshot.alt",
    "headshot.href",
    "height",
    "id",
    "injuries.0.date",
    "injuries.0.status",
    "jersey",
    "lastName",
    "position.abbreviation",
    "position.displayName",
    "position.id",
    "position.leaf",
    "position.name",
    "position.parent.abbreviation",
    "position.parent.displayName",
    "position.parent.id",
    "position.parent.leaf",
    "position.parent.name",
    "shortName",
    "slug",
    "status.abbreviation",
    "status.id",
    "status.name",
    "status.type",
    "team_id",
    "uid",
    "weight",
]


def normalize_ref(url):
    """
    ESPN Core frequently returns $ref URLs using http:// even though the
    same resources are available over HTTPS.

    Always use HTTPS for subsequent requests.
    """
    if not url:
        return url

    if url.startswith("http://"):
        return "https://" + url[len("http://"):]

    return url


def fetch_json(url, retries=4, timeout=30):
    """
    Fetch JSON from ESPN with limited retry handling for transient errors.
    Permanent HTTP errors fail with the exact URL included in the message.
    """
    url = normalize_ref(url)

    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            headers=REQUEST_HEADERS,
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw)

        except urllib.error.HTTPError as exc:
            if exc.code in RETRYABLE_HTTP_CODES and attempt < retries:
                delay = min(2 ** (attempt - 1), 8)
                print(
                    f"retry_http "
                    f"status={exc.code} "
                    f"attempt={attempt}/{retries} "
                    f"delay={delay}s "
                    f"url={url}"
                )
                time.sleep(delay)
                continue

            raise RuntimeError(
                f"ESPN HTTP error status={exc.code} url={url}"
            ) from exc

        except urllib.error.URLError as exc:
            if attempt < retries:
                delay = min(2 ** (attempt - 1), 8)
                print(
                    f"retry_url_error "
                    f"attempt={attempt}/{retries} "
                    f"delay={delay}s "
                    f"url={url} "
                    f"reason={exc.reason}"
                )
                time.sleep(delay)
                continue

            raise RuntimeError(
                f"ESPN request failed url={url} reason={exc.reason}"
            ) from exc

        except TimeoutError as exc:
            if attempt < retries:
                delay = min(2 ** (attempt - 1), 8)
                print(
                    f"retry_timeout "
                    f"attempt={attempt}/{retries} "
                    f"delay={delay}s "
                    f"url={url}"
                )
                time.sleep(delay)
                continue

            raise RuntimeError(
                f"ESPN request timed out url={url}"
            ) from exc

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"ESPN returned invalid JSON url={url}"
            ) from exc

    raise RuntimeError(f"Unable to fetch ESPN URL: {url}")


def fetch_optional_json(url):
    """
    Fetch optional enrichment metadata.

    A failed college or position-parent lookup should not destroy the entire
    roster pull. The failure is logged and the original $ref is retained.
    """
    try:
        return url, fetch_json(url)
    except Exception as exc:
        print(f"warning_optional_ref_failed url={url} error={exc}")
        return url, None


def flatten(obj, parent_key="", sep="."):
    """
    Recursively flatten a nested dict/list structure into a single-level
    dictionary using dot-separated keys.
    """
    items = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = (
                f"{parent_key}{sep}{key}"
                if parent_key
                else key
            )
            items.update(flatten(value, new_key, sep))

    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            new_key = (
                f"{parent_key}{sep}{index}"
                if parent_key
                else str(index)
            )
            items.update(flatten(value, new_key, sep))

    else:
        items[parent_key] = obj

    return items


def id_from_ref(ref, resource_name):
    """
    Extract the final numeric ID from an ESPN Core $ref URL.
    """
    ref = normalize_ref(ref)

    if not ref:
        raise ValueError(f"Missing {resource_name} $ref")

    path = urlparse(ref).path.rstrip("/")
    value = path.split("/")[-1]

    if not value.isdigit():
        raise ValueError(
            f"Unable to extract {resource_name} ID from ref={ref}"
        )

    return value


def get_team_ids():
    """
    Return the current season's NFL team IDs from the Core API.
    """
    data = fetch_json(TEAMS_URL)

    items = data.get("items", [])

    if not items:
        raise RuntimeError(
            f"No NFL teams returned for season={SEASON}"
        )

    team_ids = []

    for item in items:
        ref = item.get("$ref")

        if not ref:
            continue

        team_id = id_from_ref(ref, "team")

        if team_id not in team_ids:
            team_ids.append(team_id)

    expected_count = data.get("count")

    if expected_count is not None and len(team_ids) != expected_count:
        raise RuntimeError(
            f"Team count mismatch "
            f"expected={expected_count} "
            f"received={len(team_ids)}"
        )

    print(
        f"teams={len(team_ids)} "
        f"season={SEASON}"
    )

    return team_ids


def get_team_athlete_refs(team_id):
    """
    Return all athlete $refs currently attached to an NFL team.
    """
    url = TEAM_ATHLETES_URL_TEMPLATE.format(
        team_id=team_id
    )

    data = fetch_json(url)

    items = data.get("items", [])

    refs = []

    for item in items:
        ref = item.get("$ref")

        if ref:
            ref = normalize_ref(ref)

            if ref not in refs:
                refs.append(ref)

    expected_count = data.get("count")

    if expected_count is not None and len(refs) != expected_count:
        raise RuntimeError(
            f"Athlete count mismatch "
            f"team_id={team_id} "
            f"expected={expected_count} "
            f"received={len(refs)} "
            f"url={url}"
        )

    print(
        f"team_id={team_id} "
        f"athletes={len(refs)}"
    )

    return refs


def fetch_athlete_entry(entry):
    """
    Fetch one athlete while retaining the roster team ID from which the
    athlete was discovered.
    """
    team_id, athlete_ref = entry

    athlete = fetch_json(athlete_ref)

    athlete_id = athlete.get("id")

    if not athlete_id:
        raise RuntimeError(
            f"Athlete response missing id "
            f"team_id={team_id} "
            f"url={athlete_ref}"
        )

    return {
        "team_id": team_id,
        "athlete": athlete,
    }


def fetch_reference_map(refs):
    """
    Fetch a unique collection of optional ESPN metadata references.
    """
    refs = sorted(
        {
            normalize_ref(ref)
            for ref in refs
            if ref
        }
    )

    if not refs:
        return {}

    result = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for ref, data in executor.map(fetch_optional_json, refs):
            if data is not None:
                result[ref] = data

    return result


def main():
    print(
        f"pull_raw_roster.py started "
        f"season={SEASON} "
        f"workers={MAX_WORKERS}"
    )

    # ------------------------------------------------------------------
    # NFL SEASON METADATA
    # ------------------------------------------------------------------

    season_data = fetch_json(SEASON_URL)

    season_year = season_data.get("year", SEASON)
    season_start_date = season_data.get("startDate", "")
    season_end_date = season_data.get("endDate", "")

    print(
        f"season_year={season_year} "
        f"season_start={season_start_date} "
        f"season_end={season_end_date}"
    )

    # ------------------------------------------------------------------
    # TEAM LIST
    # ------------------------------------------------------------------

    team_ids = get_team_ids()

    # ------------------------------------------------------------------
    # TEAM ROSTER REFERENCES
    # ------------------------------------------------------------------

    athlete_entries = []

    for team_id in team_ids:
        athlete_refs = get_team_athlete_refs(team_id)

        for athlete_ref in athlete_refs:
            athlete_entries.append(
                (
                    team_id,
                    athlete_ref,
                )
            )

    if not athlete_entries:
        raise RuntimeError(
            f"No NFL athletes found for season={SEASON}"
        )

    print(
        f"athlete_refs={len(athlete_entries)}"
    )

    # ------------------------------------------------------------------
    # ATHLETE DETAILS
    # ------------------------------------------------------------------

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        athlete_records = list(
            executor.map(
                fetch_athlete_entry,
                athlete_entries,
            )
        )

    print(
        f"athlete_records={len(athlete_records)}"
    )

    # ------------------------------------------------------------------
    # COLLECT OPTIONAL ENRICHMENT REFERENCES
    # ------------------------------------------------------------------

    college_refs = set()
    position_parent_refs = set()

    for record in athlete_records:
        athlete = record["athlete"]

        college = athlete.get("college")

        if isinstance(college, dict):
            college_ref = college.get("$ref")

            if college_ref:
                college_refs.add(
                    normalize_ref(college_ref)
                )

        position = athlete.get("position")

        if isinstance(position, dict):
            parent = position.get("parent")

            if isinstance(parent, dict):
                parent_ref = parent.get("$ref")

                if parent_ref:
                    position_parent_refs.add(
                        normalize_ref(parent_ref)
                    )

    # ------------------------------------------------------------------
    # FETCH COLLEGE + POSITION PARENT DETAILS
    # ------------------------------------------------------------------

    college_map = fetch_reference_map(
        college_refs
    )

    position_parent_map = fetch_reference_map(
        position_parent_refs
    )

    print(
        f"college_refs={len(college_refs)} "
        f"college_records={len(college_map)}"
    )

    print(
        f"position_parent_refs={len(position_parent_refs)} "
        f"position_parent_records={len(position_parent_map)}"
    )

    # ------------------------------------------------------------------
    # BUILD RAW ROSTER ROWS
    # ------------------------------------------------------------------

    all_rows = []
    all_columns = set(COMPATIBILITY_COLUMNS)

    for record in athlete_records:
        team_id = record["team_id"]
        athlete = record["athlete"]

        # --------------------------------------------------------------
        # College enrichment
        # --------------------------------------------------------------

        college = athlete.get("college")

        if isinstance(college, dict):
            college_ref = normalize_ref(
                college.get("$ref")
            )

            if college_ref and college_ref in college_map:
                athlete["college"] = college_map[college_ref]

        # --------------------------------------------------------------
        # Position parent enrichment
        # --------------------------------------------------------------

        position = athlete.get("position")

        if isinstance(position, dict):
            parent = position.get("parent")

            if isinstance(parent, dict):
                parent_ref = normalize_ref(
                    parent.get("$ref")
                )

                if (
                    parent_ref
                    and parent_ref in position_parent_map
                ):
                    position["parent"] = (
                        position_parent_map[parent_ref]
                    )

        # --------------------------------------------------------------
        # Contract season enrichment
        #
        # ESPN embeds current contract data for players who have contract
        # information. The embedded season value is only a $ref, so add
        # the season fields expected by roster_cleanup.py.
        # --------------------------------------------------------------

        contract = athlete.get("contract")

        if isinstance(contract, dict):
            contract_season = contract.get("season")

            if not isinstance(contract_season, dict):
                contract_season = {}

            contract_season["year"] = season_year
            contract_season["startDate"] = season_start_date
            contract_season["endDate"] = season_end_date

            contract["season"] = contract_season

        # --------------------------------------------------------------
        # Flatten athlete
        # --------------------------------------------------------------

        flat_row = flatten(athlete)

        # Preserve roster membership team ID exactly as the old script did.
        flat_row["team_id"] = team_id

        # ESPN Core does not currently expose the old hand.* values in the
        # athlete payload. Preserve the downstream schema without guessing.
        flat_row.setdefault(
            "hand.abbreviation",
            "",
        )
        flat_row.setdefault(
            "hand.displayValue",
            "",
        )
        flat_row.setdefault(
            "hand.type",
            "",
        )

        # Guarantee every column required by roster_cleanup.py exists even
        # when a field is legitimately absent for every player in a pull.
        for column in COMPATIBILITY_COLUMNS:
            flat_row.setdefault(
                column,
                "",
            )

        all_rows.append(flat_row)
        all_columns.update(flat_row.keys())

    if not all_rows:
        raise RuntimeError(
            f"No roster rows created for season={SEASON}"
        )

    # ------------------------------------------------------------------
    # WRITE RAW CSV
    # ------------------------------------------------------------------

    fieldnames = sorted(all_columns)

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in all_rows:
            writer.writerow(row)

    print(
        f"rows={len(all_rows)} "
        f"columns={len(fieldnames)} "
        f"teams={len(team_ids)} "
        f"season={SEASON} "
        f"output={OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()