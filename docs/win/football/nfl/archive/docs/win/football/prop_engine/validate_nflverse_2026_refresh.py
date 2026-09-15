#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "scripts" / "build" / "refresh_nflverse_player_data.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--season",
        type=int,
        default=2026,
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(
        f"NFLVERSE 2026 REFRESH VALIDATION: FAIL - {message}"
    )


def main() -> int:
    args = parse_args()
    season = int(args.season)

    if not SCRIPT.exists():
        fail(
            f"missing refresh script: {SCRIPT}"
        )

    text = SCRIPT.read_text(
        encoding="utf-8"
    )

    static_required = [
        "def load_nflverse_release(",
        '"nflverse_release",',
        "load_nflverse_release,",
        '"official_release_fallback"',
        '"fallback_order"',
        "urllib.request.urlopen(",
        "shutil.copyfileobj(",
    ]

    missing = [
        marker
        for marker in static_required
        if marker not in text
    ]

    if missing:
        fail(
            f"missing release fallback markers: {missing}"
        )

    try:
        nflreadpy_pos = text.index(
            '"nflreadpy",\n            load_nflreadpy,'
        )
        release_pos = text.index(
            '"nflverse_release",\n            load_nflverse_release,'
        )
        nfl_data_py_pos = text.index(
            '"nfl_data_py",\n            load_nfl_data_py,'
        )
    except ValueError as exc:
        fail(
            f"unable to verify loader order: {exc}"
        )

    if not (
        nflreadpy_pos
        < release_pos
        < nfl_data_py_pos
    ):
        fail(
            "loader order is not nflreadpy -> official release -> nfl_data_py"
        )

    log_path = (
        HERE
        / "logs"
        / f"refresh_nflverse_player_data_{season}.json"
    )

    if not log_path.exists():
        fail(
            f"missing refresh log: {log_path}"
        )

    payload = json.loads(
        log_path.read_text(
            encoding="utf-8"
        )
    )

    policy = payload.get(
        "policy",
        {}
    )

    if policy.get(
        "nflreadpy_first"
    ) is not True:
        fail(
            "nflreadpy_first is not true"
        )

    if policy.get(
        "fabricate_missing_rows"
    ) is not False:
        fail(
            "fabricate_missing_rows is not false"
        )

    if policy.get(
        "official_release_fallback"
    ) is not True:
        fail(
            "official_release_fallback is not true"
        )

    if policy.get(
        "fallback_order"
    ) != [
        "nflverse_release",
        "nfl_data_py",
    ]:
        fail(
            "fallback_order is incorrect"
        )

    families = payload.get(
        "families",
        {}
    )

    required_families = {
        "player_stats",
        "weekly_rosters",
        "snap_counts",
        "participation",
        "players",
    }

    if set(
        families
    ) != required_families:
        fail(
            f"family set mismatch: {sorted(families)}"
        )

    for family, result in families.items():
        attempts = result.get(
            "attempts",
            []
        )

        if not attempts:
            fail(
                f"{family}: no source attempts recorded"
            )

        if attempts[0].get(
            "source"
        ) != "nflreadpy":
            fail(
                f"{family}: first source was not nflreadpy"
            )

        status = result.get(
            "status"
        )

        if status == "unavailable":
            if result.get(
                "output_written"
            ) is True:
                fail(
                    f"{family}: unavailable family wrote output"
                )
            if int(
                result.get(
                    "row_count",
                    0,
                )
                or 0
            ) != 0:
                fail(
                    f"{family}: unavailable family has nonzero row_count"
                )

        if (
            attempts[0].get("status")
            != "success"
            and len(attempts) >= 2
            and attempts[1].get("source")
            != "nflverse_release"
        ):
            fail(
                f"{family}: second attempt was not official nflverse release"
            )

    roster = families[
        "weekly_rosters"
    ]

    if roster.get(
        "status"
    ) != "success":
        fail(
            "2026 weekly roster is published but refresh did not load it"
        )

    if roster.get(
        "source"
    ) not in {
        "nflreadpy",
        "nflverse_release",
        "nfl_data_py",
    }:
        fail(
            f"unexpected weekly roster source: {roster.get('source')}"
        )

    if int(
        roster.get(
            "row_count",
            0,
        )
        or 0
    ) <= 0:
        fail(
            "weekly roster row_count is zero"
        )

    players = families[
        "players"
    ]

    if players.get(
        "status"
    ) != "success":
        fail(
            "players.parquet refresh did not succeed"
        )

    counts = payload.get(
        "counts",
        {}
    )

    success_count = int(
        counts.get(
            "successful_families",
            0,
        )
        or 0
    )

    unavailable_count = int(
        counts.get(
            "unavailable_families",
            0,
        )
        or 0
    )

    if success_count < 2:
        fail(
            f"expected at least 2 successful families, got {success_count}"
        )

    print(f"season={season}")
    print(f"status={payload.get('status')}")
    print(f"successful_families={success_count}")
    print(f"unavailable_families={unavailable_count}")
    print(
        "weekly_rosters_source="
        f"{roster.get('source')}"
    )
    print(
        "weekly_rosters_rows="
        f"{roster.get('row_count')}"
    )
    print(
        "players_source="
        f"{players.get('source')}"
    )
    print(
        "players_rows="
        f"{players.get('row_count')}"
    )

    for family in (
        "player_stats",
        "snap_counts",
        "participation",
    ):
        result = families[
            family
        ]
        print(
            f"{family}_status="
            f"{result.get('status')}"
        )

    print("nflreadpy_first=true")
    print("fabricate_missing_rows=false")
    print("NFLVERSE 2026 REFRESH VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
