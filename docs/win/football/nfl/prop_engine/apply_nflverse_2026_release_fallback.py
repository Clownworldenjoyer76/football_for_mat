#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import py_compile

HERE = Path(__file__).resolve().parent
TARGET = HERE / "scripts" / "build" / "refresh_nflverse_player_data.py"

FUNCTION_BLOCK = r'''
def nflverse_release_url(
    family: str,
    season: int,
) -> str:
    releases = {
        "player_stats": (
            "stats_player",
            f"stats_player_week_{season}.parquet",
        ),
        "weekly_rosters": (
            "weekly_rosters",
            f"roster_weekly_{season}.parquet",
        ),
        "snap_counts": (
            "snap_counts",
            f"snap_counts_{season}.parquet",
        ),
        "participation": (
            "pbp_participation",
            f"pbp_participation_{season}.parquet",
        ),
        "players": (
            "players",
            "players.parquet",
        ),
    }

    if family not in releases:
        raise ValueError(
            f"Unsupported family: {family}"
        )

    tag, filename = releases[family]

    return (
        "https://github.com/"
        "nflverse/nflverse-data/releases/download/"
        f"{tag}/{filename}"
    )


def load_nflverse_release(
    family: str,
    season: int,
) -> tuple[pd.DataFrame, str]:
    # Official nflverse GitHub release fallback. This is attempted
    # only after nflreadpy, so nflreadpy remains the first source.
    url = nflverse_release_url(
        family,
        season,
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "football_for_mat-prop-engine/1.0"
            ),
        },
    )

    temp_path: Path | None = None

    try:
        with urllib.request.urlopen(
            request,
            timeout=60,
        ) as response:
            handle = tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=(
                    f".nflverse_{family}_{season}."
                ),
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

        return (
            pd.read_parquet(
                temp_path
            ),
            "official-github-release",
        )

    finally:
        if (
            temp_path is not None
            and temp_path.exists()
        ):
            temp_path.unlink()
'''

OLD_LOADERS = '''    loaders = (
        (
            "nflreadpy",
            load_nflreadpy,
        ),
        (
            "nfl_data_py",
            load_nfl_data_py,
        ),
    )
'''

NEW_LOADERS = '''    loaders = (
        (
            "nflreadpy",
            load_nflreadpy,
        ),
        (
            "nflverse_release",
            load_nflverse_release,
        ),
        (
            "nfl_data_py",
            load_nfl_data_py,
        ),
    )
'''

OLD_POLICY = '''            "fallback_source": (
                "nfl_data_py"
            ),
'''

NEW_POLICY = '''            "fallback_source": (
                "nfl_data_py"
            ),
            "official_release_fallback": (
                True
            ),
            "fallback_order": [
                "nflverse_release",
                "nfl_data_py",
            ],
'''


def main() -> int:
    if not TARGET.exists():
        raise FileNotFoundError(
            f"Missing target: {TARGET}"
        )

    text = TARGET.read_text(
        encoding="utf-8"
    )
    original = text

    if "import shutil\n" not in text:
        marker = "import os\n"
        if marker not in text:
            raise ValueError(
                "Unable to locate import insertion point."
            )
        text = text.replace(
            marker,
            marker + "import shutil\n",
            1,
        )

    if "import urllib.request\n" not in text:
        marker = "import tempfile\n"
        if marker not in text:
            raise ValueError(
                "Unable to locate urllib import insertion point."
            )
        text = text.replace(
            marker,
            marker + "import urllib.request\n",
            1,
        )

    if "def load_nflverse_release(" not in text:
        anchor = "\ndef load_nfl_data_py(\n"
        if anchor not in text:
            raise ValueError(
                "Unable to locate release-loader insertion point."
            )
        text = text.replace(
            anchor,
            "\n" + FUNCTION_BLOCK + anchor,
            1,
        )

    if '"nflverse_release",' not in text:
        if OLD_LOADERS not in text:
            raise ValueError(
                "Unable to locate loader-order block."
            )
        text = text.replace(
            OLD_LOADERS,
            NEW_LOADERS,
            1,
        )

    if '"official_release_fallback"' not in text:
        if OLD_POLICY not in text:
            raise ValueError(
                "Unable to locate refresh policy block."
            )
        text = text.replace(
            OLD_POLICY,
            NEW_POLICY,
            1,
        )

    if text == original:
        print(
            "NFLVERSE 2026 RELEASE FALLBACK PATCH: ALREADY APPLIED"
        )
    else:
        TARGET.write_text(
            text,
            encoding="utf-8",
            newline="\n",
        )
        print(
            "changed=scripts/build/refresh_nflverse_player_data.py"
        )

    py_compile.compile(
        str(TARGET),
        doraise=True,
    )

    check = TARGET.read_text(
        encoding="utf-8"
    )

    required = [
        "def load_nflverse_release(",
        '"nflverse_release",',
        "load_nflverse_release,",
        '"official_release_fallback"',
        '"fallback_order"',
        "urllib.request.urlopen(",
        "shutil.copyfileobj(",
    ]

    missing = [
        item
        for item in required
        if item not in check
    ]

    if missing:
        raise ValueError(
            f"Patch verification failed; missing={missing}"
        )

    nflreadpy_pos = check.index(
        '"nflreadpy",\n            load_nflreadpy,'
    )
    release_pos = check.index(
        '"nflverse_release",\n            load_nflverse_release,'
    )
    nfl_data_py_pos = check.index(
        '"nfl_data_py",\n            load_nfl_data_py,'
    )

    if not (
        nflreadpy_pos
        < release_pos
        < nfl_data_py_pos
    ):
        raise ValueError(
            "Loader order is not nflreadpy -> official release -> nfl_data_py."
        )

    print("nflreadpy_first=true")
    print("official_nflverse_release_fallback=true")
    print("nfl_data_py_last=true")
    print("fabricated_rows_allowed=false")
    print("NFLVERSE 2026 RELEASE FALLBACK PATCH: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
