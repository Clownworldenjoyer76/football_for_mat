#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile
from pathlib import Path

PROP = Path(__file__).resolve().parent
IDENTITY = PROP / "scripts/build/build_player_identity.py"
UNIVERSE = PROP / "scripts/project/build_current_universe.py"


def read_required(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    return path.read_text(encoding="utf-8-sig")


def atomic_write(path: Path, text: str) -> None:
    h = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp = Path(h.name)
    try:
        with h:
            h.write(text)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}.")
    return source.replace(old, new, 1)


def patch_identity(source: str) -> str:
    if '"unresolved_identity_policy": "skip_and_continue"' in source:
        if "Identity validation failed:" in source:
            raise RuntimeError("build_player_identity.py is partially patched.")
        return source

    old_status = '''        "status": (
            "failed"
            if critical_unresolved
            else "passed"
        ),
'''
    new_status = '''        "status": "passed",
        "identity_warning_status": (
            "unresolved_players_skipped"
            if critical_unresolved
            else "clear"
        ),
        "unresolved_identity_policy": "skip_and_continue",
'''
    source = replace_once(source, old_status, new_status, "identity status block")

    start_marker = '''    if critical_unresolved:
        sample = (
            critical_unresolved[
                :20
            ]
        )
'''
    start = source.find(start_marker)
    if start < 0:
        raise RuntimeError("Identity fatal block start not found.")
    end = source.find("\n    return 0", start)
    if end < 0:
        raise RuntimeError("Identity return anchor not found.")

    replacement = '''    if critical_unresolved:
        for item in critical_unresolved:
            print(
                "IDENTITY WARNING: unresolved player excluded; "
                f"name={item.get('display_name', '')} "
                f"espn_id={item.get('current_espn_id', '')} "
                f"team={item.get('current_team', '')} "
                f"position={item.get('position', '')} "
                f"reason={item.get('resolution_method', '')}",
                file=sys.stderr,
            )

        print(
            "IDENTITY WARNING: unresolved identities were skipped; "
            f"count={len(critical_unresolved)}. See {log_path}",
            file=sys.stderr,
        )
'''
    return source[:start] + replacement + source[end:]


def patch_universe(source: str) -> str:
    if '"unresolved_identity_policy": "skip_and_continue"' in source:
        if "Issue 29 unresolved starter identity failure." in source:
            raise RuntimeError("build_current_universe.py is partially patched.")
        return source

    start_marker = '''    if critical_unresolved:
        log_path = prop / "logs" / f"current_universe_{season}_week_{week}.json"
'''
    start = source.find(start_marker)
    if start < 0:
        raise RuntimeError("Current-universe fatal block start not found.")
    end = source.find("\n    player_teams:", start)
    if end < 0:
        raise RuntimeError("Current-universe player_teams anchor not found.")

    replacement = '''    for item in [*critical_unresolved, *unresolved_skipped]:
        print(
            "IDENTITY WARNING: unresolved current-week player excluded; "
            f"names={item.get('player_names', [])} "
            f"espn_ids={item.get('espn_ids', [])} "
            f"team={item.get('team', '')} "
            f"position={item.get('position', '')} "
            f"starter={item.get('depth_starter_flag', 0)} "
            f"reason={item.get('reason', '')}",
            file=sys.stderr,
        )
'''
    source = source[:start] + replacement + source[end:]

    source = replace_once(
        source,
        '        "critical_unresolved_starters": [],\n',
        '        "critical_unresolved_starters": critical_unresolved,\n',
        "current-universe critical list",
    )

    source = replace_once(
        source,
        '        "skipped_unresolved_count": len(unresolved_skipped),\n',
        '''        "skipped_unresolved_count": len(unresolved_skipped),
        "critical_unresolved_count": len(critical_unresolved),
        "total_unresolved_skipped_count": (
            len(critical_unresolved) + len(unresolved_skipped)
        ),
        "unresolved_identity_policy": "skip_and_continue",
''',
        "current-universe unresolved counts",
    )

    source = replace_once(
        source,
        '''            "unresolved_starter_identity_fails": True,
            "noncritical_unresolved_backup_skipped_and_logged": True,
''',
        '''            "unresolved_starter_identity_fails": False,
            "unresolved_identity_skipped_and_logged": True,
            "noncritical_unresolved_backup_skipped_and_logged": True,
''',
        "current-universe rules",
    )

    summary_field = '            "skipped_unresolved": len(unresolved_skipped),\n'
    count = source.count(summary_field)
    if count != 2:
        raise RuntimeError(
            f"Expected two current-universe skipped_unresolved summaries, found {count}."
        )
    source = source.replace(
        summary_field,
        '''            "skipped_unresolved": (
                len(critical_unresolved) + len(unresolved_skipped)
            ),
''',
        2,
    )

    return source


def main() -> int:
    identity_old = read_required(IDENTITY)
    universe_old = read_required(UNIVERSE)

    identity_new = patch_identity(identity_old)
    universe_new = patch_universe(universe_old)

    compile(identity_new, str(IDENTITY), "exec")
    compile(universe_new, str(UNIVERSE), "exec")

    if "Identity validation failed:" in identity_new:
        raise RuntimeError("Fatal identity exception still present.")
    if "Issue 29 unresolved starter identity failure." in universe_new:
        raise RuntimeError("Fatal current-universe identity exception still present.")
    if '"unresolved_starter_identity_fails": False' not in universe_new:
        raise RuntimeError("Current-universe nonfatal rule missing.")

    changed = []
    if identity_new != identity_old:
        atomic_write(IDENTITY, identity_new)
        changed.append(IDENTITY.relative_to(PROP).as_posix())
    if universe_new != universe_old:
        atomic_write(UNIVERSE, universe_new)
        changed.append(UNIVERSE.relative_to(PROP).as_posix())

    print("unresolved_identity_policy=skip_and_continue")
    print("fatal_identity_exits=0")
    print("unresolved_players_excluded_from_projection=true")
    print("fabricated_gsis_allowed=false")
    print(f"changed_files={len(changed)}")
    for path in changed:
        print(f"changed={path}")
    print("UNRESOLVED IDENTITY CONTINUE PATCH: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
