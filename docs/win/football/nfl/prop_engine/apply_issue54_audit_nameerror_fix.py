#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE / "scripts" / "validate" / "audit_market_exclusion.py"


def atomic_write(path: Path, text: str) -> None:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp = Path(handle.name)
    try:
        with handle:
            handle.write(text)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def main() -> int:
    if not AUDIT.is_file():
        raise FileNotFoundError(f"Missing audit script: {AUDIT}")

    text = AUDIT.read_text(encoding="utf-8-sig")
    original = text

    scan_start = text.find("def scan_one_file(")
    if scan_start < 0:
        raise RuntimeError("scan_one_file() not found")

    scan_end = text.find("\ndef ", scan_start + 4)
    if scan_end < 0:
        raise RuntimeError("scan_one_file() end not found")

    scan_block = text[scan_start:scan_end]

    bad_init = '''    source_hits: list[str] = configured_path_forbidden_hits(
        config,
        deny_refs,
    )
    feature_hits: list[str] = []
'''
    good_init = '''    source_hits: list[str] = []
    feature_hits: list[str] = []
'''

    if bad_init in scan_block:
        scan_block = scan_block.replace(bad_init, good_init, 1)
        text = text[:scan_start] + scan_block + text[scan_end:]

    audit_start = text.find("def audit_paths(")
    if audit_start < 0:
        raise RuntimeError("audit_paths() not found")

    audit_end = text.find("\ndef ", audit_start + 4)
    if audit_end < 0:
        audit_end = len(text)

    audit_block = text[audit_start:audit_end]

    plain_init = '''    source_hits: list[str] = []
    feature_hits: list[str] = []
'''
    required_init = '''    source_hits: list[str] = configured_path_forbidden_hits(
        config,
        deny_refs,
    )
    feature_hits: list[str] = []
'''

    if required_init not in audit_block:
        if plain_init not in audit_block:
            raise RuntimeError(
                "Unable to locate audit_paths source_hits initialization"
            )
        audit_block = audit_block.replace(plain_init, required_init, 1)
        text = text[:audit_start] + audit_block + text[audit_end:]

    compile(text, str(AUDIT), "exec")

    scan_start = text.find("def scan_one_file(")
    scan_end = text.find("\ndef ", scan_start + 4)
    scan_block = text[scan_start:scan_end]

    if "configured_path_forbidden_hits(" in scan_block:
        raise RuntimeError(
            "scan_one_file still references configured_path_forbidden_hits"
        )
    if "source_hits: list[str] = []" not in scan_block:
        raise RuntimeError("scan_one_file source_hits initialization is incorrect")

    audit_start = text.find("def audit_paths(")
    audit_end = text.find("\ndef ", audit_start + 4)
    if audit_end < 0:
        audit_end = len(text)
    audit_block = text[audit_start:audit_end]

    if required_init not in audit_block:
        raise RuntimeError(
            "audit_paths does not enforce configured forbidden paths"
        )

    if text != original:
        atomic_write(AUDIT, text)
        print("changed=scripts/validate/audit_market_exclusion.py")
    else:
        print("changed_files=0")

    py_compile.compile(str(AUDIT), doraise=True)

    print("scan_one_file_config_scope=PASS")
    print("configured_path_guard_location=audit_paths")
    print("ISSUE 54 AUDIT NAMEERROR FIX: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
