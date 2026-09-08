#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOC = HERE / "IMPLEMENTATION_SEQUENCE.md"

REPLACEMENTS = {
    "`train/select_architecture.py`": "`train/select_model_architecture.py`",
    "`train/build_model_report.py`": "`report/build_model_report.py`",
}


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
    if not DOC.is_file():
        raise FileNotFoundError(f"Missing Issue 55 document: {DOC}")

    text = DOC.read_text(encoding="utf-8-sig")
    original = text

    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)

    required = (
        "`train/select_model_architecture.py`",
        "`report/build_model_report.py`",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise ValueError(
            f"IMPLEMENTATION_SEQUENCE.md missing corrected training path(s): {missing}"
        )

    forbidden = (
        "`train/select_architecture.py`",
        "`train/build_model_report.py`",
    )
    remaining = [marker for marker in forbidden if marker in text]
    if remaining:
        raise ValueError(
            f"IMPLEMENTATION_SEQUENCE.md still contains wrong training path(s): {remaining}"
        )

    if text != original:
        atomic_write(DOC, text)
        print("changed=IMPLEMENTATION_SEQUENCE.md")
    else:
        print("changed_files=0")

    print("training_architecture_path=scripts/train/select_model_architecture.py")
    print("model_report_path=scripts/report/build_model_report.py")
    print("ISSUE 55 TRAINING PATH FIX: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
