#!/usr/bin/env python3
from __future__ import annotations
import os, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE / "scripts" / "validate" / "audit_market_exclusion.py"

def atomic_write(path: Path, text: str) -> None:
    h = tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n",
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False)
    temp = Path(h.name)
    try:
        with h:
            h.write(text)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()

def main() -> int:
    if not AUDIT.is_file():
        raise FileNotFoundError(AUDIT)
    text = AUDIT.read_text(encoding="utf-8-sig")
    original = text
    old_sig = "def forbidden_source_references(config: dict) -> list[str]:"
    new_sig = "def forbidden_source_references(config: dict | None = None) -> list[str]:"
    if old_sig in text:
        text = text.replace(old_sig, new_sig, 1)
    if new_sig not in text:
        raise RuntimeError("forbidden_source_references signature not found")

    start = text.find(new_sig)
    end = text.find("\ndef ", start + len(new_sig))
    if end < 0:
        end = len(text)
    block = text[start:end]

    if "active = common.load_config() if config is None else config" not in block:
        needle = '    values = config.get("forbidden_input_paths")\n'
        replacement = (
            '    active = common.load_config() if config is None else config\n'
            '    values = active.get("forbidden_input_paths")\n'
        )
        if needle not in block:
            raise RuntimeError("forbidden_input_paths lookup not found")
        block = block.replace(needle, replacement, 1)
        text = text[:start] + block + text[end:]

    compile(text, str(AUDIT), "exec")
    if text != original:
        atomic_write(AUDIT, text)
        print("changed=scripts/validate/audit_market_exclusion.py")
    else:
        print("changed_files=0")
    print("forbidden_source_references_backward_compatible=true")
    print("config_driven_forbidden_inputs=true")
    print("ISSUE 56 VALIDATOR COMPATIBILITY PATCH: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
