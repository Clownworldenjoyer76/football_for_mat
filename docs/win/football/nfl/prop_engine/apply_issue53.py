#!/usr/bin/env python3
"""
Apply Issue 53 deterministic training controls to the existing Prop Engine.

Changes:
- config/prop_engine.yaml:
    training.random_seed = 76076
    training.deterministic = true
    training.num_threads = 1
- train_opportunity_models.py SEED = 76076
- train_efficiency_models.py SEED = 76076
- train_direct_models.py SEED = 76076

The three trainers already persist the seed in metadata and already pass
deterministic=True and num_threads=1 to LightGBM. This patch makes standalone
training use the same seed as run_training.py's config-driven execution.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

CONFIG = HERE / "config" / "prop_engine.yaml"

TRAINERS = {
    HERE / "scripts" / "train" / "train_opportunity_models.py": 22022,
    HERE / "scripts" / "train" / "train_efficiency_models.py": 23023,
    HERE / "scripts" / "train" / "train_direct_models.py": 24024,
}

NEW_SEED = 76076


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".issue53.tmp",
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


def patch_training_block(text: str) -> tuple[str, bool]:
    lines = text.splitlines()
    start = None
    end = None

    for i, line in enumerate(lines):
        if line.startswith("training:"):
            start = i
            break

    if start is None:
        raise ValueError("prop_engine.yaml has no top-level training section.")

    for i in range(start + 1, len(lines)):
        line = lines[i]
        if line and not line[0].isspace() and re.match(r"^[A-Za-z_][A-Za-z0-9_]*:", line):
            end = i
            break

    if end is None:
        end = len(lines)

    replacement = [
        "training:",
        "  random_seed: 76076",
        "  deterministic: true",
        "  num_threads: 1",
        "",
    ]

    # Avoid accumulating more than one blank line before the next top-level key.
    while end > start + 1 and lines[end - 1] == "":
        end -= 1

    new_lines = lines[:start] + replacement + lines[end:]
    new_text = "\n".join(new_lines).rstrip() + "\n"
    return new_text, new_text != text.replace("\r\n", "\n")


def patch_seed(path: Path, expected_old: int) -> tuple[int, bool]:
    if not path.is_file():
        raise FileNotFoundError(f"Required trainer missing: {path}")

    original = path.read_text(encoding="utf-8-sig")
    matches = list(re.finditer(r"(?m)^SEED\s*=\s*(\d+)\s*$", original))

    if len(matches) != 1:
        raise ValueError(
            f"{path.name}: expected exactly one module-level numeric SEED assignment; "
            f"found {len(matches)}."
        )

    observed = int(matches[0].group(1))
    if observed not in {expected_old, NEW_SEED}:
        raise ValueError(
            f"{path.name}: unexpected existing seed {observed}; "
            f"expected accepted prior seed {expected_old} or Issue 53 seed {NEW_SEED}."
        )

    updated = re.sub(
        r"(?m)^SEED\s*=\s*\d+\s*$",
        f"SEED = {NEW_SEED}",
        original,
        count=1,
    )
    changed = updated != original
    if changed:
        atomic_write(path, updated)
    return observed, changed


def main() -> int:
    if not CONFIG.is_file():
        raise FileNotFoundError(f"Required config missing: {CONFIG}")

    config_original = CONFIG.read_text(encoding="utf-8-sig")
    config_updated, config_changed = patch_training_block(config_original)
    if config_changed:
        atomic_write(CONFIG, config_updated)

    changed_files = []
    if config_changed:
        changed_files.append(str(CONFIG.relative_to(HERE)).replace("\\", "/"))

    for path, old_seed in TRAINERS.items():
        observed, changed = patch_seed(path, old_seed)
        status = "updated" if changed else "already_compliant"
        print(f"{path.name}: old_seed={observed} new_seed={NEW_SEED} status={status}")
        if changed:
            changed_files.append(str(path.relative_to(HERE)).replace("\\", "/"))

    print("config_training.random_seed=76076")
    print("config_training.deterministic=true")
    print("config_training.num_threads=1")
    print(f"changed_files={len(changed_files)}")
    for item in changed_files:
        print(f"changed={item}")
    print("ISSUE 53 PATCH: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
