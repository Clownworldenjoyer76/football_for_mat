#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import os
import re
import tempfile
from pathlib import Path

EXPECTED_SEED = 24024
TARGETS = [
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
    "kicking_points",
    "tackles",
    "sacks",
]

PROP = Path(__file__).resolve().parent
TRAINER = PROP / "scripts/train/train_direct_models.py"


def atomic_write(path: Path, text: str) -> None:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    )
    temp = Path(handle.name)
    try:
        with handle:
            handle.write(text)
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def trainer_seed(text: str) -> int | None:
    tree = ast.parse(text, filename=str(TRAINER))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "SEED":
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
                        return int(node.value.value)
    return None


def metadata_seed_contract_is_seed(text: str) -> bool:
    pattern = re.compile(r'"random_seed"\s*:\s*SEED\s*,', re.MULTILINE)
    return bool(pattern.search(text))


def patch_trainer() -> bool:
    original = TRAINER.read_text(encoding="utf-8")
    updated = original

    seed = trainer_seed(updated)
    if seed is None:
        raise RuntimeError("Could not resolve integer SEED assignment in train_direct_models.py")
    if seed != EXPECTED_SEED:
        updated, n = re.subn(
            r"(?m)^SEED\s*=\s*\d+\s*$",
            f"SEED = {EXPECTED_SEED}",
            updated,
            count=1,
        )
        if n != 1:
            raise RuntimeError("Could not patch trainer SEED assignment")

    if not metadata_seed_contract_is_seed(updated):
        updated, n = re.subn(
            r'("random_seed"\s*:\s*)([^,\n]+(?:\n\s*[^,\n]+)?)(\s*,)',
            r'\1SEED\3',
            updated,
            count=1,
        )
        if n != 1:
            raise RuntimeError("Could not patch metadata random_seed assignment to SEED")

    ast.parse(updated, filename=str(TRAINER))
    if trainer_seed(updated) != EXPECTED_SEED:
        raise RuntimeError("Trainer SEED contract still incorrect after patch")
    if not metadata_seed_contract_is_seed(updated):
        raise RuntimeError("Metadata random_seed is not sourced from SEED after patch")

    if updated == original:
        print("UNCHANGED: scripts/train/train_direct_models.py")
        return False

    atomic_write(TRAINER, updated)
    print("PATCHED: scripts/train/train_direct_models.py")
    return True


def inspect_metadata() -> tuple[bool, list[dict]]:
    rows = []
    all_ok = True
    for target in TARGETS:
        path = PROP / "models" / target / "metadata.json"
        if not path.is_file():
            rows.append({"target": target, "random_seed": None, "status": "missing"})
            all_ok = False
            continue
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        value = payload.get("random_seed")
        primary_params = payload.get("primary", {}).get("params", {})
        param_seed = primary_params.get("seed")
        ok = (
            type(value) is int
            and value == EXPECTED_SEED
            and type(param_seed) is int
            and param_seed == EXPECTED_SEED
        )
        rows.append({
            "target": target,
            "random_seed": value,
            "random_seed_type": type(value).__name__,
            "primary_param_seed": param_seed,
            "ok": ok,
        })
        all_ok = all_ok and ok

    print("DIRECT MODEL SEED METADATA:")
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    return all_ok, rows


def main() -> int:
    changed = patch_trainer()
    metadata_ok, _ = inspect_metadata()
    print(
        "ISSUE 56 ISSUE24 SEED CONTRACT: "
        f"trainer_changed={str(changed).lower()} metadata_ok={str(metadata_ok).lower()}"
    )
    # Exit 10 tells the PowerShell runner that deterministic retraining is required.
    return 0 if metadata_ok and not changed else 10


if __name__ == "__main__":
    raise SystemExit(main())
