#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import os
import re
import tempfile
from pathlib import Path

PROP = Path(__file__).resolve().parent
TRAINER = PROP / "scripts/train/train_direct_models.py"
VALIDATOR = PROP / "validate_issue24.py"

EXPECTED_ASSIGNMENTS = {
    "SEED": "24024",
    "MODEL_SELECTION_TRAIN_END": "2023",
    "DEVELOPMENT_VALIDATION_SEASON": "2024",
    "FINAL_TRAIN_END": "2024",
    "UNTOUCHED_TEST_SEASON": "2025",
}

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

EXPECTED_OBJECTIVE = {
    "passing_yards": "regression",
    "passing_tds": "poisson",
    "rushing_yards": "regression",
    "rushing_tds": "poisson",
    "receiving_yards": "regression",
    "receiving_tds": "poisson",
    "kicking_points": "regression",
    "tackles": "poisson",
    "sacks": "poisson",
}


def atomic_write(path: Path, text: str) -> None:
    h = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    )
    tmp = Path(h.name)
    try:
        with h:
            h.write(text)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def literal_assignments(source: str) -> dict[str, object]:
    tree = ast.parse(source, filename=str(TRAINER))
    out: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in EXPECTED_ASSIGNMENTS:
                if isinstance(node.value, ast.Constant):
                    out[target.id] = node.value.value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id in EXPECTED_ASSIGNMENTS:
                if isinstance(node.value, ast.Constant):
                    out[target.id] = node.value.value
    return out


def primary_objective(source: str) -> dict[str, str]:
    tree = ast.parse(source, filename=str(TRAINER))
    for node in tree.body:
        value = None
        name = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            value = node.value
        if name == "PRIMARY_OBJECTIVE" and isinstance(value, ast.Dict):
            result: dict[str, str] = {}
            for k, v in zip(value.keys, value.values):
                if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                    result[str(k.value)] = str(v.value)
            return result
    raise AssertionError("PRIMARY_OBJECTIVE dictionary not found")


def required_markers_from_validator() -> list[str]:
    source = VALIDATOR.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(VALIDATOR))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "required_markers" for t in node.targets):
                if not isinstance(node.value, ast.List):
                    raise AssertionError("validate_issue24.py required_markers is not a literal list")
                markers = []
                for item in node.value.elts:
                    if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
                        raise AssertionError("validate_issue24.py required_markers contains non-string")
                    markers.append(item.value)
                return markers
    raise AssertionError("validate_issue24.py required_markers list not found")


def normalize_policy_assignment_lines(source: str) -> tuple[str, list[str]]:
    lines = source.splitlines(keepends=True)
    changed: list[str] = []

    for name, expected in EXPECTED_ASSIGNMENTS.items():
        matches = []
        pattern = re.compile(rf"^(?P<indent>[ \t]*){re.escape(name)}(?:\s*:\s*[^=]+)?\s*=\s*.*?(?P<eol>\r?\n)?$")
        for i, line in enumerate(lines):
            m = pattern.match(line)
            if m:
                matches.append((i, m))

        if len(matches) != 1:
            raise AssertionError(
                f"{name}: expected exactly one top-level assignment line; found {len(matches)}"
            )

        i, m = matches[0]
        eol = m.group("eol") or ""
        desired = f"{name} = {expected}{eol}"
        if lines[i] != desired:
            lines[i] = desired
            changed.append(name)

    updated = "".join(lines)
    ast.parse(updated, filename=str(TRAINER))
    return updated, changed


def validate_semantics(source: str) -> None:
    assignments = literal_assignments(source)
    for name, expected_text in EXPECTED_ASSIGNMENTS.items():
        expected = int(expected_text)
        actual = assignments.get(name)
        if actual != expected:
            raise AssertionError(f"{name}: expected {expected}, found {actual!r}")

    objective = primary_objective(source)
    if objective != EXPECTED_OBJECTIVE:
        raise AssertionError(
            f"PRIMARY_OBJECTIVE mismatch: expected={EXPECTED_OBJECTIVE} actual={objective}"
        )


def validate_all_static_markers(source: str) -> None:
    markers = required_markers_from_validator()
    missing = [marker for marker in markers if marker not in source]
    if missing:
        raise AssertionError(
            "Issue 24 static marker preflight failed; missing: " + repr(missing)
        )
    print(f"PASS: all {len(markers)} Issue 24 static source markers are present")


def validate_model_seed_metadata() -> None:
    failures = []
    for target in TARGETS:
        path = PROP / "models" / target / "metadata.json"
        if not path.is_file():
            failures.append(f"{target}: metadata missing")
            continue
        with path.open("r", encoding="utf-8-sig") as h:
            meta = json.load(h)

        random_seed = meta.get("random_seed")
        param_seed = meta.get("primary", {}).get("params", {}).get("seed")
        ff_seed = meta.get("primary", {}).get("params", {}).get("feature_fraction_seed")
        bag_seed = meta.get("primary", {}).get("params", {}).get("bagging_seed")
        data_seed = meta.get("primary", {}).get("params", {}).get("data_random_seed")

        values = {
            "random_seed": random_seed,
            "params.seed": param_seed,
            "params.feature_fraction_seed": ff_seed,
            "params.bagging_seed": bag_seed,
            "params.data_random_seed": data_seed,
        }
        bad = {k: v for k, v in values.items() if type(v) is not int or v != 24024}
        if bad:
            failures.append(f"{target}: {bad}")

    if failures:
        raise AssertionError(
            "Model seed metadata preflight failed. No retraining will be started. "
            + " | ".join(failures)
        )
    print("PASS: all nine direct-model metadata/LightGBM seed fields equal 24024")


def main() -> int:
    original = TRAINER.read_text(encoding="utf-8")
    updated, changed = normalize_policy_assignment_lines(original)
    validate_semantics(updated)
    validate_all_static_markers(updated)

    if updated != original:
        atomic_write(TRAINER, updated)
        print("PATCHED: scripts/train/train_direct_models.py")
        print("normalized=" + ",".join(changed))
    else:
        print("UNCHANGED: scripts/train/train_direct_models.py")

    # Re-read after atomic write and repeat every cheap check.
    final_source = TRAINER.read_text(encoding="utf-8")
    ast.parse(final_source, filename=str(TRAINER))
    validate_semantics(final_source)
    validate_all_static_markers(final_source)
    validate_model_seed_metadata()

    print("ISSUE 56 ISSUE24 STATIC CONTRACT PREFLIGHT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
