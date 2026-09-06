#!/usr/bin/env python3
"""Independent acceptance validator for Prop Engine Issue 28."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import ast
import importlib.util
import json
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common

AUDIT_PATH = ROOT / "scripts" / "validate" / "audit_market_exclusion.py"
WIRE_PATH = ROOT / "scripts" / "validate" / "wire_market_exclusion.py"
OUTPUT_PATH = ROOT / "evaluation" / "market_exclusion_audit.json"
MARKER = "# ISSUE28_MARKET_EXCLUSION_PREFLIGHT"

EXPECTED_OUTPUT_KEYS = [
    "passed",
    "forbidden_source_references",
    "forbidden_feature_columns",
    "files_scanned",
]

EXPECTED_FORBIDDEN_REFS = [
    "docs/win/football/nfl/data/historic_data/odds/",
    "docs/win/football/nfl/scripts/00_intake/pull_odds.py",
    "docs/win/football/nfl/scripts/00_intake/pull_opening_odds.py",
    "docs/win/football/nfl/scripts/00_intake/enrich_moneyline.py",
    "docs/win/football/nfl/scripts/00_intake/enrich_spread.py",
    "docs/win/football/nfl/scripts/00_intake/enrich_totals.py",
    "docs/win/football/nfl/scripts/00_intake/pull_market_futures.py",
    "docs/win/football/nfl/data/historic_data/predictions/drat/",
    "docs/win/football/nfl/data/historic_data/predictions/epred/",
]

SCAN_RELATIVE_ROOTS = [
    "config",
    "scripts",
    "data/historical/features",
    "models",
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize(text: str) -> str:
    return text.replace("\\", "/").casefold()


def independent_files() -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for relative in SCAN_RELATIVE_ROOTS:
        root = ROOT / relative
        if not root.is_dir():
            raise FileNotFoundError(f"Required scan root missing: {root}")
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.name in {".DS_Store", "Thumbs.db"}:
                continue
            if "__pycache__" in path.parts or path.suffix.casefold() in {".pyc", ".pyo", ".tmp", ".lock"}:
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(resolved)
    return files




def parquet_schema_and_metadata(path: Path) -> tuple[list[str], list[str]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        pq = None
    if pq is not None:
        parquet = pq.ParquetFile(path)
        names = list(parquet.schema_arrow.names)
        texts: list[str] = []
        for raw in (parquet.metadata.metadata or {}).values():
            try:
                texts.append(raw.decode("utf-8"))
            except (UnicodeDecodeError, AttributeError):
                pass
        return [str(name) for name in names], texts
    try:
        from fastparquet import ParquetFile  # type: ignore
    except ImportError:
        return [], []
    parquet = ParquetFile(str(path))
    return [str(name) for name in (getattr(parquet, "columns", []) or [])], []


def independent_source_reference_hits(files: list[Path]) -> list[str]:
    hits: list[str] = []
    for path in files:
        if path.suffix.casefold() == ".parquet":
            _, texts = parquet_schema_and_metadata(path)
        else:
            try:
                texts = [path.read_text(encoding="utf-8")]
            except UnicodeDecodeError:
                continue
        joined = normalize("\n".join(texts))
        for reference in EXPECTED_FORBIDDEN_REFS:
            if normalize(reference) in joined:
                hits.append(f"{path.relative_to(ROOT)} -> {reference}")
    return sorted(set(hits))


def collect_feature_lists(payload: Any) -> list[str]:
    names: list[str] = []
    keys = {"feature_columns", "numeric_features", "categorical_features", "feature_names", "features"}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).casefold() in keys and isinstance(child, list):
                    names.extend(str(x) for x in child if isinstance(x, (str, int, float)))
                if isinstance(child, (dict, list)):
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, (dict, list)):
                    walk(child)

    walk(payload)
    return names


def independent_feature_hits(files: list[Path], forbidden_tokens: list[str]) -> list[str]:
    hits: list[str] = []
    for path in files:
        suffix = path.suffix.casefold()
        names: list[str] = []
        if suffix == ".parquet":
            schema_names, _ = parquet_schema_and_metadata(path)
            for name in schema_names:
                lower = str(name).casefold()
                if lower.startswith("audit_") or lower.startswith("target_"):
                    continue
                names.append(str(name))
        elif suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            names.extend(collect_feature_lists(payload))
        elif suffix == ".txt":
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for line in text.splitlines():
                if line.strip().startswith("feature_names="):
                    names.extend(line.split("=", 1)[1].strip().split())

        for name in names:
            lower = name.casefold()
            if any(token in lower for token in forbidden_tokens):
                hits.append(f"{path.relative_to(ROOT)}:{name}")
    return sorted(set(hits))


def top_level_main_source(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            lines = text.splitlines()
            start = node.lineno - 1
            end = int(node.end_lineno or node.lineno)
            return "\n".join(lines[start:end])
    return None


def current_training_entrypoints() -> list[Path]:
    paths: list[Path] = []
    for path in sorted((ROOT / "scripts" / "train").glob("*.py")):
        if top_level_main_source(path) is not None:
            paths.append(path)
    return paths


def current_weekly_inference_entrypoints() -> list[Path]:
    candidates: list[Path] = []
    patterns = ("weekly", "project", "projection", "infer", "inference")
    for path in sorted((ROOT / "scripts").rglob("*.py")):
        relative = path.relative_to(ROOT / "scripts")
        if relative.parts[0] in {"build", "train", "validate", "__pycache__"}:
            continue
        if not any(token in path.name.casefold() for token in patterns):
            continue
        if top_level_main_source(path) is not None:
            candidates.append(path)
    return candidates


def require_wired(paths: list[Path], label: str) -> None:
    bad: list[str] = []
    for path in paths:
        source = top_level_main_source(path) or ""
        if MARKER not in source or "audit_market_exclusion.py" not in source or "--preflight" not in source:
            bad.append(str(path.relative_to(ROOT)))
    if bad:
        raise AssertionError(f"{label} missing Issue 28 preflight: {bad}")


def main() -> int:
    print("CHECK 01: required audit, wiring helper, and output")
    for path in (AUDIT_PATH, WIRE_PATH, OUTPUT_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required Issue 28 artifact missing: {path}")

    print("CHECK 02: execute audit and validate exact required output contract")
    result = subprocess.run([sys.executable, str(AUDIT_PATH)], cwd=common.repo_root(), check=False)
    if result.returncode != 0:
        raise AssertionError(f"Market exclusion audit returned {result.returncode}")
    payload = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    if list(payload.keys()) != EXPECTED_OUTPUT_KEYS:
        raise AssertionError(f"Audit output keys mismatch: {list(payload.keys())}")
    if payload["passed"] is not True:
        raise AssertionError("Audit output passed must be true")
    if payload["forbidden_source_references"] != []:
        raise AssertionError(f"Forbidden source refs found: {payload['forbidden_source_references']}")
    if payload["forbidden_feature_columns"] != []:
        raise AssertionError(f"Forbidden feature columns found: {payload['forbidden_feature_columns']}")
    if not isinstance(payload["files_scanned"], int) or payload["files_scanned"] <= 0:
        raise AssertionError("files_scanned must be a positive integer")

    print("CHECK 03: independent scan of required roots")
    files = independent_files()
    if payload["files_scanned"] != len(files):
        raise AssertionError(f"files_scanned mismatch audit={payload['files_scanned']} independent={len(files)}")
    source_hits = independent_source_reference_hits(files)
    if source_hits:
        raise AssertionError(f"Independent forbidden source references found: {source_hits}")

    config = common.load_config()
    forbidden_tokens = sorted({str(x).strip().casefold() for x in config["forbidden_features"] if str(x).strip()})
    feature_hits = independent_feature_hits(files, forbidden_tokens)
    if feature_hits:
        raise AssertionError(f"Independent forbidden feature columns found: {feature_hits}")

    print("CHECK 04: exact forbidden source deny-list and rejection self-test")
    audit = load_module(AUDIT_PATH, "issue28_audit")
    if audit.forbidden_source_references() != EXPECTED_FORBIDDEN_REFS:
        raise AssertionError("Audit forbidden source reference list does not exactly match Issue 28")
    with tempfile.TemporaryDirectory(prefix="issue28_selftest_") as temp:
        root = Path(temp)
        (root / "bad_source.py").write_text(f'x = "{EXPECTED_FORBIDDEN_REFS[0]}"\n', encoding="utf-8")
        (root / "feature_manifest.json").write_text(
            json.dumps({"feature_columns": ["market_implied_projection"]}), encoding="utf-8"
        )
        selftest = audit.audit_paths([root], config=config, repo=None)
        if selftest["passed"] is not False:
            raise AssertionError("Audit self-test did not fail")
        if not selftest["forbidden_source_references"]:
            raise AssertionError("Audit self-test missed forbidden source reference")
        if not selftest["forbidden_feature_columns"]:
            raise AssertionError("Audit self-test missed forbidden feature column")

    print("CHECK 05: every current training entrypoint calls the validator")
    training = current_training_entrypoints()
    if not training:
        raise AssertionError("No current training entrypoints found")
    require_wired(training, "Training entrypoint")

    print("CHECK 06: weekly inference integration contract")
    weekly = current_weekly_inference_entrypoints()
    if weekly:
        require_wired(weekly, "Weekly inference entrypoint")
        weekly_status = "wired"
    else:
        weekly_status = "not_created_at_issue28_stage"

    print("CHECK 07: summarize independently validated market exclusion")
    print(f"files_scanned={len(files)}")
    print(f"training_entrypoints={len(training)}")
    print(f"weekly_inference_entrypoints={len(weekly)}")
    print(f"weekly_inference_status={weekly_status}")
    print("forbidden_source_references=0")
    print("forbidden_feature_columns=0")
    print("rejection_self_test=true")
    print("ISSUE 28 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
