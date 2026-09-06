#!/usr/bin/env python3
"""Audit NFL Prop Engine artifacts for sportsbook/market contamination.

Scans the required Issue 28 roots for:
- forbidden source-path references;
- forbidden model feature columns in manifests, parquet schemas, and LightGBM models.

The audit always writes evaluation/market_exclusion_audit.json when run against
its production roots. A failed audit returns exit code 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import argparse
import json
import os
import tempfile
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common


SCAN_RELATIVE_ROOTS = [
    "docs/win/football/nfl/prop_engine/config/",
    "docs/win/football/nfl/prop_engine/scripts/",
    "docs/win/football/nfl/prop_engine/data/historical/features/",
    "docs/win/football/nfl/prop_engine/models/",
]

OUTPUT_RELATIVE_PATH = (
    "docs/win/football/nfl/prop_engine/evaluation/"
    "market_exclusion_audit.json"
)

FEATURE_LIST_KEYS = {
    "feature_columns",
    "numeric_features",
    "categorical_features",
    "feature_names",
    "features",
}

SKIP_SUFFIXES = {".pyc", ".pyo", ".tmp", ".lock"}
SKIP_NAMES = {".DS_Store", "Thumbs.db"}


def forbidden_source_references() -> list[str]:
    """Return the exact Issue 28 deny-list without self-matching its source."""
    nfl = "docs/win/football/nfl/"
    historic = nfl + "data/historic_data/"
    intake = nfl + "scripts/00_intake/"
    predictions = historic + "predictions/"

    return [
        historic + "odds/",
        intake + "pull_" + "odds.py",
        intake + "pull_opening_" + "odds.py",
        intake + "enrich_" + "moneyline.py",
        intake + "enrich_" + "spread.py",
        intake + "enrich_" + "totals.py",
        intake + "pull_market_" + "futures.py",
        predictions + "drat/",
        predictions + "epred/",
    ]


def normalize_reference_text(value: str) -> str:
    return value.replace("\\", "/").casefold()


def forbidden_feature_tokens(config: dict) -> list[str]:
    tokens: list[str] = []
    for value in config.get("forbidden_features", []):
        text = str(value).strip().casefold()
        if text:
            tokens.append(text)
    if not tokens:
        raise ValueError("Config forbidden_features must be non-empty.")
    return sorted(set(tokens))


def is_forbidden_feature_name(name: Any, tokens: Iterable[str]) -> bool:
    text = str(name).strip().casefold()
    if not text:
        return False
    return any(token in text for token in tokens)


def should_skip_file(path: Path) -> bool:
    if path.name in SKIP_NAMES:
        return True
    if path.suffix.casefold() in SKIP_SUFFIXES:
        return True
    if "__pycache__" in path.parts:
        return True
    return False


def iter_scan_files(roots: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Required Issue 28 scan root does not exist: {resolved}")
        for path in sorted(resolved.rglob("*")):
            if not path.is_file() or should_skip_file(path):
                continue
            real = path.resolve()
            if real not in seen:
                seen.add(real)
                files.append(real)
    return files


def scan_text_for_source_references(
    text: str,
    *,
    display_path: str,
    deny_refs: Iterable[str],
) -> list[str]:
    normalized = normalize_reference_text(text)
    hits: list[str] = []
    for reference in deny_refs:
        if normalize_reference_text(reference) in normalized:
            hits.append(f"{display_path} -> {reference}")
    return hits


def collect_json_feature_names(payload: Any) -> list[str]:
    names: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key).strip().casefold()
                if key_text in FEATURE_LIST_KEYS and isinstance(child, list):
                    names.extend(str(item) for item in child if isinstance(item, (str, int, float)))
                if isinstance(child, (dict, list)):
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, (dict, list)):
                    walk(child)

    walk(payload)
    return names


def lightgbm_feature_names(text: str) -> list[str]:
    names: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("feature_names="):
            payload = stripped.split("=", 1)[1].strip()
            if payload:
                names.extend(payload.split())
    return names




def parquet_schema_and_metadata(path: Path) -> tuple[list[str], list[str]]:
    """Inspect parquet metadata without loading row data when an engine is available."""
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        pq = None

    if pq is not None:
        parquet = pq.ParquetFile(path)
        names = list(parquet.schema_arrow.names)
        metadata_text: list[str] = []
        metadata = parquet.metadata.metadata or {}
        for raw_value in metadata.values():
            try:
                metadata_text.append(raw_value.decode("utf-8"))
            except (UnicodeDecodeError, AttributeError):
                continue
        return names, metadata_text

    try:
        from fastparquet import ParquetFile  # type: ignore
    except ImportError:
        return [], []

    parquet = ParquetFile(str(path))
    names = list(getattr(parquet, "columns", []) or [])
    return [str(name) for name in names], []


def relative_display(path: Path, repo: Path | None) -> str:
    if repo is None:
        return str(path)
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return str(path)


def scan_one_file(
    path: Path,
    *,
    repo: Path | None,
    deny_refs: list[str],
    feature_tokens: list[str],
) -> tuple[list[str], list[str]]:
    source_hits: list[str] = []
    feature_hits: list[str] = []
    display = relative_display(path, repo)
    suffix = path.suffix.casefold()

    if suffix == ".parquet":
        schema_names, metadata_text = parquet_schema_and_metadata(path)
        for name in schema_names:
            lowered = str(name).casefold()
            if lowered.startswith("audit_") or lowered.startswith("target_"):
                continue
            if is_forbidden_feature_name(name, feature_tokens):
                feature_hits.append(f"{display}:{name}")

        for text in metadata_text:
            source_hits.extend(
                scan_text_for_source_references(
                    text,
                    display_path=display,
                    deny_refs=deny_refs,
                )
            )
        return source_hits, feature_hits

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            # Non-text files have no inspectable source reference contract here.
            return source_hits, feature_hits

    source_hits.extend(
        scan_text_for_source_references(
            text,
            display_path=display,
            deny_refs=deny_refs,
        )
    )

    if suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON inside Issue 28 scan roots: {display}: {exc}") from exc
        for name in collect_json_feature_names(payload):
            if is_forbidden_feature_name(name, feature_tokens):
                feature_hits.append(f"{display}:{name}")

    if suffix == ".txt":
        for name in lightgbm_feature_names(text):
            if is_forbidden_feature_name(name, feature_tokens):
                feature_hits.append(f"{display}:{name}")

    return source_hits, feature_hits


def audit_paths(
    roots: Iterable[Path],
    *,
    config: dict,
    repo: Path | None = None,
) -> dict:
    deny_refs = forbidden_source_references()
    feature_tokens = forbidden_feature_tokens(config)
    files = iter_scan_files(roots)

    source_hits: list[str] = []
    feature_hits: list[str] = []
    for path in files:
        one_source, one_feature = scan_one_file(
            path,
            repo=repo,
            deny_refs=deny_refs,
            feature_tokens=feature_tokens,
        )
        source_hits.extend(one_source)
        feature_hits.extend(one_feature)

    source_hits = sorted(set(source_hits))
    feature_hits = sorted(set(feature_hits))

    return {
        "passed": not source_hits and not feature_hits,
        "forbidden_source_references": source_hits,
        "forbidden_feature_columns": feature_hits,
        "files_scanned": int(len(files)),
    }


def write_json_atomic(payload: dict, path: Path) -> None:
    root = common.prop_root().resolve()
    destination = path.resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Issue 28 output must remain under Prop Engine: {destination}") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=False, ensure_ascii=False)
            handle.write("\n")
        os.replace(temp_path, destination)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def run_production_audit(*, write_output: bool = True) -> dict:
    config = common.load_config()
    repo = common.repo_root().resolve()
    roots = [(repo / relative).resolve() for relative in SCAN_RELATIVE_ROOTS]
    payload = audit_paths(roots, config=config, repo=repo)
    if write_output:
        write_json_atomic(payload, repo / OUTPUT_RELATIVE_PATH)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Prop Engine market exclusion.")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run the same production audit as a training/inference preflight.",
    )
    args = parser.parse_args(argv)
    _ = args.preflight

    payload = run_production_audit(write_output=True)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    print("MARKET EXCLUSION AUDIT: PASS" if payload["passed"] else "MARKET EXCLUSION AUDIT: FAIL")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
