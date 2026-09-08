#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config" / "prop_engine.yaml"
COMMON = HERE / "scripts" / "common.py"
AUDIT = HERE / "scripts" / "validate" / "audit_market_exclusion.py"

REQUIRED_FORBIDDEN_INPUTS = [
    "docs/win/football/nfl/data/historic_data/odds/",
    "docs/win/football/nfl/scripts/00_intake/pull_odds.py",
    "docs/win/football/nfl/scripts/00_intake/pull_opening_odds.py",
    "docs/win/football/nfl/scripts/00_intake/enrich_moneyline.py",
    "docs/win/football/nfl/scripts/00_intake/enrich_spread.py",
    "docs/win/football/nfl/scripts/00_intake/enrich_totals.py",
    "docs/win/football/nfl/scripts/00_intake/pull_market_futures.py",
    "docs/win/football/nfl/data/historic_data/predictions/drat/",
    "docs/win/football/nfl/data/historic_data/predictions/epred/",
    "docs/win/football/nfl/training/",
    "docs/win/football/nfl/01_merge/",
]


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


def read_required(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    return path.read_text(encoding="utf-8-sig")


def patch_config(text: str) -> str:
    parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict):
        raise ValueError("prop_engine.yaml must be a mapping")

    existing = parsed.get("forbidden_input_paths")
    if existing is not None:
        normalized = [str(x).replace("\\", "/") for x in existing]
        if normalized != REQUIRED_FORBIDDEN_INPUTS:
            raise ValueError(
                "Existing forbidden_input_paths does not exactly match Issue 54. "
                f"existing={normalized}"
            )
        return text

    block = (
        "\nforbidden_input_paths:\n"
        + "".join(f"  - {item}\n" for item in REQUIRED_FORBIDDEN_INPUTS)
        + "\n"
    )
    marker = "\nforbidden_features:\n"
    if marker not in text:
        raise ValueError("Unable to find forbidden_features insertion anchor")
    return text.replace(marker, block + "forbidden_features:\n", 1)


def patch_common(text: str) -> str:
    if '"forbidden_input_paths",' not in text:
        marker = '    "forbidden_features",\n'
        if marker not in text:
            raise ValueError("Unable to find common.py required-config anchor")
        text = text.replace(marker, marker + '    "forbidden_input_paths",\n', 1)

    if "normalized_forbidden_inputs" not in text:
        old = """    forbidden = config.get("forbidden_features")

    if not isinstance(forbidden, list) or not forbidden:
        raise ValueError(
            "Config section 'forbidden_features' "
            "must be a non-empty list."
        )

    return config
"""
        new = """    forbidden = config.get("forbidden_features")

    if not isinstance(forbidden, list) or not forbidden:
        raise ValueError(
            "Config section 'forbidden_features' "
            "must be a non-empty list."
        )

    forbidden_inputs = config.get("forbidden_input_paths")

    if not isinstance(forbidden_inputs, list) or not forbidden_inputs:
        raise ValueError(
            "Config section 'forbidden_input_paths' "
            "must be a non-empty list."
        )

    normalized_forbidden_inputs = [
        str(value).strip().replace("\\\\", "/")
        for value in forbidden_inputs
        if str(value).strip()
    ]

    if len(normalized_forbidden_inputs) != len(forbidden_inputs):
        raise ValueError(
            "Config forbidden_input_paths cannot contain blank values."
        )

    if len(set(normalized_forbidden_inputs)) != len(normalized_forbidden_inputs):
        raise ValueError(
            "Config forbidden_input_paths cannot contain duplicates."
        )

    return config
"""
        if old not in text:
            raise ValueError("Unable to find common.py config validation block")
        text = text.replace(old, new, 1)

    if "def reject_forbidden_input_path(" not in text:
        anchor = "\ndef require_columns(\n"
        if anchor not in text:
            raise ValueError("Unable to find common.py insertion anchor")
        helper = r"""

def forbidden_input_paths(
    config: Mapping[str, Any] | None = None,
) -> tuple[tuple[Path, bool, str], ...]:
    active = load_config() if config is None else config
    values = active.get("forbidden_input_paths")

    if not isinstance(values, list) or not values:
        raise ValueError(
            "Config forbidden_input_paths must be a non-empty list."
        )

    rules: list[tuple[Path, bool, str]] = []

    for raw in values:
        reference = str(raw).strip().replace("\\", "/")

        if not reference:
            raise ValueError(
                "Config forbidden_input_paths cannot contain blank values."
            )

        rules.append(
            (
                _resolve_repo_path(reference),
                reference.endswith("/"),
                reference,
            )
        )

    return tuple(rules)


def reject_forbidden_input_path(
    path: str | os.PathLike[str],
    config: Mapping[str, Any] | None = None,
) -> None:
    resolved = _resolve_repo_path(path)

    for forbidden, is_directory, reference in forbidden_input_paths(config):
        blocked = (
            resolved == forbidden
            or (
                is_directory
                and forbidden in resolved.parents
            )
        )

        if blocked:
            raise ValueError(
                "Prop Engine direct input is forbidden by Issue 54: "
                f"{reference} (requested {resolved})"
            )
"""
        text = text.replace(anchor, helper + anchor, 1)

    for function_name in ("read_csv_required", "read_parquet_required"):
        start = text.find(f"def {function_name}(")
        if start < 0:
            raise ValueError(f"Unable to find {function_name}")
        end = text.find("\ndef ", start + 4)
        if end < 0:
            end = len(text)
        block = text[start:end]
        if "reject_forbidden_input_path(resolved)" not in block:
            needle = "    resolved = _resolve_repo_path(path)\n"
            if needle not in block:
                raise ValueError(
                    f"Unable to find resolved-path line inside {function_name}"
                )
            block = block.replace(
                needle,
                needle + "    reject_forbidden_input_path(resolved)\n",
                1,
            )
            text = text[:start] + block + text[end:]

    return text


def patch_audit(text: str) -> str:
    start = text.find("def forbidden_source_references() -> list[str]:")
    if start >= 0:
        end = text.find("\ndef normalize_reference_text(", start)
        if end < 0:
            raise ValueError("Unable to locate end of forbidden_source_references")
        new_func = """def forbidden_source_references(config: dict) -> list[str]:
    values = config.get("forbidden_input_paths")
    if not isinstance(values, list) or not values:
        raise ValueError(
            "Config forbidden_input_paths must be a non-empty list."
        )

    normalized = [
        str(value).strip().replace("\\\\", "/")
        for value in values
        if str(value).strip()
    ]

    if len(normalized) != len(values):
        raise ValueError(
            "Config forbidden_input_paths cannot contain blank values."
        )

    return list(dict.fromkeys(normalized))

"""
        text = text[:start] + new_func + text[end + 1:]

    text = text.replace(
        "deny_refs = forbidden_source_references()\n",
        "deny_refs = forbidden_source_references(config)\n",
    )

    if "FORBIDDEN_INPUT_CONTRACT_RELATIVE_PATH" not in text:
        marker = """OUTPUT_RELATIVE_PATH = (
    "docs/win/football/nfl/prop_engine/evaluation/"
    "market_exclusion_audit.json"
)
"""
        if marker not in text:
            raise ValueError("Unable to locate audit output-path anchor")
        text = text.replace(
            marker,
            marker + """
FORBIDDEN_INPUT_CONTRACT_RELATIVE_PATH = (
    "docs/win/football/nfl/prop_engine/config/prop_engine.yaml"
)
""",
            1,
        )

    if "def configured_path_forbidden_hits(" not in text:
        anchor = "\ndef audit_paths(\n"
        if anchor not in text:
            raise ValueError("Unable to locate audit_paths insertion anchor")
        helper = r"""

def configured_path_forbidden_hits(
    config: dict,
    deny_refs: Iterable[str],
) -> list[str]:
    hits: list[str] = []
    paths = config.get("paths", {})

    if not isinstance(paths, dict):
        raise ValueError("Config paths must be a mapping.")

    normalized_rules = [
        (
            normalize_reference_text(reference),
            str(reference).replace("\\", "/").endswith("/"),
            str(reference),
        )
        for reference in deny_refs
    ]

    for key, value in paths.items():
        candidate = normalize_reference_text(str(value))
        for rule, is_directory, reference in normalized_rules:
            blocked = (
                candidate == rule
                or (
                    is_directory
                    and candidate.startswith(rule)
                )
            )
            if blocked:
                hits.append(
                    f"config.paths.{key} -> {reference}"
                )

    return sorted(set(hits))
"""
        text = text.replace(anchor, helper + anchor, 1)

    old_scan = """    source_hits.extend(
        scan_text_for_source_references(
            text,
            display_path=display,
            deny_refs=deny_refs,
        )
    )
"""
    new_scan = """    if display != FORBIDDEN_INPUT_CONTRACT_RELATIVE_PATH:
        source_hits.extend(
            scan_text_for_source_references(
                text,
                display_path=display,
                deny_refs=deny_refs,
            )
        )
"""
    if old_scan in text:
        text = text.replace(old_scan, new_scan, 1)

    old_init = """    source_hits: list[str] = []
    feature_hits: list[str] = []
"""
    new_init = """    source_hits: list[str] = configured_path_forbidden_hits(
        config,
        deny_refs,
    )
    feature_hits: list[str] = []
"""
    if old_init in text:
        text = text.replace(old_init, new_init, 1)

    return text


def validate_static(config_text: str, common_text: str, audit_text: str) -> None:
    compile(common_text, str(COMMON), "exec")
    compile(audit_text, str(AUDIT), "exec")

    config = yaml.safe_load(config_text)
    if config.get("forbidden_input_paths") != REQUIRED_FORBIDDEN_INPUTS:
        raise ValueError("Forbidden input contract mismatch")

    for marker in (
        '"forbidden_input_paths",',
        "def forbidden_input_paths(",
        "def reject_forbidden_input_path(",
        "reject_forbidden_input_path(resolved)",
    ):
        if marker not in common_text:
            raise ValueError(f"common.py missing marker: {marker}")

    for marker in (
        "def forbidden_source_references(config: dict)",
        "forbidden_source_references(config)",
        "def configured_path_forbidden_hits(",
        "FORBIDDEN_INPUT_CONTRACT_RELATIVE_PATH",
    ):
        if marker not in audit_text:
            raise ValueError(f"audit missing marker: {marker}")


def main() -> int:
    config_old = read_required(CONFIG)
    common_old = read_required(COMMON)
    audit_old = read_required(AUDIT)

    config_new = patch_config(config_old)
    common_new = patch_common(common_old)
    audit_new = patch_audit(audit_old)

    validate_static(config_new, common_new, audit_new)

    changed = []
    for path, old, new in (
        (CONFIG, config_old, config_new),
        (COMMON, common_old, common_new),
        (AUDIT, audit_old, audit_new),
    ):
        if old != new:
            atomic_write(path, new)
            changed.append(path.relative_to(HERE).as_posix())

    print(f"forbidden_input_paths={len(REQUIRED_FORBIDDEN_INPUTS)}")
    print("runtime_shared_read_guard=true")
    print("configured_path_guard=true")
    print("audit_source_reference_guard=true")
    print("market_data_allowed=false")
    print(f"changed_files={len(changed)}")
    for item in changed:
        print(f"changed={item}")
    print("ISSUE 54 FORBIDDEN INPUT PATCH: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
