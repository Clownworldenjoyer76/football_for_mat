#!/usr/bin/env python3
"""Independent acceptance validator for NFL Prop Engine Issue 50."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOC = HERE / "docs" / "FEATURE_DICTIONARY.md"
CANONICAL = HERE / "data" / "historical" / "features" / "feature_manifest.json"

REQUIRED_FIELDS = [
    "feature_name",
    "feature_family",
    "data_type",
    "source_file",
    "source_column_or_formula",
    "lag_rule",
    "positions_used",
    "targets_used",
    "missing_value_rule",
    "leakage_risk",
]

MARKET_TOKENS = [
    "odds", "moneyline", "spread", "total_line", "over_odds", "under_odds",
    "market", "sportsbook", "prop_line", "drat", "epred", "projected_pts",
    "win_probability", "cover_probability",
]


def fail(message: str) -> None:
    raise AssertionError(message)


def unescape_cell(value: str) -> str:
    return value.strip().replace("\\|", "|").replace("\\\\", "\\")


def parse_table(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("| feature_name | feature_family |"):
            start = i
            break
    if start is None:
        fail("Feature dictionary table header not found.")

    header = [unescape_cell(x) for x in lines[start].strip().strip("|").split("|")]
    if header != REQUIRED_FIELDS:
        fail(f"Feature dictionary fields/order mismatch: {header}")

    rows: list[dict[str, str]] = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        # Split only on pipes not escaped with backslash.
        cells = []
        current = []
        escaped = False
        for ch in line.strip().strip("|"):
            if ch == "|" and not escaped:
                cells.append(unescape_cell("".join(current)))
                current = []
                continue
            current.append(ch)
            escaped = (ch == "\\" and not escaped)
            if ch != "\\":
                escaped = False
        cells.append(unescape_cell("".join(current)))
        if len(cells) != len(REQUIRED_FIELDS):
            fail(f"Malformed feature row with {len(cells)} cells: {line[:200]}")
        rows.append(dict(zip(REQUIRED_FIELDS, cells)))
    return rows


def manifest_features(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        fail(f"Invalid feature manifest: {path}")
    names = list(payload.get("numeric_features", [])) + list(payload.get("categorical_features", []))
    return {str(x) for x in names if str(x).strip()}


def main() -> int:
    if not DOC.is_file():
        fail(f"Missing required document: {DOC}")
    if not CANONICAL.is_file():
        fail(f"Missing canonical feature manifest: {CANONICAL}")

    canonical_payload = json.loads(CANONICAL.read_text(encoding="utf-8-sig"))
    canonical = list(canonical_payload.get("feature_columns", []))
    if len(canonical) != 774:
        fail(f"Expected accepted canonical feature count 774; found {len(canonical)}")
    if len(canonical) != len(set(canonical)):
        fail("Canonical feature manifest contains duplicate feature names.")

    rows = parse_table(DOC.read_text(encoding="utf-8"))
    if not rows:
        fail("Feature dictionary has no rows.")

    documented = [row["feature_name"] for row in rows]
    if len(documented) != len(set(documented)):
        dupes = sorted({x for x in documented if documented.count(x) > 1})
        fail(f"Duplicate feature dictionary names: {dupes[:20]}")

    for row in rows:
        empty = [field for field in REQUIRED_FIELDS if not row[field].strip()]
        if empty:
            fail(f"{row.get('feature_name')}: blank required field(s): {empty}")

    # Production features are the canonical feature table plus every persisted
    # LightGBM feature manifest. This catches model-local efficiency features.
    expected = set(canonical)
    model_manifests: list[Path] = []
    model_root = HERE / "models"
    if model_root.is_dir():
        for path in sorted(model_root.rglob("feature_manifest.json")):
            # Exclude data/current feature manifests; model_root confines this.
            model_manifests.append(path)
            expected |= manifest_features(path)

    documented_set = set(documented)
    missing = sorted(expected - documented_set)
    if missing:
        fail(
            "Production feature(s) absent from FEATURE_DICTIONARY.md: "
            + ", ".join(missing[:50])
        )

    extras = sorted(documented_set - expected)
    if extras:
        fail(
            "Dictionary contains feature names absent from canonical/model manifests: "
            + ", ".join(extras[:50])
        )

    # Canonical family/data type consistency.
    family_map = {}
    for family, names in canonical_payload.get("column_families", {}).items():
        for name in names:
            family_map[str(name)] = str(family)
    numeric = set(canonical_payload.get("numeric_features", []))
    categorical = set(canonical_payload.get("categorical_features", []))

    row_map = {row["feature_name"]: row for row in rows}
    for name in canonical:
        row = row_map[name]
        if name in categorical and "categorical" not in row["data_type"]:
            fail(f"{name}: dictionary data_type does not mark canonical categorical feature.")
        if name in numeric and "numeric" not in row["data_type"]:
            fail(f"{name}: dictionary data_type does not mark canonical numeric feature.")
        if name in family_map:
            fam = family_map[name]
            # Player defensive/kicking subfamilies are intentionally more specific.
            actual = row["feature_family"]
            if fam == "player":
                if actual not in {"player", "player_defensive", "player_kicking"}:
                    fail(f"{name}: invalid player subfamily {actual!r}")
            elif actual != fam:
                fail(f"{name}: feature_family {actual!r} != canonical {fam!r}")

    for name in expected:
        low = name.casefold()
        if any(token in low for token in MARKET_TOKENS):
            fail(f"Forbidden market-derived production feature found: {name}")

    eff_expected = {
        "eff_player_prior_rate", "eff_position_prior_rate", "eff_league_prior_rate",
        "eff_shrunk_rate", "eff_player_prior_exposure", "eff_prior_weight",
        "eff_rookie_position_prior_flag",
    }
    eff_present = eff_expected & expected
    if eff_present and not eff_present.issubset(documented_set):
        fail("One or more model-local efficiency features are undocumented.")

    print("document=docs/FEATURE_DICTIONARY.md")
    print(f"required_fields={len(REQUIRED_FIELDS)}")
    print(f"canonical_features={len(canonical)}")
    print(f"model_feature_manifests={len(model_manifests)}")
    print(f"production_feature_union={len(expected)}")
    print(f"documented_features={len(documented_set)}")
    print(f"undocumented_production_features={len(expected - documented_set)}")
    print("all_required_fields_nonblank=true")
    print("canonical_family_types_verified=true")
    print("model_local_efficiency_features_documented=true")
    print("market_features_used=false")
    print("FEATURE DICTIONARY VALIDATION: PASS")
    print("ISSUE 50 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FEATURE DICTIONARY VALIDATION: FAIL - {exc}", file=sys.stderr)
        raise SystemExit(1)
