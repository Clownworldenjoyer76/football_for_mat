#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import os
import tempfile
from pathlib import Path

OLD = "matchup_player_sack_rate_x_opp_dropbacks"
NEW = "matchup_player_sack_rate_x_opp_plays"
OLD_DEN = "matchup_expected_opponent_dropbacks"
NEW_DEN = "matchup_expected_opponent_plays"

SOURCE_FILES = [
    "scripts/build/build_historical_features.py",
    "scripts/project/build_current_features.py",
    "validate_issue17.py",
]
CONFIG_FILE = "config/features/sacks.json"


def prop_root() -> Path:
    return Path(__file__).resolve().parent


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


def patch_feature_formula(text: str, label: str) -> str:
    text = text.replace(OLD, NEW)

    if label == "scripts/build/build_historical_features.py":
        old = (
            'out["matchup_player_sack_rate_x_opp_plays"] = safe_product(\n'
            '        out["player_sack_rate_per_def_play_roll5_mean"],\n'
            '        out["matchup_expected_opponent_dropbacks"],\n'
            '    )'
        )
        new = (
            'out["matchup_player_sack_rate_x_opp_plays"] = safe_product(\n'
            '        out["player_sack_rate_per_def_play_roll5_mean"],\n'
            '        out["matchup_expected_opponent_plays"],\n'
            '    )'
        )
        text = text.replace(old, new, 1)
        text = text.replace(
            '"matchup_player_sack_rate_x_opp_plays": '
            '"player sack_rate roll5 * expected opponent dropbacks"',
            '"matchup_player_sack_rate_x_opp_plays": '
            '"player sack_rate roll5 * expected opponent plays"',
            1,
        )

    elif label == "scripts/project/build_current_features.py":
        old = (
            'if exists("matchup_player_sack_rate_x_opp_plays", '
            '"player_sack_rate_per_def_play_roll5_mean", '
            '"matchup_expected_opponent_dropbacks"):\n'
            '        out["matchup_player_sack_rate_x_opp_plays"] = safe_product(\n'
            '            out["player_sack_rate_per_def_play_roll5_mean"], '
            'out["matchup_expected_opponent_dropbacks"]\n'
            '        )'
        )
        new = (
            'if exists("matchup_player_sack_rate_x_opp_plays", '
            '"player_sack_rate_per_def_play_roll5_mean", '
            '"matchup_expected_opponent_plays"):\n'
            '        out["matchup_player_sack_rate_x_opp_plays"] = safe_product(\n'
            '            out["player_sack_rate_per_def_play_roll5_mean"], '
            'out["matchup_expected_opponent_plays"]\n'
            '        )'
        )
        text = text.replace(old, new, 1)

    elif label == "validate_issue17.py":
        old = (
            'expected["matchup_player_sack_rate_x_opp_plays"] = (\n'
            '        pd.to_numeric(\n'
            '            calc["player_sack_rate_per_def_play_roll5_mean"],\n'
            '            errors="coerce",\n'
            '        )\n'
            '        * expected["matchup_expected_opponent_dropbacks"]'
        )
        new = (
            'expected["matchup_player_sack_rate_x_opp_plays"] = (\n'
            '        pd.to_numeric(\n'
            '            calc["player_sack_rate_per_def_play_roll5_mean"],\n'
            '            errors="coerce",\n'
            '        )\n'
            '        * expected["matchup_expected_opponent_plays"]'
        )
        text = text.replace(old, new, 1)

    return text


def validate_formula_context(text: str, label: str) -> None:
    if OLD in text:
        raise RuntimeError(f"{label}: stale feature name remains")
    if NEW not in text:
        raise RuntimeError(f"{label}: corrected feature name missing")

    if label == "scripts/build/build_historical_features.py":
        anchor = text.find(f'out["{NEW}"]')
        chunk = text[anchor:anchor + 500] if anchor >= 0 else ""
        if NEW_DEN not in chunk or OLD_DEN in chunk:
            raise RuntimeError(f"{label}: corrected formula not found")
        if '"player sack_rate roll5 * expected opponent plays"' not in text:
            raise RuntimeError(f"{label}: manifest formula description is stale")

    elif label == "scripts/project/build_current_features.py":
        anchor = text.find(f'if exists("{NEW}"')
        chunk = text[anchor:anchor + 650] if anchor >= 0 else ""
        if NEW_DEN not in chunk or OLD_DEN in chunk:
            raise RuntimeError(f"{label}: corrected formula not found")

    elif label == "validate_issue17.py":
        anchor = text.find(f'expected["{NEW}"]')
        chunk = text[anchor:anchor + 500] if anchor >= 0 else ""
        if NEW_DEN not in chunk or OLD_DEN in chunk:
            raise RuntimeError(f"{label}: corrected reconstruction not found")


def patch_python(path: Path, label: str) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = patch_feature_formula(original, label)
    ast.parse(updated, filename=str(path))
    validate_formula_context(updated, label)
    if updated == original:
        print(f"UNCHANGED: {label}")
        return False
    atomic_write(path, updated)
    print(f"PATCHED: {label}")
    return True


def patch_config(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = original.replace(OLD, NEW)
    parsed = json.loads(updated)

    numeric = list(parsed.get("numeric_features", []))
    optional = list(parsed.get("optional_features", []))
    required = list(parsed.get("required_features", []))
    all_features = numeric + optional + required

    if OLD in all_features:
        raise RuntimeError("config/features/sacks.json: stale feature remains")
    if NEW not in numeric:
        raise RuntimeError("config/features/sacks.json: corrected feature not in numeric_features")
    if NEW not in optional and NEW not in required:
        raise RuntimeError("config/features/sacks.json: corrected feature not classified")

    if updated == original:
        print(f"UNCHANGED: {CONFIG_FILE}")
        return False
    atomic_write(path, updated)
    print(f"PATCHED: {CONFIG_FILE}")
    return True


def main() -> int:
    prop = prop_root()
    changed = 0
    for rel in SOURCE_FILES:
        path = prop / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        changed += int(patch_python(path, rel))

    config = prop / CONFIG_FILE
    if not config.is_file():
        raise FileNotFoundError(config)
    changed += int(patch_config(config))

    print(
        "ISSUE 56 SACKS DIRECT FEATURE FIX: PASS "
        f"(files_changed={changed}, feature={NEW}, denominator={NEW_DEN})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
