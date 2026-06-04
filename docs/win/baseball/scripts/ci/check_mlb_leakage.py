#!/usr/bin/env python3
# docs/win/baseball/scripts/ci/check_mlb_leakage.py

from pathlib import Path
import sys

FORBIDDEN_TERMS = [
    "05_final_scores",
    "final_scores",
    "graded",
    "bet_result",
    "final_home_score",
    "final_away_score",
    "final_total",
]

PRE_SELECTION_FILES = [
    Path("docs/win/baseball/scripts/00_intake/transform_baseball.py"),
    Path("docs/win/baseball/scripts/01_merge/merge_intake.py"),
    Path("docs/win/baseball/scripts/03_edges/compute_edges.py"),
    Path("docs/win/baseball/scripts/03_edges/compute_ev_kelly.py"),
    Path("docs/win/baseball/scripts/04_select/baseball_select_bets.py"),
    Path(".github/workflows/odds_mlb.yml"),
    Path(".github/workflows/intake_mlb.yml"),
    Path(".github/workflows/pipeline_mlb.yml"),
]

ALLOWED_LINE_MARKERS = {
    "LEAKAGE_GUARD_ALLOWED_REFERENCE",
}

PIPELINE_WORKFLOW = Path(".github/workflows/pipeline_mlb.yml")
SELECT_SCRIPT = "docs/win/baseball/scripts/04_select/baseball_select_bets.py"
FINAL_SCORE_SCRIPT_DIR = "docs/win/baseball/scripts/05_final_scores/"


def line_allowed(line: str) -> bool:
    return any(marker in line for marker in ALLOWED_LINE_MARKERS)


def scan_for_forbidden_terms() -> list[str]:
    failures = []

    for path in PRE_SELECTION_FILES:
        if not path.exists():
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if line_allowed(line):
                continue

            lower_line = line.lower()
            for term in FORBIDDEN_TERMS:
                if term.lower() in lower_line:
                    failures.append(f"{path}:{line_no}: forbidden pre-selection reference '{term}' -> {line.strip()}")

    return failures


def check_pipeline_order() -> list[str]:
    failures = []

    if not PIPELINE_WORKFLOW.exists():
        return failures

    text = PIPELINE_WORKFLOW.read_text(encoding="utf-8", errors="replace")
    select_idx = text.find(SELECT_SCRIPT)

    if select_idx < 0:
        failures.append(f"{PIPELINE_WORKFLOW}: missing required baseball_select_bets.py step")
        return failures

    first_final_idx = text.find(FINAL_SCORE_SCRIPT_DIR)

    if first_final_idx >= 0 and first_final_idx < select_idx:
        failures.append(
            f"{PIPELINE_WORKFLOW}: a 05_final_scores script appears before baseball_select_bets.py"
        )

    return failures


def main() -> int:
    failures = []
    failures.extend(scan_for_forbidden_terms())
    failures.extend(check_pipeline_order())

    if failures:
        print("MLB leakage CI check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("MLB leakage CI check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
