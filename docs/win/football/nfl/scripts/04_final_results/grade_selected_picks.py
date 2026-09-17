#!/usr/bin/env python3
"""Compatibility entry point for the NFL graded-bets reporting grader."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).resolve().with_name("01_nfl_results_grade.py")
    runpy.run_path(str(target), run_name="__main__")
