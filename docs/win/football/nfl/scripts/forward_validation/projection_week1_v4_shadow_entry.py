#!/usr/bin/env python3
"""Entry point that imports the legacy Week 1 helper through normal import semantics."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import projection_week1_v4_shadow as shadow


def load_week1_helper_fixed():
    helper_dir = shadow.nfl_root() / "scripts/01_merge"
    if str(helper_dir) not in sys.path:
        sys.path.insert(0, str(helper_dir))
    return importlib.import_module("projection_week1")


shadow.load_week1_helper = load_week1_helper_fixed

if __name__ == "__main__":
    raise SystemExit(shadow.main())
