#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any


def completed_game_keys(
    frame: Any,
    path: Path,
    done: Any,
) -> set[str]:
    if not done.any():
        return set()

    if "game_id" in frame.columns:
        ids = (
            frame.loc[done, "game_id"]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )
        return {
            value
            for value in ids
            if value
        }

    return {
        f"{path.name}:{index}"
        for index in frame.index[done].tolist()
    }
