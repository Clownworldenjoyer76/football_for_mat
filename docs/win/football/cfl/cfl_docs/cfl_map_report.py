#!/usr/bin/env python3
"""
CFL structure-map REPORT aggregator (READ-ONLY).

Reads every <file>.map.json produced by cfl_structure_map.py and writes ONE
consolidated CSV: _map_report.csv. This is the single file to send back.

Compared to _blocks_summary.csv it adds, per block:
  - title_line        the line above the column header (e.g. "OPPONENT PASSING
                      DATA") - this is what disambiguates look-alike tables and
                      identifies blocks whose column header came through blank
  - row_column_counts the per-row cell counts (reveals dual-schema tables)

Input : every *.map.json in IN_DIR  (default below; overridable as argv[1]).
Output: <IN_DIR>/_map_report.csv     (overridable as argv[2]).
Stdlib only. Inputs opened read-only.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# --- Knobs (override via argv, or edit here) -------------------------------
IN_DIR = Path("docs/win/football/cfl/data_dump/structure_map")
OUT_FILE = Path("docs/win/football/cfl/data_dump/structure_map/_map_report.csv")
# ---------------------------------------------------------------------------

FIELDS = [
    "file", "week", "games_played", "block_index",
    "start_line", "end_line", "looks_like_data_table",
    "distinct_team_count", "missing_teams", "logical_rows",
    "modal_columns_non_cfl", "columns_consistent",
    "title_line", "column_header", "row_column_counts", "flags",
]


def block_rows(m: dict):
    for bi, b in enumerate(m.get("blocks", [])):
        headers = b.get("header_lines", [])
        # header_lines is [title, column_header] when two are present; the
        # column header is the last one (matches b["column_header"]).
        title = headers[0]["text"] if len(headers) >= 2 else ""
        yield {
            "file": m.get("file", ""),
            "week": m.get("week", ""),
            "games_played": m.get("games_played", ""),
            "block_index": bi,
            "start_line": b.get("start_line", ""),
            "end_line": b.get("end_line", ""),
            "looks_like_data_table": b.get("looks_like_data_table", ""),
            "distinct_team_count": b.get("distinct_team_count", ""),
            "missing_teams": ",".join(b.get("missing_teams", [])),
            "logical_rows": b.get("logical_rows", ""),
            "modal_columns_non_cfl": b.get("modal_columns_non_cfl", ""),
            "columns_consistent": b.get("columns_consistent", ""),
            "title_line": title,
            "column_header": b.get("column_header", ""),
            "row_column_counts": "|".join(str(c) for c in b.get("row_column_counts", [])),
            "flags": ";".join(b.get("flags", [])),
        }


def main(argv: list[str]) -> int:
    in_dir = Path(argv[1]) if len(argv) > 1 else IN_DIR
    out_file = Path(argv[2]) if len(argv) > 2 else OUT_FILE

    if not in_dir.exists():
        print(f"Input dir not found: {in_dir}")
        return 1

    maps = sorted(in_dir.glob("*.map.json"))
    if not maps:
        print(f"No *.map.json files in {in_dir}")
        return 1

    out_file.parent.mkdir(parents=True, exist_ok=True)
    n_blocks = 0
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for p in maps:
            m = json.loads(p.read_text(encoding="utf-8"))
            for row in block_rows(m):
                writer.writerow(row)
                n_blocks += 1

    print(f"files_read={len(maps)}")
    print(f"blocks_written={n_blocks}")
    print(f"report={out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
