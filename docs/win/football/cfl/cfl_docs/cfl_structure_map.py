#!/usr/bin/env python3
"""
CFL Game-Analysis TXT - structure mapper (READ-ONLY diagnostic).

What it does
------------
Maps the raw section structure of every raw TXT, with no value parsing and no
writes to the parsed-output tree. For each file it locates "team-row blocks"
(runs of lines that begin with a team code: BC..CFL) purely from structure, and
reports, per block: line range, the team codes present (and any missing), the
verbatim header line(s) directly above it, and an approximate per-row
column-count signature.

It deliberately does NOT anchor on section names or numbers, and does NOT try to
name the tables - that naming is exactly what we want to redesign from this map.

Known limitation (by design)
-----------------------------
It maps team-ROW tables only. Tables laid out with teams as COLUMNS (e.g. the
penalties-by-category table, where rows are penalty types and the team codes run
across the top) will NOT appear as blocks. Their absence is itself a signal that
they need separate handling in the rebuild.

Inputs : every *.txt in INPUT_DIR (default below; overridable as argv[1]).
Outputs: written ONLY to OUTPUT_DIR (default below; overridable as argv[2]):
           - <stem>.map.json     full per-file map
           - _blocks_summary.csv  one row per block, all files (the diff view)
         plus a compact summary to stdout.

Stdlib only. Input files are opened read-only.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

# --- Knobs (override via argv, or edit here) -------------------------------
INPUT_DIR = Path("docs/win/football/cfl/data_dump/raw_txt")
OUTPUT_DIR = Path("docs/win/football/cfl/data_dump/structure_map")
# ---------------------------------------------------------------------------

TEAM_CODES = ["BC", "CGY", "EDM", "HAM", "MTL", "OTT", "SSK", "TOR", "WPG", "CFL"]
TEAM_RE = re.compile(r"^(BC|CGY|EDM|HAM|MTL|OTT|SSK|TOR|WPG|CFL)\b")
NUM_RE = re.compile(
    r"#DIV/0!|###|T\d+|\([-+]?\d+(?:,\d{3})*(?:\.\d+)?\)|[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?"
)
ALPHA_RE = re.compile(r"[A-Za-z]{2,}")
CONT_RE = re.compile(r"^(?:[(\-+]?\d|T\d|###|#DIV)")
SECTION_HDR_RE = re.compile(r"^\d+[a-z]?\.\s")   # "3. ", "4a. ", "14c. "
PAGEBREAK_RE = re.compile(r"^\d{4}\s+CFL\b")      # "2023 CFL GAME ANALYSIS"

DATA_TABLE_MIN_TEAMS = 5  # >= this many distinct teams => looks like a real table


def classify(stripped: str) -> tuple[str, dict]:
    """Classify one stripped line: 'blank' | 'team' | 'cont' | 'hard'."""
    if not stripped:
        return "blank", {}

    m = TEAM_RE.match(stripped)
    if m:
        body = stripped[m.end():]
        n_num = len(NUM_RE.findall(body))
        n_alpha = len(ALPHA_RE.findall(body))
        # A data row is number-dominated. Lines such as
        # "CFL FIELD GOALS - DATA 2017 TO 2023" begin with a code but are
        # headers -> classify as 'hard', not a data row. The >=1 (not >=2)
        # floor lets a wrapped row whose first physical line carries a single
        # value (e.g. "BC -4") still register as a row start.
        if n_num >= 1 and n_num >= n_alpha:
            return "team", {"code": m.group(1), "n_num": n_num, "n_alpha": n_alpha}
        return "hard", {}

    # Numbered section headers ("3. TURNOVER ANALYSIS") and the repeating
    # page-break line ("2023 CFL GAME ANALYSIS") begin with digits but must end
    # a block, not be glued to it as a continuation.
    if SECTION_HDR_RE.match(stripped) or PAGEBREAK_RE.match(stripped):
        return "hard", {}

    if CONT_RE.match(stripped):
        return "cont", {}

    return "hard", {}


def row_columns(glued: str) -> int:
    """Approx cell count for a glued logical row (leading team code stripped)."""
    m = TEAM_RE.match(glued)
    body = glued[m.end():] if m else glued
    return len(NUM_RE.findall(body))


def glue_row(lines: list[str], row: list[int]) -> str:
    _code, a, b = row
    return " ".join(lines[k].strip() for k in range(a, b))


def preceding_nonblank(lines: list[str], start_line: int, k: int = 2) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    p = start_line - 1
    while p >= 0 and len(out) < k:
        s = lines[p].strip()
        if s:
            out.append((p, s))
        p -= 1
    return list(reversed(out))


def build_block(lines: list[str], start_line: int, rows: list[list[int]]) -> dict:
    team_seq = [r[0] for r in rows]

    distinct: list[str] = []
    for c in team_seq:
        if c not in distinct:
            distinct.append(c)

    missing = [c for c in TEAM_CODES if c not in distinct]
    col_counts = [row_columns(glue_row(lines, r)) for r in rows]
    wrapped = any((r[2] - r[1]) > 1 for r in rows)
    end_line = (rows[-1][2] - 1) if rows else start_line

    noncfl = [c for r, c in zip(rows, col_counts) if r[0] != "CFL"]
    if noncfl:
        modal = max(set(noncfl), key=noncfl.count)
        consistent = len(set(noncfl)) <= 1
    else:
        modal = col_counts[0] if col_counts else 0
        consistent = True

    headers = preceding_nonblank(lines, start_line, k=2)

    flags: list[str] = []
    looks_like_table = len(distinct) >= DATA_TABLE_MIN_TEAMS
    if not looks_like_table:
        flags.append("FEW_TEAMS")
    if len(team_seq) != len(distinct):
        flags.append("DUPLICATE_TEAM")
    if missing and looks_like_table:
        flags.append("MISSING:" + ",".join(missing))
    if not consistent:
        flags.append("INCONSISTENT_COLS")
    if wrapped:
        flags.append("WRAPPED")

    return {
        "start_line": start_line + 1,  # 1-indexed for humans
        "end_line": end_line + 1,
        "header_lines": [{"line": p + 1, "text": t} for p, t in headers],
        "column_header": headers[-1][1] if headers else "",
        "team_sequence": team_seq,
        "distinct_team_count": len(distinct),
        "missing_teams": missing,
        "logical_rows": len(rows),
        "row_column_counts": col_counts,
        "modal_columns_non_cfl": modal,
        "columns_consistent": consistent,
        "looks_like_data_table": looks_like_table,
        "flags": flags,
    }


def find_blocks(lines: list[str]) -> list[dict]:
    cls = [classify(s.strip()) for s in lines]  # index-aligned with lines
    blocks: list[dict] = []
    n = len(lines)
    i = 0

    while i < n:
        kind, _info = cls[i]
        if kind != "team":
            i += 1
            continue

        start = i
        rows: list[list[int]] = []
        cur_row: list[int] | None = None
        blank_run = 0
        j = i

        while j < n:
            jkind, jinfo = cls[j]
            if jkind == "team":
                if cur_row is not None:
                    rows.append(cur_row)
                cur_row = [jinfo["code"], j, j + 1]
                blank_run = 0
            elif jkind == "cont":
                if cur_row is None:
                    break
                cur_row[2] = j + 1
                blank_run = 0
            elif jkind == "blank":
                blank_run += 1
                if blank_run >= 2:
                    break
            else:  # hard
                break
            j += 1

        if cur_row is not None:
            rows.append(cur_row)

        blocks.append(build_block(lines, start, rows))
        i = rows[-1][2] if rows else i + 1

    return blocks


def map_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    blocks = find_blocks(lines)

    trends = next(
        (i + 1 for i, s in enumerate(lines) if "CFL-WIDE FOOTBALL TRENDS" in s), None
    )
    page_headers = [
        i + 1 for i, s in enumerate(lines) if re.search(r"GAME ANALYSIS AS OF\b", s)
    ]
    wk = re.search(r"WK #:\s*(\d+)", text)
    gp = re.search(r"GMS PLAYED:\s*(\d+)", text)
    gm = re.search(r"TO GAME #:\s*(\d+)", text) or re.search(r"TO GM #:\s*(\d+)", text)
    data_blocks = [b for b in blocks if b["looks_like_data_table"]]

    return {
        "file": path.name,
        "total_lines": len(lines),
        "week": wk.group(1) if wk else "",
        "games_played": gp.group(1) if gp else "",
        "to_game": gm.group(1) if gm else "",
        "trends_page_line": trends,
        "page_header_lines": page_headers,
        "block_count": len(blocks),
        "data_table_block_count": len(data_blocks),
        "blocks": blocks,
    }


def write_outputs(maps: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    for m in maps:
        stem = Path(m["file"]).stem
        (out_dir / f"{stem}.map.json").write_text(
            json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    summary_path = out_dir / "_blocks_summary.csv"
    fields = [
        "file", "week", "games_played", "block_index", "start_line", "end_line",
        "looks_like_data_table", "distinct_team_count", "missing_teams",
        "logical_rows", "modal_columns_non_cfl", "columns_consistent",
        "column_header", "flags",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for m in maps:
            for bi, b in enumerate(m["blocks"]):
                writer.writerow({
                    "file": m["file"],
                    "week": m["week"],
                    "games_played": m["games_played"],
                    "block_index": bi,
                    "start_line": b["start_line"],
                    "end_line": b["end_line"],
                    "looks_like_data_table": b["looks_like_data_table"],
                    "distinct_team_count": b["distinct_team_count"],
                    "missing_teams": ",".join(b["missing_teams"]),
                    "logical_rows": b["logical_rows"],
                    "modal_columns_non_cfl": b["modal_columns_non_cfl"],
                    "columns_consistent": b["columns_consistent"],
                    "column_header": b["column_header"],
                    "flags": ";".join(b["flags"]),
                })

    return summary_path


def main(argv: list[str]) -> int:
    in_dir = Path(argv[1]) if len(argv) > 1 else INPUT_DIR
    out_dir = Path(argv[2]) if len(argv) > 2 else OUTPUT_DIR

    if not in_dir.exists():
        print(f"Input dir not found: {in_dir}")
        return 1

    files = sorted(in_dir.glob("*.txt"))
    if not files:
        print(f"No .txt files in {in_dir}")
        return 1

    maps = [map_file(p) for p in files]
    summary = write_outputs(maps, out_dir)

    print(f"input_dir={in_dir}")
    print(f"output_dir={out_dir}")
    print(f"files_mapped={len(maps)}")
    print(f"summary_csv={summary}")
    print("")
    print(f"{'file':<54} {'wk':>3} {'data_tbls':>9} {'blocks':>6}  flagged_data_tbls")
    for m in maps:
        flagged = sum(
            1 for b in m["blocks"] if b["flags"] and b["looks_like_data_table"]
        )
        print(
            f"{m['file'][:53]:<54} {m['week']:>3} "
            f"{m['data_table_block_count']:>9} {m['block_count']:>6}  {flagged}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
