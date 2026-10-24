#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union


MANIFEST_PATH = Path("docs/win/football/cfl/data_dump/manifest/cfl_raw_dump_manifest.csv")
OUT_DIR = Path("docs/win/football/cfl/01_parsed/game_analysis")

TEAM_ORDER = ["BC", "CGY", "EDM", "HAM", "MTL", "OTT", "SSK", "TOR", "WPG", "CFL"]
TEAM_SORT = {team: idx for idx, team in enumerate(TEAM_ORDER)}

TEAM_RE = re.compile(r"^(BC|CGY|EDM|HAM|MTL|OTT|SSK|TOR|WPG|CFL)\b")

DUMP_ID_RE = re.compile(
    r"^(?P<season>\d{4})_week_(?P<week>\d{2})_(?P<report_slug>.+)_(?P<timestamp>\d{8}_\d{6})$"
)

BASE_COLUMNS = ["dump_id", "season", "week", "report_type", "timestamp", "team"]

SOURCE_COLUMNS = [
    "source_txt_path",
    "source_pdf_path",
    "pdf_pages",
    "row_parse_status",
]

YEAR_POLLUTION_VALUES = {
    "2026",
    "2025",
    "2024",
    "2023",
    "2022",
    "2021",
    "2019",
    "2018",
    "2017",
    "2016",
    "2015",
}

YEAR_POLLUTION_GUARD_FILES = {
    "first_down_offence.csv",
    "first_downs_made.csv",
    "passing_base.csv",
    "passing_depth.csv",
    "opponent_passing_base.csv",
    "opponent_passing_depth.csv",
    "rushing_analysis.csv",
    "second_down_conversions.csv",
    "third_short_results.csv",
    "penalties_team_report.csv",
    "special_teams_field_goals.csv",
    "special_teams_converts.csv",
    "special_teams_punts.csv",
    "special_teams_kickoffs.csv",
}

SECTION_VALIDATION_MODES = {
    "field_goals",
    "converts",
    "kickoffs",
    "penalties",
    "punts",
    "second_down",
    "third_short",
}

MarkerInput = Union[str, list[str]]


@dataclass(frozen=True)
class DumpContext:
    dump_id: str
    season: str
    week: str
    report_type: str
    timestamp: str
    txt_path: Path
    pdf_path: Path
    pdf_pages: str


@dataclass(frozen=True)
class TableSpec:
    filename: str
    start_marker: MarkerInput
    end_marker: MarkerInput
    value_columns: list[str]
    mode: str = "default"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8") as f:
        _qodana_return_value = list(csv.DictReader(f))
    return _qodana_return_value


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def upsert_csv(path: Path, fieldnames: list[str], new_rows: list[dict[str, str]], dump_ids: set[str]) -> None:
    old_rows = read_csv(path)
    kept_rows = [row for row in old_rows if row.get("dump_id", "") not in dump_ids]
    all_rows = kept_rows + new_rows

    def sort_key(row: dict[str, str]) -> tuple[int, int, str, int, str]:
        try:
            season = int(row.get("season", "0") or "0")
        except ValueError:
            season = 0

        try:
            week = int(row.get("week", "0") or "0")
        except ValueError:
            week = 0

        return (
            season,
            week,
            row.get("dump_id", ""),
            TEAM_SORT.get(row.get("team", ""), 999),
            row.get("file_name", ""),
        )

    all_rows.sort(key=sort_key)
    write_csv(path, fieldnames, all_rows)


def parse_dump_id(dump_id: str) -> dict[str, str]:
    match = DUMP_ID_RE.match(dump_id)

    if not match:
        raise ValueError(f"Bad dump_id format: {dump_id}")

    return {
        "season": match.group("season"),
        "week": str(int(match.group("week"))),
        "report_type": match.group("report_slug").replace("_", " ").upper(),
        "timestamp": match.group("timestamp"),
    }


def count_pdf_pages(pdf_path: Path) -> str:
    data = pdf_path.read_bytes()
    return str(len(re.findall(rb"/Type\s*/Page(?!s)\b", data)))


def build_context(manifest_row: dict[str, str]) -> DumpContext:
    dump_id = manifest_row.get("dump_id", "").strip()

    if not dump_id:
        raise ValueError("Manifest row missing dump_id")

    parsed = parse_dump_id(dump_id)

    txt_path = Path(manifest_row.get("txt_path", "").strip())
    pdf_path = Path(manifest_row.get("pdf_path", "").strip())

    if not txt_path.exists():
        raise FileNotFoundError(f"Missing TXT file: {txt_path}")

    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing PDF file: {pdf_path}")

    pdf_pages = manifest_row.get("pdf_pages", "").strip()

    if not pdf_pages:
        pdf_pages = count_pdf_pages(pdf_path)

    return DumpContext(
        dump_id=dump_id,
        season=manifest_row.get("season", "").strip() or parsed["season"],
        week=manifest_row.get("week", "").strip() or parsed["week"],
        report_type=manifest_row.get("report_type", "").strip() or parsed["report_type"],
        timestamp=manifest_row.get("timestamp", "").strip() or parsed["timestamp"],
        txt_path=txt_path,
        pdf_path=pdf_path,
        pdf_pages=pdf_pages,
    )


def read_manifest_rows() -> list[dict[str, str]]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")

    rows = read_csv(MANIFEST_PATH)

    return [
        row
        for row in rows
        if row.get("report_type", "").strip().upper() == "GAME ANALYSIS REPORT"
        and row.get("status", "").strip().upper() == "READY"
    ]


def parse_games_played(text: str) -> str:
    match = re.search(r"GMS PLAYED:\s*(\d+)", text)
    return match.group(1) if match else ""


def marker_list(markers: MarkerInput) -> list[str]:
    if isinstance(markers, str):
        return [markers]

    return markers


def normalize_marker_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", text.upper()).strip()


def line_has_marker(line: str, marker: str) -> bool:
    marker_norm = normalize_marker_text(marker)
    line_norm = normalize_marker_text(line)

    if not marker_norm:
        return False

    return line_norm.startswith(marker_norm)


def find_marker_idx(lines: list[str], markers: MarkerInput, start_at: int = 0) -> int | None:
    marker_options = marker_list(markers)

    for idx in range(start_at, len(lines)):
        for marker in marker_options:
            if line_has_marker(lines[idx], marker):
                return idx

    return None


def extract_section_lines(text: str, start_marker: MarkerInput, end_marker: MarkerInput) -> list[str]:
    lines = text.splitlines()

    start_idx = find_marker_idx(lines, start_marker, 0)

    if start_idx is None:
        raise ValueError(f"Missing section start: {marker_list(start_marker)}")

    end_idx = find_marker_idx(lines, end_marker, start_idx + 1)

    if end_idx is None:
        raise ValueError(f"Missing section end after {marker_list(start_marker)}: {marker_list(end_marker)}")

    return lines[start_idx:end_idx]


def count_team_row_lines(lines: list[str]) -> int:
    teams_seen: set[str] = set()

    for raw_line in lines:
        line = raw_line.strip()
        match = TEAM_RE.match(line)

        if match:
            teams_seen.add(match.group(1))

    return len(teams_seen)


def section_passes_mode_validation(lines: list[str], mode: str) -> bool:
    if mode not in SECTION_VALIDATION_MODES:
        return True

    if count_team_row_lines(lines) >= 5:
        return True

    return False


def extract_section_lines_checked(
    text: str,
    start_marker: MarkerInput,
    end_marker: MarkerInput,
    mode: str,
) -> list[str]:
    lines = text.splitlines()
    start_at = 0
    rejected = 0

    while True:
        start_idx = find_marker_idx(lines, start_marker, start_at)

        if start_idx is None:
            if rejected:
                raise ValueError(
                    f"Missing valid section start: {marker_list(start_marker)} "
                    f"(rejected_candidates={rejected})"
                )

            raise ValueError(f"Missing section start: {marker_list(start_marker)}")

        end_idx = find_marker_idx(lines, end_marker, start_idx + 1)

        if end_idx is None:
            raise ValueError(f"Missing section end after {marker_list(start_marker)}: {marker_list(end_marker)}")

        section_lines = lines[start_idx:end_idx]

        if section_passes_mode_validation(section_lines, mode):
            return section_lines

        rejected += 1
        start_at = start_idx + 1


def collect_team_blocks(lines: list[str]) -> dict[str, str]:
    blocks: dict[str, str] = {}
    current_team: str | None = None
    current_lines: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            continue

        match = TEAM_RE.match(line)

        if match:
            team = match.group(1)

            if current_team is not None and team != current_team:
                if current_team not in blocks or current_team == "CFL":
                    blocks[current_team] = " ".join(current_lines)

            if current_team is None or team != current_team:
                current_team = team
                current_lines = [line]
            else:
                current_lines.append(line)

            continue

        if current_team is not None:
            current_lines.append(line)

    if current_team is not None and (current_team not in blocks or current_team == "CFL"):
        blocks[current_team] = " ".join(current_lines)

    return blocks


def team_segment_from_block(block: str, team: str) -> str:
    marker = f"{team} "

    if block.count(marker) > 1:
        return f"{team} " + block.rsplit(marker, 1)[1]

    return block


def strip_team_tokens(text: str) -> str:
    return re.sub(r"\b(BC|CGY|EDM|HAM|MTL|OTT|SSK|TOR|WPG|CFL)\b", " ", text)


def parse_numeric_values(text: str, parentheses_negative: bool = False) -> list[str]:
    text = strip_team_tokens(text)

    tokens = re.findall(
        r"#DIV/0!|###|T\d+|\([-+]?\d+(?:,\d{3})*(?:\.\d+)?\)|[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?",
        text,
    )

    values: list[str] = []

    for token in tokens:
        if token == "#DIV/0!":
            values.append("0")
        elif token == "###":
            values.append("")
        elif token.startswith("T") and token[1:].isdigit():
            values.append(token[1:])
        elif token.startswith("(") and token.endswith(")"):
            inner = token.strip("()").replace(",", "").replace("%", "")
            if parentheses_negative and inner not in {"", "0", "0.0"}:
                values.append(f"-{inner.lstrip('+-')}")
            else:
                values.append(inner)
        else:
            values.append(token.replace(",", "").replace("%", ""))

    return values


def parse_gp_map(text: str) -> dict[str, str]:
    gp_map: dict[str, str] = {}

    try:
        section_lines = extract_section_lines(
            text,
            "SCORING BREAKDOWN Home/Away",
            "TEAM SCORING TEAM TOUCHDOWNS",
        )
        blocks = collect_team_blocks(section_lines)
    except Exception:
        return gp_map

    for team in TEAM_ORDER:
        if team == "CFL":
            continue

        block = blocks.get(team, "")

        if not block:
            gp_map[team] = ""
            continue

        segment = team_segment_from_block(block, team)
        values = parse_numeric_values(segment)
        gp_map[team] = values[0] if values else ""

    return gp_map


def clean_block(block: str, mode: str) -> str:
    out = block
    stops: list[str] = []

    if mode == "big_play":
        stops = ["Notes:", "1. Offence", "2. Returns", "K/O returns"]

    elif mode == "coaches":
        stops = [
            "Def. Pass",
            "Reception",
            "Roughing",
            "Fumble?",
            "No DPI",
            "No Yards?",
            "All Others",
            "BY TYPE",
        ]

    elif mode == "field_goals":
        out = re.sub(r"\s+[-+]?\d+(?:\.\d+)?\s+yds.*$", "", out)
        out = re.sub(r"\s+[-+]?\d+(?:\.\d+)?\s+PerG.*$", "", out)
        out = re.sub(r"\s+4th Quarter.*$", "", out)
        out = re.sub(r"\s+FG Atts.*$", "", out)
        out = re.sub(r"\s+Made\s+\d+.*$", "", out)
        out = re.sub(r"\s+%\s+100%.*$", "", out)
        out = re.sub(r"\s+Atts\s+\d+.*$", "", out)

    elif mode == "kickoffs":
        stops = [
            "'OT Rec'",
            "* Regular Kickoffs",
            "1. Post K/O possessions",
            "Notes 1.",
            "SPECIAL TEAMS COVER PENALTIES",
            "SPECIAL TEAMS RETURN PENALTIES",
        ]

    elif mode == "penalties":
        stops = [
            "PER GAME",
            "IN 2022",
            "IN 2021",
            "SPECIAL TEAMS",
            "FIELD GOALS",
        ]

    elif mode == "punts":
        stops = ["* NET YARDS", "** FIELD POSITION", "KICKOFF COVER TEAM"]

    elif mode == "red_zone":
        stops = ["CFL RED ZONE FREQUENCY", "RED ZONE LEGEND:"]

    elif mode == "second_down":
        stops = [
            "2ND DOWN CONVERSION HISTORY",
            "2ND DOWN CONVERSION - HISTORY",
            "OPPONENT 2ND DN CONV",
            "OPPONENT 2ND DOWN CONV",
        ]

    elif mode == "third_short":
        stops = ["3RD & SHORT RESULTS LEGEND:", "PCT", "Legend:", "CANADIAN FOOTBALL LEAGUE"]

    elif mode == "turnover":
        stops = ["Notes:", "2023 T/Os by Quarter:"]

    else:
        stops = ["Notes:"]

    for stop in stops:
        pos = out.find(stop)
        if pos != -1:
            out = out[:pos]

    return out


def all_zero_or_blank(values: list[str]) -> bool:
    if not values:
        return False

    for value in values:
        if value == "":
            continue

        try:
            if float(value) != 0:
                return False
        except ValueError:
            return False

    return True


def zero_row(count: int) -> list[str]:
    return ["0"] * count


def normalize_converts(values: list[str], team: str, expected_count: int) -> tuple[list[str], str]:
    if team != "CFL" and values[:3] == ["0", "0", "0"] and len(values) >= 9 and values[3] != "0":
        values = values[:3] + ["0"] + values[3:8]
        return values[:expected_count], "OK"

    if len(values) >= expected_count:
        return values[:expected_count], "OK_AGGREGATE" if team == "CFL" else "OK"

    return (values + [""] * expected_count)[:expected_count], f"NEEDS_REVIEW_{len(values)}_OF_{expected_count}"


def normalize_for_spec(
    values: list[str],
    team: str,
    spec: TableSpec,
    games_played: str,
    gp_map: dict[str, str],
) -> tuple[list[str], str]:
    expected_count = len(spec.value_columns)
    mode = spec.mode

    if team != "CFL" and gp_map.get(team) == "0" and all_zero_or_blank(values[:expected_count]):
        return zero_row(expected_count), "OK_ZERO_GP"

    if team != "CFL" and values and values[0] == "0" and all_zero_or_blank(values[:expected_count]):
        status = "OK_ZERO_GP" if gp_map.get(team) == "0" else "OK"
        return zero_row(expected_count), status

    if mode == "team_scoring" and team != "CFL" and values and values[0] == "0":
        return zero_row(expected_count), "OK_ZERO_GP"

    if mode == "team_scoring" and team == "CFL" and len(values) == 11:
        values = values[:5] + ["0"] + values[5:]
        return values[:expected_count], "OK_AGGREGATE"

    if mode == "turnover" and team == "CFL" and values and re.fullmatch(r"\d{4}", values[0]):
        values = ["", ""] + values[1:13]
        return (values + [""] * expected_count)[:expected_count], "OK_AGGREGATE"

    if mode == "big_play" and team == "CFL" and len(values) == expected_count - 1:
        values = [values[0], ""] + values[1:]
        return values[:expected_count], "OK_AGGREGATE"

    if mode == "coaches" and values and values[0] == "0":
        return zero_row(expected_count), "OK"

    if mode == "second_down" and team == "CFL":
        if len(values) >= expected_count:
            values = values[:expected_count]
            values[3] = ""
            values[15] = ""
            return values, "OK_AGGREGATE"

        if len(values) == expected_count - 2:
            values = values[:3] + [""] + values[3:] + [""]
            return values[:expected_count], "OK_AGGREGATE"

    if mode == "converts":
        return normalize_converts(values, team, expected_count)

    if mode == "kickoffs" and team == "CFL" and len(values) == expected_count - 2:
        values = values[:15] + [""] + values[15:] + [""]
        return values[:expected_count], "OK_AGGREGATE"

    if mode == "punts" and team == "CFL" and len(values) == expected_count - 1:
        values = values[:14] + [""] + values[14:]
        return values[:expected_count], "OK_AGGREGATE"

    if mode == "field_goals" and len(values) == expected_count - 1:
        if len(values) > 7 and values[5] == "0" and values[6] == "0":
            values = values[:7] + ["0"] + values[7:]
            return values[:expected_count], "OK"

        if (
            len(values) > 12
            and values[8] == "0"
            and values[9] == "0"
            and values[10] == "0"
            and values[11] == "0"
        ):
            values = values[:12] + ["0"] + values[12:]
            return values[:expected_count], "OK"

    if mode == "passing_base" and len(values) == expected_count - 1:
        if len(values) > 4 and values[4] == "0":
            values = values[:13] + [""] + values[13:]
            return values[:expected_count], "OK"

    if mode == "opponent_passing_base" and len(values) == expected_count - 1:
        if len(values) > 4 and values[4] == "0":
            values = values[:13] + [""] + values[13:]
            return values[:expected_count], "OK"

        if team == "CFL":
            values = values[:17] + [""] + values[17:]
            return values[:expected_count], "OK_AGGREGATE"

    if mode == "net_offence" and team == "CFL" and len(values) == 15:
        values = [
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            "",
            values[6],
            values[7],
            "",
            values[8],
            values[9],
            values[10],
            values[11],
            values[12],
            values[13],
            values[14],
        ]
        return values, "OK_AGGREGATE"

    if mode == "red_zone" and team == "CFL" and len(values) == 15:
        values = [
            values[0],
            values[1],
            values[2],
            "",
            values[3],
            values[4],
            values[5],
            values[6],
            values[7],
            values[8],
            values[9],
            values[10],
            "",
            values[11],
            values[12],
            values[13],
            values[14],
        ]
        return values, "OK_AGGREGATE"

    if mode == "penalties" and team == "CFL":
        if values and values[0] in {"2023", "2024", "2025", "2026"}:
            values = [games_played] + values[1:]
            return (values + [""] * expected_count)[:expected_count], "OK_AGGREGATE"

    if mode == "kick_returns" and team != "CFL" and len(values) == expected_count - 1:
        values = values[:6] + [""] + values[6:]
        return values[:expected_count], "OK"

    if mode == "kick_returns" and team == "CFL" and len(values) == expected_count - 2:
        values = values[:6] + [""] + values[6:13] + [""] + values[13:]
        return values[:expected_count], "OK_AGGREGATE"

    if team == "CFL" and len(values) < expected_count:
        return (values + [""] * expected_count)[:expected_count], "OK_AGGREGATE"

    if len(values) >= expected_count:
        return values[:expected_count], "OK_AGGREGATE" if team == "CFL" else "OK"

    if all_zero_or_blank(values):
        if team == "CFL":
            return zero_row(expected_count), "OK_AGGREGATE"

        status = "OK_ZERO_GP" if gp_map.get(team) == "0" else "OK"
        return zero_row(expected_count), status

    return (values + [""] * expected_count)[:expected_count], (
        f"NEEDS_REVIEW_{len(values)}_OF_{expected_count}"
    )


def base_row(context: DumpContext, team: str, status: str) -> dict[str, str]:
    return {
        "dump_id": context.dump_id,
        "season": context.season,
        "week": context.week,
        "report_type": context.report_type,
        "timestamp": context.timestamp,
        "team": team,
        "source_txt_path": str(context.txt_path),
        "source_pdf_path": str(context.pdf_path),
        "pdf_pages": context.pdf_pages,
        "row_parse_status": status,
    }


def append_status(status: str, issue: str) -> str:
    if status in {"OK", "OK_ZERO_GP", "OK_AGGREGATE"}:
        return issue

    if issue in status:
        return status

    return f"{status}|{issue}"


def row_value_columns(row: dict[str, str]) -> list[str]:
    excluded = set(BASE_COLUMNS + SOURCE_COLUMNS)
    return [column for column in row if column not in excluded]


def has_year_pollution(row: dict[str, str]) -> bool:
    for column in row_value_columns(row):
        if row.get(column, "") in YEAR_POLLUTION_VALUES:
            return True

    return False


def validate_table_rows(filename: str, rows: list[dict[str, str]]) -> None:
    for row in rows:
        team = row.get("team", "")
        status = row.get("row_parse_status", "")

        if filename == "second_down_conversions.csv" and team == "CFL":
            row["rank"] = ""
            row["yards_to_go_rank"] = ""
            if status in {"OK", "OK_AGGREGATE"}:
                row["row_parse_status"] = "OK_AGGREGATE"

        elif filename == "special_teams_kickoffs.csv" and team == "CFL":
            if row.get("regular_kickoff_rank", "") or row.get("post_kickoff_rank", ""):
                row["row_parse_status"] = append_status(status, "NEEDS_REVIEW_CFL_RANK_SHIFT")

        elif filename == "turnover_analysis.csv" and team == "CFL":
            if row.get("turnover_ratio", "") or row.get("rank", ""):
                row["row_parse_status"] = append_status(status, "NEEDS_REVIEW_CFL_HISTORY_SHIFT")

        elif filename == "team_scoring_breakdown.csv" and team == "CFL":
            if row.get("point_diff", "") == "":
                row["row_parse_status"] = append_status(status, "NEEDS_REVIEW_CFL_POINT_DIFF_BLANK")

        elif filename == "special_teams_converts.csv" and team != "CFL":
            if (
                row.get("convert_1_attempts") == "0"
                and row.get("convert_1_made") == "0"
                and row.get("convert_1_pct") == "0"
                and row.get("convert_1_missed") not in {"0", ""}
            ):
                row["row_parse_status"] = append_status(status, "NEEDS_REVIEW_CONVERT_SHIFT")

        if filename in YEAR_POLLUTION_GUARD_FILES and has_year_pollution(row):
            row["row_parse_status"] = append_status(
                row.get("row_parse_status", ""),
                "NEEDS_REVIEW_YEAR_POLLUTION",
            )


def audit_entry(context: DumpContext, filename: str, rows: list[dict[str, str]]) -> dict[str, str]:
    validate_table_rows(filename, rows)
    statuses = [row.get("row_parse_status", "") for row in rows]

    return {
        "dump_id": context.dump_id,
        "season": context.season,
        "week": context.week,
        "report_type": context.report_type,
        "timestamp": context.timestamp,
        "file_name": filename,
        "rows_written": str(len(rows)),
        "ok_rows": str(sum(status == "OK" for status in statuses)),
        "zero_gp_rows": str(sum(status == "OK_ZERO_GP" for status in statuses)),
        "aggregate_rows": str(sum(status == "OK_AGGREGATE" for status in statuses)),
        "needs_review_rows": str(sum("NEEDS_REVIEW" in status for status in statuses)),
        "missing_rows": str(sum(status == "MISSING_ROW" or status.startswith("MISSING_SECTION") for status in statuses)),
        "status": (
            "OK"
            if all(status in {"OK", "OK_ZERO_GP", "OK_AGGREGATE"} for status in statuses)
            else "NEEDS_REVIEW"
        ),
    }


def parse_generic_table(
    text: str,
    context: DumpContext,
    spec: TableSpec,
    games_played: str,
    gp_map: dict[str, str],
) -> tuple[str, list[str], list[dict[str, str]], dict[str, str]]:
    rows: list[dict[str, str]] = []

    try:
        section_lines = extract_section_lines_checked(text, spec.start_marker, spec.end_marker, spec.mode)
        blocks = collect_team_blocks(section_lines)
        missing_status = ""
    except Exception as exc:
        blocks = {}
        missing_status = f"MISSING_SECTION: {exc}"

    for team in TEAM_ORDER:
        block = blocks.get(team, "")

        if not block:
            values = [""] * len(spec.value_columns)
            status = missing_status or "MISSING_ROW"
        else:
            cleaned = clean_block(block, spec.mode)

            if spec.mode in {"team_scoring", "kickoffs"}:
                cleaned = team_segment_from_block(cleaned, team)

            raw_values = parse_numeric_values(cleaned, parentheses_negative=(spec.mode == "punts"))
            values, status = normalize_for_spec(raw_values, team, spec, games_played, gp_map)

        row = base_row(context, team, status)

        for column, value in zip(spec.value_columns, values):
            row[column] = value

        rows.append(row)

    fieldnames = BASE_COLUMNS + spec.value_columns + SOURCE_COLUMNS
    return spec.filename, fieldnames, rows, audit_entry(context, spec.filename, rows)


def parse_top_block(
    block: str,
    expected_count: int,
    team: str,
    gp_map: dict[str, str],
) -> tuple[str, list[str], str]:
    if team != "CFL" and gp_map.get(team) == "0":
        return "", zero_row(expected_count), "OK_ZERO_GP"

    match = re.search(r"\b\d{1,2}:\d{2}\b", block)
    top_value = match.group(0) if match else ""

    cleaned = block.replace(top_value, " ", 1) if top_value else block
    values = parse_numeric_values(cleaned)

    if team == "CFL" and len(values) >= 8:
        values = [
            values[0],
            values[1],
            values[2],
            "",
            values[3],
            values[4],
            "",
            values[5],
            values[6],
            values[7],
        ]
        return top_value, values[:expected_count], "OK_AGGREGATE"

    if len(values) >= expected_count:
        return top_value, values[:expected_count], "OK"

    if all_zero_or_blank(values):
        status = "OK_ZERO_GP" if gp_map.get(team) == "0" else "OK"
        return top_value, zero_row(expected_count), status

    return top_value, (values + [""] * expected_count)[:expected_count], (
        f"NEEDS_REVIEW_{len(values)}_OF_{expected_count}"
    )


def parse_time_of_possession_table(
    text: str,
    context: DumpContext,
    gp_map: dict[str, str],
) -> tuple[str, list[str], list[dict[str, str]], dict[str, str]]:
    value_columns = [
        "time_of_possession",
        "team_possessions",
        "team_field_position_yards",
        "team_avg_start_yard_line",
        "team_start_yard_line_rank",
        "opponent_end_start_count",
        "opponent_end_start_td",
        "start_yard_line_gap",
        "opponent_possessions",
        "opponent_field_position_yards",
        "opponent_avg_start_yard_line",
    ]

    numeric_columns = value_columns[1:]
    rows: list[dict[str, str]] = []

    try:
        section_lines = extract_section_lines(
            text,
            "4c. TIME OF POSSESSION & FIELD POSITION",
            ["5. BIG PLAY ANALYSIS", "5. 2ND DOWN CONVERSIONS"],
        )
        blocks = collect_team_blocks(section_lines)
        missing_status = ""
    except Exception as exc:
        blocks = {}
        missing_status = f"MISSING_SECTION: {exc}"

    for team in TEAM_ORDER:
        block = blocks.get(team, "")

        if not block:
            top_value = ""
            values = [""] * len(numeric_columns)
            status = missing_status or "MISSING_ROW"
        else:
            top_value, values, status = parse_top_block(
                block,
                len(numeric_columns),
                team,
                gp_map,
            )

        row = base_row(context, team, status)
        row["time_of_possession"] = top_value

        for column, value in zip(numeric_columns, values):
            row[column] = value

        rows.append(row)

    filename = "time_of_possession_field_position.csv"
    fieldnames = BASE_COLUMNS + value_columns + SOURCE_COLUMNS

    return filename, fieldnames, rows, audit_entry(context, filename, rows)


def split_first_down_values(values: list[str], team: str) -> tuple[list[str], list[str], str]:
    offence_values = [""] * 6
    made_values = [""] * 10

    if team == "CFL":
        if len(values) >= 14:
            offence_values = [
                values[0],
                values[1],
                "",
                values[2],
                values[3],
                "",
            ]
            made_values = (values[4:14] + [""] * 10)[:10]
            return offence_values, made_values, "OK_AGGREGATE"

        if len(values) >= 4:
            offence_values = [
                values[0],
                values[1],
                "",
                values[2],
                values[3],
                "",
            ]
            if len(values) > 4:
                made_values = (values[4:14] + [""] * 10)[:10]
            return offence_values, made_values, "OK_AGGREGATE"

        return offence_values, made_values, f"NEEDS_REVIEW_{len(values)}_OF_4"

    if len(values) >= 16:
        return values[:6], values[6:16], "OK"

    if len(values) >= 6:
        offence_values = values[:6]
        if len(values) > 6:
            made_values = (values[6:16] + [""] * 10)[:10]
        return offence_values, made_values, "OK"

    if all_zero_or_blank(values):
        return zero_row(6), zero_row(10), "OK"

    return (values + [""] * 6)[:6], made_values, f"NEEDS_REVIEW_{len(values)}_OF_6"


def parse_first_down_tables(
    text: str,
    context: DumpContext,
) -> list[tuple[str, list[str], list[dict[str, str]], dict[str, str]]]:
    offence_columns = [
        "team_first_down_plays",
        "team_first_down_avg_yards",
        "team_first_down_rank",
        "opponent_first_down_plays",
        "opponent_first_down_avg_yards",
        "opponent_first_down_rank",
    ]

    made_columns = [
        "first_down_call_rush",
        "first_down_call_pass_plus",
        "team_first_downs_total",
        "team_first_downs_rush",
        "team_first_downs_pass",
        "team_first_downs_penalty",
        "opponent_first_downs_total",
        "opponent_first_downs_rush",
        "opponent_first_downs_pass",
        "opponent_first_downs_penalty",
    ]

    offence_rows: list[dict[str, str]] = []
    made_rows: list[dict[str, str]] = []

    try:
        section_lines = extract_section_lines(
            text,
            ["8c. NET OFFENCE - ON 1ST DOWN", "10c. NET OFFENCE - ON 1ST DOWN"],
            ["9 RUSHING ANALYSIS", "9. RUSHING ANALYSIS", "11. RUSHING ANALYSIS"],
        )
        blocks = collect_team_blocks(section_lines)
        missing_status = ""
    except Exception as exc:
        blocks = {}
        missing_status = f"MISSING_SECTION: {exc}"

    for team in TEAM_ORDER:
        block = blocks.get(team, "")

        if not block:
            offence_values = [""] * 6
            made_values = [""] * 10
            status = missing_status or "MISSING_ROW"

        else:
            values = parse_numeric_values(clean_block(block, "default"))

            if team != "CFL" and values and values[0] == "0":
                offence_values = zero_row(6)
                made_values = zero_row(10)
                status = "OK_ZERO_GP"

            else:
                offence_values, made_values, status = split_first_down_values(values, team)

        offence_row = base_row(context, team, status)
        made_row = base_row(context, team, status)

        for column, value in zip(offence_columns, offence_values):
            offence_row[column] = value

        for column, value in zip(made_columns, made_values):
            made_row[column] = value

        offence_rows.append(offence_row)
        made_rows.append(made_row)

    return [
        (
            "first_down_offence.csv",
            BASE_COLUMNS + offence_columns + SOURCE_COLUMNS,
            offence_rows,
            audit_entry(context, "first_down_offence.csv", offence_rows),
        ),
        (
            "first_downs_made.csv",
            BASE_COLUMNS + made_columns + SOURCE_COLUMNS,
            made_rows,
            audit_entry(context, "first_downs_made.csv", made_rows),
        ),
    ]


def make_specs() -> list[TableSpec]:
    touchdown_columns = [
        "td",
        "convert_1",
        "convert_2",
        "field_goals",
        "singles",
        "special_teams_td",
        "rush_td",
        "pass_td",
        "interception_return_td",
        "fumble_return_td",
        "punt_return_td",
        "kickoff_return_td",
        "missed_fg_return_td",
        "team_td",
        "q1_points",
        "q2_points",
        "q3_points",
        "q4_points",
        "ot_points",
    ]

    possession_columns = [
        "possessions",
        "td_drives",
        "fg_drives",
        "missed_fg_drives",
        "turnover_drives",
        "punt_drives",
        "other_drives",
        "own_1_20_drives",
        "own_1_20_td",
        "own_21_40_drives",
        "own_21_40_td",
        "own_41_54_drives",
        "own_41_54_td",
        "opp_55_41_drives",
        "opp_55_41_td",
        "opp_40_21_drives",
        "opp_40_21_td",
        "opp_20_1_drives",
        "opp_20_1_td",
        "two_and_outs",
        "two_and_out_pct",
    ]

    passing_depth_columns = [
        "depth_0_9_att",
        "depth_0_9_comp",
        "depth_0_9_comp_pct",
        "depth_0_9_yards",
        "depth_0_9_int",
        "depth_0_9_td",
        "depth_0_9_efficiency",
        "depth_10_19_att",
        "depth_10_19_comp",
        "depth_10_19_comp_pct",
        "depth_10_19_yards",
        "depth_10_19_int",
        "depth_10_19_td",
        "depth_10_19_efficiency",
        "depth_20_plus_att",
        "depth_20_plus_comp",
        "depth_20_plus_comp_pct",
        "depth_20_plus_yards",
        "depth_20_plus_int",
        "depth_20_plus_td",
        "depth_20_plus_efficiency",
    ]

    net_columns = [
        "gp",
        "net_yards",
        "pass_yards",
        "rush_yards",
        "team_losses",
        "yards_per_game",
        "yards_per_game_rank",
        "plays",
        "yards_per_play",
        "yards_per_play_rank",
        "pass_attempts",
        "sacks_allowed",
        "rush_attempts",
        "other_team_losses",
        "rush_call_pct",
        "pass_plus_call_pct",
        "other_call_pct",
    ]

    opponent_net_columns = [
        "gp",
        "opponent_net_yards",
        "opponent_pass_yards",
        "opponent_rush_yards",
        "opponent_team_losses",
        "opponent_yards_per_game",
        "opponent_yards_per_game_rank",
        "opponent_plays",
        "opponent_yards_per_play",
        "opponent_yards_per_play_rank",
        "opponent_pass_attempts",
        "opponent_sacks_allowed",
        "opponent_rush_attempts",
        "opponent_other_team_losses",
        "opponent_rush_call_pct",
        "opponent_pass_plus_call_pct",
        "opponent_other_call_pct",
    ]

    passing_base_columns = [
        "pass_attempts",
        "pass_completions",
        "completion_pct",
        "pass_yards",
        "interceptions",
        "pass_td",
        "longest_pass",
        "passes_30_plus",
        "second_down_conversion_receptions",
        "pass_efficiency",
        "interception_pct",
        "yards_per_attempt",
        "yac_yards",
        "td_int_ratio",
        "pass_yards_per_game",
        "average_depth",
        "sacks_allowed",
        "sack_pct",
        "qb_escape_runs",
        "dropbacks",
    ]

    opponent_passing_base_columns = [
        "opponent_pass_attempts",
        "opponent_pass_completions",
        "opponent_completion_pct",
        "opponent_pass_yards",
        "opponent_interceptions",
        "opponent_pass_td",
        "opponent_longest_pass",
        "opponent_passes_30_plus",
        "opponent_second_down_conversion_receptions",
        "opponent_pass_efficiency",
        "opponent_interception_pct",
        "opponent_yards_per_attempt",
        "opponent_yac_yards",
        "opponent_td_int_ratio",
        "opponent_average_depth",
        "opponent_games",
        "opponent_dropbacks",
        "opponent_sack_pct",
        "opponent_sacks_allowed",
        "opponent_qb_escape_runs",
    ]

    return [
        TableSpec(
            "team_scoring_breakdown.csv",
            "SCORING BREAKDOWN Home/Away",
            "TEAM SCORING TEAM TOUCHDOWNS",
            [
                "gp",
                "points_for",
                "points_for_avg",
                "points_against",
                "points_against_avg",
                "point_diff",
                "home_points",
                "away_points",
                "offensive_team_points",
                "offensive_team_avg",
                "offensive_opponent_points",
                "offensive_opponent_avg",
            ],
            "team_scoring",
        ),
        TableSpec("team_touchdowns_points_by_quarter.csv", "TEAM SCORING TEAM TOUCHDOWNS", "OPPONENT SCORING OPPONENT TOUCHDOWNS", touchdown_columns),
        TableSpec("opponent_touchdowns_points_by_quarter.csv", "OPPONENT SCORING OPPONENT TOUCHDOWNS", "3. TURNOVER ANALYSIS", touchdown_columns),
        TableSpec(
            "turnover_analysis.csv",
            "3. TURNOVER ANALYSIS",
            "4a. POSSESSION ANALYSIS",
            [
                "turnover_ratio",
                "rank",
                "giveaways_fumble",
                "giveaways_interception",
                "giveaways_downs",
                "giveaways_total",
                "giveaway_points_allowed",
                "takeaway_offence",
                "takeaway_return",
                "takeaways_fumble",
                "takeaways_interception",
                "takeaways_downs",
                "takeaways_total",
                "takeaway_points",
            ],
            "turnover",
        ),
        TableSpec("team_possessions.csv", "4a. POSSESSION ANALYSIS", "4b. OPPONENT POSSESSION ANALYSIS", possession_columns),
        TableSpec(
            "opponent_possessions.csv",
            ["4b. OPPONENT POSSESSION ANALYSIS", "OPPONENT POSSESSION ANALYSIS"],
            ["4c. TIME OF POSSESSION", "TIME OF POSSESSION & FIELD POSITION"],
            possession_columns,
        ),
        TableSpec(
            "big_play_analysis.csv",
            ["5. BIG PLAY ANALYSIS", "7. BIG PLAY ANALYSIS"],
            ["6. RED ZONE RESULTS", "8. RED ZONE RESULTS"],
            [
                "team_big_plays_total",
                "team_big_play_rank",
                "team_rush_20_plus",
                "team_pass_30_plus",
                "team_punt_return_big_plays",
                "team_kickoff_return_big_plays",
                "team_fg_miss_return_big_plays",
                "opponent_big_plays_total",
                "opponent_rush_20_plus",
                "opponent_pass_30_plus",
                "opponent_punt_return_big_plays",
                "opponent_kickoff_return_big_plays",
                "opponent_fg_miss_return_big_plays",
                "big_play_diff",
            ],
            "big_play",
        ),
        TableSpec(
            "red_zone_results.csv",
            ["6. RED ZONE RESULTS", "8. RED ZONE RESULTS"],
            ["7. CFL COACHES' CHALLENGES", "9. CFL COACHES' CHALLENGES"],
            [
                "team_red_zone_att",
                "team_red_zone_td",
                "team_red_zone_td_pct",
                "team_red_zone_rank",
                "team_red_zone_fg",
                "team_red_zone_turnover",
                "team_red_zone_other",
                "team_red_zone_points",
                "team_red_zone_points_pct",
                "opponent_red_zone_att",
                "opponent_red_zone_td",
                "opponent_red_zone_td_pct",
                "opponent_red_zone_rank",
                "opponent_red_zone_fg",
                "opponent_red_zone_turnover",
                "opponent_red_zone_other",
                "opponent_red_zone_points",
            ],
            "red_zone",
        ),
        TableSpec(
            "coaches_challenges.csv",
            ["7. CFL COACHES' CHALLENGES", "9. CFL COACHES' CHALLENGES"],
            ["8a. TEAM NET OFFENCE", "10a. TEAM NET OFFENCE"],
            [
                "challenges",
                "challenges_won",
                "challenge_pct",
                "offence_challenges",
                "offence_challenges_won",
                "defence_challenges",
                "defence_challenges_won",
                "penalty_challenges",
                "no_penalty_challenges",
                "pass_challenges",
                "fumble_challenges",
                "spot_challenges",
            ],
            "coaches",
        ),
        TableSpec("team_net_offence.csv", ["8a. TEAM NET OFFENCE", "10a. TEAM NET OFFENCE"], ["8b. OPPONENT NET OFFENCE", "10b. OPPONENT NET OFFENCE"], net_columns, "net_offence"),
        TableSpec("opponent_net_offence.csv", ["8b. OPPONENT NET OFFENCE", "10b. OPPONENT NET OFFENCE"], ["8c. NET OFFENCE - ON 1ST DOWN", "10c. NET OFFENCE - ON 1ST DOWN"], opponent_net_columns, "net_offence"),
        TableSpec(
            "rushing_analysis.csv",
            ["9 RUSHING ANALYSIS", "9. RUSHING ANALYSIS", "11. RUSHING ANALYSIS"],
            ["10a. PASSING ANALYSIS", "12a. PASSING ANALYSIS"],
            [
                "team_rush_atts",
                "team_rush_yards",
                "team_rush_avg",
                "team_rush_td",
                "team_rush_10_plus",
                "team_rush_20_plus",
                "team_rush_yards_per_game",
                "first_down_rush_atts",
                "first_down_rush_yards",
                "first_down_rush_avg",
                "qb_rush_atts",
                "qb_rush_yards",
                "qb_escape_runs",
                "opponent_rush_atts",
                "opponent_rush_yards",
                "opponent_rush_avg",
                "opponent_rush_td",
                "opponent_rush_10_plus",
                "opponent_rush_20_plus",
            ],
        ),
        TableSpec("passing_base.csv", "TEAM PASSING DATA", "ATTEMPTS 0-9 YDS DEPTH DOWNFIELD", passing_base_columns, "passing_base"),
        TableSpec("passing_depth.csv", "ATTEMPTS 0-9 YDS DEPTH DOWNFIELD", ["OPPONENT PASSING DATA", "10b. PASSING ANALYSIS - BY OPPONENTS", "12b. PASSING ANALYSIS - BY OPPONENTS"], passing_depth_columns),
        TableSpec("opponent_passing_base.csv", "OPPONENT PASSING DATA", "OPPT ATTS 0-9 YDS DEPTH DOWNFIELD", opponent_passing_base_columns, "opponent_passing_base"),
        TableSpec("opponent_passing_depth.csv", "OPPT ATTS 0-9 YDS DEPTH DOWNFIELD", ["11. 2ND DOWN CONVERSIONS", "5. 2ND DOWN CONVERSIONS"], passing_depth_columns),
        TableSpec(
            "second_down_conversions.csv",
            ["11. 2ND DOWN CONVERSIONS", "5. 2ND DOWN CONVERSIONS"],
            ["12. 3RD & SHORT RESULTS", "6. 3RD & SHORT RESULTS"],
            [
                "all_att",
                "all_made",
                "all_pct",
                "rank",
                "one_to_three_att",
                "one_to_three_made",
                "one_to_three_pct",
                "four_to_six_att",
                "four_to_six_made",
                "four_to_six_pct",
                "seven_plus_att",
                "seven_plus_made",
                "seven_plus_pct",
                "yards_to_go_total",
                "yards_to_go_avg",
                "yards_to_go_rank",
            ],
            "second_down",
        ),
        TableSpec(
            "third_short_results.csv",
            ["12. 3RD & SHORT RESULTS", "6. 3RD & SHORT RESULTS"],
            ["13a. PENALTIES", "7. BIG PLAY ANALYSIS"],
            [
                "third_short_att",
                "third_short_made",
                "third_short_fail",
                "opponent_third_short_att",
                "opponent_third_short_made",
                "opponent_third_short_fail",
            ],
            "third_short",
        ),
        TableSpec(
            "penalties_team_report.csv",
            ["13b. PENALTIES - TEAM REPORTS", "PENALTIES - TEAM REPORTS", "2023 PENALTIES BY TEAM"],
            ["14. SPECIAL TEAMS", "14 SPECIAL TEAMS", "A) FIELD GOALS", "TEAM FIELD GOALS"],
            [
                "gp",
                "penalties_all",
                "penalties_avg",
                "penalties_accepted",
                "penalties_declined",
                "penalty_yards",
                "offence_penalties",
                "defence_penalties",
                "special_teams_penalties",
                "punt_cover_penalties",
                "kickoff_cover_penalties",
                "punt_return_penalties",
                "kickoff_return_penalties",
            ],
            "penalties",
        ),
        TableSpec(
            "special_teams_field_goals.csv",
            ["A) FIELD GOALS", "TEAM FIELD GOALS", "FIELD GOALS MADE"],
            "FIELD GOAL MISS RESULTS",
            [
                "fg_attempts",
                "fg_made",
                "fg_pct",
                "fg_long",
                "fg_singles",
                "fg_under_40_attempts",
                "fg_under_40_made",
                "fg_under_40_pct",
                "fg_40_plus_attempts",
                "fg_40_plus_made",
                "fg_40_plus_pct",
                "fg_50_plus_attempts",
                "fg_50_plus_made",
                "fg_made_yards",
                "fg_made_avg_yards",
                "fg_all_attempt_yards",
                "fg_all_attempt_avg_yards",
            ],
            "field_goals",
        ),
        TableSpec(
            "special_teams_converts.csv",
            ["C) CONVERTS", "CONVERTS (C-1 KICK)"],
            ["D) SPECIAL TEAMS - FAKE KICK PLAYS", "FAKE KICK PLAYS"],
            [
                "convert_1_attempts",
                "convert_1_made",
                "convert_1_pct",
                "convert_1_missed",
                "convert_2_attempts",
                "convert_2_made",
                "convert_2_pct",
                "convert_2_rush",
                "convert_2_pass",
            ],
            "converts",
        ),
        TableSpec(
            "special_teams_kick_returns.csv",
            ["E) KICK RETURN TEAMS", "I) KICK RETURN TEAMS"],
            "OPPONENT PUNT RETURNS",
            [
                "punt_return_no",
                "punt_return_yards",
                "punt_return_avg",
                "punt_return_long",
                "punt_return_td",
                "punt_return_30_plus",
                "punt_return_10_plus",
                "punt_return_rank",
                "kickoff_return_no",
                "kickoff_return_yards",
                "kickoff_return_avg",
                "kickoff_return_long",
                "kickoff_return_td",
                "kickoff_return_40_plus",
                "kickoff_return_rank",
                "fg_miss_return_no",
                "fg_miss_return_yards",
                "fg_miss_return_long",
                "fg_miss_return_td",
                "fg_miss_return_30_plus",
                "kick_return_big_plays",
                "punt_return_td_total",
                "kickoff_return_td_total",
                "fg_miss_return_td_total",
                "kick_return_td_total",
            ],
            "kick_returns",
        ),
        TableSpec(
            "special_teams_punts.csv",
            ["G) PUNT COVER", "F) PUNT COVER", "PUNT COVER - BASE"],
            ["H) KICKOFF COVER", "G) KICKOFF COVER", "KICKOFF COVER TEAM"],
            [
                "punt_no",
                "punt_yards",
                "punt_avg",
                "punt_long",
                "punt_singles",
                "punt_inside_10",
                "opponent_punt_return_no",
                "opponent_punt_return_yards",
                "opponent_punt_return_avg",
                "opponent_punt_return_long",
                "opponent_punt_return_td",
                "opponent_punt_return_30_plus",
                "net_punt_avg",
                "net_punt_yards",
                "net_punt_rank",
                "single_adjustment",
                "cover_penalty_adjustment",
                "return_penalty_adjustment",
                "field_position_yards",
                "field_position_avg",
            ],
            "punts",
        ),
        TableSpec(
            "special_teams_kickoffs.csv",
            ["H) KICKOFF COVER", "G) KICKOFF COVER", "KICKOFF COVER TEAM"],
            [
                "I) SPECIAL TEAMS COVER PENALTIES",
                "H) SPECIAL TEAMS COVER PENALTIES",
                "J) SPECIAL TEAMS RETURN PENALTIES",
                "KICK RETURN TEAMS",
            ],
            [
                "kickoff_no",
                "kickoff_yards",
                "kickoff_avg",
                "kickoff_long",
                "kickoff_singles",
                "own_team_recoveries",
                "opponent_kickoff_return_no",
                "opponent_kickoff_return_yards",
                "opponent_kickoff_return_avg",
                "opponent_kickoff_return_long",
                "opponent_kickoff_return_td",
                "opponent_kickoff_return_40_plus",
                "regular_kickoff_no",
                "regular_kickoff_yards",
                "regular_kickoff_avg",
                "regular_kickoff_rank",
                "post_kickoff_possessions",
                "post_kickoff_yards",
                "post_kickoff_avg_start_yard_line",
                "post_kickoff_rank",
            ],
            "kickoffs",
        ),
    ]


def parse_dump(context: DumpContext) -> tuple[
    dict[str, list[str]],
    dict[str, list[dict[str, str]]],
    list[dict[str, str]],
]:
    text = context.txt_path.read_text(encoding="utf-8", errors="replace")

    games_played = parse_games_played(text)
    gp_map = parse_gp_map(text)

    fieldnames_by_file: dict[str, list[str]] = {}
    rows_by_file: dict[str, list[dict[str, str]]] = {}
    audit_rows: list[dict[str, str]] = []

    for spec in make_specs():
        filename, fieldnames, rows, audit = parse_generic_table(
            text=text,
            context=context,
            spec=spec,
            games_played=games_played,
            gp_map=gp_map,
        )

        fieldnames_by_file[filename] = fieldnames
        rows_by_file.setdefault(filename, []).extend(rows)
        audit_rows.append(audit)

    filename, fieldnames, rows, audit = parse_time_of_possession_table(text, context, gp_map)
    fieldnames_by_file[filename] = fieldnames
    rows_by_file.setdefault(filename, []).extend(rows)
    audit_rows.append(audit)

    for filename, fieldnames, rows, audit in parse_first_down_tables(text, context):
        fieldnames_by_file[filename] = fieldnames
        rows_by_file.setdefault(filename, []).extend(rows)
        audit_rows.append(audit)

    return fieldnames_by_file, rows_by_file, audit_rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_manifest_rows()

    if not manifest_rows:
        print("No READY GAME ANALYSIS REPORT rows found in manifest")
        return

    contexts = [build_context(row) for row in manifest_rows]
    dump_ids = {context.dump_id for context in contexts}

    all_fieldnames_by_file: dict[str, list[str]] = {}
    all_rows_by_file: dict[str, list[dict[str, str]]] = {}
    all_audit_rows: list[dict[str, str]] = []

    for context in contexts:
        fieldnames_by_file, rows_by_file, audit_rows = parse_dump(context)

        for filename, fieldnames in fieldnames_by_file.items():
            all_fieldnames_by_file[filename] = fieldnames

        for filename, rows in rows_by_file.items():
            all_rows_by_file.setdefault(filename, []).extend(rows)

        all_audit_rows.extend(audit_rows)

    for filename, rows in all_rows_by_file.items():
        fieldnames = all_fieldnames_by_file[filename]
        upsert_csv(OUT_DIR / filename, fieldnames, rows, dump_ids)

    audit_fieldnames = [
        "dump_id",
        "season",
        "week",
        "report_type",
        "timestamp",
        "file_name",
        "rows_written",
        "ok_rows",
        "zero_gp_rows",
        "aggregate_rows",
        "needs_review_rows",
        "missing_rows",
        "status",
    ]

    upsert_csv(OUT_DIR / "_parse_audit.csv", audit_fieldnames, all_audit_rows, dump_ids)

    print(f"parsed_dir={OUT_DIR}")
    print(f"game_analysis_dumps_processed={len(contexts)}")
    print(f"tables_written={len(all_rows_by_file)}")
    print(f"audit_rows_written={len(all_audit_rows)}")


if __name__ == "__main__":
    main()
