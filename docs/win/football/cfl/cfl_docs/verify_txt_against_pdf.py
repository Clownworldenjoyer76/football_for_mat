#!/usr/bin/env python3
"""
Verify raw_txt files against their matching raw_pdf files.

For each week: extracts text from the PDF, checks whether every report
section found in the PDF also appears in the matching TXT. Anything in
the PDF but missing from the TXT gets flagged - that's how a real
source-report gap gets told apart from a bad TXT extraction.

Run on your machine (needs the PDFs and TXTs, which only exist there):
    pip install pdfplumber
    python3 verify_txt_against_pdf.py
"""
from pathlib import Path
import re
import sys

try:
    import pdfplumber
except ImportError:
    print("Missing dependency. Run: pip install pdfplumber")
    sys.exit(1)

PDF_DIR = Path("docs/win/football/cfl/data_dump/raw_pdf")
TXT_DIR = Path("docs/win/football/cfl/data_dump/raw_txt")
REPORT_OUT = Path("docs/win/football/cfl/data_dump/txt_vs_pdf_report.txt")

# (output file this section feeds, anchor phrase to search for)
ANCHORS = [
    ("team_scoring_breakdown", r"SCORING BREAKDOWN"),
    ("team_touchdowns", r"TEAM SCORING"),
    ("opponent_touchdowns", r"OPPONENT SCORING"),
    ("turnover_analysis", r"TURNOVER ANALYSIS"),
    ("team_possessions", r"POSSESSION ANALYSIS"),
    ("opponent_possessions", r"OPPONENT POSSESSION ANALYSIS"),
    ("time_of_possession", r"TIME OF POSSESSION"),
    ("big_play_analysis", r"BIG PLAY ANALYSIS"),
    ("red_zone_results", r"RED ZONE RESULTS"),
    ("coaches_challenges", r"COACHES.? CHALLENGES"),
    ("team_net_offence", r"TEAM NET OFFENCE"),
    ("opponent_net_offence", r"OPPONENT NET OFFENCE"),
    ("first_down_offence", r"NET OFFENCE - ON 1ST DOWN|1ST DOWNS MADE"),
    ("rushing_analysis", r"RUSHING ANALYSIS"),
    ("passing_base", r"TEAM PASSING DATA"),
    ("opponent_passing_base", r"OPPONENT PASSING DATA"),
    ("passing_depth", r"ATTEMPTS 0-9 YDS DEPTH DOWNFIELD"),
    ("opponent_passing_depth", r"OPPT ATTS 0-9 YDS DEPTH|OPPONENT ATTS 0"),
    ("second_down_conversions", r"2ND DN CONVERSIONS - ALL|2ND DOWN CONVERSIONS"),
    ("opponent_second_down", r"OPPONENT 2ND DN CONV"),
    ("third_short_results", r"3RD & SHORT (RESULTS|YARDS)"),
    ("special_teams_field_goals", r"FIELD GOAL ATTEMPTS"),
    ("special_teams_fg_miss", r"FG MISS RESULTS|FIELD GOAL MISS RESULTS"),
    ("special_teams_converts", r"CONVERTS \(C-1 KICK\)"),
    ("special_teams_kick_returns", r"TEAM PUNT RETURNS|KICK RETURN TEAMS"),
    ("special_teams_punts", r"TEAM PUNTING"),
    ("special_teams_kickoffs", r"KICKOFF COVER TEAM|TEAM KICKOFFS"),
    ("special_teams_cover_penalties", r"SPECIAL TEAMS COVER PENALTIES"),
    ("special_teams_return_penalties", r"SPECIAL TEAMS RETURN PENALTIES"),
    ("penalties_team_report", r"PENALTIES - TEAM REPORTS|PENALTIES BY TEAM|PENALTIES - REPORT BY TEAM"),
]


def extract_pdf_text(pdf_path):
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def check_week(pdf_path, txt_path):
    pdf_text = extract_pdf_text(pdf_path)
    txt_text = txt_path.read_text(encoding="utf-8", errors="replace")
    missing = []
    for label, pattern in ANCHORS:
        in_pdf = bool(re.search(pattern, pdf_text, re.I))
        in_txt = bool(re.search(pattern, txt_text, re.I))
        if in_pdf and not in_txt:
            missing.append(label)
    return missing


def main():
    if not PDF_DIR.exists():
        print(f"PDF dir not found: {PDF_DIR}")
        return 1
    if not TXT_DIR.exists():
        print(f"TXT dir not found: {TXT_DIR}")
        return 1

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}")
        return 1

    lines_out = []
    for pdf_path in pdf_files:
        txt_path = TXT_DIR / (pdf_path.stem + ".txt")
        if not txt_path.exists():
            lines_out.append(f"{pdf_path.name}: NO MATCHING TXT FOUND")
            continue
        missing = check_week(pdf_path, txt_path)
        if missing:
            lines_out.append(f"{pdf_path.name}: TXT MISSING SECTIONS -> {', '.join(missing)}")
        else:
            lines_out.append(f"{pdf_path.name}: OK (every PDF section found in TXT)")

    report = "\n".join(lines_out)
    print(report)
    REPORT_OUT.write_text(report, encoding="utf-8")
    print(f"\nReport written to {REPORT_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
