#!/usr/bin/env python3
"""
Creates a manifest of CFL raw TXT/PDF dump pairs.

Scans:
    docs/win/football/cfl/data_dump/raw_txt/
    docs/win/football/cfl/data_dump/raw_pdf/

Writes:
    docs/win/football/cfl/data_dump/manifest/cfl_raw_dump_manifest.csv
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


RAW_TXT_DIR = Path("docs/win/football/cfl/data_dump/raw_txt")
RAW_PDF_DIR = Path("docs/win/football/cfl/data_dump/raw_pdf")
OUT_PATH = Path("docs/win/football/cfl/data_dump/manifest/cfl_raw_dump_manifest.csv")


DUMP_ID_RE = re.compile(
    r"^(?P<season>\d{4})_week_(?P<week>\d{2})_(?P<report_slug>.+)_(?P<stamp>\d{8}_\d{6})$"
)


def report_type_from_slug(slug: str) -> str:
    return slug.replace("_", " ").upper()


def parse_dump_id(dump_id: str) -> dict[str, str]:
    match = DUMP_ID_RE.match(dump_id)

    if not match:
        return {
            "season": "",
            "week": "",
            "report_type": "",
            "timestamp": "",
            "id_parse_status": "FAILED",
        }

    return {
        "season": match.group("season"),
        "week": str(int(match.group("week"))),
        "report_type": report_type_from_slug(match.group("report_slug")),
        "timestamp": match.group("stamp"),
        "id_parse_status": "OK",
    }


def count_pdf_pages_with_pypdf(pdf_path: Path) -> str:
    try:
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader

        reader = PdfReader(str(pdf_path))
        return str(len(reader.pages))
    except Exception:
        return ""


def count_pdf_pages_with_regex(pdf_path: Path) -> str:
    try:
        data = pdf_path.read_bytes()

        # Fallback only.
        # This works for PDFs with visible uncompressed page objects.
        # It can fail on compressed/object-stream PDFs.
        matches = re.findall(rb"/Type\s*/Page(?!s)\b", data)
        return str(len(matches))
    except Exception:
        return ""


def count_pdf_pages(pdf_path: Path) -> str:
    if not pdf_path.exists():
        return ""

    pypdf_pages = count_pdf_pages_with_pypdf(pdf_path)

    if pypdf_pages:
        return pypdf_pages

    return count_pdf_pages_with_regex(pdf_path)


def file_size(path: Path) -> str:
    if not path.exists():
        return ""

    try:
        return str(path.stat().st_size)
    except Exception:
        return ""


def positive_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def collect_dump_ids() -> set[str]:
    dump_ids: set[str] = set()

    if RAW_TXT_DIR.exists():
        for path in RAW_TXT_DIR.glob("*.txt"):
            dump_ids.add(path.stem)

    if RAW_PDF_DIR.exists():
        for path in RAW_PDF_DIR.glob("*.pdf"):
            dump_ids.add(path.stem)

    return dump_ids


def status_for(
    txt_exists: bool,
    pdf_exists: bool,
    txt_size_bytes: str,
    pdf_size_bytes: str,
    pdf_pages: str,
    id_parse_status: str,
) -> str:
    if id_parse_status != "OK":
        return "BAD_DUMP_ID"

    txt_size = positive_int(txt_size_bytes)
    pdf_size = positive_int(pdf_size_bytes)
    pages = positive_int(pdf_pages)

    if txt_exists and pdf_exists and txt_size > 0 and pdf_size > 0 and pages > 0:
        return "READY"

    if txt_exists and pdf_exists and pdf_size <= 0:
        return "BAD_PDF"

    if txt_exists and pdf_exists and pages <= 0:
        return "BAD_PDF"

    if txt_exists and pdf_exists and txt_size <= 0:
        return "BAD_TXT"

    if txt_exists and not pdf_exists:
        return "MISSING_PDF"

    if pdf_exists and not txt_exists:
        return "MISSING_TXT"

    return "MISSING_BOTH"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dump_ids = sorted(collect_dump_ids())

    rows: list[dict[str, str]] = []

    for dump_id in dump_ids:
        txt_path = RAW_TXT_DIR / f"{dump_id}.txt"
        pdf_path = RAW_PDF_DIR / f"{dump_id}.pdf"

        txt_exists = txt_path.exists()
        pdf_exists = pdf_path.exists()

        parsed = parse_dump_id(dump_id)
        txt_size_bytes = file_size(txt_path)
        pdf_size_bytes = file_size(pdf_path)
        pdf_pages = count_pdf_pages(pdf_path)

        rows.append({
            "dump_id": dump_id,
            "season": parsed["season"],
            "week": parsed["week"],
            "report_type": parsed["report_type"],
            "timestamp": parsed["timestamp"],
            "txt_path": str(txt_path),
            "pdf_path": str(pdf_path),
            "txt_exists": str(txt_exists).upper(),
            "pdf_exists": str(pdf_exists).upper(),
            "txt_size_bytes": txt_size_bytes,
            "pdf_size_bytes": pdf_size_bytes,
            "pdf_pages": pdf_pages,
            "id_parse_status": parsed["id_parse_status"],
            "status": status_for(
                txt_exists=txt_exists,
                pdf_exists=pdf_exists,
                txt_size_bytes=txt_size_bytes,
                pdf_size_bytes=pdf_size_bytes,
                pdf_pages=pdf_pages,
                id_parse_status=parsed["id_parse_status"],
            ),
        })

    fieldnames = [
        "dump_id",
        "season",
        "week",
        "report_type",
        "timestamp",
        "txt_path",
        "pdf_path",
        "txt_exists",
        "pdf_exists",
        "txt_size_bytes",
        "pdf_size_bytes",
        "pdf_pages",
        "id_parse_status",
        "status",
    ]

    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ready_count = sum(row["status"] == "READY" for row in rows)
    bad_pdf_count = sum(row["status"] == "BAD_PDF" for row in rows)
    bad_txt_count = sum(row["status"] == "BAD_TXT" for row in rows)
    bad_id_count = sum(row["status"] == "BAD_DUMP_ID" for row in rows)
    missing_count = sum(row["status"].startswith("MISSING") for row in rows)

    print(f"manifest_written={OUT_PATH}")
    print(f"rows={len(rows)}")
    print(f"ready={ready_count}")
    print(f"bad_pdf={bad_pdf_count}")
    print(f"bad_txt={bad_txt_count}")
    print(f"bad_dump_id={bad_id_count}")
    print(f"missing={missing_count}")


if __name__ == "__main__":
    main()
