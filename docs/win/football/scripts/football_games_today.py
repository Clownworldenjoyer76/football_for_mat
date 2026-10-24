#!/usr/bin/env python3
"""Copy weekly CFB and NFL schedule CSVs into the frontend data directories."""

from pathlib import Path
from shutil import copy2


SCHEDULE_DIRECTORIES = (
    (
        Path("docs/win/football/cfb/00_intake/schedule/weekly"),
        Path("frontend/data/index_home/football/CFB"),
    ),
    (
        Path("docs/win/football/nfl/00_intake/schedule/weekly"),
        Path("frontend/data/index_home/football/NFL"),
    ),
)


def main() -> None:
    for source_dir, output_dir in SCHEDULE_DIRECTORIES:
        csv_files = sorted(path for path in source_dir.glob("*.csv") if path.is_file())
        if not csv_files:
            print(f"No CSV files in {source_dir}; skipping.")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        for source_file in csv_files:
            destination = output_dir / source_file.name
            copy2(source_file, destination)
            print(f"Copied {source_file} -> {destination}")


if __name__ == "__main__":
    main()
