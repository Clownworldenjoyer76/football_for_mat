#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional


PBP_PATH_TEMPLATE = "docs/win/football/nfl/00_intake/pbp/{season}_pbp.csv.gz"
FP_PATH_TEMPLATE = "docs/win/football/nfl/data/qb_data/clean_qb/{season}_fp_qb.csv"
OUT_PATH_TEMPLATE = "docs/win/football/nfl/00_intake/qb/{season}_qb_stats.csv"
LOG_PATH = Path("docs/win/football/nfl/errors/00_intake/pull_qb_stats.txt")

OUTPUT_HEADERS = [
    "season",
    "week",
    "team",
    "player_id",
    "qb_name",
    "starts",
    "dropbacks",
    "epa_per_play",
    "cpoe",
    "air_yards",
    "adjusted_completion_pct",
    "sack_rate",
    "pressure_to_sack_rate",
    "turnover_worthy_play_rate",
    "interception_rate",
    "fumble_rate",
]

PBP_REQUIRED_HEADERS = [
    "season",
    "week",
    "game_id",
    "play_id",
    "posteam",
    "passer_player_id",
    "passer_player_name",
    "rusher_player_id",
    "rusher_player_name",
    "qb_dropback",
    "qb_epa",
    "cpoe",
    "air_yards",
    "sack",
    "interception",
    "fumble",
]

FP_REQUIRED_HEADERS = [
    "season",
    "team",
    "qb_name",
    "adjusted_completion_pct",
    "pressure_to_sack_rate",
]


def write_log(lines: list[str]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line.rstrip()}\n")


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None

    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "na", "n/a", "none", "null"}:
        return None

    text = text.replace(",", "").replace("%", "")

    try:
        return float(text)
    except ValueError:
        return None


def to_int_flag(value: object) -> int:
    number = to_float(value)
    if number is None:
        return 0
    return 1 if number != 0 else 0


def is_one(value: object) -> bool:
    number = to_float(value)
    return number == 1


def fmt_number(value: Optional[float], decimals: int = 6) -> str:
    if value is None:
        return ""

    if float(value).is_integer():
        return str(int(value))

    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def fmt_rate(numerator: float, denominator: float, decimals: int = 6) -> str:
    if denominator == 0:
        return ""
    return fmt_number(numerator / denominator, decimals)


def avg(total: float, count: int) -> Optional[float]:
    if count == 0:
        return None
    return total / count


def require_headers(actual_headers: list[str], required_headers: list[str], path: Path) -> list[str]:
    return [header for header in required_headers if header not in actual_headers]


def qb_identity(row: dict[str, str]) -> tuple[str, str]:
    passer_id = row.get("passer_player_id", "").strip()
    passer_name = row.get("passer_player_name", "").strip()

    if passer_id:
        return passer_id, passer_name

    rusher_id = row.get("rusher_player_id", "").strip()
    rusher_name = row.get("rusher_player_name", "").strip()

    return rusher_id, rusher_name


def read_fantasypros(path: Path, log_lines: list[str]) -> dict[tuple[str, str, str], dict[str, str]]:
    fp_rows: dict[tuple[str, str, str], dict[str, str]] = {}
    duplicate_keys: list[tuple[str, str, str]] = []

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")

        missing_headers = require_headers(reader.fieldnames, FP_REQUIRED_HEADERS, path)
        if missing_headers:
            raise ValueError(f"{path} missing required headers: {missing_headers}")

        for row in reader:
            season = row.get("season", "").strip()
            team = row.get("team", "").strip()
            qb_name = row.get("qb_name", "").strip()

            key = (season, team, qb_name)

            if key in fp_rows:
                duplicate_keys.append(key)

            fp_rows[key] = {
                "adjusted_completion_pct": row.get("adjusted_completion_pct", "").strip(),
                "pressure_to_sack_rate": row.get("pressure_to_sack_rate", "").strip(),
            }

    if duplicate_keys:
        log_lines.append("status=failed")
        log_lines.append("reason=duplicate_fantasypros_match_key")
        for season, team, qb_name in duplicate_keys:
            log_lines.append(
                f"duplicate_fantasypros_match_key | season={season} | team={team} | qb_name={qb_name}"
            )
        write_log(log_lines)
        raise SystemExit(1)

    return fp_rows


def build_qb_stats(
    pbp_path: Path,
    fp_rows: dict[tuple[str, str, str], dict[str, str]],
    log_lines: list[str],
) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    first_qb_by_game_team: dict[tuple[str, str], tuple[float, tuple[str, str, str, str, str]]] = {}
    missing_qb_identity: list[dict[str, str]] = []

    with gzip.open(pbp_path, "rt", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"{pbp_path} has no header row")

        missing_headers = require_headers(reader.fieldnames, PBP_REQUIRED_HEADERS, pbp_path)
        if missing_headers:
            raise ValueError(f"{pbp_path} missing required headers: {missing_headers}")

        for row in reader:
            if not is_one(row.get("qb_dropback")):
                continue

            season = row.get("season", "").strip()
            week = row.get("week", "").strip()
            team = row.get("posteam", "").strip()
            game_id = row.get("game_id", "").strip()
            play_id_value = to_float(row.get("play_id"))

            player_id, qb_name = qb_identity(row)

            if not player_id or not qb_name:
                missing_qb_identity.append(
                    {
                        "season": season,
                        "week": week,
                        "game_id": game_id,
                        "team": team,
                        "play_id": row.get("play_id", "").strip(),
                    }
                )
                continue

            group_key = (season, week, team, player_id, qb_name)

            if group_key not in groups:
                groups[group_key] = {
                    "season": season,
                    "week": week,
                    "team": team,
                    "player_id": player_id,
                    "qb_name": qb_name,
                    "starts": 0,
                    "dropbacks": 0,
                    "qb_epa_total": 0.0,
                    "qb_epa_count": 0,
                    "cpoe_total": 0.0,
                    "cpoe_count": 0,
                    "air_yards_total": 0.0,
                    "sacks": 0,
                    "interceptions": 0,
                    "fumbles": 0,
                }

            group = groups[group_key]
            group["dropbacks"] = int(group["dropbacks"]) + 1
            group["sacks"] = int(group["sacks"]) + to_int_flag(row.get("sack"))
            group["interceptions"] = int(group["interceptions"]) + to_int_flag(row.get("interception"))
            group["fumbles"] = int(group["fumbles"]) + to_int_flag(row.get("fumble"))

            qb_epa = to_float(row.get("qb_epa"))
            if qb_epa is not None:
                group["qb_epa_total"] = float(group["qb_epa_total"]) + qb_epa
                group["qb_epa_count"] = int(group["qb_epa_count"]) + 1

            cpoe = to_float(row.get("cpoe"))
            if cpoe is not None:
                group["cpoe_total"] = float(group["cpoe_total"]) + cpoe
                group["cpoe_count"] = int(group["cpoe_count"]) + 1

            air_yards = to_float(row.get("air_yards"))
            if air_yards is not None:
                group["air_yards_total"] = float(group["air_yards_total"]) + air_yards

            if play_id_value is not None:
                game_team_key = (game_id, team)
                current_first = first_qb_by_game_team.get(game_team_key)

                if current_first is None or play_id_value < current_first[0]:
                    first_qb_by_game_team[game_team_key] = (play_id_value, group_key)

    if missing_qb_identity:
        log_lines.append("status=failed")
        log_lines.append("reason=missing_qb_identity")
        for item in missing_qb_identity:
            log_lines.append(
                "missing_qb_identity"
                f" | season={item['season']}"
                f" | week={item['week']}"
                f" | game_id={item['game_id']}"
                f" | team={item['team']}"
                f" | play_id={item['play_id']}"
            )
        write_log(log_lines)
        raise SystemExit(1)

    for _, group_key in first_qb_by_game_team.values():
        groups[group_key]["starts"] = int(groups[group_key]["starts"]) + 1

    output_rows: list[dict[str, str]] = []
    missing_fp_matches: list[dict[str, str]] = []

    for group_key, group in groups.items():
        season = str(group["season"])
        week = str(group["week"])
        team = str(group["team"])
        player_id = str(group["player_id"])
        qb_name = str(group["qb_name"])

        fp_key = (season, team, qb_name)
        fp_match = fp_rows.get(fp_key)

        if fp_match is None:
            missing_fp_matches.append(
                {
                    "season": season,
                    "week": week,
                    "team": team,
                    "player_id": player_id,
                    "qb_name": qb_name,
                }
            )
            continue

        dropbacks = int(group["dropbacks"])
        sacks = int(group["sacks"])
        interceptions = int(group["interceptions"])
        fumbles = int(group["fumbles"])

        output_rows.append(
            {
                "season": season,
                "week": week,
                "team": team,
                "player_id": player_id,
                "qb_name": qb_name,
                "starts": str(int(group["starts"])),
                "dropbacks": str(dropbacks),
                "epa_per_play": fmt_number(
                    avg(float(group["qb_epa_total"]), int(group["qb_epa_count"]))
                ),
                "cpoe": fmt_number(
                    avg(float(group["cpoe_total"]), int(group["cpoe_count"]))
                ),
                "air_yards": fmt_number(float(group["air_yards_total"])),
                "adjusted_completion_pct": fp_match["adjusted_completion_pct"],
                "sack_rate": fmt_rate(sacks, dropbacks),
                "pressure_to_sack_rate": fp_match["pressure_to_sack_rate"],
                "turnover_worthy_play_rate": fmt_rate(interceptions + fumbles, dropbacks),
                "interception_rate": fmt_rate(interceptions, dropbacks),
                "fumble_rate": fmt_rate(fumbles, dropbacks),
            }
        )

    if missing_fp_matches:
        log_lines.append("status=failed")
        log_lines.append("reason=missing_fantasypros_match")
        for item in missing_fp_matches:
            log_lines.append(
                "missing_fantasypros_match"
                f" | season={item['season']}"
                f" | week={item['week']}"
                f" | team={item['team']}"
                f" | player_id={item['player_id']}"
                f" | qb_name={item['qb_name']}"
            )
        write_log(log_lines)
        raise SystemExit(1)

    output_rows.sort(key=lambda row: to_float(row["week"]) if to_float(row["week"]) is not None else 999)

    return output_rows


def run(season: int) -> int:
    pbp_path = Path(PBP_PATH_TEMPLATE.format(season=season))
    fp_path = Path(FP_PATH_TEMPLATE.format(season=season))
    out_path = Path(OUT_PATH_TEMPLATE.format(season=season))

    log_lines = [
        f"pull_qb_stats.py started | season={season}",
        f"pbp_input={pbp_path}",
        f"fantasypros_input={fp_path}",
        f"output={out_path}",
    ]

    try:
        if not pbp_path.exists():
            log_lines.append("status=failed")
            log_lines.append(f"reason=missing_pbp_input | path={pbp_path}")
            write_log(log_lines)
            return 1

        if not fp_path.exists():
            log_lines.append("status=failed")
            log_lines.append(f"reason=missing_fantasypros_input | path={fp_path}")
            write_log(log_lines)
            return 1

        fp_rows = read_fantasypros(fp_path, log_lines)
        output_rows = build_qb_stats(pbp_path, fp_rows, log_lines)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADERS)
            writer.writeheader()
            writer.writerows(output_rows)

        log_lines.append("status=success")
        log_lines.append(f"output_rows={len(output_rows)}")
        write_log(log_lines)

        print(f"output={out_path}")
        print(f"output_rows={len(output_rows)}")
        print(f"log={LOG_PATH}")

        return 0

    except SystemExit as exc:
        return int(exc.code)

    except Exception as exc:
        log_lines.append("status=failed")
        log_lines.append(f"reason=unexpected_error | error={type(exc).__name__}: {exc}")
        write_log(log_lines)
        print(f"log={LOG_PATH}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True, type=int)
    args = parser.parse_args()

    return run(args.season)


if __name__ == "__main__":
    raise SystemExit(main())
