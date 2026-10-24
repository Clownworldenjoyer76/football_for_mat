#!/usr/bin/env python3
"""
CFL Game-Analysis parser - v2 (registry-driven, signature-anchored).

Design change from v1:
  v1 located each table by searching the flat TXT for section markers/numbers.
  That broke because section numbers and order reshuffle week to week.
  v2 locates tables the way the structure mapper does - by finding team-row
  blocks and matching each table on its column-header/title *signature*
  (order-independent). The proven row-level parsing from v1 (block parsing,
  time-of-possession, on-1st-down split, converts) is reused unchanged.

Scope of THIS build (validated on the week-11 sample only):
  - generic signature-anchored tables (Section A of the registry)
  - reused special handlers: time-of-possession, on-1st-down (2 files), converts
  - kick returns split into team + opponent files
  - penalties (transposed), fg-miss, cover/return penalties: handlers present,
    flagged EXPERIMENTAL where the week-11 layout is the only test available
  - opp_possession_field_position_vs (weeks 12-19): declared, NOT yet handled -
    cannot be tested without a week 12-19 file

Read-only on inputs. Writes one CSV per table to OUT_DIR (overwrites per run;
v1's manifest-driven upsert/dedup is intentionally set aside for the rebuild).
Stdlib only.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

# --- Knobs (override via argv) ---------------------------------------------
IN_DIR = Path("docs/win/football/cfl/data_dump/raw_txt")
OUT_DIR = Path("docs/win/football/cfl/01_parsed/game_analysis")
# ---------------------------------------------------------------------------

TEAM_ORDER = ["BC", "CGY", "EDM", "HAM", "MTL", "OTT", "SSK", "TOR", "WPG", "CFL"]
TEAM_RE = re.compile(r"^(BC|CGY|EDM|HAM|MTL|OTT|SSK|TOR|WPG|CFL)\b")
NUM_RE = re.compile(
    r"#DIV/0!|###|T\d+|\([-+]?\d+(?:,\d{3})*(?:\.\d+)?\)|[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?"
)
ALPHA_RE = re.compile(r"[A-Za-z]{2,}")
SECTION_HDR_RE = re.compile(r"^\d+[a-z]?\.\s")
PAGEBREAK_RE = re.compile(r"^\d{4}\s+CFL\b")
CONT_RE = re.compile(r"^(?:[(\-+]?\d|T\d|###|#DIV)")

BASE_COLUMNS = ["dump_id", "season", "week", "report_type", "timestamp", "team"]
SOURCE_COLUMNS = ["source_txt_path", "source_pdf_path", "pdf_pages", "row_parse_status"]

FOLD = {
    "atts": "att", "attempts": "att", "plys": "play", "plays": "play",
    "yds": "yards", "oppt": "opp", "opponents": "opp", "opponent": "opp",
    "rnk": "rank", "pnt": "punt",
}


# =========================================================================
# Block finder (from the structure mapper) - replaces marker search
# =========================================================================
def classify(stripped: str):
    if not stripped:
        return "blank", {}
    m = TEAM_RE.match(stripped)
    if m:
        body = stripped[m.end():]
        n_num = len(NUM_RE.findall(body))
        n_alpha = len(ALPHA_RE.findall(body))
        if n_num >= 1 and n_num >= n_alpha:
            return "team", {"code": m.group(1)}
        return "hard", {}
    if SECTION_HDR_RE.match(stripped) or PAGEBREAK_RE.match(stripped):
        return "hard", {}
    if CONT_RE.match(stripped):
        return "cont", {}
    return "hard", {}


def preceding_nonblank(lines, start_line, k=2):
    out, p = [], start_line - 1
    while p >= 0 and len(out) < k:
        s = lines[p].strip()
        if s:
            out.append(s)
        p -= 1
    return list(reversed(out))


def find_blocks(lines):
    """Return blocks: {start,end (1-idx), title, header, teams}."""
    cls = [classify(s.strip()) for s in lines]
    blocks, n, i = [], len(lines), 0
    while i < n:
        if cls[i][0] != "team":
            i += 1
            continue
        start = i
        rows, cur, blank = [], None, 0
        j = i
        while j < n:
            kind, info = cls[j]
            if kind == "team":
                if cur is not None:
                    rows.append(cur)
                cur = [info["code"], j, j + 1]
                blank = 0
            elif kind == "cont":
                if cur is None:
                    break
                cur[2] = j + 1
                blank = 0
            elif kind == "blank":
                blank += 1
                if blank >= 2:
                    break
            else:
                break
            j += 1
        if cur is not None:
            rows.append(cur)
        teams = []
        for r in rows:
            if r[0] not in teams:
                teams.append(r[0])
        hdrs = preceding_nonblank(lines, start, 2)
        blocks.append({
            "start": start + 1,
            "end": (rows[-1][2] if rows else start + 1),
            "title": hdrs[0] if len(hdrs) >= 2 else "",
            "header": hdrs[-1] if hdrs else "",
            "teams": teams,
        })
        i = rows[-1][2] if rows else i + 1
    return blocks


def fold_tokens(s):
    s = re.sub(r"[^A-Za-z ]", " ", s).lower()
    return [FOLD.get(t, t) for t in s.split() if len(t) > 1]


# =========================================================================
# Reused row-level helpers (verbatim from v1 - the parts that worked)
# =========================================================================
def strip_team_tokens(text):
    return re.sub(r"\b(BC|CGY|EDM|HAM|MTL|OTT|SSK|TOR|WPG|CFL)\b", " ", text)


def parse_numeric_values(text, parentheses_negative=False):
    text = strip_team_tokens(text)
    tokens = NUM_RE.findall(text)
    values = []
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


def collect_team_blocks(lines):
    blocks, current_team, current_lines = {}, None, []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        m = TEAM_RE.match(line)
        if m:
            team = m.group(1)
            if current_team is not None and team != current_team:
                if current_team not in blocks or current_team == "CFL":
                    blocks[current_team] = " ".join(current_lines)
            if current_team is None or team != current_team:
                current_team, current_lines = team, [line]
            else:
                current_lines.append(line)
            continue
        if current_team is not None:
            current_lines.append(line)
    if current_team is not None and (current_team not in blocks or current_team == "CFL"):
        blocks[current_team] = " ".join(current_lines)
    return blocks


def team_segment_from_block(block, team):
    marker = f"{team} "
    if block.count(marker) > 1:
        return f"{team} " + block.rsplit(marker, 1)[1]
    return block


def zero_row(count):
    return ["0"] * count


def all_zero_or_blank(values):
    if not values:
        return False
    for v in values:
        if v == "":
            continue
        try:
            if float(v) != 0:
                return False
        except ValueError:
            return False
    return True


def clean_block(block, mode):
    out, stops = block, []
    if mode == "big_play":
        stops = ["Notes:", "1. Offence", "2. Returns", "K/O returns"]
    elif mode == "coaches":
        stops = ["Def. Pass", "Reception", "Roughing", "Fumble?", "No DPI",
                 "No Yards?", "All Others", "BY TYPE"]
    elif mode == "field_goals":
        out = re.sub(r"\s+[-+]?\d+(?:\.\d+)?\s+yds.*$", "", out)
        out = re.sub(r"\s+[-+]?\d+(?:\.\d+)?\s+PerG.*$", "", out)
        out = re.sub(r"\s+4th Quarter.*$", "", out)
        out = re.sub(r"\s+FG Atts.*$", "", out)
        out = re.sub(r"\s+Made\s+\d+.*$", "", out)
        out = re.sub(r"\s+%\s+100%.*$", "", out)
        out = re.sub(r"\s+Atts\s+\d+.*$", "", out)
    elif mode == "kickoffs":
        stops = ["'OT Rec'", "* Regular Kickoffs", "1. Post K/O possessions",
                 "Notes 1.", "SPECIAL TEAMS COVER PENALTIES", "SPECIAL TEAMS RETURN PENALTIES"]
    elif mode == "punts":
        stops = ["* NET YARDS", "** FIELD POSITION", "KICKOFF COVER TEAM"]
    elif mode == "red_zone":
        stops = ["CFL RED ZONE FREQUENCY", "RED ZONE LEGEND:"]
    elif mode == "second_down":
        stops = ["2ND DOWN CONVERSION HISTORY", "2ND DOWN CONVERSION - HISTORY",
                 "OPPONENT 2ND DN CONV", "OPPONENT 2ND DOWN CONV"]
    elif mode == "third_short":
        stops = ["3RD & SHORT RESULTS LEGEND:", "Legend:", "CANADIAN FOOTBALL LEAGUE"]
    elif mode == "turnover":
        stops = ["Notes:", "2023 T/Os by Quarter:"]
    elif mode == "kick_returns":
        stops = ['"BP" =', "long gain kick returns", "OPPONENT PUNT RETURNS",
                 "J) SPECIAL TEAMS", "NOTES & LEGEND"]
    elif mode == "return_penalties":
        stops = ["NOTES & LEGEND", "* 'Penalties'", "CFL PUNT RETURNS"]
    else:
        stops = ["Notes:"]
    for stop in stops:
        pos = out.find(stop)
        if pos != -1:
            out = out[:pos]
    return out


def normalize_converts(values, team, expected_count):
    if team != "CFL" and values[:3] == ["0", "0", "0"] and len(values) >= 9 and values[3] != "0":
        values = values[:3] + ["0"] + values[3:8]
        return values[:expected_count], "OK"
    if len(values) >= expected_count:
        return values[:expected_count], "OK_AGGREGATE" if team == "CFL" else "OK"
    return (values + [""] * expected_count)[:expected_count], f"NEEDS_REVIEW_{len(values)}_OF_{expected_count}"


def parse_top_block(block, expected_count, team, gp_map):
    if team != "CFL" and gp_map.get(team) == "0":
        return "", zero_row(expected_count), "OK_ZERO_GP"
    m = re.search(r"\b\d{1,2}:\d{2}\b", block)
    top_value = m.group(0) if m else ""
    cleaned = block.replace(top_value, " ", 1) if top_value else block
    values = parse_numeric_values(cleaned)
    if team == "CFL" and len(values) >= 8:
        values = [values[0], values[1], values[2], "", values[3], values[4], "",
                  values[5], values[6], values[7]]
        return top_value, values[:expected_count], "OK_AGGREGATE"
    if len(values) >= expected_count:
        return top_value, values[:expected_count], "OK"
    if all_zero_or_blank(values):
        return top_value, zero_row(expected_count), "OK"
    return top_value, (values + [""] * expected_count)[:expected_count], f"NEEDS_REVIEW_{len(values)}_OF_{expected_count}"


def split_first_down_values(values, team):
    offence_values = [""] * 6
    made_values = [""] * 10
    if team == "CFL":
        if len(values) >= 4:
            offence_values = [values[0], values[1], "", values[2], values[3], ""]
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


# =========================================================================
# Context (derived from filename; wire a manifest here if dump_id must match)
# =========================================================================
class Ctx:
    def __init__(self, txt_path):
        self.txt_path = txt_path
        stem = txt_path.stem
        self.dump_id = stem
        mseason = re.search(r"(20\d{2})", stem)
        mweek = re.search(r"week[_-]?(\d+)", stem)
        mts = re.search(r"(\d{8}_\d{6})$", stem)
        self.season = mseason.group(1) if mseason else ""
        self.week = mweek.group(1).lstrip("0") if mweek else ""
        self.timestamp = mts.group(1) if mts else ""
        self.report_type = "game_analysis"
        self.pdf_path = txt_path.with_suffix(".pdf")
        self.pdf_pages = ""


def base_row(ctx, team, status):
    return {
        "dump_id": ctx.dump_id, "season": ctx.season, "week": ctx.week,
        "report_type": ctx.report_type, "timestamp": ctx.timestamp, "team": team,
        "source_txt_path": str(ctx.txt_path), "source_pdf_path": str(ctx.pdf_path),
        "pdf_pages": ctx.pdf_pages, "row_parse_status": status,
    }


# =========================================================================
# Column schemas (harvested from v1)
# =========================================================================
COLS = {
    "team_scoring_breakdown": ["gp","points_for","points_for_avg","points_against","points_against_avg","point_diff","home_points","away_points","offensive_team_points","offensive_team_avg","offensive_opponent_points","offensive_opponent_avg"],
    "touchdown": ["td","convert_1","convert_2","field_goals","singles","special_teams_td","rush_td","pass_td","interception_return_td","fumble_return_td","punt_return_td","kickoff_return_td","missed_fg_return_td","team_td","q1_points","q2_points","q3_points","q4_points","ot_points"],
    "turnover": ["turnover_ratio","rank","giveaways_fumble","giveaways_interception","giveaways_downs","giveaways_total","giveaway_points_allowed","takeaway_offence","takeaway_return","takeaways_fumble","takeaways_interception","takeaways_downs","takeaways_total","takeaway_points"],
    "possession": ["possessions","td_drives","fg_drives","missed_fg_drives","turnover_drives","punt_drives","other_drives","own_1_20_drives","own_1_20_td","own_21_40_drives","own_21_40_td","own_41_54_drives","own_41_54_td","opp_55_41_drives","opp_55_41_td","opp_40_21_drives","opp_40_21_td","opp_20_1_drives","opp_20_1_td","two_and_outs","two_and_out_pct"],
    "top": ["time_of_possession","team_possessions","team_field_position_yards","team_avg_start_yard_line","team_start_yard_line_rank","opponent_end_start_count","opponent_end_start_td","start_yard_line_gap","opponent_possessions","opponent_field_position_yards","opponent_avg_start_yard_line"],
    "big_play": ["team_big_plays_total","team_big_play_rank","team_rush_20_plus","team_pass_30_plus","team_punt_return_big_plays","team_kickoff_return_big_plays","team_fg_miss_return_big_plays","opponent_big_plays_total","opponent_rush_20_plus","opponent_pass_30_plus","opponent_punt_return_big_plays","opponent_kickoff_return_big_plays","opponent_fg_miss_return_big_plays","big_play_diff"],
    "red_zone": ["team_red_zone_att","team_red_zone_td","team_red_zone_td_pct","team_red_zone_rank","team_red_zone_fg","team_red_zone_turnover","team_red_zone_other","team_red_zone_points","team_red_zone_points_pct","opponent_red_zone_att","opponent_red_zone_td","opponent_red_zone_td_pct","opponent_red_zone_rank","opponent_red_zone_fg","opponent_red_zone_turnover","opponent_red_zone_other","opponent_red_zone_points"],
    "coaches": ["challenges","challenges_won","challenge_pct","offence_challenges","offence_challenges_won","defence_challenges","defence_challenges_won","penalty_challenges","no_penalty_challenges","pass_challenges","fumble_challenges","spot_challenges"],
    "net": ["gp","net_yards","pass_yards","rush_yards","team_losses","yards_per_game","yards_per_game_rank","plays","yards_per_play","yards_per_play_rank","pass_attempts","sacks_allowed","rush_attempts","other_team_losses","rush_call_pct","pass_plus_call_pct","other_call_pct"],
    "opp_net": ["gp","opponent_net_yards","opponent_pass_yards","opponent_rush_yards","opponent_team_losses","opponent_yards_per_game","opponent_yards_per_game_rank","opponent_plays","opponent_yards_per_play","opponent_yards_per_play_rank","opponent_pass_attempts","opponent_sacks_allowed","opponent_rush_attempts","opponent_other_team_losses","opponent_rush_call_pct","opponent_pass_plus_call_pct","opponent_other_call_pct"],
    "rushing": ["team_rush_atts","team_rush_yards","team_rush_avg","team_rush_td","team_rush_10_plus","team_rush_20_plus","team_rush_yards_per_game","first_down_rush_atts","first_down_rush_yards","first_down_rush_avg","qb_rush_atts","qb_rush_yards","qb_escape_runs","opponent_rush_atts","opponent_rush_yards","opponent_rush_avg","opponent_rush_td","opponent_rush_10_plus","opponent_rush_20_plus"],
    "passing_base": ["pass_attempts","pass_completions","completion_pct","pass_yards","interceptions","pass_td","longest_pass","passes_30_plus","second_down_conversion_receptions","pass_efficiency","interception_pct","yards_per_attempt","yac_yards","td_int_ratio","pass_yards_per_game","average_depth","sacks_allowed","sack_pct","qb_escape_runs","dropbacks"],
    "opp_passing_base": ["opponent_pass_attempts","opponent_pass_completions","opponent_completion_pct","opponent_pass_yards","opponent_interceptions","opponent_pass_td","opponent_longest_pass","opponent_passes_30_plus","opponent_second_down_conversion_receptions","opponent_pass_efficiency","opponent_interception_pct","opponent_yards_per_attempt","opponent_yac_yards","opponent_td_int_ratio","opponent_average_depth","opponent_games","opponent_dropbacks","opponent_sack_pct","opponent_sacks_allowed","opponent_qb_escape_runs"],
    "passing_depth": ["depth_0_9_att","depth_0_9_comp","depth_0_9_comp_pct","depth_0_9_yards","depth_0_9_int","depth_0_9_td","depth_0_9_efficiency","depth_10_19_att","depth_10_19_comp","depth_10_19_comp_pct","depth_10_19_yards","depth_10_19_int","depth_10_19_td","depth_10_19_efficiency","depth_20_plus_att","depth_20_plus_comp","depth_20_plus_comp_pct","depth_20_plus_yards","depth_20_plus_int","depth_20_plus_td","depth_20_plus_efficiency"],
    "second_down": ["all_att","all_made","all_pct","rank","one_to_three_att","one_to_three_made","one_to_three_pct","four_to_six_att","four_to_six_made","four_to_six_pct","seven_plus_att","seven_plus_made","seven_plus_pct","yards_to_go_total","yards_to_go_avg","yards_to_go_rank"],
    "third_short": ["third_short_att","third_short_made","third_short_fail","opponent_third_short_att","opponent_third_short_made","opponent_third_short_fail"],
    "field_goals": ["fg_attempts","fg_made","fg_pct","fg_long","fg_singles","fg_under_40_attempts","fg_under_40_made","fg_under_40_pct","fg_40_plus_attempts","fg_40_plus_made","fg_40_plus_pct","fg_50_plus_attempts","fg_50_plus_made","fg_made_yards","fg_made_avg_yards","fg_all_attempt_yards","fg_all_attempt_avg_yards"],
    "fg_miss": ["fg_missed","fg_missed_att_yards","fg_missed_avg","fg_missed_singles","opponent_fg_miss_return_no","opponent_fg_miss_return_yards","opponent_fg_miss_return_long","opponent_fg_miss_return_td"],
    "converts": ["convert_1_attempts","convert_1_made","convert_1_pct","convert_1_missed","convert_2_attempts","convert_2_made","convert_2_pct","convert_2_rush","convert_2_pass"],
    "kick_returns": ["punt_return_no","punt_return_yards","punt_return_avg","punt_return_long","punt_return_td","punt_return_30_plus","punt_return_10_plus","punt_return_rank","kickoff_return_no","kickoff_return_yards","kickoff_return_avg","kickoff_return_long","kickoff_return_td","kickoff_return_40_plus","kickoff_return_rank","fg_miss_return_no","fg_miss_return_yards","fg_miss_return_long","fg_miss_return_td","fg_miss_return_30_plus","kick_return_big_plays","punt_return_td_total","kickoff_return_td_total","fg_miss_return_td_total","kick_return_td_total"],
    "punts": ["punt_no","punt_yards","punt_avg","punt_long","punt_singles","punt_inside_10","opponent_punt_return_no","opponent_punt_return_yards","opponent_punt_return_avg","opponent_punt_return_long","opponent_punt_return_td","opponent_punt_return_30_plus","net_punt_avg","net_punt_yards","net_punt_rank","single_adjustment","cover_penalty_adjustment","return_penalty_adjustment","field_position_yards","field_position_avg"],
    "kickoffs": ["kickoff_no","kickoff_yards","kickoff_avg","kickoff_long","kickoff_singles","own_team_recoveries","opponent_kickoff_return_no","opponent_kickoff_return_yards","opponent_kickoff_return_avg","opponent_kickoff_return_long","opponent_kickoff_return_td","opponent_kickoff_return_40_plus","regular_kickoff_no","regular_kickoff_yards","regular_kickoff_avg","regular_kickoff_rank","post_kickoff_possessions","post_kickoff_yards","post_kickoff_avg_start_yard_line","post_kickoff_rank"],
    "cover_penalties": ["punt_cover_penalties","kickoff_cover_penalties","total_cover_penalties","cover_penalty_rank","illegal_punt_penalties","illegal_kickoff_penalties"],
    "return_penalties": ["punt_return_penalties","kickoff_return_penalties","total_return_penalties","return_penalty_per_game","return_penalty_rank"],
    "penalties": ["gp","penalties_all","penalties_avg","penalties_accepted","penalties_declined","penalty_yards","offence_penalties","defence_penalties","special_teams_penalties","punt_cover_penalties","kickoff_cover_penalties","punt_return_penalties","kickoff_return_penalties"],
}

# =========================================================================
# Registry: each entry = (file, sig tokens (all must be in title+header),
#           title_re (raw, on title line; None=any), handler, mode, cols)
# =========================================================================
R = lambda file, sig, title_re, handler, mode, cols: {
    "file": file, "sig": sig, "title_re": title_re,
    "handler": handler, "mode": mode, "cols": cols}

REGISTRY = [
    R("team_scoring_breakdown.csv", ["scoring","breakdown","pf","pa"], None, "generic", "team_scoring", COLS["team_scoring_breakdown"]),
    R("team_touchdowns_points_by_quarter.csv", ["rush","pass","fgm","kor"], r"TEAM SCORING", "generic", "default", COLS["touchdown"]),
    R("opponent_touchdowns_points_by_quarter.csv", ["rush","pass","fgm","kor"], r"OPP.*SCORING|OPPONENT SCORING", "generic", "default", COLS["touchdown"]),
    R("turnover_analysis.csv", ["ratio","fum","int","dns"], None, "generic", "turnover", COLS["turnover"]),
    R("team_possessions.csv", ["poss","punt","drv"], r"TEAM POSSESS", "generic", "default", COLS["possession"]),
    R("opponent_possessions.csv", ["poss","punt","drv"], r"OPP.*POSSESS|OPPONENT POSSESS", "generic", "default", COLS["possession"]),
    R("time_of_possession_field_position.csv", ["top","poss","fp","ydl"], None, "top", "default", COLS["top"]),
    R("big_play_analysis.csv", ["total","rush","pass","rets"], None, "generic", "big_play", COLS["big_play"]),
    R("red_zone_results.csv", ["opportunities","td","fg"], None, "generic", "red_zone", COLS["red_zone"]),
    R("coaches_challenges.csv", ["chall","won","pct"], None, "generic", "coaches", COLS["coaches"]),
    R("team_net_offence.csv", ["yards","pass","rush","play","rank"], r"TEAM OFFENCE|TEAM NET", "generic", "net_offence", COLS["net"]),
    R("opponent_net_offence.csv", ["yards","pass","rush","play","rank"], r"OPP.*OFFENCE|OPPT OFFENCE|OPPONENT NET", "generic", "net_offence", COLS["opp_net"]),
    R("first_down_offence.csv", ["play","avg","yd","rank","rush"], None, "first_down", "default", None),  # produces 2 files
    R("rushing_analysis.csv", ["yards","avg","perg","esc"], None, "generic", "default", COLS["rushing"]),
    R("passing_base.csv", ["att","com","effic","yac"], r"TEAM PASSING DATA|2023 TEAM PASSING", "generic", "passing_base", COLS["passing_base"]),
    R("opponent_passing_base.csv", ["att","com","effic"], r"OPPONENT PASSING DATA|OPP.*PASSING DATA", "generic", "opponent_passing_base", COLS["opp_passing_base"]),
    R("opponent_passing_depth.csv", ["att","com","effic"], r"OPPT", "generic", "default", COLS["passing_depth"]),
    R("passing_depth.csv", ["att","com","effic"], None, "generic", "default", COLS["passing_depth"]),
    R("second_down_conversions.csv", ["att","md","yards","avg"], r"2ND DN CONVERSIONS - ALL|2ND DN CONVERSIONS", "generic", "second_down", COLS["second_down"]),
    R("opponent_second_down_conversions.csv", ["att","md","yards","avg"], r"OPPONENT 2ND DN", "generic", "second_down", COLS["second_down"]),
    R("third_short_results.csv", ["att","md","fail"], None, "third_short", "third_short", COLS["third_short"]),
    R("special_teams_field_goals.csv", ["fga","md","lg"], None, "generic", "field_goals", COLS["field_goals"]),
    R("special_teams_fg_miss.csv", ["ms","missed","out"], None, "generic", "default", COLS["fg_miss"]),
    R("special_teams_converts.csv", ["miss","rsh","pass","def"], None, "converts", "default", COLS["converts"]),
    R("special_teams_kick_returns_team.csv", ["kor","fgm","tot","bp"], r"TEAM PUNT RETURNS", "generic", "kick_returns", COLS["kick_returns"]),
    R("special_teams_kick_returns_opponent.csv", ["kor","fgm","tot","bp"], r"OPPONENT PUNT RETURNS|OPP.*PUNT RETURNS", "generic", "kick_returns", COLS["kick_returns"]),
    R("special_teams_punts.csv", ["netyd","cover","retn","fp"], None, "generic", "punts", COLS["punts"]),
    R("special_teams_kickoffs.csv", ["rec","poss","yl"], None, "generic", "kickoffs", COLS["kickoffs"]),
    R("special_teams_cover_penalties.csv", ["punt","total","illeg"], None, "generic", "default", COLS["cover_penalties"]),
    R("special_teams_return_penalties.csv", ["pr","kor","perg"], None, "generic", "return_penalties", COLS["return_penalties"]),
]


# =========================================================================
# Matcher + region computation
# =========================================================================
def assign(blocks):
    used, result = set(), []
    for e in REGISTRY:
        chosen = None
        for bi, b in enumerate(blocks):
            if bi in used:
                continue
            toks = set(fold_tokens(b["title"] + " " + b["header"]))
            if not all(t in toks for t in e["sig"]):
                continue
            if e["title_re"] and not re.search(e["title_re"], b["title"], re.I):
                continue
            chosen = (bi, b)
            break
        if chosen:
            used.add(chosen[0])
            result.append((e, chosen[1]))
        else:
            result.append((e, None))
    return result


def regions(assigned, total_lines):
    starts = sorted(b["start"] for e, b in assigned if b)
    out = {}
    for e, b in assigned:
        if not b:
            out[id(e)] = None
            continue
        nxt = next((s for s in starts if s > b["start"]), total_lines + 1)
        out[id(e)] = (b["start"], nxt)
    return out


# =========================================================================
# Handlers
# =========================================================================
def normalize_for_spec(values, team, mode, expected_count, gp, gp_map):
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
    if mode == "kickoffs" and team != "CFL" and len(values) == expected_count - 1:
        # post_kickoff_rank is genuinely absent from the source text some weeks
        return (values + [""])[:expected_count], "OK_MISSING_RANK"
    if mode == "punts" and team == "CFL" and len(values) == expected_count - 1:
        values = values[:14] + [""] + values[14:]
        return values[:expected_count], "OK_AGGREGATE"
    if mode == "field_goals" and len(values) == expected_count - 1:
        if len(values) > 7 and values[5] == "0" and values[6] == "0":
            values = values[:7] + ["0"] + values[7:]
            return values[:expected_count], "OK"
        if len(values) > 12 and values[8] == "0" and values[9] == "0" and values[10] == "0" and values[11] == "0":
            values = values[:12] + ["0"] + values[12:]
            return values[:expected_count], "OK"
    if mode in ("passing_base", "opponent_passing_base") and len(values) == expected_count - 1:
        if len(values) > 4 and values[4] == "0":
            values = values[:13] + [""] + values[13:]
            return values[:expected_count], "OK"
    if mode == "opponent_passing_base" and team == "CFL":
        values = values[:17] + [""] + values[17:]
        return values[:expected_count], "OK_AGGREGATE"
    if mode == "net_offence" and team == "CFL" and len(values) == 15:
        values = values[:6] + [""] + values[6:8] + [""] + values[8:]
        return values[:expected_count], "OK_AGGREGATE"
    if mode == "red_zone" and team == "CFL" and len(values) == 15:
        values = values[:3] + [""] + values[3:12] + [""] + values[12:]
        return values[:expected_count], "OK_AGGREGATE"
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
    return (values + [""] * expected_count)[:expected_count], f"NEEDS_REVIEW_{len(values)}_OF_{expected_count}"


def handle_generic(region_lines, e, ctx, gp_map):
    blocks = collect_team_blocks(region_lines)
    cols, mode = e["cols"], e["mode"]
    rows = []
    for team in TEAM_ORDER:
        block = blocks.get(team, "")
        if not block:
            row = base_row(ctx, team, "MISSING_ROW")
            for c in cols:
                row[c] = ""
            rows.append(row)
            continue
        cleaned = clean_block(block, mode)
        if mode in ("team_scoring", "kickoffs"):
            cleaned = team_segment_from_block(cleaned, team)
        values = parse_numeric_values(cleaned, parentheses_negative=(mode == "punts"))
        vals, status = normalize_for_spec(values, team, mode, len(cols), gp_map.get(team, ""), gp_map)
        row = base_row(ctx, team, status)
        for c, v in zip(cols, vals):
            row[c] = v
        rows.append(row)
    return [(e["file"], cols, rows)]


def handle_top(region_lines, e, ctx, gp_map):
    blocks = collect_team_blocks(region_lines)
    cols = e["cols"]
    numeric = cols[1:]
    rows = []
    for team in TEAM_ORDER:
        block = blocks.get(team, "")
        if not block:
            row = base_row(ctx, team, "MISSING_ROW")
            for c in cols:
                row[c] = ""
            rows.append(row)
            continue
        top_value, values, status = parse_top_block(team_segment_from_block(block, team), len(numeric), team, gp_map)
        row = base_row(ctx, team, status)
        row[cols[0]] = top_value
        for c, v in zip(numeric, values):
            row[c] = v
        rows.append(row)
    return [(e["file"], cols, rows)]


def handle_first_down(region_lines, e, ctx, gp_map):
    off_cols = ["team_first_down_plays","team_first_down_avg_yards","team_first_down_rank","opponent_first_down_plays","opponent_first_down_avg_yards","opponent_first_down_rank"]
    made_cols = ["first_down_call_rush","first_down_call_pass_plus","team_first_downs_total","team_first_downs_rush","team_first_downs_pass","team_first_downs_penalty","opponent_first_downs_total","opponent_first_downs_rush","opponent_first_downs_pass","opponent_first_downs_penalty"]
    blocks = collect_team_blocks(region_lines)
    off_rows, made_rows = [], []
    for team in TEAM_ORDER:
        block = blocks.get(team, "")
        if not block:
            ov, mv, status = [""] * 6, [""] * 10, "MISSING_ROW"
        else:
            values = parse_numeric_values(clean_block(team_segment_from_block(block, team), "default"))
            if team != "CFL" and values and values[0] == "0":
                ov, mv, status = zero_row(6), zero_row(10), "OK_ZERO_GP"
            else:
                ov, mv, status = split_first_down_values(values, team)
        orow = base_row(ctx, team, status)
        mrow = base_row(ctx, team, status)
        for c, v in zip(off_cols, ov):
            orow[c] = v
        for c, v in zip(made_cols, mv):
            mrow[c] = v
        off_rows.append(orow)
        made_rows.append(mrow)
    return [("first_down_offence.csv", off_cols, off_rows),
            ("first_downs_made.csv", made_cols, made_rows)]


def _slash_groups(block):
    return list(re.finditer(r"(\d+)\s*/\s*(\d+)", block))


def _nums_until_year(text):
    nums = parse_numeric_values(text)
    cut = next((i for i, v in enumerate(nums) if re.fullmatch(r"20\d\d", v)), len(nums))
    return nums[:cut]


def handle_converts(region_lines, e, ctx, gp_map):
    # Row = C-1 "att / md" ... [optional YEAR history] ... C-2 "att / md" ...
    # Anchor on the two slash groups; skip interleaved year-history columns.
    blocks = collect_team_blocks(region_lines)
    cols = e["cols"]
    rows = []
    for team in TEAM_ORDER:
        block = blocks.get(team, "")
        g = _slash_groups(block)
        if not block or len(g) < 2:
            vals = [""] * 9
            status = "MISSING_ROW" if not block else f"NEEDS_REVIEW_{len(g)}_GROUPS"
        else:
            c1a, c1m = g[0].group(1), g[0].group(2)
            c2a, c2m = g[1].group(1), g[1].group(2)
            mid = _nums_until_year(block[g[0].end():g[1].start()])
            tail = parse_numeric_values(block[g[1].end():])
            c1pct = mid[0] if len(mid) > 0 else ""
            c1miss = mid[1] if len(mid) > 1 else ""
            c2pct = tail[0] if len(tail) > 0 else ""
            c2rush = tail[1] if len(tail) > 1 else ""
            c2pass = tail[2] if len(tail) > 2 else ""
            vals = [c1a, c1m, c1pct, c1miss, c2a, c2m, c2pct, c2rush, c2pass]
            status = "OK_AGGREGATE" if team == "CFL" else "OK"
        row = base_row(ctx, team, status)
        for c, v in zip(cols, vals):
            row[c] = v
        rows.append(row)
    return [(e["file"], cols, rows)]


def handle_third_short(region_lines, e, ctx, gp_map):
    # Row = team "att / md" ... opponent "att / md" ...
    # Fail is always att-md; computing it avoids layout differences (some
    # weeks show a fail count after the slash group, others a percentage).
    blocks = collect_team_blocks(region_lines)
    cols = e["cols"]
    rows = []
    for team in TEAM_ORDER:
        block = blocks.get(team, "")
        g = _slash_groups(block)
        if not block or len(g) < 2:
            vals = [""] * 6
            status = "MISSING_ROW" if not block else f"NEEDS_REVIEW_{len(g)}_GROUPS"
        else:
            ta, tm = int(g[0].group(1)), int(g[0].group(2))
            oa, om = int(g[1].group(1)), int(g[1].group(2))
            vals = [ta, tm, ta - tm, oa, om, oa - om]
            status = "OK_AGGREGATE" if team == "CFL" else "OK"
        row = base_row(ctx, team, status)
        for c, v in zip(cols, vals):
            row[c] = v
        rows.append(row)
    return [(e["file"], cols, rows)]


# --- Penalties: transposed table (teams as columns). EXPERIMENTAL. ---------
PEN_LABEL_MAP = [
    (r"TEAM TOTALS", "penalties_all"),
    (r"PENALTY YARDS", "penalty_yards"),
    (r"GAMES PLAYED", "gp"),
    (r"PENALTIES PER GAME", "penalties_avg"),
    (r"^\s*OFFENCE\b", "offence_penalties"),
    (r"^\s*DEFENCE\b", "defence_penalties"),
    (r"^\s*SPECIAL TEAMS\b", "special_teams_penalties"),
    (r"Cover:?\s*Punts", "punt_cover_penalties"),
    (r"Return:?\s*Punts", "punt_return_penalties"),
]
# rows whose label is just "Kickoffs" - disambiguated by the preceding Cover/Return row
PEN_LOOKAHEAD = {"punt_cover_penalties": "kickoff_cover_penalties",
                 "punt_return_penalties": "kickoff_return_penalties"}
PEN_TEAM_ORDER = ["CFL", "BC", "CGY", "EDM", "HAM", "MTL", "OTT", "SSK", "TOR", "WPG"]


PEN_ROW_HDR_RE = re.compile(r"\bGP\b.*\bALL\b.*\bAcc\b.*\bDecl\b.*\b(?:YDS|Yards)\b", re.I)


def handle_penalties_rowform(lines, ctx):
    # Weeks 1-10 layout: "PENALTIES - TEAM REPORTS" - one row per team,
    # columns already in COLS["penalties"] order: GP ALL Avg Acc Decl YDS
    # Off Def SpTm Punt K/Os PtR KOR.
    cols = COLS["penalties"]
    hdr_idx = next((i for i, s in enumerate(lines) if PEN_ROW_HDR_RE.search(s)), None)
    rows = []
    if hdr_idx is None:
        for team in TEAM_ORDER:
            row = base_row(ctx, team, "MISSING_SECTION")
            for c in cols:
                row[c] = ""
            rows.append(row)
        return rows
    window = lines[hdr_idx + 1: hdr_idx + 40]
    stop_at = next((i for i, s in enumerate(window)
                    if re.search(r"^PER GAME|CANADIAN FOOTBALL LEAGUE", s)), len(window))
    window = window[:stop_at]
    blocks = collect_team_blocks(window)
    for team in TEAM_ORDER:
        block = blocks.get(team, "")
        if not block:
            row = base_row(ctx, team, "MISSING_ROW")
            for c in cols:
                row[c] = ""
            rows.append(row)
            continue
        values = parse_numeric_values(team_segment_from_block(block, team))
        if team == "CFL" and values and values[0] == "2023":
            values = values[1:]  # CFL row has no GP column, starts with the year label
            vals = [""] + values[:len(cols) - 1]
            vals = (vals + [""] * len(cols))[:len(cols)]
            status = "OK_AGGREGATE"
        elif all_zero_or_blank(values):
            vals, status = zero_row(len(cols)), "OK_ZERO_GP"
        elif len(values) >= len(cols):
            vals, status = values[:len(cols)], ("OK_AGGREGATE" if team == "CFL" else "OK")
        else:
            vals = (values + [""] * len(cols))[:len(cols)]
            status = f"NEEDS_REVIEW_{len(values)}_OF_{len(cols)}"
        row = base_row(ctx, team, status)
        for c, v in zip(cols, vals):
            row[c] = v
        rows.append(row)
    return rows


def handle_penalties(lines, ctx):
    cols = COLS["penalties"]
    hdr_idx = next((i for i, s in enumerate(lines)
                    if re.search(r"PENALTIES BY TEAM.*\bBC\b.*\bWPG\b", s)), None)
    if hdr_idx is None:
        # Older report layout (weeks 1-10) has no transposed table at all -
        # use the team-row summary instead so these weeks aren't blank.
        return [("penalties_team_report.csv", cols, handle_penalties_rowform(lines, ctx))]
    data = {t: {} for t in TEAM_ORDER}
    status = "OK"
    window = lines[hdr_idx: hdr_idx + 40]
    for i, line in enumerate(window):
        for pat, col in PEN_LABEL_MAP:
            if re.search(pat, line):
                vals = parse_numeric_values(line)
                for t, v in zip(PEN_TEAM_ORDER, vals):
                    data[t][col] = v
                # the row immediately after a Cover/Return line is its Kickoffs row
                if col in PEN_LOOKAHEAD and i + 1 < len(window):
                    kvals = parse_numeric_values(window[i + 1])
                    for t, v in zip(PEN_TEAM_ORDER, kvals[:10]):
                        data[t][PEN_LOOKAHEAD[col]] = v
                break
    rows = []
    for team in TEAM_ORDER:
        row = base_row(ctx, team, status if data[team] else "MISSING_ROW")
        for c in cols:
            row[c] = data[team].get(c, "")
        rows.append(row)
    return [("penalties_team_report.csv", cols, rows)]


HANDLERS = {"generic": handle_generic, "top": handle_top,
            "first_down": handle_first_down, "converts": handle_converts,
            "third_short": handle_third_short}


# =========================================================================
# Engine
# =========================================================================
def compute_gp_map(assigned, lines):
    for e, b in assigned:
        if e["file"] == "team_scoring_breakdown.csv" and b:
            reg = lines[b["start"] - 1:]
            blocks = collect_team_blocks(reg)
            gp = {}
            for team in TEAM_ORDER:
                seg = blocks.get(team, "")
                v = parse_numeric_values(team_segment_from_block(seg, team)) if seg else []
                gp[team] = v[0] if v else ""
            return gp
    return {}


def parse_file(txt_path, sink):
    ctx = Ctx(txt_path)
    lines = txt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    blocks = find_blocks(lines)
    assigned = assign(blocks)
    reg = regions(assigned, len(lines))
    gp_map = compute_gp_map(assigned, lines)

    for e, b in assigned:
        handler = HANDLERS.get(e["handler"])
        if not b:
            continue
        span = reg[id(e)]
        region_lines = lines[span[0] - 1: span[1] - 1]
        for fname, cols, rows in handler(region_lines, e, ctx, gp_map):
            sink.setdefault(fname, (cols, []))
            sink[fname][1].extend(rows)

    for fname, cols, rows in handle_penalties(lines, ctx):
        sink.setdefault(fname, (cols, []))
        sink[fname][1].extend(rows)

    found = sum(1 for e, b in assigned if b)
    missing = [e["file"] for e, b in assigned if not b]
    return found, missing


def main(argv):
    in_dir = Path(argv[1]) if len(argv) > 1 else IN_DIR
    out_dir = Path(argv[2]) if len(argv) > 2 else OUT_DIR
    if not in_dir.exists():
        print(f"Input dir not found: {in_dir}")
        return 1
    files = sorted(in_dir.glob("*.txt"))
    if not files:
        print(f"No .txt in {in_dir}")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    sink = {}
    for p in files:
        found, missing = parse_file(p, sink)
        print(f"{p.name[:50]:<50} matched={found}/{len(REGISTRY)}  missing={','.join(m.replace('.csv','') for m in missing) or '-'}")
    for fname, (cols, rows) in sorted(sink.items()):
        fieldnames = BASE_COLUMNS + cols + SOURCE_COLUMNS
        with (out_dir / fname).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    print(f"\nwrote {len(sink)} CSVs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))