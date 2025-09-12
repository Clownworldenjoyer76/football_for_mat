#!/usr/bin/env python3
# /scripts/02b_enrich_features.py
# Enrich weekly_clean.csv.gz with additional columns used as model targets.
# - Adds/derives: pass_attempts, rush_attempts, qb_sacks_taken
# - Adds longs if present: rushing_long, longest_reception
# - Adds kicker stats when source columns exist: field_goals_made/attempted,
#   longest_field_goal, kicking_points
# - Adds defensive stats if present: tackles_combined, tackles_assists,
#   interceptions_def, passes_defended
# - Adds team/game points if possible: team_points_for, game_total_points
# Writes in-place to data/features/weekly_clean.csv.gz and logs to output/logs/step02b_enrich.log

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path('.')
FEAT = ROOT / 'data' / 'features' / 'weekly_clean.csv.gz'
LOG  = ROOT / 'output' / 'logs' / 'step02b_enrich.log'
LOG.parent.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    print(msg)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(msg + "\n")

def find_col(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def ensure_col(df: pd.DataFrame, name: str, values):
    if name in df.columns:
        log(f"[KEEP] {name} already exists; not overwriting.")
        return df[name]
    df[name] = values
    log(f"[ADD]  {name} created.")
    return df[name]

def looks_like_qb_row(row: pd.Series) -> bool:
    return (row.get('completions', 0) > 0) or (row.get('passing_yards', 0) > 0)

def looks_like_rush_row(row: pd.Series) -> bool:
    return (row.get('rushing_yards', 0) > 0) and not looks_like_qb_row(row)

def main():
    if not FEAT.exists():
        log(f"[FATAL] Features not found: {FEAT}")
        sys.exit(2)

    df = pd.read_csv(FEAT, low_memory=False)
    created, passthru, placeholders = [], [], []

    # ---- Attempts (separate pass vs. rush) ----
    src_pass = find_col(df, ['pass_attempts', 'attempts_pass', 'attempts_passing'])
    if src_pass:
        ensure_col(df, 'pass_attempts', df[src_pass])
        passthru.append(f"pass_attempts <- {src_pass}")
    else:
        generic_attempts = find_col(df, ['attempts', 'att'])
        if generic_attempts is not None:
            vals = df.apply(lambda r: r[generic_attempts] if looks_like_qb_row(r) else np.nan, axis=1)
            ensure_col(df, 'pass_attempts', vals)
            created.append("pass_attempts (from generic attempts + QB heuristic)")
        else:
            ensure_col(df, 'pass_attempts', np.nan)
            placeholders.append('pass_attempts')
            log("[WARN] pass_attempts not derivable (no attempts). Placeholder created.")

    src_rush = find_col(df, ['rush_attempts', 'rushing_attempts', 'carries'])
    if src_rush:
        ensure_col(df, 'rush_attempts', df[src_rush])
        passthru.append(f"rush_attempts <- {src_rush}")
    else:
        generic_attempts = find_col(df, ['attempts', 'att'])
        if generic_attempts is not None:
            vals = df.apply(lambda r: r[generic_attempts] if looks_like_rush_row(r) else np.nan, axis=1)
            ensure_col(df, 'rush_attempts', vals)
            created.append("rush_attempts (from generic attempts + rush heuristic)")
        else:
            ensure_col(df, 'rush_attempts', np.nan)
            placeholders.append('rush_attempts')
            log("[WARN] rush_attempts not derivable (no attempts). Placeholder created.")

    # ---- QB sacks taken ----
    if 'qb_sacks_taken' in df.columns:
        log("[KEEP] qb_sacks_taken exists.")
        passthru.append("qb_sacks_taken")
    else:
        src_taken = find_col(df, ['sacks_taken', 'qb_sacks', 'sacked'])
        if src_taken:
            ensure_col(df, 'qb_sacks_taken', df[src_taken])
            passthru.append(f"qb_sacks_taken <- {src_taken}")
        else:
            src_def_sacks = find_col(df, ['sacks'])
            if src_def_sacks:
                vals = df.apply(lambda r: r[src_def_sacks] if looks_like_qb_row(r) else 0, axis=1)
                ensure_col(df, 'qb_sacks_taken', vals)
                created.append("qb_sacks_taken (from sacks for QB-like rows)")
            else:
                ensure_col(df, 'qb_sacks_taken', np.nan)
                placeholders.append('qb_sacks_taken')
                log("[WARN] qb_sacks_taken not derivable (no sacks). Placeholder created.")

    # ---- Long plays ----
    long_rush_src = find_col(df, ['rushing_long', 'long_rush', 'rush_long'])
    if long_rush_src:
        ensure_col(df, 'rushing_long', df[long_rush_src])
        passthru.append(f"rushing_long <- {long_rush_src}")
    else:
        ensure_col(df, 'rushing_long', np.nan)
        placeholders.append('rushing_long')
        log("[WARN] rushing_long missing; placeholder.")

    long_rec_src = find_col(df, ['longest_reception', 'rec_long', 'long_rec', 'receiving_long'])
    if long_rec_src:
        ensure_col(df, 'longest_reception', df[long_rec_src])
        passthru.append(f"longest_reception <- {long_rec_src}")
    else:
        ensure_col(df, 'longest_reception', np.nan)
        placeholders.append('longest_reception')
        log("[WARN] longest_reception missing; placeholder.")

    # ---- Kickers ----
    fgm_src = find_col(df, ['field_goals_made', 'fg_made', 'fgm'])
    fga_src = find_col(df, ['field_goals_attempted', 'fg_attempts', 'fga'])
    xpm_src = find_col(df, ['xp_made', 'xpm'])
    fgl_src = find_col(df, ['longest_field_goal', 'fg_long', 'fg_longest'])

    if fgm_src: ensure_col(df, 'field_goals_made', df[fgm_src]); passthru.append(f"field_goals_made <- {fgm_src}")
    else: ensure_col(df, 'field_goals_made', np.nan); placeholders.append('field_goals_made')

    if fga_src: ensure_col(df, 'field_goals_attempted', df[fga_src]); passthru.append(f"field_goals_attempted <- {fga_src}")
    else: ensure_col(df, 'field_goals_attempted', np.nan); placeholders.append('field_goals_attempted')

    if fgl_src: ensure_col(df, 'longest_field_goal', df[fgl_src]); passthru.append(f"longest_field_goal <- {fgl_src}")
    else: ensure_col(df, 'longest_field_goal', np.nan); placeholders.append('longest_field_goal')

    if 'kicking_points' not in df.columns:
        if fgm_src or xpm_src:
            fgm_vals = df[fgm_src] if fgm_src else 0
            xpm_vals = df[xpm_src] if xpm_src else 0
            ensure_col(df, 'kicking_points',
                       3 * pd.to_numeric(fgm_vals, errors='coerce').fillna(0) +
                       pd.to_numeric(xpm_vals, errors='coerce').fillna(0))
            created.append("kicking_points (3*FGM + XP_made)")
        else:
            ensure_col(df, 'kicking_points', np.nan)
            placeholders.append('kicking_points')

    # ---- Defense ----
    for name, cands in {
        'tackles_combined': ['tackles_combined', 'tackles_total', 'total_tackles'],
        'tackles_assists' : ['tackles_assists', 'assist_tackles', 'tackles_ast'],
        'interceptions_def': ['interceptions_def', 'def_interceptions', 'interceptions_defense'],
        'passes_defended' : ['passes_defended', 'pass_deflections', 'pd'],
    }.items():
        src = find_col(df, cands)
        if src:
            ensure_col(df, name, df[src]); passthru.append(f"{name} <- {src}")
        else:
            ensure_col(df, name, np.nan); placeholders.append(name)

    # ---- Team/Game points ----
    team_src = find_col(df, ['team_points_for', 'points_for', 'team_points'])
    if team_src:
        ensure_col(df, 'team_points_for', df[team_src]); passthru.append(f"team_points_for <- {team_src}")
    else:
        ensure_col(df, 'team_points_for', np.nan); placeholders.append('team_points_for')

    if 'game_total_points' in df.columns:
        log("[KEEP] game_total_points exists.")
        passthru.append("game_total_points")
    else:
        if 'team_points_for' in df.columns and df['team_points_for'].notna().any() and 'game_id' in df.columns:
            totals = df[['game_id', 'team_points_for']].groupby('game_id', as_index=False)['team_points_for'].sum()
            totals.rename(columns={'team_points_for':'game_total_points'}, inplace=True)
            df = df.merge(totals, on='game_id', how='left')
            log("[ADD]  game_total_points derived by summing team_points_for per game_id.")
            created.append("game_total_points (from team_points_for)")
        else:
            ensure_col(df, 'game_total_points', np.nan)
            placeholders.append('game_total_points')
            log("[WARN] game_total_points not derivable. Placeholder created.")

    # Save
    df.to_csv(FEAT, index=False)
    log(f"[OK] Wrote enriched features -> {FEAT}")

    log("---- SUMMARY ----")
    for c in created: log(f"[MADE] {c}")
    for c in passthru: log(f"[PASS] {c}")
    for c in placeholders: log(f"[TODO] {c}")

if __name__ == '__main__':
    main()
