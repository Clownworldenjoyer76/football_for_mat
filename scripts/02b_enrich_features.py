#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02b_enrich_features.py
Optional feature enrichment pass.

- Loads:  data/features/weekly_clean.csv.gz
- Adds/normalizes commonly referenced columns used by models/props:
    pass_attempts, rush_attempts,
    qb_sacks_taken,
    rushing_long, longest_reception,
    field_goals_made, field_goals_attempted, longest_field_goal,
    extra_points_made (xpm)

All operations are defensive:
- Works whether inputs are Series or scalars
- Never calls .fillna() on scalars
- Missing sources become 0 with a warning
- Overwrites weekly_clean.csv.gz in-place

Outputs:
- data/features/weekly_clean.csv.gz  (overwritten)
- Prints a short log of what was added/normalized
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd

FEATURES_PATH = Path("data/features/weekly_clean.csv.gz")


# ----------------------------- helpers -----------------------------

def to_num(obj) -> Union[pd.Series, float, int]:
    """
    Convert obj to numeric.
    - If obj is a pandas Series: returns a numeric Series (NaNs for non-numeric)
    - If obj is scalar-like: returns a numeric scalar (NaN if not convertible)
    """
    if isinstance(obj, pd.Series):
        return pd.to_numeric(obj, errors="coerce")
    try:
        # scalar / array-like
        arr = pd.to_numeric(obj, errors="coerce")
        # if it's a 0-dim numpy or python scalar, keep scalar
        if np.ndim(arr) == 0:
            return arr.item() if hasattr(arr, "item") else arr
        # otherwise return as Series
        return pd.Series(arr)
    except Exception:
        return np.nan


def fillna_safe(obj, value=0):
    """
    Fill NaNs only if obj supports .fillna; otherwise:
    - if scalar NaN -> return 'value'
    - else return obj unchanged
    """
    if hasattr(obj, "fillna"):
        return obj.fillna(value)
    # scalar path
    if obj is None:
        return value
    try:
        if np.isnan(obj):
            return value
    except Exception:
        pass
    return obj


def get_first_existing(df: pd.DataFrame, names: Iterable[str]) -> Optional[pd.Series]:
    """Return the first column present in df from 'names', else None."""
    for n in names:
        if n in df.columns:
            return df[n]
    return None


def add_or_normalize(df: pd.DataFrame,
                     out_col: str,
                     candidates: Iterable[str],
                     expr: Optional[str] = None,
                     required: Tuple[str, ...] = (),
                     warn_if_missing: bool = True) -> Tuple[pd.DataFrame, bool]:
    """
    Create/normalize a column.

    - If out_col exists: ensure numeric; done.
    - Else if expr is provided: evaluate using required columns (if any present), else warn/create zeros.
    - Else: use the first available candidate column, numeric; else zeros.

    Returns (df, created_flag)
    """
    created = False

    if out_col in df.columns:
        # normalize to numeric
        df[out_col] = fillna_safe(to_num(df[out_col]), 0)
        return df, created

    if expr:
        # check prerequisites
        if all(req in df.columns for req in required):
            try:
                # evaluate expression in a safe local namespace with numeric coercion
                local = {col: fillna_safe(to_num(df[col]), 0) for col in required}
                df[out_col] = eval(expr, {}, local)
                df[out_col] = fillna_safe(to_num(df[out_col]), 0)
                print(f"[ADD]  {out_col} created.")
                created = True
                return df, created
            except Exception as e:
                if warn_if_missing:
                    print(f"[WARN] {out_col} expression failed ({e}); placeholder created.", file=sys.stderr)

    # fallback: first existing candidate
    src = get_first_existing(df, candidates)
    if src is not None:
        df[out_col] = fillna_safe(to_num(src), 0)
        print(f"[ADD]  {out_col} created.")
        created = True
    else:
        df[out_col] = 0
        if warn_if_missing:
            print(f"[WARN] {out_col} missing; placeholder created.", file=sys.stderr)
        else:
            print(f"[ADD]  {out_col} created (zeros).")
        created = True

    return df, created


# ----------------------------- main -----------------------------

def main():
    if not FEATURES_PATH.exists():
        print(f"ERROR: {FEATURES_PATH} not found. Run 02_build_features.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(FEATURES_PATH, low_memory=False)

    # Normalize IDs/types early
    for c in ("season", "week"):
        if c in df.columns:
            df[c] = fillna_safe(to_num(df[c]), 0).astype(int)

    # --------- passing / rushing attempts ----------
    # pass_attempts: prefer pass_attempts / attempts / pass_att; else derive from comps + incomps
    df, _ = add_or_normalize(
        df,
        out_col="pass_attempts",
        candidates=("pass_attempts", "attempts", "pass_att", "passing_attempts"),
        expr="completions + incompletions",
        required=("completions", "incompletions"),
        warn_if_missing=True,
    )

    # rush_attempts: prefer rush_attempts / rushing_attempts / carries
    df, _ = add_or_normalize(
        df,
        out_col="rush_attempts",
        candidates=("rush_attempts", "rushing_attempts", "carries", "rush_att"),
        warn_if_missing=True,
    )

    # --------- QB sacks taken ----------
    # Many sources lack a per-player 'qb_sacks_taken'; default to 0 if not derivable.
    df, created = add_or_normalize(
        df,
        out_col="qb_sacks_taken",
        candidates=("qb_sacks_taken", "sacks_taken", "qb_sacked"),
        warn_if_missing=True,
    )
    if created and "qb_sacks_taken" in df.columns and df["qb_sacks_taken"].sum() == 0:
        print("[WARN] qb_sacks_taken not derivable (no sacks). Placeholder created.", file=sys.stderr)

    # --------- Longest plays ----------
    df, _ = add_or_normalize(
        df,
        out_col="rushing_long",
        candidates=("rushing_long", "long_rush", "rush_long"),
        warn_if_missing=True,
    )
    df, _ = add_or_normalize(
        df,
        out_col="longest_reception",
        candidates=("longest_reception", "rec_long", "long_rec", "receiving_long"),
        warn_if_missing=True,
    )

    # --------- Kicking ----------
    df, _ = add_or_normalize(
        df,
        out_col="field_goals_made",
        candidates=("field_goals_made", "fgm"),
        warn_if_missing=True,
    )
    df, _ = add_or_normalize(
        df,
        out_col="field_goals_attempted",
        candidates=("field_goals_attempted", "fga"),
        warn_if_missing=True,
    )
    df, _ = add_or_normalize(
        df,
        out_col="longest_field_goal",
        candidates=("longest_field_goal", "fg_long"),
        warn_if_missing=True,
    )
    # Extra points made (xpm)
    df, _ = add_or_normalize(
        df,
        out_col="extra_points_made",
        candidates=("extra_points_made", "xpm", "xp_made"),
        warn_if_missing=True,
    )

    # --------- Sanity: coerce all added numeric columns ---------
    force_numeric = [
        "pass_attempts", "rush_attempts",
        "qb_sacks_taken",
        "rushing_long", "longest_reception",
        "field_goals_made", "field_goals_attempted", "longest_field_goal",
        "extra_points_made",
    ]
    for c in force_numeric:
        if c in df.columns:
            df[c] = fillna_safe(to_num(df[c]), 0)

    # --------- Write back ---------
    df.to_csv(FEATURES_PATH, index=False, compression="gzip")
    print(f"[OK]  Enrichment complete. Wrote {FEATURES_PATH}  rows={len(df)}")

if __name__ == "__main__":
    main()
