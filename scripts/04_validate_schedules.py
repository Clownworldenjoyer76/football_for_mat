#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full script path: /mnt/data/football_for_mat-main/scripts/04_validate_schedules.py

Purpose:
  Validate normalized schedules against schema (config/schemas/schedule_schema.yml)
  and emit CSV reports for errors, duplicates, and a high-level summary.

Inputs:
  - data/processed/schedules/schedules_normalized.csv
  - config/schemas/schedule_schema.yml

Outputs:
  - output/reports/schedule_validation_errors.csv
  - output/reports/schedule_dupes.csv
  - output/reports/schedule_summary.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_YML = ROOT / "config" / "schemas" / "schedule_schema.yml"
NORM_CSV = ROOT / "data" / "processed" / "schedules" / "schedules_normalized.csv"

REPORTS_DIR = ROOT / "output" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ERR_CSV = REPORTS_DIR / "schedule_validation_errors.csv"
DUP_CSV = REPORTS_DIR / "schedule_dupes.csv"
SUM_CSV = REPORTS_DIR / "schedule_summary.csv"


def load_schema(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _enum_ok(values: List[str], val: str) -> bool:
    return str(val).strip().lower() in {str(v).strip().lower() for v in values}


def validate_types(df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
    """Validate field presence and basic types/enums per schema."""
    errors = []
    fields = schema.get("fields", [])
    # Build lookups
    type_map = {f.get("name"): f.get("type") for f in fields}
    required_map = {f.get("name"): bool(f.get("required", False)) for f in fields}
    enum_map = {f.get("name"): f.get("values", []) for f in fields if f.get("type") == "enum"}

    for name in type_map:
        if name not in df.columns:
            # Missing column => if required, record a file-level error (row = -1)
            if required_map.get(name):
                errors.append({"row": -1, "field": name, "issue": "missing_required_column", "value": ""})
            continue

        series = df[name]
        # Required non-null/non-empty
        if required_map.get(name):
            if series.dtype.name.startswith("Int"):
                bad_idx = series.isna()
            else:
                bad_idx = series.astype(str).str.strip().eq("") | series.isna()
            for i in df.index[bad_idx]:
                errors.append({"row": int(i), "field": name, "issue": "required_missing", "value": str(df.at[i, name])})

        # Type checks (lightweight; do not coerce in-place)
        t = type_map[name]
        if t == "int":
            bad = pd.to_numeric(series, errors="coerce").isna()
            # allow NA only if not required and empty
            if required_map.get(name):
                for i in df.index[bad]:
                    errors.append({"row": int(i), "field": name, "issue": "invalid_int", "value": str(df.at[i, name])})
        elif t == "date":
            # Expect YYYY-MM-DD
            mask_len = series.astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
            for i in df.index[~mask_len & series.notna()]:
                errors.append({"row": int(i), "field": name, "issue": "invalid_date", "value": str(df.at[i, name])})
        elif t == "time":
            mask_len = series.astype(str).str.match(r"^\d{2}:\d{2}:\d{2}$", na=False)
            for i in df.index[~mask_len & series.notna()]:
                errors.append({"row": int(i), "field": name, "issue": "invalid_time", "value": str(df.at[i, name])})
        elif t == "enum":
            allowed = enum_map.get(name, [])
            for i in df.index[series.notna()]:
                val = str(df.at[i, name])
                if val.strip() == "":
                    # required_missing already captured; skip
                    continue
                if not _enum_ok(allowed, val):
                    errors.append({"row": int(i), "field": name, "issue": "invalid_enum", "value": val})

    return pd.DataFrame(errors)


def validate_uniques(df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
    """Check uniqueness constraints; return duplicate rows."""
    constraints = schema.get("constraints", {}) or {}
    unique_sets = constraints.get("unique", []) or []
    all_dupes = []

    for cols in unique_sets:
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        dup_mask = df.duplicated(subset=cols, keep=False)
        if dup_mask.any():
            d = df.loc[dup_mask, cols + [c for c in ("canonical_game_id",) if c in df.columns]].copy()
            d["_violated_unique_on"] = "+".join(cols)
            all_dupes.append(d)

    if all_dupes:
        return pd.concat(all_dupes, ignore_index=True)
    return pd.DataFrame(columns=["_violated_unique_on"])  # empty


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    parts = []

    # High-level counts
    parts.append(pd.DataFrame({
        "metric": ["rows"],
        "value": [len(df)]
    }))

    # By season/type/week
    for cols in [["season"], ["season","season_type"], ["season","season_type","week"]]:
        have = [c for c in cols if c in df.columns]
        if not have:
            continue
        g = df.groupby(have, dropna=False).size().reset_index(name="count")
        g["metric"] = "group:" + "+".join(have)
        parts.append(g)

    # Presence of canonical fields
    for col in ["home_abbr","away_abbr","canonical_game_id","canonical_hash"]:
        if col in df.columns:
            parts.append(pd.DataFrame({"metric":[f"non_empty:{col}"], "value":[int((df[col].astype(str)!="").sum())]}))

    return pd.concat(parts, ignore_index=True)


def main() -> int:
    # Load inputs
    if not NORM_CSV.exists():
        # Write empty reports if input missing
        for p in [ERR_CSV, DUP_CSV, SUM_CSV]:
            pd.DataFrame().to_csv(p, index=False)
        print(f"Missing normalized schedules: {NORM_CSV}. Wrote empty reports.")
        return 0

    df = pd.read_csv(NORM_CSV, dtype=str).fillna("")
    schema = load_schema(SCHEMA_YML)

    # Validate
    df_err = validate_types(df, schema)
    df_dup = validate_uniques(df, schema)
    df_sum = summarize(df)

    # Write reports
    df_err.to_csv(ERR_CSV, index=False)
    df_dup.to_csv(DUP_CSV, index=False)
    df_sum.to_csv(SUM_CSV, index=False)

    print(f"Wrote reports:\n  {ERR_CSV}\n  {DUP_CSV}\n  {SUM_CSV}\n  errors={len(df_err)} dup_rows={len(df_dup)} rows={len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
