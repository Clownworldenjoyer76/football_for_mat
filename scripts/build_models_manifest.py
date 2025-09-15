# --- add near imports ---
import re
import json
from pathlib import Path

# --- replace your current "infer/extract target" logic with this ---
def infer_target_from_filename(filename: str) -> str:
    """
    Return the full target name from a model filename.
    Handles both legacy 4 models and version-suffixed artifacts.
    Examples:
      qb_passing_yards.joblib -> qb_passing_yards
      passing_yards_20250912_8eb60df.joblib -> passing_yards
    """
    stem = Path(filename).stem  # strip .joblib
    # If it already matches the legacy 4, keep as-is
    legacy = {"qb_passing_yards", "rb_rushing_yards", "wr_rec_yards", "wrte_receptions"}
    if stem in legacy:
        return stem
    # Else strip a trailing version/hash suffix if present
    # matches: _YYYYMMDD_<7+ hex>  OR  _<7+ hex>
    m = re.match(r"^(.*?)(?:_(\d{8}_[0-9a-f]{7,})|_[0-9a-f]{7,})$", stem)
    return m.group(1) if m else stem

# --- wherever you build rows, use: ---
target = infer_target_from_filename(row["filename"])

# --- merge metrics (kept robust to missing keys) ---
metrics_path = Path("output/metrics_summary.json")
metrics = {}
if metrics_path.exists():
    with metrics_path.open() as f:
        metrics = json.load(f)

mt = metrics.get(target, {})
row["mae"]  = mt.get("MAE")
row["rmse"] = mt.get("RMSE")
row["rows"] = mt.get("rows")

# Optional: fill actual_column if you track it per target
actual_map = {
    "qb_passing_yards": "passing_yards",
    "rb_rushing_yards": "rushing_yards",
    "wr_rec_yards": "receiving_yards",
    "wrte_receptions": "receptions",
}
row["actual_column"] = actual_map.get(target, row.get("actual_column"))
