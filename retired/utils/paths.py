
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

DATA_RAW        = BASE / "data/raw"
DATA_WAREHOUSE  = BASE / "data/warehouse"
DATA_FEATURES   = BASE / "data/features"
DATA_PRED       = BASE / "data/predictions"
DATA_ODDS       = BASE / "data/odds"
OUTPUT_DIR      = BASE / "output"

for p in [DATA_RAW, DATA_WAREHOUSE, DATA_FEATURES, DATA_PRED, DATA_ODDS, OUTPUT_DIR]:
    p.mkdir(parents=True, exist_ok=True)
