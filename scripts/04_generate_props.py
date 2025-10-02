# === Season Guard (auto-added) ===
# Ensures outputs are for TARGET_SEASON (default 2025). Non-invasive: runs at process exit.
try:
    import os, atexit, sys
    import pandas as _pd
    TARGET_SEASON = int(os.getenv("TARGET_SEASON", "2025"))
    _SEASON_GUARD_PATHS = ["data/props/props_current.csv"]

    def _season_guard_check():
        for _p in _SEASON_GUARD_PATHS:
            try:
                if not _p or not isinstance(_p, str):
                    continue
                if not os.path.exists(_p):
                    continue
                _df = _pd.read_csv(_p)
                if "season" not in _df.columns:
                    continue
                _mx = _pd.to_numeric(_df["season"], errors="coerce").max()
                if int(_mx) != TARGET_SEASON:
                    print(f"ERROR: {_p} max(season)={int(_mx)} != {TARGET_SEASON}. Rebuild inputs for {TARGET_SEASON}.", file=sys.stderr)
                    sys.exit(1)
            except Exception as _e:
                print(f"WARNING: season guard issue for {_p}: {_e}", file=sys.stderr)

    atexit.register(_season_guard_check)
except Exception:
    pass
# === End Season Guard ===
