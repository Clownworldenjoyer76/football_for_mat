#!/usr/bin/env python3
# /scripts/03_train_models.py
# Deterministic trainer
# - Reads config/models.yml
# - Trains one model per target
# - Saves only VERSIONED artifacts: models/pregame/<target>_<YYYYMMDD>_<sha>.joblib
# - Writes <target>_<tag>.meta.json sidecar with full provenance
# - NO '.latest.joblib' copies
# - Cleans up legacy/unversioned/`.latest` artifacts from older runs
from __future__ import annotations

import json, os, random, sys, subprocess, hashlib, shutil, re
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np
import joblib, yaml

REPO = Path('.').resolve()
CONFIG = REPO / 'config' / 'models.yml'
FEATS = REPO / 'data' / 'features' / 'weekly_clean.csv.gz'
MODELS_DIR = REPO / 'models' / 'pregame'
OUT_DIR = REPO / 'output' / 'models'
LOG_DIR = REPO / 'output' / 'logs'
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = OUT_DIR / 'metrics_summary.csv'
FEATURES_LIST = LOG_DIR / 'features_columns.txt'

# ----------------------------- helpers -----------------------------
def git_commit_short() -> str:
    try:
        return subprocess.check_output(['git','rev-parse','--short','HEAD'], text=True).strip()
    except Exception:
        return 'nogit'

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def sha256_file(p: Path, chunk_size: int = 1<<20) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(chunk_size), b''):
            h.update(b)
    return h.hexdigest()

def load_config() -> dict:
    if not CONFIG.exists():
        print(f'[FATAL] Missing config: {CONFIG}', file=sys.stderr)
        sys.exit(2)
    with CONFIG.open() as f:
        return yaml.safe_load(f) or {}

def set_global_seed(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(seed)
    np.random.seed(seed)

def build_version_tag(tag_cfg: str) -> str:
    if tag_cfg and tag_cfg != 'auto':
        return str(tag_cfg)
    return datetime.now(timezone.utc).strftime('%Y%m%d') + '_' + git_commit_short()

def get_features_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f'[FATAL] Features file not found: {path}', file=sys.stderr)
        sys.exit(2)
    df = pd.read_csv(path, low_memory=False)
    with FEATURES_LIST.open('w', encoding='utf-8') as f:
        for c in df.columns:
            f.write(c + '\n')
    print(f'[INFO] Wrote feature column list -> {FEATURES_LIST}')
    return df

def normkey(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower())

ALIAS = {
    'qb_passing_yards': ['qbpassyards','passingyards','passyards','qbpassyds','passyds','qb_passing_yds','qb_pass_yds'],
    'rb_rushing_yards': ['rbrushyards','rushingyards','rushyards','rb_rush_yds','rb_rushing_yds'],
    'wr_rec_yards'    : ['wrrecyards','receivingyards','recyards','wr_rec_yds','wr_receiving_yds','wrte_rec_yards'],
    'wrte_receptions' : ['wrtereceptions','receptions','rec','wr_receptions','wrte_rec'],
}

def resolve_targets(df_cols, desired):
    cols_norm = {normkey(c): c for c in df_cols}
    mapping = {}
    for want in desired:
        wn = normkey(want)
        if wn in cols_norm:
            mapping[want] = cols_norm[wn]
            continue
        for alt in ALIAS.get(want, []):
            an = normkey(alt)
            if an in cols_norm:
                mapping[want] = cols_norm[an]
                break
    return mapping

def numeric_X(df: pd.DataFrame, target_col: str, id_cols):
    cols_drop = [c for c in id_cols if c in df.columns] + [target_col]
    base = df.drop(columns=cols_drop, errors='ignore')
    X = base.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return X

def get_estimator(cfg: dict, seed: int):
    est_name = cfg.get('estimator','random_forest')
    if est_name == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        params = dict(cfg.get('random_forest',{}))
        params['random_state'] = seed
        return RandomForestRegressor(**params), {'name':'RandomForestRegressor','params':params}
    raise SystemExit(f'Unsupported estimator: {est_name}')

def cleanup_legacy_files():
    # remove *.latest.joblib and old unversioned artifacts if present
    removed = []
    for p in MODELS_DIR.glob('*.latest.joblib'):
        try:
            p.unlink()
            removed.append(p.name)
        except Exception:
            pass
    # common old unversioned names to purge
    legacy = ['qb_passing_yards.joblib','rb_rushing_yards.joblib','wr_rec_yards.joblib','wrte_receptions.joblib']
    for name in legacy:
        p = MODELS_DIR / name
        if p.exists():
            try:
                p.unlink()
                removed.append(p.name)
            except Exception:
                pass
    if removed:
        print('[CLEANUP] Removed legacy files:', ', '.join(removed))

# ----------------------------- train -----------------------------
def train_all() -> None:
    cfg = load_config()
    seed = int(cfg.get('seed',42))
    set_global_seed(seed)
    tag = build_version_tag(str(cfg.get('version_tag','auto')))
    commit = git_commit_short()
    built_utc = utc_now_iso()

    desired_targets = list(cfg.get('targets',[]))
    if not desired_targets:
        print('[FATAL] No targets specified in config/models.yml -> targets: []', file=sys.stderr)
        sys.exit(2)
    id_cols = list(cfg.get('id_columns', []))

    df = get_features_df(FEATS)
    mapping = resolve_targets(list(df.columns), desired_targets)
    if not mapping:
        print(f'[FATAL] None of your desired targets are present in features. Desired={desired_targets}', file=sys.stderr)
        print('[HINT] See output/logs/features_columns.txt for available columns.', file=sys.stderr)
        sys.exit(3)

    rows = []
    for want, actual_col in mapping.items():
        y = df[actual_col]
        X = numeric_X(df, actual_col, id_cols)
        est, est_info = get_estimator(cfg, seed)
        est.fit(X, y)

        versioned = MODELS_DIR / f'{want}_{tag}.joblib'
        joblib.dump(est, versioned, compress=3)

        meta = {
            'target': want,
            'actual_column': actual_col,
            'built_at_utc': built_utc,
            'git_commit': commit,
            'random_seed': seed,
            'features_path': str(FEATS),
            'data_sha256': sha256_file(FEATS) if FEATS.exists() else None,
            'estimator': est_info['name'],
            'hyperparameters': est_info['params'],
            'feature_columns': list(X.columns),
            'artifact_path': str(versioned),
            'artifact_sha256': sha256_file(versioned),
            'version_tag': tag,
        }
        (MODELS_DIR / f'{want}_{tag}.meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))

        preds = est.predict(X)
        mae = float(np.mean(np.abs(preds - y)))
        ss_res = float(np.sum((preds - y) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2)) if len(y) else 0.0
        r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0

        rows.append({
            'target': want,
            'actual_column': actual_col,
            'artifact_path': str(versioned),
            'artifact_sha256': meta['artifact_sha256'],
            'git_commit': commit,
            'random_seed': seed,
            'built_at_utc': built_utc,
            'mae': mae,
            'r2': r2,
            'version_tag': tag,
        })
        print(f"[OK] Saved {versioned.name} for '{want}' (actual='{actual_col}') sha256={meta['artifact_sha256'][:10]}...  mae={mae:.3f} r2={r2:.3f}")

    # write metrics
    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)
    print(f'[INFO] Wrote metrics -> {SUMMARY_CSV}')
    print(f'[INFO] Models dir -> {MODELS_DIR}')

    # cleanup after successful training
    cleanup_legacy_files()

if __name__ == '__main__':
    train_all()
