
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal, Tuple
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold
import joblib

Method = Literal["isotonic","sigmoid"]

@dataclass
class CalibrationReport:
    method: str
    n: int
    brier_before: float
    brier_after: float
    logloss_before: float | None
    logloss_after: float | None

def _ensure_arrays(p_hat: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p_hat = np.asarray(p_hat, dtype=float)
    y = np.asarray(y, dtype=int)
    return p_hat, y

def fit_calibrator(p_hat: pd.Series | np.ndarray, y: pd.Series | np.ndarray, method: Method="isotonic"):
    """
    Fit a probability calibrator mapping raw probabilities to calibrated probabilities.
    p_hat: uncalibrated probability estimates for the positive class (over=1).
    y: binary outcomes (0/1).
    """
    p_hat, y = _ensure_arrays(p_hat, y)
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_hat, y)
        model = ("isotonic", iso)
    elif method == "sigmoid":
        # Platt scaling via logistic regression on logit(p)
        eps = 1e-6
        z = np.log((p_hat + eps) / (1 - p_hat + eps)).reshape(-1, 1)
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(z, y)
        model = ("sigmoid", lr)
    else:
        raise ValueError("method must be 'isotonic' or 'sigmoid'")
    return model

def apply_calibrator(model, p_hat: pd.Series | np.ndarray) -> np.ndarray:
    kind, est = model
    p_hat = np.asarray(p_hat, dtype=float)
    if kind == "isotonic":
        return np.clip(est.predict(p_hat), 0, 1)
    elif kind == "sigmoid":
        eps = 1e-6
        z = np.log((p_hat + eps) / (1 - p_hat + eps)).reshape(-1, 1)
        return est.predict_proba(z)[:, 1]
    else:
        raise ValueError("Unknown calibrator kind")

def evaluate_calibration(p_hat_before, p_hat_after, y) -> CalibrationReport:
    p_hat_before, y = _ensure_arrays(p_hat_before, y)
    p_hat_after, _ = _ensure_arrays(p_hat_after, y)
    rep = CalibrationReport(
        method = "n/a",
        n = int(len(y)),
        brier_before = float(brier_score_loss(y, p_hat_before)),
        brier_after = float(brier_score_loss(y, p_hat_after)),
        logloss_before = float(log_loss(y, p_hat_before, labels=[0,1], eps=1e-12)) if 0 < p_hat_before.min() and p_hat_before.max() < 1 else None,
        logloss_after = float(log_loss(y, p_hat_after, labels=[0,1], eps=1e-12)) if 0 < p_hat_after.min() and p_hat_after.max() < 1 else None,
    )
    return rep

def cross_validated_method_choice(p_hat, y, n_splits: int=5) -> str:
    """
    Simple CV to choose between isotonic and sigmoid by Brier score.
    """
    p_hat, y = _ensure_arrays(p_hat, y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = {"isotonic": [], "sigmoid": []}
    for tr, va in skf.split(p_hat, y):
        ph_tr, y_tr = p_hat[tr], y[tr]
        ph_va, y_va = p_hat[va], y[va]
        for method in ["isotonic", "sigmoid"]:
            mdl = fit_calibrator(ph_tr, y_tr, method)
            ph_va_cal = apply_calibrator(mdl, ph_va)
            scores[method].append(brier_score_loss(y_va, ph_va_cal))
    # lower is better
    iso = np.mean(scores["isotonic"])
    sig = np.mean(scores["sigmoid"])
    return "isotonic" if iso <= sig else "sigmoid"

def save_calibrator(model, path: str | Path):
    joblib.dump(model, path)

def load_calibrator(path: str | Path):
    return joblib.load(path)
