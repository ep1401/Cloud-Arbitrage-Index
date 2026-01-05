# src/cai_model/utils.py
import numpy as np
import pandas as pd

def safe_parse_dt(s):
    try:
        return pd.to_datetime(s, utc=True)
    except Exception:
        return pd.NaT

def log_loss_fractional(y_true, p_pred):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

def find_best_scale_for_logloss(y_true, raw_rate, lo=0.01, hi=3.0, num_steps=400):
    y = np.asarray(y_true, dtype=float)
    r = np.asarray(raw_rate, dtype=float)
    scales = np.linspace(lo, hi, num_steps)
    best_scale = 1.0
    best_ll = float("inf")
    eps = 1e-15

    for s in scales:
        p = np.clip(r * s, eps, 1.0 - eps)
        ll = log_loss_fractional(y, p)
        if ll < best_ll:
            best_ll = ll
            best_scale = float(s)

    return best_scale, best_ll
