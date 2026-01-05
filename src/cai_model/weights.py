# src/cai_model/weights.py
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

from .config import USE_BALANCED_WEIGHTS, BALANCE_SCHEME

def make_sample_weights_from_labels(y_rate):
    y = np.asarray(y_rate)
    if not USE_BALANCED_WEIGHTS:
        return np.ones(len(y), dtype=float)
    y_bin = (y > 0).astype(int)
    w = compute_sample_weight(BALANCE_SCHEME, y_bin)
    w = w * (len(w) / w.sum())
    return w.astype(float)
