# src/cai_model/subset.py
import numpy as np
from sklearn.metrics import roc_auc_score

from .model import make_model
from .utils import find_best_scale_for_logloss, log_loss_fractional

def find_best_training_subset(
    X_train, X_test,
    y_train_count, y_train_rate,
    y_test_rate, y_test_bin,
    w_train,
    block_hours,
    subset_size=600,
    n_candidates=50,
    random_state=42,
    label="training instances",
):
    n_samples = X_train.shape[0]
    if n_samples == 0:
        print(f"\n[Best-subset search] No {label} available.")
        return None, None

    if subset_size >= n_samples:
        idx_all = np.arange(n_samples, dtype=int)
        mdl = make_model()
        mdl.fit(X_train, y_train_count, sample_weight=w_train)

        lambda_test = np.maximum(mdl.predict(X_test), 0.0)
        raw_rate_test = lambda_test / float(block_hours)

        best_scale_all, _ = find_best_scale_for_logloss(
            y_true=y_train_rate,
            raw_rate=np.maximum(mdl.predict(X_train), 0.0) / float(block_hours),
            lo=0.01, hi=3.0, num_steps=200
        )

        eps = 1e-15
        p_test_rate_all = np.clip(raw_rate_test * best_scale_all, eps, 1.0 - eps)

        test_ll_all = log_loss_fractional(y_test_rate, p_test_rate_all)
        brier_all = float(np.mean((p_test_rate_all - y_test_rate) ** 2))
        try:
            roc_all = roc_auc_score(y_test_bin, p_test_rate_all)
        except Exception:
            roc_all = np.nan

        summary = {
            "test_log_loss": float(test_ll_all),
            "test_brier": float(brier_all),
            "test_roc_auc": float(roc_all),
            "subset_size": int(n_samples),
            "candidate": 0,
            "note": "subset_size >= n_samples; using all training examples",
        }
        return idx_all, summary

    rng = np.random.RandomState(random_state)
    best_idx = None
    best_test_ll = float("inf")
    best_summary = None

    print(f"\n=== Best-subset search over {label}: subset_size={subset_size}, n_candidates={n_candidates} ===")
    for c in range(n_candidates):
        idx = rng.choice(n_samples, size=subset_size, replace=False)
        X_sub = X_train.iloc[idx] if hasattr(X_train, "iloc") else X_train[idx]

        y_count_sub = y_train_count[idx]
        y_rate_sub = y_train_rate[idx]
        w_sub = w_train[idx]

        mdl = make_model()
        mdl.fit(X_sub, y_count_sub, sample_weight=w_sub)

        lambda_train_sub = np.maximum(mdl.predict(X_sub), 0.0)
        lambda_test = np.maximum(mdl.predict(X_test), 0.0)

        raw_rate_train_sub = lambda_train_sub / float(block_hours)
        raw_rate_test = lambda_test / float(block_hours)

        best_scale_sub, _ = find_best_scale_for_logloss(
            y_true=y_rate_sub,
            raw_rate=raw_rate_train_sub,
            lo=0.01, hi=3.0, num_steps=200
        )

        eps = 1e-15
        p_test_rate = np.clip(raw_rate_test * best_scale_sub, eps, 1.0 - eps)

        test_ll = log_loss_fractional(y_test_rate, p_test_rate)
        brier = float(np.mean((p_test_rate - y_test_rate) ** 2))
        try:
            roc = roc_auc_score(y_test_bin, p_test_rate)
        except Exception:
            roc = np.nan

        print(f"  Candidate {c+1:2d}/{n_candidates:2d} ({label}): log_loss={test_ll:.6f}, Brier={brier:.6f}, ROC={roc:.3f}")

        if test_ll < best_test_ll:
            best_test_ll = test_ll
            best_idx = idx
            best_summary = {
                "test_log_loss": float(test_ll),
                "test_brier": float(brier),
                "test_roc_auc": float(roc),
                "subset_size": int(subset_size),
                "candidate": int(c + 1),
            }

    return best_idx, best_summary
