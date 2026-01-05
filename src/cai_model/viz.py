# src/cai_model/viz.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from .model import make_model
from .utils import find_best_scale_for_logloss

def plot_accuracy_vs_training_examples(
    X_train, y_train_count, y_train_rate, y_train_bin, w_train,
    X_test, y_test_rate, y_test_bin,
    block_hours,
    random_state=42,
    n_points=8,
    min_frac=0.1,
    output_prefix=None,
):
    n_samples = X_train.shape[0]
    if n_samples < 10:
        print("Not enough training samples to plot a meaningful learning curve.")
        return

    rng = np.random.RandomState(random_state)

    min_size = max(10, int(n_samples * min_frac))
    training_sizes = np.linspace(min_size, n_samples, n_points, dtype=int)
    training_sizes = np.unique(training_sizes)

    roc_scores = []
    brier_scores = []

    print("\n=== Learning curve: ROC-AUC & Brier vs number of training examples ===")
    for n in training_sizes:
        idx = rng.choice(n_samples, size=n, replace=False)
        X_sub = X_train.iloc[idx] if hasattr(X_train, "iloc") else X_train[idx]

        y_count_sub = y_train_count[idx]
        y_rate_sub = y_train_rate[idx]
        w_sub = w_train[idx]

        model_i = make_model()
        model_i.fit(X_sub, y_count_sub, sample_weight=w_sub)

        lambda_train_i = np.maximum(model_i.predict(X_sub), 0.0)
        lambda_test_i  = np.maximum(model_i.predict(X_test), 0.0)

        raw_rate_train_i = lambda_train_i / float(block_hours)
        raw_rate_test_i  = lambda_test_i / float(block_hours)

        best_scale_i, _ = find_best_scale_for_logloss(
            y_true=y_rate_sub,
            raw_rate=raw_rate_train_i,
            lo=0.01,
            hi=3.0,
            num_steps=200,
        )
        eps = 1e-15
        p_test_rate_i = np.clip(raw_rate_test_i * best_scale_i, eps, 1.0 - eps)

        try:
            roc_i = roc_auc_score(y_test_bin, p_test_rate_i)
        except Exception:
            roc_i = np.nan

        brier_i = float(np.mean((p_test_rate_i - y_test_rate) ** 2))

        roc_scores.append(roc_i)
        brier_scores.append(brier_i)

        print(f"  n_train={n:6d}  ROC-AUC={roc_i:.3f}  Brier={brier_i:.6f}")

    plt.figure()
    plt.plot(training_sizes, roc_scores, marker="o")
    plt.xlabel("Number of training examples")
    plt.ylabel("ROC-AUC on test (any>0)")
    plt.title(f"Learning curve (ROC-AUC, block_hours={block_hours})")
    plt.grid(True)
    if output_prefix is not None:
        plt.savefig(f"{output_prefix}_roc.png", bbox_inches="tight")

    plt.figure()
    plt.plot(training_sizes, brier_scores, marker="o")
    plt.xlabel("Number of training examples")
    plt.ylabel("Brier score vs rate (test)")
    plt.title(f"Learning curve (Brier, block_hours={block_hours})")
    plt.grid(True)
    if output_prefix is not None:
        plt.savefig(f"{output_prefix}_brier.png", bbox_inches="tight")

    plt.show()
