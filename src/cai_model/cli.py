# src/cai_model/cli.py
import argparse
import warnings
import numpy as np
import pandas as pd

from .config import (
    CSV_PATH_DEFAULT, TEST_FRACTION, N_BINS, RANDOM_STATE,
    EVENT_TIME_COL, POOL_COLS,
    TARGET_RATE_COL, TARGET_COUNT_COL, TARGET_BIN_COL, TARGET_HORIZON_COL,
    USE_THINNING_FOR_SANITY_CHECK, MAX_EFFECTIVE_H_FOR_THINNING
)
from .io import load_raw
from .features import make_hour_slices, add_extra_features_on_slices, feature_lists
from .rolling import assemble_events_for_rolling, add_arm_launch_lag_2h_on_events
from .targets import add_pool_block_rate_target
from .split import time_split_with_purge
from .weights import make_sample_weights_from_labels
from .model import make_model
from .utils import find_best_scale_for_logloss
from .eval import reliability_table, print_global_metrics_block, build_train_climatology, per_pool_basic_brier_rate, get_train_clim_probs
from .viz import plot_accuracy_vs_training_examples
from .subset import find_best_training_subset

warnings.filterwarnings("ignore", category=UserWarning)

def main():
    parser = argparse.ArgumentParser(description="POOL-WIDE block-horizon termination-rate model (Poisson on counts).")
    parser.add_argument("--csv-path", type=str, default=CSV_PATH_DEFAULT)
    parser.add_argument("--block-hours", type=int, default=6)
    parser.add_argument("--outdir", type=str, default="outputs")
    args = parser.parse_args()

    block_hours = args.block_hours
    if block_hours < 1:
        raise ValueError("--block-hours must be >= 1")

    print(f"\n=== Using block_hours = {block_hours} (POOL-WIDE) ===")

    # 1) Raw
    df_raw = load_raw(args.csv_path)

    # 2) Slices
    slices = make_hour_slices(df_raw)
    if len(slices) == 0:
        print("No sliceable rows found.")
        return

    # 3) Rolling features on events
    events = assemble_events_for_rolling(slices, df_raw)
    events = add_arm_launch_lag_2h_on_events(events, alpha=0.5, beta=0.5)

    # 4) keep slices
    d = events[events["is_slice"] == True].copy()

    # 5) extra features
    d = add_extra_features_on_slices(d)

    # 6) targets
    d = add_pool_block_rate_target(d, block_hours=block_hours)

    RATE_COL_EFF = TARGET_RATE_COL
    COUNT_COL_EFF = TARGET_COUNT_COL
    HORIZON_COL_EFF = TARGET_HORIZON_COL

    # baseline print
    if {"region","instance_type", RATE_COL_EFF}.issubset(d.columns):
        base = (d.groupby(["region","instance_type"])
                  .agg(n=(RATE_COL_EFF,"size"), interrupt_rate=(RATE_COL_EFF,"mean"))
                  .reset_index())
        print("\nSlice-level POOL-WIDE block rate per pool (overall):")
        print(base.to_string(index=False))

    # 7) split
    train, test = time_split_with_purge(d, test_fraction=TEST_FRACTION)
    if len(test) == 0:
        print("Empty test set after split.")
        return

    # prune test rows lacking full horizon
    t_max = pd.to_datetime(test[EVENT_TIME_COL], utc=True).max()
    cutoff_time = t_max - pd.Timedelta(hours=block_hours)
    before = len(test)
    test = test[pd.to_datetime(test[EVENT_TIME_COL], utc=True) <= cutoff_time].copy()
    print(f"\nPruned {before - len(test)} test rows with < {block_hours}h remaining horizon.")
    if len(test) == 0:
        print("All test rows were pruned; nothing left to evaluate.")
        return

    # 8) features
    cat_cols, num_cols = feature_lists()
    feat_cols = cat_cols + num_cols

    X_train_raw = train[feat_cols].copy()
    X_test_raw  = test[feat_cols].copy()

    X_train = pd.get_dummies(X_train_raw, columns=cat_cols, dummy_na=True)
    X_test  = pd.get_dummies(X_test_raw,  columns=cat_cols, dummy_na=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    y_train_count = train[COUNT_COL_EFF].astype(float).values
    y_train_rate  = train[RATE_COL_EFF].astype(float).values
    y_train_bin   = (y_train_count > 0).astype(int)

    y_test_count  = test[COUNT_COL_EFF].astype(float).values
    y_test_rate   = test[RATE_COL_EFF].astype(float).values
    y_test_bin    = (y_test_count > 0).astype(int)
    y_test_horizon = test[HORIZON_COL_EFF].astype(float).values

    # 9) climatology baseline
    train_clim_map, global_q_train = build_train_climatology(train, pool_cols=POOL_COLS, y_col=RATE_COL_EFF)
    p_train_clim_test = get_train_clim_probs(test, train_clim_map, global_q_train, pool_cols=POOL_COLS)

    # 10) weights
    w_train = make_sample_weights_from_labels(y_train_rate)

    # 11) fit model
    model = make_model()
    model.fit(X_train, y_train_count, sample_weight=w_train)

    # 12) predict -> rates -> calibrate
    lambda_train = np.maximum(model.predict(X_train), 0.0)
    lambda_test  = np.maximum(model.predict(X_test), 0.0)
    raw_rate_train = lambda_train / float(block_hours)
    raw_rate_test  = lambda_test  / float(block_hours)

    best_scale, best_ll_train = find_best_scale_for_logloss(
        y_true=y_train_rate, raw_rate=raw_rate_train, lo=0.01, hi=3.0, num_steps=400
    )
    print(f"\nGlobal calibration factor: {best_scale:.3f}")
    print(f"Train fractional log loss at best scale: {best_ll_train:.6f}")

    eps = 1e-15
    p_train_rate = np.clip(raw_rate_train * best_scale, eps, 1.0 - eps)
    p_test_rate  = np.clip(raw_rate_test  * best_scale, eps, 1.0 - eps)

    print_global_metrics_block(y_train_rate, y_train_bin, p_train_rate, title_suffix=f" (TRAIN, block-{block_hours}h)")
    print_global_metrics_block(y_test_rate,  y_test_bin,  p_test_rate,  title_suffix=f" (TEST, block-{block_hours}h)")

    # learning curve
    plot_accuracy_vs_training_examples(
        X_train=X_train,
        y_train_count=y_train_count,
        y_train_rate=y_train_rate,
        y_train_bin=y_train_bin,
        w_train=w_train,
        X_test=X_test,
        y_test_rate=y_test_rate,
        y_test_bin=y_test_bin,
        block_hours=block_hours,
        random_state=RANDOM_STATE,
        n_points=8,
        min_frac=0.1,
        output_prefix=f"{args.outdir}/learning_curve",
    )

    # best subset
    subset_size = min(300, len(X_train))
    best_idx, best_summary = find_best_training_subset(
        X_train=X_train,
        X_test=X_test,
        y_train_count=y_train_count,
        y_train_rate=y_train_rate,
        y_test_rate=y_test_rate,
        y_test_bin=y_test_bin,
        w_train=w_train,
        block_hours=block_hours,
        subset_size=subset_size,
        n_candidates=50,
        random_state=RANDOM_STATE,
        label="instances (1-hour slices)",
    )

    if best_idx is None:
        useful_df = train.copy()
    else:
        useful_df = train.iloc[best_idx].copy()

    out_path = f"{args.outdir}/best_instances.csv"
    useful_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(useful_df)} 1-hour instances (slices) to {out_path}")
    if best_summary is not None:
        print("Best-instance-subset summary:", best_summary)

    # per-pool brier table
    per_pool_basic_brier_rate(
        test_df=test,
        p_model_rate=p_test_rate,
        train_clim_map=train_clim_map,
        global_q=global_q_train,
        pool_cols=POOL_COLS,
        y_col=RATE_COL_EFF,
    )

    tab, ece, mce = reliability_table(y_test_rate, p_test_rate, bins=N_BINS)
    print("\nReliability table (TEST):")
    print(tab.to_string(index=False))
    print(f"ECE={ece:.6f}, MCE={mce:.6f}")

    # per-block output
    out = test[[EVENT_TIME_COL, "provider", "region", "instance_type"]].copy()
    out["block_hours_param"] = block_hours
    out["terminated"] = y_test_count.astype(int)
    out["total_in_block"] = y_test_horizon.astype(int)
    out["block_fraction"] = out["terminated"].astype(str) + "/" + out["total_in_block"].astype(str)
    out["block_rate_actual"] = y_test_rate
    out["pred_rate"] = p_test_rate

    print("\nPer-block POOL-WIDE interruption fraction vs predicted rate (TEST):")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

if __name__ == "__main__":
    main()
