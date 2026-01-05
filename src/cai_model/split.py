# src/cai_model/split.py
import pandas as pd

from .config import EVENT_TIME_COL, TIME_COL

def time_split_with_purge(d, test_fraction=0.2):
    d = d.sort_values(EVENT_TIME_COL).reset_index(drop=True)
    n = len(d)
    split_idx = int((1 - test_fraction) * max(n, 1))
    split_idx = min(max(split_idx, 0), n - 1)
    split_time = d.loc[split_idx, EVENT_TIME_COL]

    run_id_col = "instance_id" if "instance_id" in d.columns else ("id" if "id" in d.columns else None)
    if run_id_col is None:
        d["__run_key__"] = (
            d["provider"].astype(str) + "|" +
            d["region"].astype(str) + "|" +
            d["instance_type"].astype(str) + "|" +
            d[TIME_COL].astype(str)
        )
        run_id_col = "__run_key__"

    group = d.groupby(run_id_col)
    run_min = group[EVENT_TIME_COL].min()
    run_max = group[EVENT_TIME_COL].max()
    straddle = run_min.le(split_time) & run_max.gt(split_time)
    bad_ids = set(run_min[straddle].index.astype(str))

    d["__run_id__"] = d[run_id_col].astype(str)
    d = d[~d["__run_id__"].isin(bad_ids)].copy()
    print(f"Purged {len(bad_ids)} runs that straddled the split.")

    d = d.sort_values(EVENT_TIME_COL)
    split_idx = int((1 - test_fraction) * len(d))
    train = d.iloc[:split_idx].copy()
    test = d.iloc[split_idx:].copy()

    print(f"\nTrain size: {len(train):4d}  Test size: {len(test):4d}")
    if len(train) and len(test):
        print(f"Train time range: {train[EVENT_TIME_COL].min()} → {train[EVENT_TIME_COL].max()}")
        print(f"Test  time range: {test[EVENT_TIME_COL].min()} → {test[EVENT_TIME_COL].max()}")

    return train, test
