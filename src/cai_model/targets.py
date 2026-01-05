# src/cai_model/targets.py
import numpy as np
import pandas as pd

from .config import POOL_COLS, EVENT_TIME_COL, TARGET_COUNT_COL, TARGET_RATE_COL, TARGET_BIN_COL, TARGET_HORIZON_COL, RANDOM_STATE

def add_pool_block_rate_target(d, block_hours: int):
    if block_hours < 1:
        raise ValueError("block_hours must be >= 1")

    d = d.copy()
    d["__orig_idx__"] = np.arange(len(d))
    d = d.sort_values(POOL_COLS + [EVENT_TIME_COL]).reset_index(drop=True)

    times = pd.to_datetime(d[EVENT_TIME_COL], utc=True).values
    y1 = d["y_1h"].astype(int).values

    n = len(d)
    y_block_count = np.zeros(n, dtype=float)
    y_block_horizon = np.zeros(n, dtype=float)

    H = pd.Timedelta(hours=block_hours)

    grp = d.groupby(POOL_COLS, sort=False, group_keys=False)
    for _, g_idx in grp.groups.items():
        idx_arr = np.asarray(g_idx, dtype=int)
        if len(idx_arr) == 0:
            continue

        t_pool = times[idx_arr]
        y_pool = y1[idx_arr]
        L = len(idx_arr)

        cum = np.concatenate([[0], np.cumsum(y_pool)])

        e = 0
        for i in range(L):
            t_end = t_pool[i] + H
            while e < L and t_pool[e] < t_end:
                e += 1
            horizon = e - i
            count = float(cum[e] - cum[i])

            y_block_horizon[idx_arr[i]] = float(horizon)
            y_block_count[idx_arr[i]] = count

    with np.errstate(divide="ignore", invalid="ignore"):
        y_block_rate = np.where(y_block_horizon > 0, y_block_count / y_block_horizon, 0.0)

    y_block_bin = (y_block_count > 0).astype(int)

    d[TARGET_COUNT_COL] = y_block_count
    d[TARGET_RATE_COL] = y_block_rate
    d[TARGET_BIN_COL] = y_block_bin
    d[TARGET_HORIZON_COL] = y_block_horizon
    d["block_hours"] = float(block_hours)

    d = d.sort_values("__orig_idx__").drop(columns=["__orig_idx__"]).reset_index(drop=True)
    return d
