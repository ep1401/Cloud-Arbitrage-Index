# src/cai_model/features.py
import numpy as np
import pandas as pd

from .config import (
    TIME_COL, EVENT_TIME_COL, MAX_HOURS_PER_RUN,
    ALPHA_PRIOR, BETA_PRIOR, PRIOR_RATE,
    DROP_USELESS_FEATURES, DROP_CAT, DROP_NUM
)
from .rolling import roll_sum_excluding_current, roll_mean_excluding_current

def make_hour_slices(df):
    need = [TIME_COL, "duration_minutes", "interrupted"]
    for c in need:
        if c not in df.columns:
            df[c] = np.nan

    launched_mask = df["is_launch_failed_int"] == 0
    elig = df[launched_mask & df[TIME_COL].notna()].copy()

    rows = []
    for _, r in elig.iterrows():
        t0 = r[TIME_COL]
        dur = r.get("duration_minutes", np.nan)
        intr = bool(r.get("interrupted", False))
        if pd.isna(dur):
            continue

        total_minutes = float(dur)
        max_slices = int(np.ceil(min(total_minutes, MAX_HOURS_PER_RUN * 60) / 60.0))
        if max_slices <= 0:
            continue

        for k in range(max_slices):
            slice_start = t0 + pd.Timedelta(hours=k)
            lo = k * 60.0
            hi = (k + 1) * 60.0

            if intr and (lo < total_minutes <= hi):
                y = 1
            elif total_minutes > hi or (not intr and total_minutes >= hi):
                y = 0
            else:
                continue

            row = r.to_dict()
            row[EVENT_TIME_COL] = slice_start
            row["slice_k"] = k
            row["y_1h"] = y
            row["is_slice"] = True
            rows.append(row)

    slices = pd.DataFrame(rows)
    print(f"make_hour_slices → {len(slices)} rows")
    return slices

def add_extra_features_on_slices(d):
    d = d.copy()

    dt = pd.to_datetime(d[EVENT_TIME_COL], utc=True)
    hour = dt.dt.hour.astype(float)
    dow = dt.dt.weekday.astype(float)
    d["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    d["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    d["sin_dow"]  = np.sin(2 * np.pi * dow / 7.0)
    d["cos_dow"]  = np.cos(2 * np.pi * dow / 7.0)

    win3, win12 = "3h", "12h"
    d = d.sort_values(["arm_key", EVENT_TIME_COL])
    grp = d.groupby("arm_key", group_keys=False)

    cnt3  = grp.apply(lambda g: roll_sum_excluding_current(g, g["y_1h"].astype(float), win3))\
               .reset_index(level=0, drop=True)
    att3  = grp.apply(lambda g: roll_sum_excluding_current(g, pd.Series(1.0, index=g.index), win3))\
               .reset_index(level=0, drop=True)
    cnt12 = grp.apply(lambda g: roll_sum_excluding_current(g, g["y_1h"].astype(float), win12))\
               .reset_index(level=0, drop=True)
    att12 = grp.apply(lambda g: roll_sum_excluding_current(g, pd.Series(1.0, index=g.index), win12))\
               .reset_index(level=0, drop=True)

    d["arm_intr_lag_3h"] = ((cnt3 + ALPHA_PRIOR) / (att3 + ALPHA_PRIOR + BETA_PRIOR)).fillna(PRIOR_RATE).astype(float)
    d["arm_intr_lag_12h"] = ((cnt12 + ALPHA_PRIOR) / (att12 + ALPHA_PRIOR + BETA_PRIOR)).fillna(PRIOR_RATE).astype(float)

    win2, win6 = "2h", "6h"
    att2 = grp.apply(lambda g: roll_sum_excluding_current(g, pd.Series(1.0, index=g.index), win2))\
              .reset_index(level=0, drop=True)
    d["arm_attempts_2h"] = att2.fillna(0).astype(float)

    intr6 = grp.apply(lambda g: roll_sum_excluding_current(g, g["y_1h"].astype(float), win6))\
               .reset_index(level=0, drop=True)
    att6 = grp.apply(lambda g: roll_sum_excluding_current(g, pd.Series(1.0, index=g.index), win6))\
              .reset_index(level=0, drop=True)
    d["arm_intr_rate_6h"] = ((intr6 + ALPHA_PRIOR) / (att6 + ALPHA_PRIOR + BETA_PRIOR)).fillna(PRIOR_RATE).astype(float)

    fam = d["instance_type"].astype(str).str.split(".", n=1, expand=True)[0]
    d["family"] = fam
    d["family_region_key"] = d["family"].astype(str) + "|" + d["region"].astype(str)

    grpfr = d.groupby("family_region_key", group_keys=False)
    fam_reg_mean = grpfr.apply(lambda g: roll_mean_excluding_current(g, g["y_1h"].astype(float), "6h"))\
                        .reset_index(level=0, drop=True)
    d["family_region_intr_lag_6h"] = fam_reg_mean.fillna(PRIOR_RATE).astype(float)

    return d

def feature_lists():
    cat_all = ["instance_type", "region"]
    num_all = [
        "slice_k",
        "arm_launch_lag_2h",
        "arm_intr_lag_3h", "arm_intr_lag_12h",
        "family_region_intr_lag_6h",
        "arm_attempts_2h", "arm_intr_rate_6h",
        "sin_hour", "cos_hour", "sin_dow", "cos_dow",
        "spot_price_usd", "price_delta_6h",
        "fs_launch_local_hour", "fs_launch_utc_hour",
        "fs_launch_dow", "fs_price_zscore_6h",
    ]

    if DROP_USELESS_FEATURES:
        cat = [c for c in cat_all if c not in DROP_CAT]
        num = [c for c in num_all if c not in DROP_NUM]
    else:
        cat, num = cat_all, num_all

    return cat, num
