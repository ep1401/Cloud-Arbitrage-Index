# src/cai_model/rolling.py
import numpy as np
import pandas as pd

from .config import EVENT_TIME_COL, CREATED_COL, TIME_COL, ALPHA_PRIOR, BETA_PRIOR, PRIOR_RATE

def roll_sum_excluding_current(g, value_series, window):
    s = value_series.copy()
    s.index = pd.to_datetime(g[EVENT_TIME_COL].values, utc=True)
    try:
        return s.rolling(window, closed="left").sum()
    except TypeError:
        return s.rolling(window).sum() - s

def roll_mean_excluding_current(g, value_series, window):
    s = value_series.copy()
    s.index = pd.to_datetime(g[EVENT_TIME_COL].values, utc=True)
    try:
        return s.rolling(window, closed="left").mean()
    except TypeError:
        roll_sum = s.rolling(window).sum() - s
        roll_cnt = s.rolling(window).count() - 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            return roll_sum / roll_cnt

def assemble_events_for_rolling(df_slices, df_raw):
    lf = df_raw[df_raw["is_launch_failed_int"] == 1].copy()
    lf_evt = lf[CREATED_COL].where(lf[CREATED_COL].notna(), lf[TIME_COL])
    lf = lf.loc[lf_evt.notna()].copy()
    lf[EVENT_TIME_COL] = lf_evt
    lf["is_slice"] = False
    lf["y_1h"] = np.nan

    slices_cols = set(df_slices.columns)
    lf_cols = set(lf.columns)

    for c in slices_cols - lf_cols:
        lf[c] = np.nan
    for c in lf_cols - slices_cols:
        if c not in df_slices.columns:
            df_slices[c] = np.nan

    events = pd.concat([df_slices, lf], ignore_index=True)
    events = events.sort_values(EVENT_TIME_COL).reset_index(drop=True)
    print(f"assemble_events_for_rolling → events: {events.shape}")
    return events

def add_arm_launch_lag_2h_on_events(events, alpha=ALPHA_PRIOR, beta=BETA_PRIOR):
    ev = events.copy()
    ev["is_attempt"] = 1.0

    win = "2h"
    ev = ev.sort_values(["arm_key", EVENT_TIME_COL])
    grp = ev.groupby("arm_key", group_keys=False)

    arm_attempt = grp.apply(lambda g: roll_sum_excluding_current(g, pd.Series(1.0, index=g.index), win))\
                     .reset_index(level=0, drop=True)
    arm_fail = grp.apply(lambda g: roll_sum_excluding_current(g, g["is_launch_failed_int"].astype(float), win))\
                  .reset_index(level=0, drop=True)

    arm_rate = (arm_fail + alpha) / (arm_attempt + alpha + beta)
    prior = float(alpha / (alpha + beta))
    ev["arm_launch_lag_2h"] = arm_rate.fillna(prior).values
    ev.drop(columns=["is_attempt"], inplace=True)
    return ev
