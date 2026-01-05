# src/cai_model/eval.py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import N_BINS, POOL_COLS, ALPHA_PRIOR, BETA_PRIOR, PRIOR_RATE
from .utils import log_loss_fractional

def reliability_table(y_true, p_pred, bins=10):
    y_true = np.asarray(y_true).astype(float)
    p_pred = np.asarray(p_pred).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p_pred, edges, right=True)
    idx = np.clip(idx, 1, bins)

    rows = []
    for b in range(1, bins + 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            rows.append((edges[b-1], edges[b], 0, np.nan, np.nan, np.nan))
        else:
            obs = float(y_true[m].mean())
            avg = float(p_pred[m].mean())
            rows.append((edges[b-1], edges[b], n, obs, avg, abs(obs - avg)))

    tab = pd.DataFrame(rows, columns=["bin_lo","bin_hi","n","obs_rate","avg_pred","gap"])
    ece = np.nansum((tab["n"]/np.nansum(tab["n"])) * tab["gap"])
    mce = np.nanmax(tab["gap"])
    return tab, float(ece), float(mce)

def print_global_metrics_block(y_rate_true, y_bin_true, p_pred_rate, title_suffix=""):
    y_rate_true = np.asarray(y_rate_true, dtype=float)
    y_bin_true  = np.asarray(y_bin_true, dtype=int)
    p_pred_rate = np.asarray(p_pred_rate, dtype=float)
    p_pred_rate = np.clip(p_pred_rate, 0.0, 1.0)

    brier_model = float(np.mean((p_pred_rate - y_rate_true) ** 2))
    ll = log_loss_fractional(y_rate_true, p_pred_rate)

    try:
        roc = roc_auc_score(y_bin_true, p_pred_rate)
    except Exception:
        roc = np.nan

    print("\n==============================")
    print(f"Metrics{title_suffix}")
    print("==============================")
    print(f"Brier (model vs rate) : {brier_model:.6f}")
    print(f"Log loss (fractional) : {ll:.6f}")
    print(f"ROC-AUC (any>0)       : {roc:.3f}")

def build_train_climatology(train_df, pool_cols=POOL_COLS, y_col="y_block_rate"):
    g = train_df.groupby(pool_cols, dropna=False)[y_col]
    cnt = g.count().astype(float)
    pos = g.sum().astype(float)
    q = (pos + ALPHA_PRIOR) / (cnt + ALPHA_PRIOR + BETA_PRIOR)
    mapping = {tuple(k): float(v) for k, v in q.reset_index().set_index(pool_cols)[y_col].items()}

    global_q = float(((train_df[y_col].sum() + ALPHA_PRIOR) /
                     (len(train_df) + ALPHA_PRIOR + BETA_PRIOR)) if len(train_df) else PRIOR_RATE)
    return mapping, global_q

def get_train_clim_probs(df_any, train_clim_map, global_q, pool_cols=POOL_COLS):
    keys = list(map(tuple, df_any[pool_cols].astype(str).values))
    return np.array([train_clim_map.get(k, global_q) for k in keys], dtype=float)

def per_pool_basic_brier_rate(test_df, p_model_rate, train_clim_map, global_q, pool_cols=POOL_COLS, y_col="y_block_rate"):
    df = test_df.copy()
    df["__p__"] = np.clip(np.asarray(p_model_rate).astype(float), 0.0, 1.0)
    df["__y__"] = df[y_col].astype(float)
    df["__p_clim_train__"] = get_train_clim_probs(df, train_clim_map, global_q, pool_cols)

    grp = df.groupby(pool_cols, dropna=False)
    rows = []
    for key, g in grp:
        y = g["__y__"].values
        p = g["__p__"].values
        p_clm = g["__p_clim_train__"].values
        if len(y) == 0:
            continue

        rows.append({
            "provider": key[0],
            "region": key[1],
            "instance_type": key[2],
            "n": int(len(g)),
            "rate_mean": float(y.mean()),
            "brier_model": float(np.mean((p - y) ** 2)),
            "brier_zero": float(np.mean((0.0 - y) ** 2)),
            "brier_train_average": float(np.mean((p_clm - y) ** 2)),
        })

    res = pd.DataFrame(rows)
    if res.empty:
        print("\nPer-pool Brier table: no pools found in TEST.")
        return res

    res = res.sort_values(["region","instance_type","provider","n"], ascending=[True, True, True, False]).reset_index(drop=True)

    print("\nPer-pool Brier vs actual termination rate (TEST):")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    return res
