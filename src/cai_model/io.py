# src/cai_model/io.py
import json
import numpy as np
import pandas as pd

from .config import TIME_COL, CREATED_COL
from .utils import safe_parse_dt

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded rows: {len(df)}")

    if TIME_COL in df.columns:
        df[TIME_COL] = df[TIME_COL].apply(safe_parse_dt)
    if CREATED_COL in df.columns:
        df[CREATED_COL] = df[CREATED_COL].apply(safe_parse_dt)

    if "features_snapshot" in df.columns:
        def unpack_json(x):
            if isinstance(x, dict):
                return x
            if pd.isna(x):
                return {}
            try:
                return json.loads(x)
            except Exception:
                return {}

        fs = df["features_snapshot"].apply(unpack_json)
        df["fs_launch_local_hour"] = fs.apply(lambda d: d.get("launch_local_hour", np.nan))
        df["fs_launch_utc_hour"]   = fs.apply(lambda d: d.get("launch_utc_hour", np.nan))
        df["fs_launch_dow"]        = fs.apply(lambda d: d.get("launch_dow", np.nan))
        df["fs_price_zscore_6h"]   = fs.apply(lambda d: d.get("price_zscore_6h", np.nan))

    out_norm = df.get("outcome", "").astype(str).str.strip().str.lower()
    df["is_launch_failed_int"] = out_norm.str.contains("launchfailed", na=False).astype(int)

    for c in ["provider", "region", "instance_type"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["arm_key"] = df[["provider", "region", "instance_type"]].astype(str).agg(":".join, axis=1)

    print("Raw columns:", list(df.columns))
    return df
