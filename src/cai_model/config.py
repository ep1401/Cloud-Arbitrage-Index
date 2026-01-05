# src/cai_model/config.py

CSV_PATH_DEFAULT = "data/probe_results_rows.csv"

TIME_COL = "start_time_utc"
CREATED_COL = "created_at"
EVENT_TIME_COL = "event_time"

HORIZON_MIN = 60
MAX_HOURS_PER_RUN = 5
TEST_FRACTION = 0.20
N_BINS = 10
RANDOM_STATE = 42

USE_BALANCED_WEIGHTS = True
BALANCE_SCHEME = "balanced"

DROP_USELESS_FEATURES = True
DROP_CAT = []

DROP_NUM = [
    "cos_dow",
    "fs_launch_utc_hour",
    "sin_hour",
    "cos_hour",
    "fs_launch_local_hour",
    "arm_intr_rate_6h",
    "arm_attempts_2h",
    "price_delta_6h",
    "slice_k",
]

ALPHA_PRIOR = 0.5
BETA_PRIOR = 2.0
PRIOR_RATE = ALPHA_PRIOR / (ALPHA_PRIOR + BETA_PRIOR)

POOL_COLS = ["provider", "region", "instance_type"]

TARGET_RATE_COL = "y_block_rate"
TARGET_COUNT_COL = "y_block_count"
TARGET_BIN_COL = "y_block_bin"
TARGET_HORIZON_COL = "y_block_horizon"

USE_THINNING_FOR_SANITY_CHECK = False
MAX_EFFECTIVE_H_FOR_THINNING = 6
