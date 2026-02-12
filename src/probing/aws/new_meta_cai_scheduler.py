"""
Meta-bandit CAI Scheduler (AWS) — Ensemble + 1-hour probes + Option B persistence + MIN-4 top-up per cycle
--------------------------------------------------------------------------------------------------------
This is your ORIGINAL working scheduler with ONE functional change: per-cycle min-probe backfilling.

Adds an hourly "top-up" rule:

- Run the bandit plan exactly as before (TARGET 24 probes/cycle).
- Keep a per-cycle (per-hour) in-memory count of probe ATTEMPTS per arm (pool).
  * A "probe attempt" counts even if launch fails.
- After the bandit launches, if ANY arm has < MIN_PROBES_PER_ARM_PER_CYCLE attempts this cycle,
  launch additional "top-up" probes until it reaches the minimum.
  * This can push the total probes above 24 (allowed).
  * Top-up probe results are logged to a DIFFERENT table: probe_results_topup
  * For top-ups, we try launch up to TOPUP_LAUNCH_TRIES times.
    - If still fails, we log LaunchFailed to probe_results_topup and STILL increment the count by 1
      so we never get stuck trying forever.

IMPORTANT FIX:
- When restoring persistence snapshots, normalize posterior keys:
    blocks "0","1","2","3" -> ints 0,1,2,3
    horizons "1" -> int 1
  and fill any missing blocks/arms/horizons with (alpha0,beta0),
  so new_cai_bandit never KeyErrors at:
      self.tod_posterior[block][arm_key][1]

Requires Supabase tables:
  - probe_results (existing)
  - probe_results_topup (new; same schema as probe_results)
  - cai_ensemble_state (existing for persistence)

NOTE: This only enforces MIN-4 for the CURRENT cycle/hour (in-memory), not across restarts.
"""

import os
import time
import argparse
import datetime
import threading
import traceback
from collections import defaultdict, deque
from typing import Dict, Tuple, List, Optional, Any, Callable

import boto3
import numpy as np
from botocore.exceptions import ClientError
from supabase import create_client, Client

from new_cai_bandit import CAIBandit, Arm, ProbeResult


# ============ Drain controls ============
DRAIN_SENTINEL_PATH = "/home/ec2-user/.cai_drain"


def drain_enabled(cli_drain: bool = False) -> bool:
    return (
        cli_drain
        or os.environ.get("CAI_DRAIN", "").strip() in ("1", "true", "True")
        or os.path.exists(DRAIN_SENTINEL_PATH)
    )


# ========================
# Config (edit to taste)
# ========================
PROVIDER = "aws"
REGIONS = ["us-east-1", "us-west-2"]
INSTANCE_TYPES = ["t3a.large", "m6a.large", "c6i.large"]

AMI_IDS = {
    "us-east-1": "ami-08982f1c5bf93d976",
    "us-west-2": "ami-06a974f9b8a97ecf2",
}
KEY_NAMES = {"us-east-1": "my-spot-key", "us-west-2": "my-spot-key"}
SECURITY_GROUP_IDS = {
    "us-east-1": "sg-098bfbec48d19166d",
    "us-west-2": "sg-03b7cc83a5e0f65c2",
}

# Main cadence
INTERVAL_MIN = 60
STATUS_CHECK_SEC = 60

# 1-hour-only modeling horizon
H = 1

# Probes
TOTAL_PROBES_PER_INTERVAL = 24  # total probes launched each cycle
PROBE_MIN = 60                  # minutes to monitor before censoring/terminating (set 60 for true 1h probes)

# NEW: min attempts per arm per cycle + top-up launch behavior
MIN_PROBES_PER_ARM_PER_CYCLE = 4
TOPUP_LAUNCH_TRIES = 2

# Ensemble / learning knobs
RECENT_WINDOW_HOURS = 6        # sliding window length for RECENT expert
TOD_BLOCK_HOURS = 6            # UTC blocks: 0-6,6-12,12-18,18-24

HEDGE_ETA = 0.06               # learning rate
HEDGE_DECAY = 0.995            # loss decay
WEIGHT_FLOOR = 0.05            # keep experts alive (regimes flip)

# Planning knobs
PLAN_MC_SAMPLES = 600
TOPK_CANDIDATES = 12

# Coverage quotas (rolling 7 days) at granularity (provider, region, family, local_hour)
COVERAGE_WINDOW_DAYS = 7
MIN_COVERAGE_PER_SLICE = 6
MAX_COVERAGE_PENALTY_AFTER = 24
RHO_COVERAGE = 0.5

# Event compatibility (optional; off by default)
EVENT_THRESHOLD = 9999

# Supabase
SUPABASE_URL = "https://udrjcsighueuyyivsvnq.supabase.co"
SUPABASE_KEY = "sb_publishable_K7YuRphF5F0k5b8I5LqpBQ_bVsbnjvZ"
POLICY_VERSION = "meta_v3_ensemble_h1_hedge_persist"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL_STATE_TABLE = os.environ.get("CAI_MODEL_STATE_TABLE", "cai_ensemble_state")
TOPUP_TABLE = os.environ.get("CAI_TOPUP_TABLE", "probe_results_topup")

# Snapshot cadence
SAVE_SNAPSHOT_EVERY_CYCLE = True
SAVE_SNAPSHOT_AFTER_EACH_UPDATE = True  # safest (writes more)

# AWS
EC2_CLIENTS = {r: boto3.client("ec2", region_name=r) for r in REGIONS}
EC2_RESOURCES = {r: boto3.resource("ec2", region_name=r) for r in REGIONS}


# ========================
# Arms
# ========================
def build_arms() -> List[Arm]:
    return [Arm(PROVIDER, r, it) for r in REGIONS for it in INSTANCE_TYPES]


ARMS = build_arms()
ARM_BY_KEY = {a.key(): a for a in ARMS}


def neighbors_fn(arm: Arm) -> List[Arm]:
    return [a for a in ARMS if a.provider == arm.provider and a.family == arm.family and a.region != arm.region]


# ========================
# Rolling coverage & rates
# ========================
launches: Dict[str, deque] = {a.key(): deque() for a in ARMS}
interrupts: Dict[str, deque] = {a.key(): deque() for a in ARMS}
launch_failures: Dict[str, deque] = {a.key(): deque() for a in ARMS}
coverage_slice: Dict[Tuple[str, str, str, int], deque] = defaultdict(deque)  # (prov, reg, fam, local_hour) -> timestamps


def _prune_old(dq: deque, now: float, window_sec: int):
    while dq and (now - dq[0]) > window_sec:
        dq.popleft()


# ========================
# Helpers: AWS pricing & deltas
# ========================
def spot_price_latest(region: str, instance_type: str) -> float:
    try:
        end = datetime.datetime.utcnow()
        start = end - datetime.timedelta(hours=3)
        resp = EC2_CLIENTS[region].describe_spot_price_history(
            InstanceTypes=[instance_type],
            ProductDescriptions=["Linux/UNIX"],
            StartTime=start,
            EndTime=end,
            MaxResults=10,
        )
        hist = sorted(resp.get("SpotPriceHistory", []), key=lambda x: x["Timestamp"])
        if not hist:
            return 0.0
        return float(hist[-1]["SpotPrice"])
    except Exception:
        traceback.print_exc()
        return 0.0


def spot_price_delta(region: str, instance_type: str) -> float:
    try:
        end = datetime.datetime.utcnow()
        start = end - datetime.timedelta(hours=6)
        resp = EC2_CLIENTS[region].describe_spot_price_history(
            InstanceTypes=[instance_type],
            ProductDescriptions=["Linux/UNIX"],
            StartTime=start,
            EndTime=end,
            MaxResults=50,
        )
        prices = sorted(resp.get("SpotPriceHistory", []), key=lambda x: x["Timestamp"])
        if len(prices) < 2:
            return 0.0
        p1 = float(prices[-2]["SpotPrice"])
        p2 = float(prices[-1]["SpotPrice"])
        return p2 - p1
    except Exception:
        traceback.print_exc()
        return 0.0


def price_zscore6(region: str, instance_type: str) -> float:
    delta = spot_price_delta(region, instance_type)
    return float(delta / max(1e-4, abs(delta) + 0.0002))


# ========================
# DB Logging (probe_results schema)
# ========================
def _insert_probe_row(table: str, row: dict):
    try:
        supabase.table(table).insert(row).execute()
        print(f"[DB:{table}] Logged: outcome={row.get('outcome')} instance={row.get('instance_id') or 'N/A'} arm={row.get('provider')}:{row.get('region')}:{row.get('instance_type')}")
    except Exception:
        traceback.print_exc()


def log_probe_result(
    provider: str,
    region: str,
    instance_type: str,
    probe_kind: str,
    meta_policy: str,
    max_minutes: int,
    outcome: str,
    instance_id: Optional[str],
    start_time: Optional[datetime.datetime],
    end_time: Optional[datetime.datetime],
    duration_minutes: float,
    interrupted: bool,
    interrupt_bin: Optional[int],
    survived_hours: int,
    spot_price_usd: Optional[float],
    price_delta_6h: Optional[float],
    sampling_propensity: Optional[float],
    policy_version: Optional[str],
    pred_h1_at_launch: Optional[float],
    pred_risk_5h_at_launch: Optional[float],
    features_snapshot: Optional[dict],
    launch_error: Optional[str] = None,  # not stored (schema compat)
):
    row = {
        "provider": provider,
        "region": region,
        "instance_type": instance_type,
        "probe_kind": probe_kind,
        "meta_policy": meta_policy,
        "max_minutes": max_minutes,
        "policy_version": policy_version,
        "instance_id": instance_id if instance_id else None,
        "outcome": outcome,
        "start_time_utc": start_time.isoformat() if start_time else None,
        "end_time_utc": end_time.isoformat() if end_time else None,
        "duration_minutes": round(float(duration_minutes or 0.0), 2),
        "interrupted": bool(interrupted),
        "interrupt_bin": interrupt_bin,
        "survived_hours": survived_hours,
        "spot_price_usd": None if spot_price_usd is None else float(spot_price_usd),
        "price_delta_6h": None if price_delta_6h is None else float(price_delta_6h),
        "sampling_propensity": None if sampling_propensity is None else float(sampling_propensity),
        "pred_h1_at_launch": None if pred_h1_at_launch is None else float(pred_h1_at_launch),
        "pred_risk_5h_at_launch": None if pred_risk_5h_at_launch is None else float(pred_risk_5h_at_launch),
        "features_snapshot": features_snapshot,
    }
    _insert_probe_row("probe_results", row)


def log_probe_result_topup(**kwargs):
    # exact same schema, just different table
    provider = kwargs["provider"]
    region = kwargs["region"]
    instance_type = kwargs["instance_type"]
    probe_kind = kwargs["probe_kind"]
    meta_policy = kwargs["meta_policy"]
    max_minutes = kwargs["max_minutes"]
    outcome = kwargs["outcome"]
    instance_id = kwargs.get("instance_id")
    start_time = kwargs.get("start_time")
    end_time = kwargs.get("end_time")
    duration_minutes = kwargs.get("duration_minutes", 0.0)
    interrupted = kwargs.get("interrupted", False)
    interrupt_bin = kwargs.get("interrupt_bin")
    survived_hours = kwargs.get("survived_hours", 0)
    spot_price_usd = kwargs.get("spot_price_usd")
    price_delta_6h = kwargs.get("price_delta_6h")
    sampling_propensity = kwargs.get("sampling_propensity")
    policy_version = kwargs.get("policy_version")
    pred_h1_at_launch = kwargs.get("pred_h1_at_launch")
    pred_risk_5h_at_launch = kwargs.get("pred_risk_5h_at_launch")
    features_snapshot = kwargs.get("features_snapshot")

    row = {
        "provider": provider,
        "region": region,
        "instance_type": instance_type,
        "probe_kind": probe_kind,
        "meta_policy": meta_policy,
        "max_minutes": max_minutes,
        "policy_version": policy_version,
        "instance_id": instance_id if instance_id else None,
        "outcome": outcome,
        "start_time_utc": start_time.isoformat() if start_time else None,
        "end_time_utc": end_time.isoformat() if end_time else None,
        "duration_minutes": round(float(duration_minutes or 0.0), 2),
        "interrupted": bool(interrupted),
        "interrupt_bin": interrupt_bin,
        "survived_hours": survived_hours,
        "spot_price_usd": None if spot_price_usd is None else float(spot_price_usd),
        "price_delta_6h": None if price_delta_6h is None else float(price_delta_6h),
        "sampling_propensity": None if sampling_propensity is None else float(sampling_propensity),
        "pred_h1_at_launch": None if pred_h1_at_launch is None else float(pred_h1_at_launch),
        "pred_risk_5h_at_launch": None if pred_risk_5h_at_launch is None else float(pred_risk_5h_at_launch),
        "features_snapshot": features_snapshot,
    }
    _insert_probe_row(TOPUP_TABLE, row)


# ========================
# Option B: Persist model state to Supabase
# ========================
def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def extract_engine_state(engine: CAIBandit) -> dict:
    """
    JSON-serializable snapshot.

    Important: we also capture weights_by_arm from engine.cai() because
    that's the one place we KNOW the weights exist (you print them).
    """
    now_ts = time.time()

    # Guaranteed weights: pull from cai()
    try:
        cai_out = engine.cai(launch_ts=now_ts, mc_samples=300)
        weights_by_arm = {
            arm_key: (cai_out.get(arm_key, {}).get("weights") or {})
            for arm_key in cai_out.keys()
        }
    except Exception:
        traceback.print_exc()
        weights_by_arm = {}

    hedge_logw_by_arm = (
        _safe_getattr(engine, "hedge_logw_by_arm", None)
        or _safe_getattr(engine, "_hedge_logw_by_arm", None)
        or _safe_getattr(engine, "hedge_loss", None)
        or _safe_getattr(engine, "losses", None)
        or {}
    )
    if not isinstance(hedge_logw_by_arm, dict):
        hedge_logw_by_arm = {}

    global_post = _safe_getattr(engine, "global_posterior", None) or _safe_getattr(engine, "global_post", None)
    tod_post = _safe_getattr(engine, "tod_posterior", None) or _safe_getattr(engine, "tod_post", None)

    state = {
        "provider": PROVIDER,
        "policy_version": POLICY_VERSION,
        "created_at_utc": datetime.datetime.utcnow().isoformat(),
        "engine_meta": {
            "H": int(_safe_getattr(engine, "H", 1)),
            "alpha0": float(_safe_getattr(engine, "alpha0", 1.0)),
            "beta0": float(_safe_getattr(engine, "beta0", 19.0)),
            "recent_window_hours": float(
                _safe_getattr(engine, "recent_window_hours", None)
                or (_safe_getattr(engine, "recent_window_sec", RECENT_WINDOW_HOURS * 3600) / 3600.0)
            ),
            "tod_block_hours": int(_safe_getattr(engine, "tod_block_hours", TOD_BLOCK_HOURS)),
            "hedge_eta": float(_safe_getattr(engine, "hedge_eta", HEDGE_ETA)),
            "hedge_decay": float(_safe_getattr(engine, "hedge_decay", HEDGE_DECAY)),
            "weight_floor": float(_safe_getattr(engine, "weight_floor", WEIGHT_FLOOR)),
        },
        "global_posterior": global_post,
        "tod_posterior": tod_post,
        "weights_by_arm": weights_by_arm,
        "hedge_logw_by_arm": hedge_logw_by_arm,
    }
    return state


def _maybe_int_key(k: Any) -> Any:
    # Convert "0" -> 0, "1" -> 1; leave others alone.
    if isinstance(k, int):
        return k
    if isinstance(k, str) and k.isdigit():
        try:
            return int(k)
        except Exception:
            return k
    return k


def _normalize_arm_posterior(
    arm_post: Any,
    H: int,
    alpha0: float,
    beta0: float,
) -> Dict[int, List[float]]:
    """
    arm_post expected shape:
        { 1: [a,b] } or { "1": [a,b] }
    Returns:
        { 1: [a,b], 2: [a,b], ... } (only ensures H exists; doesn't invent more)
    """
    out: Dict[int, List[float]] = {}
    if isinstance(arm_post, dict):
        for hk, hv in arm_post.items():
            hk2 = _maybe_int_key(hk)
            if isinstance(hk2, int):
                out[hk2] = hv
    # ensure horizon H exists
    if H not in out:
        out[H] = [float(alpha0), float(beta0)]
    return out


def _normalize_global_posterior(
    glob: Any,
    arms: List[Arm],
    H: int,
    alpha0: float,
    beta0: float,
) -> Dict[str, Dict[int, List[float]]]:
    out: Dict[str, Dict[int, List[float]]] = {}
    if isinstance(glob, dict):
        for arm_key, arm_post in glob.items():
            out[str(arm_key)] = _normalize_arm_posterior(arm_post, H, alpha0, beta0)

    # ensure every arm exists
    for a in arms:
        k = a.key()
        if k not in out:
            out[k] = {H: [float(alpha0), float(beta0)]}
        else:
            if H not in out[k]:
                out[k][H] = [float(alpha0), float(beta0)]
    return out


def _normalize_tod_posterior(
    tod: Any,
    arms: List[Arm],
    tod_block_hours: int,
    H: int,
    alpha0: float,
    beta0: float,
) -> Dict[int, Dict[str, Dict[int, List[float]]]]:
    """
    Expected shape from DB:
      {
        "0": { "aws:...": {"1":[a,b]} , ... },
        "1": { ... },
        ...
      }

    Returns:
      {
        0: { "aws:...": {1:[a,b]} , ... },
        1: ...
      }
    and ensures ALL blocks 0..num_blocks-1 exist, and all arms have horizon H.
    """
    num_blocks = int(24 // max(1, int(tod_block_hours)))
    out: Dict[int, Dict[str, Dict[int, List[float]]]] = {}

    # ingest existing
    if isinstance(tod, dict):
        for bk, bmap in tod.items():
            b = _maybe_int_key(bk)
            if not isinstance(b, int):
                continue
            if not isinstance(bmap, dict):
                continue
            out[b] = {}
            for arm_key, arm_post in bmap.items():
                out[b][str(arm_key)] = _normalize_arm_posterior(arm_post, H, alpha0, beta0)

    # ensure all blocks + all arms
    for b in range(num_blocks):
        if b not in out:
            out[b] = {}
        for a in arms:
            k = a.key()
            if k not in out[b]:
                out[b][k] = {H: [float(alpha0), float(beta0)]}
            else:
                if H not in out[b][k]:
                    out[b][k][H] = [float(alpha0), float(beta0)]
    return out


def apply_engine_state(engine: CAIBandit, state: dict) -> bool:
    """
    Restore engine state from snapshot dict.

    CRITICAL: normalize keys so new_cai_bandit can index with ints:
      tod_posterior[block:int][arm_key][H:int]
      global_posterior[arm_key][H:int]
    """
    if not state:
        return False

    alpha0 = float(_safe_getattr(engine, "alpha0", 1.0))
    beta0 = float(_safe_getattr(engine, "beta0", 19.0))
    H_engine = int(_safe_getattr(engine, "H", 1))
    tod_block_hours = int(_safe_getattr(engine, "tod_block_hours", TOD_BLOCK_HOURS))

    global_post_raw = state.get("global_posterior")
    tod_post_raw = state.get("tod_posterior")

    try:
        if hasattr(engine, "global_posterior"):
            engine.global_posterior = _normalize_global_posterior(
                global_post_raw, ARMS, H_engine, alpha0, beta0
            )
        if hasattr(engine, "tod_posterior"):
            engine.tod_posterior = _normalize_tod_posterior(
                tod_post_raw, ARMS, tod_block_hours, H_engine, alpha0, beta0
            )
    except Exception:
        traceback.print_exc()
        print("[STATE] Failed to normalize/apply posterior state; continuing with fresh priors.")

    print("[STATE] Applied model state snapshot into engine (with key normalization).")
    return True


def save_engine_state_to_db(engine: CAIBandit):
    try:
        state = extract_engine_state(engine)

        payload = {
            "provider": PROVIDER,
            "policy_version": POLICY_VERSION,
            "snapshot_time_utc": datetime.datetime.utcnow().isoformat(),
            "state": state,
            "weights_by_arm": state.get("weights_by_arm", {}) or {},
            "hedge_logw_by_arm": state.get("hedge_logw_by_arm", {}) or {},
            "hedge_eta": HEDGE_ETA,
            "hedge_decay": HEDGE_DECAY,
            "weight_floor": WEIGHT_FLOOR,
            "recent_window_hours": int(RECENT_WINDOW_HOURS),
            "tod_block_hours": int(TOD_BLOCK_HOURS),
        }

        supabase.table(MODEL_STATE_TABLE).insert(payload).execute()
        print(f"[STATE] Saved model state snapshot -> {MODEL_STATE_TABLE}")
    except Exception:
        traceback.print_exc()


def load_latest_engine_state_from_db() -> Optional[dict]:
    """Load latest snapshot from MODEL_STATE_TABLE for this provider + policy_version."""
    try:
        resp = (
            supabase.table(MODEL_STATE_TABLE)
            .select("state,snapshot_time_utc")
            .eq("provider", PROVIDER)
            .eq("policy_version", POLICY_VERSION)
            .order("snapshot_time_utc", desc=True)
            .limit(1)
            .execute()
        )
        data = getattr(resp, "data", None) or []
        if not data:
            print(f"[STATE] No snapshot found in {MODEL_STATE_TABLE} for provider={PROVIDER} policy={POLICY_VERSION}")
            return None
        row = data[0]
        st = row.get("state")
        ts = row.get("snapshot_time_utc")
        print(f"[STATE] Loaded snapshot from {MODEL_STATE_TABLE} at {ts}")
        return st if isinstance(st, dict) else None
    except Exception:
        traceback.print_exc()
        return None


# ========================
# Coverage penalty
# ========================
def coverage_penalty(region: str, family: str) -> float:
    now = time.time()
    window_sec = COVERAGE_WINDOW_DAYS * 86400
    penalties = []
    for hour in range(24):
        key = (PROVIDER, region, family, hour)
        dq = coverage_slice.get(key, deque())
        _prune_old(dq, now, window_sec)
        cnt = len(dq)
        if cnt < MIN_COVERAGE_PER_SLICE:
            penalties.append(-(MIN_COVERAGE_PER_SLICE - cnt))
        else:
            over = max(0, cnt - MAX_COVERAGE_PENALTY_AFTER)
            penalties.append(over)
    return float(sum(penalties))


# ========================
# Features snapshot
# ========================
def build_features_snapshot(region: str, instance_type: str, start_time: datetime.datetime) -> dict:
    local_hour = datetime.datetime.now().hour
    utc_hour = start_time.hour
    dow = int(start_time.weekday())
    z6 = price_zscore6(region, instance_type)
    return {
        "launch_local_hour": local_hour,
        "launch_utc_hour": utc_hour,
        "launch_dow": dow,
        "price_zscore_6h": z6,
    }


# ========================
# Launch / monitor
# ========================
def launch_spot_probe(region: str, instance_type: str):
    now_ts = time.time()
    arm_key = f"{PROVIDER}:{region}:{instance_type}"
    launches[arm_key].append(now_ts)

    price_at_launch = spot_price_latest(region, instance_type)
    price_delta6 = spot_price_delta(region, instance_type)

    try:
        response = EC2_CLIENTS[region].run_instances(
            ImageId=AMI_IDS[region],
            InstanceType=instance_type,
            KeyName=KEY_NAMES[region],
            SecurityGroupIds=[SECURITY_GROUP_IDS[region]],
            InstanceMarketOptions={"MarketType": "spot", "SpotOptions": {"SpotInstanceType": "one-time"}},
            MinCount=1,
            MaxCount=1,
        )
        instance = response["Instances"][0]
        instance_id = instance["InstanceId"]
        start_time = datetime.datetime.utcnow()

        try:
            EC2_CLIENTS[region].get_waiter("instance_exists").wait(
                InstanceIds=[instance_id],
                WaiterConfig={"Delay": 3, "MaxAttempts": 10},
            )
        except Exception as werr:
            print(f"[WARN] Waiter instance_exists failed for {instance_id}: {werr}")

        local_hour = datetime.datetime.now().hour
        coverage_slice[(PROVIDER, region, instance_type, local_hour)].append(now_ts)

        features = build_features_snapshot(region, instance_type, start_time)
        print(f"[{start_time}] Launched {instance_type} in {region}: {instance_id}")
        return instance_id, start_time, price_at_launch, price_delta6, features, None

    except Exception as e:
        err = str(e)
        print(f"Launch failed in {region} ({instance_type}): {err}")
        launch_failures[arm_key].append(now_ts)
        start_time = datetime.datetime.utcnow()
        features = build_features_snapshot(region, instance_type, start_time)
        return None, None, price_at_launch, price_delta6, features, err


def launch_spot_probe_with_retries(region: str, instance_type: str, tries: int):
    last_err = None
    last_features = None
    last_price = None
    last_delta = None

    for attempt in range(1, max(1, int(tries)) + 1):
        iid, st, price_at_launch, price_delta6, features, err = launch_spot_probe(region, instance_type)
        if iid and st:
            return iid, st, price_at_launch, price_delta6, features, None
        last_err = err
        last_features = features
        last_price = price_at_launch
        last_delta = price_delta6
        if attempt < tries:
            time.sleep(1.5)

    return None, None, last_price, last_delta, last_features, last_err


def terminate_if_exists(region: str, instance_id: str):
    try:
        EC2_CLIENTS[region].terminate_instances(InstanceIds=[instance_id])
    except Exception:
        pass


def monitor_probe(
    region: str,
    instance_type: str,
    instance_id: str,
    start_time: datetime.datetime,
    engine: CAIBandit,
    max_minutes: int,
    spot_price_at_launch: Optional[float],
    price_delta6: Optional[float],
    probe_kind: str,
    meta_policy: str,
    sampling_propensity: float,
    policy_version: str,
    pred_h1_at_launch: Optional[float],
    features_snapshot: dict,
    log_fn: Callable[..., None] = log_probe_result,  # default preserves original behavior
):
    instance = EC2_RESOURCES[region].Instance(instance_id)
    arm_key = f"{PROVIDER}:{region}:{instance_type}"

    # Warm-up: eventual consistency
    appeared = False
    for _ in range(10):
        try:
            instance.load()
            appeared = True
            break
        except ClientError as ce:
            code = ce.response.get("Error", {}).get("Code", "")
            if code in ("InvalidInstanceID.NotFound", "InvalidInstanceID.Malformed"):
                time.sleep(3)
                continue
            print(f"[WARN] Unexpected ClientError on load() {instance_id}: {code} -> {ce}")
            break
        except Exception as e:
            print(f"[WARN] Error loading state for {instance_id}: {e}")
            time.sleep(2)

    if not appeared:
        end_time = datetime.datetime.utcnow()
        log_fn(
            provider=PROVIDER,
            region=region,
            instance_type=instance_type,
            probe_kind=probe_kind,
            meta_policy=meta_policy,
            max_minutes=max_minutes,
            outcome="LaunchFailed",
            instance_id=instance_id,
            start_time=None,
            end_time=end_time,
            duration_minutes=0.0,
            interrupted=False,
            interrupt_bin=None,
            survived_hours=0,
            spot_price_usd=spot_price_at_launch,
            price_delta_6h=price_delta6,
            sampling_propensity=sampling_propensity,
            policy_version=policy_version,
            pred_h1_at_launch=pred_h1_at_launch,
            pred_risk_5h_at_launch=None,
            features_snapshot=features_snapshot,
        )
        print(f"[WARN] {instance_id} never became visible; recorded as LaunchFailed.")
        return

    elapsed = 0.0
    while elapsed < max_minutes:
        try:
            instance.load()
            state = instance.state["Name"]
        except ClientError as ce:
            code = ce.response.get("Error", {}).get("Code", "")
            if code in ("InvalidInstanceID.NotFound", "InvalidInstanceID.Malformed"):
                time.sleep(3)
                elapsed += 3.0 / 60.0
                continue
            print(f"[WARN] ClientError loading state for {instance_id}: {code} -> {ce}")
            break
        except Exception as e:
            print(f"[WARN] Error loading state for {instance_id}: {e}")
            break

        if state in ["shutting-down", "terminated", "stopping", "stopped"]:
            end_time = datetime.datetime.utcnow()
            dur_min = (end_time - start_time).total_seconds() / 60.0
            interrupts[arm_key].append(time.time())

            pr = ProbeResult(
                arm=ARM_BY_KEY[arm_key],
                interrupted=True,
                launch_ts=start_time.timestamp(),
            )
            engine.update_from_probe_results([pr])

            if SAVE_SNAPSHOT_AFTER_EACH_UPDATE:
                save_engine_state_to_db(engine)

            log_fn(
                provider=PROVIDER,
                region=region,
                instance_type=instance_type,
                probe_kind=probe_kind,
                meta_policy=meta_policy,
                max_minutes=max_minutes,
                outcome="Interrupted",
                instance_id=instance_id,
                start_time=start_time,
                end_time=end_time,
                duration_minutes=dur_min,
                interrupted=True,
                interrupt_bin=1,
                survived_hours=0,
                spot_price_usd=spot_price_at_launch,
                price_delta_6h=price_delta6,
                sampling_propensity=sampling_propensity,
                policy_version=policy_version,
                pred_h1_at_launch=pred_h1_at_launch,
                pred_risk_5h_at_launch=None,
                features_snapshot=features_snapshot,
            )
            print(f"[{end_time}] {instance_id} INTERRUPTED after {dur_min:.1f} min")
            return

        time.sleep(STATUS_CHECK_SEC)
        elapsed += STATUS_CHECK_SEC / 60.0

    # Censored at max_minutes -> terminate
    end_time = datetime.datetime.utcnow()
    terminate_if_exists(region, instance_id)
    dur_min = (end_time - start_time).total_seconds() / 60.0

    pr = ProbeResult(
        arm=ARM_BY_KEY[arm_key],
        interrupted=False,
        launch_ts=start_time.timestamp(),
    )
    engine.update_from_probe_results([pr])

    if SAVE_SNAPSHOT_AFTER_EACH_UPDATE:
        save_engine_state_to_db(engine)

    log_fn(
        provider=PROVIDER,
        region=region,
        instance_type=instance_type,
        probe_kind=probe_kind,
        meta_policy=meta_policy,
        max_minutes=max_minutes,
        outcome="Censored (Stopped by Design)",
        instance_id=instance_id,
        start_time=start_time,
        end_time=end_time,
        duration_minutes=dur_min,
        interrupted=False,
        interrupt_bin=None,
        survived_hours=1 if max_minutes >= 60 else 0,
        spot_price_usd=spot_price_at_launch,
        price_delta_6h=price_delta6,
        sampling_propensity=sampling_propensity,
        policy_version=policy_version,
        pred_h1_at_launch=pred_h1_at_launch,
        pred_risk_5h_at_launch=None,
        features_snapshot=features_snapshot,
    )
    print(f"[{end_time}] {instance_id} SURVIVED (censored at {max_minutes} min)")


# ========================
# Metrics ingestion (optional)
# ========================
def ingest_simple_metrics_into_engine(engine: CAIBandit):
    price_delta_by_arm = {}
    fail_rate_by_arm = {}
    eviction_rate_by_arm = {}
    cp_by_arm = {}

    now = time.time()
    window_sec = 12 * 3600

    for a in ARMS:
        k = a.key()
        price_delta_by_arm[k] = spot_price_delta(a.region, a.family)

        dqL = launches[k]
        dqF = launch_failures[k]
        launches[k] = deque([ts for ts in dqL if now - ts <= window_sec], maxlen=len(dqL))
        launch_failures[k] = deque([ts for ts in dqF if now - ts <= window_sec], maxlen=len(dqF))
        tot = len(launches[k])
        fls = len(launch_failures[k])
        fail_rate_by_arm[k] = (fls / tot) if tot > 0 else 0.0

        dqI = interrupts[k]
        interrupts[k] = deque([ts for ts in dqI if now - ts <= window_sec], maxlen=len(dqI))
        eviction_rate_by_arm[k] = (len(interrupts[k]) / tot) if tot > 0 else 0.0

        cp_by_arm[k] = 1.0 if abs(price_delta_by_arm[k]) > 0.002 else 0.0

    engine.ingest_metrics(price_delta_by_arm, fail_rate_by_arm, eviction_rate_by_arm, cp_by_arm)


# ========================
# Planning: uncertainty-driven + coverage penalty
# ========================
def weighted_sample_with_replacement(
    items: List[str], scores: List[float], n: int
) -> List[Tuple[str, float]]:
    if not items or n <= 0:
        return []

    w = np.array(scores, dtype=float)

    min_w = np.min(w)
    if min_w <= 0:
        w = w - min_w + 1e-9

    probs = w / np.sum(w)

    chosen = []
    for _ in range(n):
        idx = int(np.random.choice(len(items), p=probs))
        chosen.append((items[idx], float(probs[idx])))

    return chosen


def plan(engine: CAIBandit, now_ts: float) -> List[dict]:
    cai_out = engine.cai(launch_ts=now_ts, mc_samples=PLAN_MC_SAMPLES)

    scored = []
    for k, v in cai_out.items():
        lo, hi = v["h1_CI80"]
        unc = float(max(0.0, hi - lo))

        region = k.split(":")[1]
        family = k.split(":")[2]
        cov_pen = float(coverage_penalty(region, family))

        score = unc - (RHO_COVERAGE * cov_pen)
        scored.append((k, score, unc, cov_pen))

    scored.sort(key=lambda x: x[1], reverse=True)
    pool = scored[: min(TOPK_CANDIDATES, len(scored))]

    items = [k for k, _, _, _ in pool]
    scores = [s for _, s, _, _ in pool]

    picks = weighted_sample_with_replacement(items, scores, TOTAL_PROBES_PER_INTERVAL)

    chosen = []
    for arm_key, pi in picks:
        chosen.append(
            {
                "arm_key": arm_key,
                "max_minutes": PROBE_MIN,
                "probe_kind": "micro",
                "meta_policy": "ensemble_h1",
                "sampling_propensity": max(1e-6, float(pi)),
            }
        )
    return chosen


def current_pred_h1(engine: CAIBandit, arm_key: str, now_ts: float) -> float:
    out = engine.cai(launch_ts=now_ts, mc_samples=800).get(arm_key)
    if not out:
        return float("nan")
    return float(out["h1_mean"])


def _print_cai_table(engine: CAIBandit, now_ts: float):
    cai = engine.cai(launch_ts=now_ts, mc_samples=800)
    for k, s in cai.items():
        w = s.get("weights", {})
        if not isinstance(w, dict):
            w = {}
        print(
            f"  {k:35s} "
            f"CAI1={int(s.get('CAI1', 0)):3d}  "
            f"h1_mean={float(s.get('h1_mean', 0.0)):.3f}  "
            f"CI80={s.get('h1_CI80')}  "
            f"w={{recent:{float(w.get('recent', 0.0)):.2f}, tod:{float(w.get('tod', 0.0)):.2f}, global:{float(w.get('global', 0.0)):.2f}}}"
        )


# ========================
# NEW: Top-up enforcement
# ========================
def enforce_min_probes_per_arm_this_cycle(
    engine: CAIBandit,
    now_ts: float,
    counts_this_cycle: Dict[str, int],
    active_threads: List[threading.Thread],
):
    print(f"[TOPUP] Enforcing min={MIN_PROBES_PER_ARM_PER_CYCLE} probe-attempts per arm this cycle...")

    for arm in ARMS:
        arm_key = arm.key()
        have = int(counts_this_cycle.get(arm_key, 0))
        need = max(0, MIN_PROBES_PER_ARM_PER_CYCLE - have)
        if need <= 0:
            continue

        print(f"[TOPUP] {arm_key} has {have}, needs +{need} more attempts.")

        for _ in range(need):
            pred_h1 = current_pred_h1(engine, arm_key, now_ts=now_ts)

            iid, st, price_at_launch, price_delta6, features, err = launch_spot_probe_with_retries(
                arm.region, arm.family, tries=TOPUP_LAUNCH_TRIES
            )

            # IMPORTANT: count attempt no matter what (prevents infinite loop)
            counts_this_cycle[arm_key] = int(counts_this_cycle.get(arm_key, 0)) + 1

            if iid and st:
                th = threading.Thread(
                    target=monitor_probe,
                    args=(
                        arm.region,
                        arm.family,
                        iid,
                        st,
                        engine,
                        PROBE_MIN,
                        price_at_launch,
                        price_delta6,
                        "topup",
                        "min4_topup",
                        1.0,
                        POLICY_VERSION,
                        pred_h1,
                        features,
                        log_probe_result_topup,
                    ),
                )
                th.daemon = True
                th.start()
                active_threads.append(th)
            else:
                log_probe_result_topup(
                    provider=PROVIDER,
                    region=arm.region,
                    instance_type=arm.family,
                    probe_kind="topup",
                    meta_policy="min4_topup",
                    max_minutes=PROBE_MIN,
                    outcome="LaunchFailed",
                    instance_id=None,
                    start_time=None,
                    end_time=datetime.datetime.utcnow(),
                    duration_minutes=0.0,
                    interrupted=False,
                    interrupt_bin=None,
                    survived_hours=0,
                    spot_price_usd=price_at_launch,
                    price_delta_6h=price_delta6,
                    sampling_propensity=1.0,
                    policy_version=POLICY_VERSION,
                    pred_h1_at_launch=pred_h1,
                    pred_risk_5h_at_launch=None,
                    features_snapshot=features,
                )
                print(f"[TOPUP] LaunchFailed for {arm_key} after {TOPUP_LAUNCH_TRIES} tries; counted anyway. err={err}")

    print("[TOPUP] Done enforcing per-arm minimums.")


# ========================
# Hourly loop
# ========================
def run_meta_scheduler():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drain", action="store_true", help="Finish current probes; do not launch new ones.")
    args, _ = parser.parse_known_args()

    engine = CAIBandit(
        arms=ARMS,
        H=H,
        alpha0=1.0,
        beta0=19.0,
        recent_window_hours=RECENT_WINDOW_HOURS,
        tod_block_hours=TOD_BLOCK_HOURS,
        hedge_eta=HEDGE_ETA,
        hedge_decay=HEDGE_DECAY,
        weight_floor=WEIGHT_FLOOR,
        baseline_budget_per_hour=TOTAL_PROBES_PER_INTERVAL,
        burst_size=0,
        shadow_burst=0,
        burst_hours=0,
        event_threshold=EVENT_THRESHOLD,
        neighbors_fn=neighbors_fn,
    )

    # ---- Option B: load latest state snapshot (if exists) ----
    snap = load_latest_engine_state_from_db()
    if snap:
        apply_engine_state(engine, snap)

    active_threads: List[threading.Thread] = []

    while True:
        t0 = datetime.datetime.utcnow()
        now_ts = time.time()
        print(f"\n=== META-CAI cycle {t0.isoformat()}Z ===")

        # Per-cycle attempt counts (THIS HOUR ONLY)
        counts_this_cycle: Dict[str, int] = {a.key(): 0 for a in ARMS}

        # 1) feed metrics (safe while draining)
        ingest_simple_metrics_into_engine(engine)

        # Reap finished threads
        active_threads = [th for th in active_threads if th.is_alive()]
        print(f"Active probes still running: {len(active_threads)}")

        if drain_enabled(args.drain):
            print("[DRAIN] Draining mode is ON — no new launches will be started (including top-ups).")
            if not active_threads:
                print("[DRAIN] All probes finished. Exiting scheduler.")
                if SAVE_SNAPSHOT_EVERY_CYCLE:
                    save_engine_state_to_db(engine)
                return

            _print_cai_table(engine, now_ts=now_ts)
            time.sleep(min(300, INTERVAL_MIN * 60))
            continue

        # 2) plan probes
        selection = plan(engine, now_ts=now_ts)
        print("Planned probes:", selection)

        # 3) launch & monitor (bandit probes; unchanged logging to probe_results)
        for sel in selection:
            arm_key = sel["arm_key"]
            max_min = sel["max_minutes"]
            probe_kind = sel["probe_kind"]
            meta_pol = sel["meta_policy"]
            pi = sel["sampling_propensity"]

            # Count attempt immediately (even if launch fails)
            counts_this_cycle[arm_key] = int(counts_this_cycle.get(arm_key, 0)) + 1

            arm = ARM_BY_KEY[arm_key]
            pred_h1 = current_pred_h1(engine, arm_key, now_ts=now_ts)

            iid, st, price_at_launch, price_delta6, features, err = launch_spot_probe(arm.region, arm.family)
            if iid and st:
                th = threading.Thread(
                    target=monitor_probe,
                    args=(
                        arm.region,
                        arm.family,
                        iid,
                        st,
                        engine,
                        max_min,
                        price_at_launch,
                        price_delta6,
                        probe_kind,
                        meta_pol,
                        pi,
                        POLICY_VERSION,
                        pred_h1,
                        features,
                        log_probe_result,  # main table (original behavior)
                    ),
                )
                th.daemon = True
                th.start()
                active_threads.append(th)
            else:
                # Bandit launch failure logs to main table (unchanged behavior)
                log_probe_result(
                    provider=PROVIDER,
                    region=arm.region,
                    instance_type=arm.family,
                    probe_kind=probe_kind,
                    meta_policy=meta_pol,
                    max_minutes=max_min,
                    outcome="LaunchFailed",
                    instance_id=None,
                    start_time=None,
                    end_time=datetime.datetime.utcnow(),
                    duration_minutes=0.0,
                    interrupted=False,
                    interrupt_bin=None,
                    survived_hours=0,
                    spot_price_usd=price_at_launch,
                    price_delta_6h=price_delta6,
                    sampling_propensity=pi,
                    policy_version=POLICY_VERSION,
                    pred_h1_at_launch=pred_h1,
                    pred_risk_5h_at_launch=None,
                    features_snapshot=features,
                    launch_error=err,
                )

        # 3b) enforce per-arm minimum attempts (top-ups; logs to probe_results_topup)
        enforce_min_probes_per_arm_this_cycle(
            engine=engine,
            now_ts=now_ts,
            counts_this_cycle=counts_this_cycle,
            active_threads=active_threads,
        )

        # 4) report CAI
        _print_cai_table(engine, now_ts=now_ts)

        # 5) persist snapshot once per cycle
        if SAVE_SNAPSHOT_EVERY_CYCLE:
            save_engine_state_to_db(engine)

        # 6) sleep
        print(f"Sleeping {INTERVAL_MIN} minutes...")
        time.sleep(INTERVAL_MIN * 60)


if __name__ == "__main__":
    run_meta_scheduler()
