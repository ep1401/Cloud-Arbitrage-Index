"""
Meta-bandit CAI Scheduler (AWS) — Context-ready + DB-ready
- Two internal policies: Fail-fast micro-probes vs Long-run probes
- Meta-controller shifts mix based on uncertainty at S1 vs S5
- Info-per-dollar scoring = expected Beta variance reduction minus cost & coverage penalty
- Online hazard model via CAIBandit
- Logs rich context into Supabase probe_results:
    * features at launch (features_snapshot JSON)
    * sampling_propensity (for bandit debiasing)
    * frozen predictions at launch (pred_h1_at_launch, pred_risk_5h_at_launch)
    * policy_version

Requires:
  - boto3, numpy, supabase-py, botocore
  - cai_bandit.py in same folder
"""

import os
import time
import math
import argparse
import datetime
import threading
import traceback
from collections import defaultdict, deque
from typing import Dict, Tuple, List, Optional

import boto3
import numpy as np
from botocore.exceptions import ClientError
from supabase import create_client, Client

from cai_bandit import CAIBandit, Arm, ProbeResult

# ============ Drain controls ============
DRAIN_SENTINEL_PATH = "/home/ec2-user/.cai_drain"

def drain_enabled(cli_drain: bool = False) -> bool:
    # Any of these turns draining on:
    #   - CLI: --drain
    #   - Env:  CAI_DRAIN=1
    #   - File: /home/ec2-user/.cai_drain exists
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
    "us-west-2": "ami-06a974f9b8a97ecf2"
}
KEY_NAMES = {"us-east-1": "my-spot-key", "us-west-2": "my-spot-key"}
SECURITY_GROUP_IDS = {"us-east-1": "sg-098bfbec48d19166d", "us-west-2": "sg-03b7cc83a5e0f65c2"}

# Main cadence & horizons
INTERVAL_MIN = 60                 # run once per hour
H = 5                             # modeling horizon in hours for CAI
STATUS_CHECK_SEC = 60

# Probe mix & durations
TOTAL_PROBES_PER_INTERVAL = 10    # total probes launched each hour
MICRO_PROBE_MIN = 60              # 60 minutes (fail-fast)
LONG_PROBE_MIN = 300              # exactly 5 hours

# Meta policy mix (auto-adjusted each hour)
BASE_FAILFAST_SHARE = 0.70
MIN_FAILFAST_SHARE = 0.40
MAX_FAILFAST_SHARE = 0.85

# Info-per-dollar weights
ALPHA = 1.0   # bin 1 (1h)
BETA  = 1.0   # bin 3 (3h)
GAMMA = 1.0   # bin 5 (5h)
LAMBDA_COST = 1.0
RHO_COVERAGE = 0.5

# Candidate set sizes
TOPK_LONG = 4
TOPK_FAIL = 4

# Coverage quotas (rolling 7 days) at granularity (provider, region, family, local_hour)
COVERAGE_WINDOW_DAYS = 7
MIN_COVERAGE_PER_SLICE = 6        # if below this, auto-boost
MAX_COVERAGE_PENALTY_AFTER = 24   # start penalizing after this many probes

# Supabase (project creds)
SUPABASE_URL = "https://zpuzwrnaeceqgadwviab.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpwdXp3cm5hZWNlcWdhZHd2aWFiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjAzNzAyNTksImV4cCI6MjA3NTk0NjI1OX0.WHDkMTCGy1ITUjE8LRLdVpFthJ1v3IjzJySrJNOExWo"
POLICY_VERSION = "meta_v2_ctxready"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# AWS
EC2_CLIENTS = {r: boto3.client("ec2", region_name=r) for r in REGIONS}
EC2_RESOURCES = {r: boto3.resource("ec2", region_name=r) for r in REGIONS}

# Arms
def build_arms() -> List[Arm]:
    return [Arm(PROVIDER, r, it) for r in REGIONS for it in INSTANCE_TYPES]

ARMS = build_arms()
ARM_BY_KEY = {a.key(): a for a in ARMS}

def neighbors_fn(arm: Arm) -> List[Arm]:
    # same provider & family, other regions
    return [a for a in ARMS if a.provider == arm.provider and a.family == arm.family and a.region != arm.region]

# Rolling coverage & rates
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
            MaxResults=10
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
            MaxResults=50
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
    # simple z-like proxy using 6h delta scaled
    delta = spot_price_delta(region, instance_type)
    return float(delta / max(1e-4, abs(delta) + 0.0002))  # bounded-ish

# ========================
# Expected variance reduction (Beta-Binomial)
# ========================
def beta_var(a: float, b: float) -> float:
    s = a + b
    return (a * b) / (s * s * (s + 1.0)) if s > 1 else 0.25

def expected_delta_var_after_one(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        a = max(a, 1e-3); b = max(b, 1e-3)
    p = a / (a + b)
    v_now = beta_var(a, b)
    v_int = beta_var(a + 1.0, b)     # interruption
    v_sur = beta_var(a, b + 1.0)     # survival step
    v_next = p * v_int + (1.0 - p) * v_sur
    return max(0.0, v_now - v_next)

# ========================
# DB Logging (matches your schema)
# ========================
def log_probe_result(
    provider: str,
    region: str,
    instance_type: str,
    probe_kind: str,              # 'micro' | 'long'
    meta_policy: str,             # 'fail_fast' | 'long_run'
    max_minutes: int,
    outcome: str,                 # 'LaunchFailed' | 'Interrupted' | 'Censored (Stopped by Design)' | 'Censored (Survived Horizon)' | 'Launched'
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
    # kept in signature for call-sites, but NOT written to DB:
    launch_error: Optional[str] = None,
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
        "features_snapshot": features_snapshot
        # NOTE: 'launch_error' intentionally omitted to match your table schema
    }
    try:
        supabase.table("probe_results").insert(row).execute()
        print(f"[DB] Logged: outcome={outcome} instance={instance_id or 'N/A'} arm={provider}:{region}:{instance_type}")
    except Exception:
        traceback.print_exc()

# ========================
# Launch/monitor
# ========================
def current_pred_h1_and_risk5(engine: CAIBandit, arm_key: str) -> Tuple[float, float]:
    # Posterior means as frozen predictions
    a1,b1 = engine.posterior[arm_key][1]
    h1 = a1 / (a1 + b1)
    # Compute risk_5 from means (fast) as a frozen proxy
    s = 1.0
    for t in range(1, min(5, engine.H) + 1):
        at, bt = engine.posterior[arm_key][t]
        s *= (1.0 - (at / (at + bt)))
    risk5 = 1.0 - s
    return float(h1), float(risk5)

def build_features_snapshot(region: str, instance_type: str, start_time: datetime.datetime) -> dict:
    local_hour = datetime.datetime.now().hour  # scheduler host TZ
    utc_hour = start_time.hour
    dow = int(start_time.weekday())
    z6 = price_zscore6(region, instance_type)
    return {
        "launch_local_hour": local_hour,
        "launch_utc_hour": utc_hour,
        "launch_dow": dow,
        "price_zscore_6h": z6,
    }

def launch_spot_probe(region: str, instance_type: str):
    """Returns (instance_id, start_time, spot_price_at_launch, price_delta_6h, features_snapshot, err)"""
    now_ts = time.time()
    arm_key = f"{PROVIDER}:{region}:{instance_type}"
    launches[arm_key].append(now_ts)

    # capture price signals at launch
    price_at_launch = spot_price_latest(region, instance_type)
    price_delta6 = spot_price_delta(region, instance_type)
    try:
        response = EC2_CLIENTS[region].run_instances(
            ImageId=AMI_IDS[region],
            InstanceType=instance_type,
            KeyName=KEY_NAMES[region],
            SecurityGroupIds=[SECURITY_GROUP_IDS[region]],
            InstanceMarketOptions={"MarketType": "spot", "SpotOptions": {"SpotInstanceType": "one-time"}},
            MinCount=1, MaxCount=1
        )
        instance = response["Instances"][0]
        instance_id = instance["InstanceId"]
        start_time = datetime.datetime.utcnow()

        # best-effort waiter for visibility (won't block forever)
        try:
            EC2_CLIENTS[region].get_waiter('instance_exists').wait(
                InstanceIds=[instance_id],
                WaiterConfig={'Delay': 3, 'MaxAttempts': 10}
            )
        except Exception as werr:
            print(f"[WARN] Waiter instance_exists failed for {instance_id}: {werr}")

        # coverage slice
        local_hour = datetime.datetime.now().hour
        coverage_slice[(PROVIDER, region, instance_type, local_hour)].append(now_ts)

        features = build_features_snapshot(region, instance_type, start_time)

        print(f"[{start_time}] Launched {instance_type} in {region}: {instance_id}")
        return instance_id, start_time, price_at_launch, price_delta6, features, None
    except Exception as e:
        err = str(e)
        print(f"Launch failed in {region} ({instance_type}): {err}")
        launch_failures[arm_key].append(now_ts)
        # still return features from 'now' for logging consistency
        start_time = datetime.datetime.utcnow()
        features = build_features_snapshot(region, instance_type, start_time)
        return None, None, price_at_launch, price_delta6, features, err

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
    pred_h1_at_launch: float,
    pred_risk5_at_launch: float,
    features_snapshot: dict,
):
    instance = EC2_RESOURCES[region].Instance(instance_id)
    arm_key = f"{PROVIDER}:{region}:{instance_type}"

    # --- Warm-up: handle eventual consistency (ID not immediately queryable) ---
    appeared = False
    for _ in range(10):  # ~30s total
        try:
            instance.load()
            appeared = True
            break
        except ClientError as ce:
            code = ce.response.get("Error", {}).get("Code", "")
            if code in ("InvalidInstanceID.NotFound", "InvalidInstanceID.Malformed"):
                time.sleep(3)
                continue
            else:
                print(f"[WARN] Unexpected ClientError on load() {instance_id}: {code} -> {ce}")
                break
        except Exception as e:
            print(f"[WARN] Error loading state for {instance_id}: {e}")
            time.sleep(2)

    if not appeared:
        # Treat as a failed launch (do NOT censor at 0.0)
        end_time = datetime.datetime.utcnow()
        log_probe_result(
            provider=PROVIDER, region=region, instance_type=instance_type,
            probe_kind=probe_kind, meta_policy=meta_policy, max_minutes=max_minutes,
            outcome="LaunchFailed",
            instance_id=instance_id, start_time=None, end_time=end_time,
            duration_minutes=0.0, interrupted=False, interrupt_bin=None,
            survived_hours=0,
            spot_price_usd=spot_price_at_launch, price_delta_6h=price_delta6,
            sampling_propensity=sampling_propensity, policy_version=policy_version,
            pred_h1_at_launch=pred_h1_at_launch, pred_risk_5h_at_launch=pred_risk5_at_launch,
            features_snapshot=features_snapshot,
        )
        print(f"[WARN] {instance_id} never became visible; recorded as LaunchFailed.")
        return
    # --- end warm-up ---

    elapsed = 0.0
    while elapsed < max_minutes:
        try:
            instance.load()
            state = instance.state["Name"]
        except ClientError as ce:
            code = ce.response.get("Error", {}).get("Code", "")
            if code in ("InvalidInstanceID.NotFound", "InvalidInstanceID.Malformed"):
                # transient; retry a few times
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
            dur_h = dur_min / 60.0
            bin_idx = max(1, min(H, int(math.ceil(dur_h))))
            interrupts[arm_key].append(time.time())

            pr = ProbeResult(
                arm=ARM_BY_KEY[arm_key],
                survived_hours=int(math.floor(min(dur_h, H))),
                interrupted=True,
                interrupt_bin=bin_idx
            )
            engine.update_posteriors([pr])

            log_probe_result(
                provider=PROVIDER, region=region, instance_type=instance_type,
                probe_kind=probe_kind, meta_policy=meta_policy, max_minutes=max_minutes,
                outcome="Interrupted",
                instance_id=instance_id, start_time=start_time, end_time=end_time,
                duration_minutes=dur_min, interrupted=True, interrupt_bin=bin_idx,
                survived_hours=int(math.floor(min(dur_h, H))),
                spot_price_usd=spot_price_at_launch, price_delta_6h=price_delta6,
                sampling_propensity=sampling_propensity, policy_version=policy_version,
                pred_h1_at_launch=pred_h1_at_launch, pred_risk_5h_at_launch=pred_risk5_at_launch,
                features_snapshot=features_snapshot
            )
            print(f"[{end_time}] {instance_id} INTERRUPTED after {dur_min:.1f} min (bin={bin_idx})")
            return

        time.sleep(STATUS_CHECK_SEC)
        elapsed += STATUS_CHECK_SEC / 60.0

    # survived max_minutes -> censor
    end_time = datetime.datetime.utcnow()
    terminate_if_exists(region, instance_id)
    dur_min = (end_time - start_time).total_seconds() / 60.0
    dur_h = dur_min / 60.0

    pr = ProbeResult(
        arm=ARM_BY_KEY[arm_key],
        survived_hours=min(H, int(math.floor(dur_h))),
        interrupted=False,
        interrupt_bin=None
    )
    engine.update_posteriors([pr])

    log_probe_result(
        provider=PROVIDER, region=region, instance_type=instance_type,
        probe_kind=probe_kind, meta_policy=meta_policy, max_minutes=max_minutes,
        outcome="Censored (Stopped by Design)",
        instance_id=instance_id, start_time=start_time, end_time=end_time,
        duration_minutes=dur_min, interrupted=False, interrupt_bin=None,
        survived_hours=min(H, int(math.floor(dur_h))),
        spot_price_usd=spot_price_at_launch, price_delta_6h=price_delta6,
        sampling_propensity=sampling_propensity, policy_version=policy_version,
        pred_h1_at_launch=pred_h1_at_launch, pred_risk_5h_at_launch=pred_risk5_at_launch,
        features_snapshot=features_snapshot
    )
    print(f"[{end_time}] {instance_id} CENSORED at {dur_min:.1f} min (max={max_minutes} min)")

# ========================
# Scoring: info-per-dollar with coverage & cost
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
            penalties.append(- (MIN_COVERAGE_PER_SLICE - cnt))  # negative = boost
        else:
            over = max(0, cnt - MAX_COVERAGE_PENALTY_AFTER)
            penalties.append(over)
    return float(sum(penalties))

def info_per_dollar_score(engine: CAIBandit, arm_key: str, alpha=ALPHA, beta=BETA, gamma=GAMMA):
    post = engine.posterior[arm_key]
    t1 = 1
    t3 = min(3, H)
    t5 = min(5, H)
    a1,b1 = post[t1]
    a3,b3 = post[t3]
    a5,b5 = post[t5]

    d1 = expected_delta_var_after_one(a1,b1)
    d3 = expected_delta_var_after_one(a3,b3)
    d5 = expected_delta_var_after_one(a5,b5)

    region = arm_key.split(":")[1]
    family = arm_key.split(":")[2]

    price = spot_price_latest(region, family) or 0.0
    info_sum = alpha*d1 + beta*d3 + gamma*d5
    cov_pen = coverage_penalty(region, family)
    return info_sum, price, cov_pen

# ========================
# Meta-controller: choose mix based on uncertainty at S1 vs S5
# ========================
def compute_failfast_share(engine: CAIBandit) -> float:
    S1_w, S5_w = [], []
    for a in ARMS:
        k = a.key()
        S1 = engine._sample_S(k, horizon=1, mc_samples=400)
        S5 = engine._sample_S(k, horizon=H, mc_samples=400)
        S1_w.append(np.percentile(S1,90) - np.percentile(S1,10))
        S5_w.append(np.percentile(S5,90) - np.percentile(S5,10))
    w1 = float(np.mean(S1_w)) if S1_w else 0.0
    w5 = float(np.mean(S5_w)) if S5_w else 0.0
    share = BASE_FAILFAST_SHARE
    if w5 > w1 * 1.05:
        share = max(MIN_FAILFAST_SHARE, BASE_FAILFAST_SHARE - 0.20)  # more long
    elif w1 > w5 * 1.05:
        share = min(MAX_FAILFAST_SHARE, BASE_FAILFAST_SHARE + 0.10)  # more fail-fast
    return share

# ========================
# Candidate sets L (long) and F (fail)
# ========================
def predicted_S(engine: CAIBandit, arm_key: str, horizon: int) -> float:
    S = 1.0
    for t in range(1, horizon+1):
        a,b = engine.posterior[arm_key][t]
        p = a / (a+b)
        S *= (1.0 - p)
    return S

def build_candidates(engine: CAIBandit):
    # Early hazard rank by mean h1 (and recent deltas)
    early = []
    for a in ARMS:
        k = a.key()
        a1,b1 = engine.posterior[k][1]
        h1 = a1 / (a1+b1)
        pd = spot_price_delta(a.region, a.family)
        early.append((k, h1, pd))
    early.sort(key=lambda x: (-x[1], -abs(x[2])))
    F = [k for k,_,_ in early[:TOPK_FAIL]]

    # Long-run survival rank by S5 (or S_H), cost-aware
    longc = []
    for a in ARMS:
        k = a.key()
        S5 = predicted_S(engine, k, min(5, H))
        price = spot_price_latest(a.region, a.family) or 0.0
        longc.append((k, S5, -price))
    longc.sort(key=lambda x: (-x[1], x[2]))
    L = [k for k,_,_ in longc[:TOPK_LONG]]
    return L, F

# ========================
# Planning (weighted sampling + propensity)
# ========================
def _weighted_sample_without_replacement(items: List[str], scores: List[float], n: int) -> List[Tuple[str, float]]:
    """
    Sample n distinct items without replacement with probabilities proportional to positive scores.
    Returns list of (item, propensity_at_draw). (Approximate propensity per-draw; fine for IPW.)
    """
    eps = 1e-9
    chosen: List[Tuple[str, float]] = []
    pool = list(items)
    weights = np.array(scores, dtype=float)
    # shift up so all positive
    m = weights.min()
    if m <= 0:
        weights = weights - m + eps
    for _ in range(min(n, len(pool))):
        probs = weights / (weights.sum() if weights.sum() > 0 else len(weights))
        idx = int(np.random.choice(len(pool), p=probs))
        chosen.append((pool[idx], float(probs[idx])))
        pool.pop(idx)
        weights = np.delete(weights, idx)
        if len(weights) == 0:
            break
    return chosen

def plan(engine: CAIBandit):
    """
    Returns a list of selections, each a dict:
      {
        "arm_key": ...,
        "max_minutes": ...,
        "probe_kind": "micro" | "long",
        "meta_policy": "fail_fast" | "long_run",
        "sampling_propensity": float
      }
    """
    fail_share = compute_failfast_share(engine)
    n_fail = max(1, int(round(TOTAL_PROBES_PER_INTERVAL * fail_share)))
    n_long = max(1, TOTAL_PROBES_PER_INTERVAL - n_fail)

    L, F = build_candidates(engine)

    # Score arms using info-per-dollar. Two scores: micro vs long.
    micro_items, micro_scores = [], []
    long_items, long_scores = [], []
    for k in set(L + F):
        info_sum, price, cov_pen = info_per_dollar_score(engine, k)
        micro_cost = price * (MICRO_PROBE_MIN / 60.0)
        long_cost  = price * (LONG_PROBE_MIN / 60.0)
        cov_term = RHO_COVERAGE * cov_pen
        micro_score = info_sum - LAMBDA_COST * micro_cost - cov_term
        long_score  = info_sum - LAMBDA_COST * long_cost  - cov_term
        micro_items.append(k); micro_scores.append(micro_score)
        long_items.append(k);  long_scores.append(long_score)

    micro_picks = _weighted_sample_without_replacement(micro_items, micro_scores, n_fail)
    long_picks  = _weighted_sample_without_replacement(long_items, long_scores, n_long)

    chosen = []
    for k, pi in micro_picks:
        chosen.append({
            "arm_key": k,
            "max_minutes": MICRO_PROBE_MIN,
            "probe_kind": "micro",
            "meta_policy": "fail_fast",
            "sampling_propensity": max(1e-6, float(pi)),
        })
    for k, pi in long_picks:
        chosen.append({
            "arm_key": k,
            "max_minutes": LONG_PROBE_MIN,
            "probe_kind": "long",
            "meta_policy": "long_run",
            "sampling_propensity": max(1e-6, float(pi)),
        })
    return chosen

# ========================
# Metrics ingestion (optional; enables event compatibility)
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

        # rolling fail rate
        dqL = launches[k]
        dqF = launch_failures[k]
        _dqL = deque([ts for ts in dqL if now - ts <= window_sec], maxlen=len(dqL))
        _dqF = deque([ts for ts in dqF if now - ts <= window_sec], maxlen=len(dqF))
        launches[k] = _dqL; launch_failures[k] = _dqF
        tot = len(_dqL); fls = len(_dqF)
        fail_rate_by_arm[k] = (fls / tot) if tot > 0 else 0.0

        # rolling interrupts
        dqI = interrupts[k]
        _dqI = deque([ts for ts in dqI if now - ts <= window_sec], maxlen=len(dqI))
        interrupts[k] = _dqI
        eviction_rate_by_arm[k] = (len(_dqI) / tot) if tot > 0 else 0.0

        # change-point proxy: big price jump
        cp_by_arm[k] = 1.0 if abs(price_delta_by_arm[k]) > 0.002 else 0.0

    engine.ingest_metrics(price_delta_by_arm, fail_rate_by_arm, eviction_rate_by_arm, cp_by_arm)

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
        alpha0=1.0, beta0=19.0,
        baseline_budget_per_hour=0,     # we plan explicitly
        burst_size=0, shadow_burst=0,   # keep bursts off in this variant
        event_threshold=9999,
        neighbors_fn=neighbors_fn
    )

    # Keep track of active monitor threads across ticks
    active_threads: List[threading.Thread] = []

    while True:
        t0 = datetime.datetime.utcnow()
        print(f"\n=== META-CAI cycle {t0.isoformat()}Z ===")

        # 1) feed metrics (safe to do even while draining)
        ingest_simple_metrics_into_engine(engine)

        # Reap finished threads from previous tick
        active_threads = [th for th in active_threads if th.is_alive()]
        print(f"Active probes still running: {len(active_threads)}")

        # If draining: skip planning/launching entirely
        if drain_enabled(args.drain):
            print("[DRAIN] Draining mode is ON — no new launches will be started.")
            if not active_threads:
                print("[DRAIN] All probes finished. Exiting scheduler.")
                return
            # Optional: still print CAI for visibility
            cai = engine.cai(H=H, mc_samples=800, risk_high_is_high=True)
            for k, s in cai.items():
                print(f"  {k:35s} CAI{H}={s['CAI']:3d}  risk_mean={s['risk_mean']:.3f}  CI80={s['risk_CI80']}")
            # Sleep short while draining so we notice when threads end
            time.sleep(min(300, INTERVAL_MIN * 60))
            continue

        # 2) plan probes using meta-policy
        selection = plan(engine)
        print("Planned probes:", selection)

        # 3) launch & monitor (only if NOT draining)
        for sel in selection:
            arm_key = sel["arm_key"]
            max_min = sel["max_minutes"]
            probe_kind = sel["probe_kind"]
            meta_pol = sel["meta_policy"]
            pi = sel["sampling_propensity"]

            arm = ARM_BY_KEY[arm_key]
            pred_h1, pred_risk5 = current_pred_h1_and_risk5(engine, arm_key)

            iid, st, price_at_launch, price_delta6, features, err = launch_spot_probe(arm.region, arm.family)
            if iid and st:
                th = threading.Thread(
                    target=monitor_probe,
                    args=(
                        arm.region, arm.family, iid, st, engine, max_min,
                        price_at_launch, price_delta6, probe_kind, meta_pol,
                        pi, POLICY_VERSION, pred_h1, pred_risk5, features
                    )
                )
                th.daemon = True
                th.start()
                active_threads.append(th)
            else:
                log_probe_result(
                    provider=PROVIDER, region=arm.region, instance_type=arm.family,
                    probe_kind=probe_kind, meta_policy=meta_pol, max_minutes=max_min,
                    outcome="LaunchFailed",
                    instance_id=None, start_time=None, end_time=None,
                    duration_minutes=0.0, interrupted=False, interrupt_bin=None,
                    survived_hours=0,
                    spot_price_usd=price_at_launch, price_delta_6h=price_delta6,
                    sampling_propensity=pi, policy_version=POLICY_VERSION,
                    pred_h1_at_launch=pred_h1, pred_risk_5h_at_launch=pred_risk5,
                    features_snapshot=features
                )

        # 4) report CAI
        cai = engine.cai(H=H, mc_samples=800, risk_high_is_high=True)
        for k, s in cai.items():
            print(f"  {k:35s} CAI{H}={s['CAI']:3d}  risk_mean={s['risk_mean']:.3f}  CI80={s['risk_CI80']}")

        # 5) sleep to next hour
        print(f"Sleeping {INTERVAL_MIN} minutes...")
        time.sleep(INTERVAL_MIN * 60)

if __name__ == "__main__":
    run_meta_scheduler()
