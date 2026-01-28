"""
Meta-bandit CAI Scheduler (AZURE) — Context-ready + DB-ready
- Two internal policies: Fail-fast micro-probes vs Long-run probes
- Meta-controller shifts mix based on uncertainty at S1 vs S5
- Info-per-dollar scoring = expected Beta variance reduction minus cost & coverage penalty
- Online hazard model via CAIBandit
- Logs rich context into Supabase probe_results:
    * features at launch (features_snapshot JSON)
    * sampling_propensity (for bandit debiasing)
    * frozen predictions at launch (pred_h1_at_launch, pred_risk_5h_at_launch)
    * policy_version

Pricing:
- Uses Azure Retail Prices API to fetch *current* unitPrice for the VM SKU in-region.
- Prefers Spot meters (skuName/meterName contains "Spot"), Linux (excludes Windows).
- Falls back to non-spot Consumption price if Spot meter not found for that SKU/region.
- Caches results to avoid hammering the API and computes a 6h delta from local history.

Requires:
  - azure-identity, azure-mgmt-compute, numpy, supabase-py, requests
  - cai_bandit.py in same folder
"""

import os
import time
import math
import argparse
import datetime
import threading
import traceback
import uuid
from collections import defaultdict, deque
from typing import Dict, Tuple, List, Optional

import numpy as np
import requests
from supabase import create_client, Client

from azure.identity import ClientSecretCredential, DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.compute.models import (
    VirtualMachine,
    HardwareProfile,
    NetworkProfile,
    NetworkInterfaceReference,
    OSProfile,
    LinuxConfiguration,
    SshConfiguration,
    SshPublicKey,
    StorageProfile,
    ImageReference,
    OSDisk,
    DiskCreateOptionTypes,
    BillingProfile,
    VirtualMachineEvictionPolicyTypes,
)
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError

from cai_bandit import CAIBandit, Arm, ProbeResult

# ============ Drain controls ============
DRAIN_SENTINEL_PATH = "/home/ec2-user/.cai_drain"  # keep same sentinel path for convenience


def drain_enabled(cli_drain: bool = False) -> bool:
    return (
        cli_drain
        or os.environ.get("CAI_DRAIN", "").strip() in ("1", "true", "True")
        or os.path.exists(DRAIN_SENTINEL_PATH)
    )


# ========================
# Config (edit to taste)
# ========================
PROVIDER = "azure"

REGIONS = ["eastus", "westus2"]

INSTANCE_TYPES = [
    "Standard_D2ads_v5",
    "Standard_D2s_v3",
    "Standard_E2s_v3",
]

RESOURCE_GROUPS = {
    "eastus": "cai-probes-eastus",
    "westus2": "cai-probes-westus2",
}

# Admin + SSH
ADMIN_USERNAME = "azureuser"
SSH_PUBLIC_KEY_PATH = os.path.expanduser("~/.ssh/cai_azure.pub")

# Image (Ubuntu LTS)
IMAGE = ImageReference(
    publisher="Canonical",
    offer="0001-com-ubuntu-server-jammy",
    sku="22_04-lts-gen2",
    version="latest",
)

# Main cadence & horizons
INTERVAL_MIN = 60
H = 5
STATUS_CHECK_SEC = 60

# Probe mix & durations
TOTAL_PROBES_PER_INTERVAL = 10 
MICRO_PROBE_MIN = 60 
LONG_PROBE_MIN = 300 
# TOTAL_PROBES_PER_INTERVAL = 2 
# MICRO_PROBE_MIN = 4 
# LONG_PROBE_MIN = 6 

BASE_FAILFAST_SHARE = 0.70
MIN_FAILFAST_SHARE = 0.40
MAX_FAILFAST_SHARE = 0.85

ALPHA = 1.0
BETA = 1.0
GAMMA = 1.0
LAMBDA_COST = 1.0
RHO_COVERAGE = 0.5

TOPK_LONG = 4
TOPK_FAIL = 4

COVERAGE_WINDOW_DAYS = 7
MIN_COVERAGE_PER_SLICE = 6
MAX_COVERAGE_PENALTY_AFTER = 24

# Supabase
SUPABASE_URL = "https://lkcqegpcjpwfhtjwvtwm.supabase.co"
SUPABASE_KEY = "sb_secret_VMCPfHfIIlFsXUrT8mJnVA_czghv_Du"
POLICY_VERSION = "meta_v2_ctxready_azure"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ========================
# NIC pool config
# ========================
NIC_ID_FILES = {
    "eastus": "eastus_nic_ids.txt",
    "westus2": "westus2_nic_ids.txt",
}


def _load_nic_ids_from_files() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for region, path in NIC_ID_FILES.items():
        if not os.path.exists(path):
            raise RuntimeError(
                f"Missing NIC id file for {region}: {path}\n"
                f"Generate it with:\n"
                f"  az network nic list -g {RESOURCE_GROUPS[region]} --query \"[].id\" -o tsv > {path}"
            )
        with open(path, "r", encoding="utf-8") as f:
            ids = [ln.strip() for ln in f.readlines() if ln.strip()]
        if not ids:
            raise RuntimeError(f"NIC id file {path} is empty.")
        out[region] = ids
    return out


NIC_IDS: Dict[str, List[str]] = _load_nic_ids_from_files()

NIC_LOCK = threading.Lock()
NIC_POOL: Dict[str, deque] = {r: deque(NIC_IDS[r]) for r in REGIONS}


def acquire_nic(region: str) -> str:
    with NIC_LOCK:
        pool = NIC_POOL.get(region)
        if not pool or len(pool) == 0:
            raise RuntimeError(f"No free NICs available in {region}. (All NICs currently attached.)")
        return pool.popleft()


def release_nic(region: str, nic_id: str):
    if not nic_id:
        return
    with NIC_LOCK:
        NIC_POOL.setdefault(region, deque()).append(nic_id)


def nic_pool_status() -> Dict[str, Tuple[int, int]]:
    with NIC_LOCK:
        return {r: (len(NIC_POOL.get(r, deque())), len(NIC_IDS.get(r, []))) for r in REGIONS}


# ========================
# Azure auth + clients
# ========================
def _azure_credential():
    tid = os.environ.get("AZURE_TENANT_ID")
    cid = os.environ.get("AZURE_CLIENT_ID")
    csec = os.environ.get("AZURE_CLIENT_SECRET")
    if tid and cid and csec:
        return ClientSecretCredential(tenant_id=tid, client_id=cid, client_secret=csec)
    return DefaultAzureCredential(exclude_interactive_browser_credential=False)


AZ_SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID", "").strip()
if not AZ_SUBSCRIPTION_ID:
    raise RuntimeError("Missing AZURE_SUBSCRIPTION_ID env var.")

AZ_CRED = _azure_credential()
COMPUTE = ComputeManagementClient(AZ_CRED, AZ_SUBSCRIPTION_ID)

# ========================
# Arms
# ========================
def build_arms() -> List[Arm]:
    return [Arm(PROVIDER, r, it) for r in REGIONS for it in INSTANCE_TYPES]


ARMS = build_arms()
ARM_BY_KEY = {a.key(): a for a in ARMS}


def neighbors_fn(arm: Arm) -> List[Arm]:
    return [a for a in ARMS if a.provider == arm.provider and a.family == arm.family and a.region != arm.region]


# Rolling coverage & rates
launches: Dict[str, deque] = {a.key(): deque() for a in ARMS}
interrupts: Dict[str, deque] = {a.key(): deque() for a in ARMS}
launch_failures: Dict[str, deque] = {a.key(): deque() for a in ARMS}
coverage_slice: Dict[Tuple[str, str, str, int], deque] = defaultdict(deque)


def _prune_old(dq: deque, now: float, window_sec: int):
    while dq and (now - dq[0]) > window_sec:
        dq.popleft()


# ========================
# Live pricing via Azure Retail Prices API
# ========================
AZ_RETAIL_ENDPOINT = "https://prices.azure.com/api/retail/prices"
AZ_RETAIL_API_VERSION = "2023-01-01-preview"
AZ_PRICE_CACHE_TTL_SEC = 30 * 60  # 30 minutes
AZ_PRICE_TIMEOUT_SEC = 15

_price_cache_lock = threading.Lock()
_price_cache: Dict[Tuple[str, str], Tuple[float, float, str]] = {}  # (region, sku) -> (ts, price, source)
_price_hist_6h: Dict[Tuple[str, str], deque] = defaultdict(deque)  # (region, sku) -> deque[(ts, price)]


def _requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "cai-probes/1.0"})
    return s


_HTTP = _requests_session()


def _retail_fetch_all_items(filter_str: str, max_pages: int = 5) -> List[dict]:
    items: List[dict] = []
    params = {
        "api-version": AZ_RETAIL_API_VERSION,
        "currencyCode": "USD",
        "$filter": filter_str,
    }
    url = AZ_RETAIL_ENDPOINT
    for _ in range(max_pages):
        resp = _HTTP.get(url, params=params, timeout=AZ_PRICE_TIMEOUT_SEC)
        resp.raise_for_status()
        data = resp.json()
        items.extend(data.get("Items", []) or [])
        nxt = data.get("NextPageLink")
        if not nxt:
            break
        url = nxt
        params = None  # NextPageLink already encodes params
    return items


def _pick_best_vm_price(items: List[dict]) -> Tuple[float, str]:
    # keep only hourly meters
    hourly = [x for x in items if (x.get("unitOfMeasure") == "1 Hour")]
    linuxish = [x for x in hourly if "Windows" not in (x.get("productName") or "")]

    def is_spot(x: dict) -> bool:
        sn = (x.get("skuName") or "")
        mn = (x.get("meterName") or "")
        return ("Spot" in sn) or ("Spot" in mn)

    spot = [x for x in linuxish if is_spot(x)]
    if spot:
        best = min(spot, key=lambda x: float(x.get("unitPrice") or x.get("retailPrice") or 1e18))
        price = float(best.get("unitPrice") or best.get("retailPrice") or 0.0)
        return price, "azure_retail_spot"

    cons = [x for x in linuxish if (x.get("type") == "Consumption")]
    if cons:
        best = min(cons, key=lambda x: float(x.get("unitPrice") or x.get("retailPrice") or 1e18))
        price = float(best.get("unitPrice") or best.get("retailPrice") or 0.0)
        return price, "azure_retail_consumption_fallback"

    if hourly:
        best = min(hourly, key=lambda x: float(x.get("unitPrice") or x.get("retailPrice") or 1e18))
        price = float(best.get("unitPrice") or best.get("retailPrice") or 0.0)
        return price, "azure_retail_hourly_fallback"

    return 0.0, "azure_retail_not_found"


def spot_price_latest(region: str, vm_size: str) -> Tuple[float, str]:
    key = (region, vm_size)
    now = time.time()

    with _price_cache_lock:
        hit = _price_cache.get(key)
        if hit:
            ts, price, src = hit
            if (now - ts) <= AZ_PRICE_CACHE_TTL_SEC:
                return float(price), str(src)

    filter_str = (
        f"serviceName eq 'Virtual Machines' and "
        f"armRegionName eq '{region}' and "
        f"armSkuName eq '{vm_size}'"
    )

    try:
        items = _retail_fetch_all_items(filter_str=filter_str, max_pages=5)
        price, src = _pick_best_vm_price(items)
    except Exception as e:
        print(f"[PRICE] Retail API failed for {region}/{vm_size}: {e}")
        price, src = 0.0, "azure_retail_error"

    with _price_cache_lock:
        _price_cache[key] = (now, float(price), str(src))

        hist = _price_hist_6h[key]
        hist.append((now, float(price)))
        six_h = 6 * 3600
        while hist and (now - hist[0][0]) > six_h:
            hist.popleft()

    return float(price), str(src)


def spot_price_delta(region: str, vm_size: str) -> float:
    key = (region, vm_size)
    now = time.time()
    with _price_cache_lock:
        hist = _price_hist_6h.get(key, deque())
        six_h = 6 * 3600
        while hist and (now - hist[0][0]) > six_h:
            hist.popleft()
        if len(hist) < 2:
            return 0.0
        return float(hist[-1][1] - hist[0][1])


def price_zscore6(region: str, vm_size: str) -> float:
    d = spot_price_delta(region, vm_size)
    return float(d / max(1e-4, abs(d) + 0.0002))


# ========================
# Expected variance reduction
# ========================
def beta_var(a: float, b: float) -> float:
    s = a + b
    return (a * b) / (s * s * (s + 1.0)) if s > 1 else 0.25


def expected_delta_var_after_one(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        a = max(a, 1e-3)
        b = max(b, 1e-3)
    p = a / (a + b)
    v_now = beta_var(a, b)
    v_int = beta_var(a + 1.0, b)
    v_sur = beta_var(a, b + 1.0)
    v_next = p * v_int + (1.0 - p) * v_sur
    return max(0.0, v_now - v_next)


# ========================
# DB Logging
# ========================
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
    try:
        supabase.table("probe_results").insert(row).execute()
        print(f"[DB] Logged: outcome={outcome} instance={instance_id or 'N/A'} arm={provider}:{region}:{instance_type}")
    except Exception:
        traceback.print_exc()


# ========================
# Launch/monitor (Azure)
# ========================
def current_pred_h1_and_risk5(engine: CAIBandit, arm_key: str) -> Tuple[float, float]:
    a1, b1 = engine.posterior[arm_key][1]
    h1 = a1 / (a1 + b1)
    s = 1.0
    for t in range(1, min(5, engine.H) + 1):
        at, bt = engine.posterior[arm_key][t]
        s *= (1.0 - (at / (at + bt)))
    risk5 = 1.0 - s
    return float(h1), float(risk5)


def build_features_snapshot(region: str, vm_size: str, start_time: datetime.datetime) -> dict:
    local_hour = datetime.datetime.now().hour
    utc_hour = start_time.hour
    dow = int(start_time.weekday())

    price_now, price_src = spot_price_latest(region, vm_size)
    z6 = price_zscore6(region, vm_size)

    return {
        "launch_local_hour": local_hour,
        "launch_utc_hour": utc_hour,
        "launch_dow": dow,
        "price_zscore_6h": z6,
        "price_source": price_src,
        "price_per_hour_usd_snapshot": price_now,
        "nic_pool_free_total": nic_pool_status().get(region, (None, None)),
    }


def _read_ssh_key() -> str:
    if not os.path.exists(SSH_PUBLIC_KEY_PATH):
        raise RuntimeError(f"SSH public key not found at {SSH_PUBLIC_KEY_PATH}")
    with open(SSH_PUBLIC_KEY_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()


def launch_spot_probe(region: str, vm_size: str):
    now_ts = time.time()
    arm_key = f"{PROVIDER}:{region}:{vm_size}"
    launches[arm_key].append(now_ts)

    price_at_launch, price_src = spot_price_latest(region, vm_size)
    price_delta6 = spot_price_delta(region, vm_size)

    rg = RESOURCE_GROUPS[region]
    vm_name = f"cai-{region}-{vm_size.lower().replace('_','-')}-{uuid.uuid4().hex[:8]}"

    start_time = datetime.datetime.utcnow()

    nic_id: Optional[str] = None
    try:
        nic_id = acquire_nic(region)
    except Exception as e:
        features = build_features_snapshot(region, vm_size, start_time)
        err = str(e)
        print(f"[NIC] No NIC available in {region} for {vm_size}: {err}")
        launch_failures[arm_key].append(now_ts)
        return None, None, price_at_launch, price_delta6, features, None, err

    features = build_features_snapshot(region, vm_size, start_time)

    try:
        ssh_key = _read_ssh_key()

        vm_params = VirtualMachine(
            location=region,
            hardware_profile=HardwareProfile(vm_size=vm_size),

            # IMPORTANT: avoid PriorityTypes (not present in some azure-mgmt-compute versions)
            priority="Spot",

            eviction_policy=VirtualMachineEvictionPolicyTypes.deallocate,

            # IMPORTANT: use float
            billing_profile=BillingProfile(max_price=-1.0),

            network_profile=NetworkProfile(
                network_interfaces=[NetworkInterfaceReference(id=nic_id, primary=True)]
            ),
            os_profile=OSProfile(
                computer_name=vm_name,
                admin_username=ADMIN_USERNAME,
                linux_configuration=LinuxConfiguration(
                    disable_password_authentication=True,
                    ssh=SshConfiguration(
                        public_keys=[
                            SshPublicKey(
                                path=f"/home/{ADMIN_USERNAME}/.ssh/authorized_keys",
                                key_data=ssh_key,
                            )
                        ]
                    ),
                ),
            ),
            storage_profile=StorageProfile(
                image_reference=IMAGE,
                os_disk=OSDisk(
                    create_option=DiskCreateOptionTypes.from_image,
                    delete_option="Delete",
                    managed_disk={"storage_account_type": "Standard_LRS"},
                ),
            ),
        )

        poller = COMPUTE.virtual_machines.begin_create_or_update(rg, vm_name, vm_params)
        poller.result()

        local_hour = datetime.datetime.now().hour
        coverage_slice[(PROVIDER, region, vm_size, local_hour)].append(now_ts)

        print(
            f"[{start_time}] Launched {vm_size} in {region}: {vm_name} "
            f"(nic={nic_id.split('/')[-1]}) price=${price_at_launch:.6f}/h src={price_src}"
        )
        return vm_name, start_time, price_at_launch, price_delta6, features, nic_id, None

    except Exception as e:
        err = str(e)
        print(f"Launch failed in {region} ({vm_size}): {err}")
        launch_failures[arm_key].append(now_ts)
        release_nic(region, nic_id)
        return None, None, price_at_launch, price_delta6, features, None, err


def terminate_if_exists(region: str, vm_name: str):
    rg = RESOURCE_GROUPS[region]
    try:
        COMPUTE.virtual_machines.begin_deallocate(rg, vm_name).result()
    except Exception:
        pass
    try:
        COMPUTE.virtual_machines.begin_delete(rg, vm_name).result()
    except Exception:
        pass


def _get_power_state(region: str, vm_name: str) -> Optional[str]:
    rg = RESOURCE_GROUPS[region]
    try:
        iv = COMPUTE.virtual_machines.instance_view(rg, vm_name)
        for st in iv.statuses or []:
            if st.code and st.code.startswith("PowerState/"):
                return st.code
        return None
    except ResourceNotFoundError:
        return None
    except HttpResponseError:
        return None


def monitor_probe(
    region: str,
    vm_size: str,
    vm_name: str,
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
    nic_id: str,
):
    arm_key = f"{PROVIDER}:{region}:{vm_size}"

    try:
        for _ in range(10):
            ps = _get_power_state(region, vm_name)
            if ps is not None:
                break
            time.sleep(3)

        elapsed = 0.0
        while elapsed < max_minutes:
            ps = _get_power_state(region, vm_name)

            if ps is None or ps in ("PowerState/deallocated", "PowerState/stopped"):
                end_time = datetime.datetime.utcnow()
                dur_min = (end_time - start_time).total_seconds() / 60.0
                dur_h = dur_min / 60.0
                bin_idx = max(1, min(H, int(math.ceil(dur_h))))
                interrupts[arm_key].append(time.time())

                pr = ProbeResult(
                    arm=ARM_BY_KEY[arm_key],
                    survived_hours=int(math.floor(min(dur_h, H))),
                    interrupted=True,
                    interrupt_bin=bin_idx,
                )
                engine.update_posteriors([pr])

                log_probe_result(
                    provider=PROVIDER,
                    region=region,
                    instance_type=vm_size,
                    probe_kind=probe_kind,
                    meta_policy=meta_policy,
                    max_minutes=max_minutes,
                    outcome="Interrupted",
                    instance_id=vm_name,
                    start_time=start_time,
                    end_time=end_time,
                    duration_minutes=dur_min,
                    interrupted=True,
                    interrupt_bin=bin_idx,
                    survived_hours=int(math.floor(min(dur_h, H))),
                    spot_price_usd=spot_price_at_launch,
                    price_delta_6h=price_delta6,
                    sampling_propensity=sampling_propensity,
                    policy_version=policy_version,
                    pred_h1_at_launch=pred_h1_at_launch,
                    pred_risk_5h_at_launch=pred_risk5_at_launch,
                    features_snapshot=features_snapshot,
                )
                print(f"[{end_time}] {vm_name} INTERRUPTED after {dur_min:.1f} min (bin={bin_idx}) ps={ps}")
                return

            time.sleep(STATUS_CHECK_SEC)
            elapsed += STATUS_CHECK_SEC / 60.0

        end_time = datetime.datetime.utcnow()
        terminate_if_exists(region, vm_name)
        dur_min = (end_time - start_time).total_seconds() / 60.0
        dur_h = dur_min / 60.0

        pr = ProbeResult(
            arm=ARM_BY_KEY[arm_key],
            survived_hours=min(H, int(math.floor(dur_h))),
            interrupted=False,
            interrupt_bin=None,
        )
        engine.update_posteriors([pr])

        log_probe_result(
            provider=PROVIDER,
            region=region,
            instance_type=vm_size,
            probe_kind=probe_kind,
            meta_policy=meta_policy,
            max_minutes=max_minutes,
            outcome="Censored (Stopped by Design)",
            instance_id=vm_name,
            start_time=start_time,
            end_time=end_time,
            duration_minutes=dur_min,
            interrupted=False,
            interrupt_bin=None,
            survived_hours=min(H, int(math.floor(dur_h))),
            spot_price_usd=spot_price_at_launch,
            price_delta_6h=price_delta6,
            sampling_propensity=sampling_propensity,
            policy_version=policy_version,
            pred_h1_at_launch=pred_h1_at_launch,
            pred_risk_5h_at_launch=pred_risk5_at_launch,
            features_snapshot=features_snapshot,
        )
        print(f"[{end_time}] {vm_name} CENSORED at {dur_min:.1f} min (max={max_minutes} min)")

    finally:
        release_nic(region, nic_id)


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
            penalties.append(-(MIN_COVERAGE_PER_SLICE - cnt))
        else:
            over = max(0, cnt - MAX_COVERAGE_PENALTY_AFTER)
            penalties.append(over)
    return float(sum(penalties))


def info_per_dollar_score(engine: CAIBandit, arm_key: str, alpha=ALPHA, beta=BETA, gamma=GAMMA):
    post = engine.posterior[arm_key]
    t1 = 1
    t3 = min(3, H)
    t5 = min(5, H)
    a1, b1 = post[t1]
    a3, b3 = post[t3]
    a5, b5 = post[t5]

    d1 = expected_delta_var_after_one(a1, b1)
    d3 = expected_delta_var_after_one(a3, b3)
    d5 = expected_delta_var_after_one(a5, b5)

    region = arm_key.split(":")[1]
    family = arm_key.split(":")[2]

    price, _src = spot_price_latest(region, family)
    info_sum = alpha * d1 + beta * d3 + gamma * d5
    cov_pen = coverage_penalty(region, family)
    return info_sum, float(price), cov_pen


# ========================
# Meta-controller
# ========================
def compute_failfast_share(engine: CAIBandit) -> float:
    S1_w, S5_w = [], []
    for a in ARMS:
        k = a.key()
        S1 = engine._sample_S(k, horizon=1, mc_samples=400)
        S5 = engine._sample_S(k, horizon=H, mc_samples=400)
        S1_w.append(np.percentile(S1, 90) - np.percentile(S1, 10))
        S5_w.append(np.percentile(S5, 90) - np.percentile(S5, 10))
    w1 = float(np.mean(S1_w)) if S1_w else 0.0
    w5 = float(np.mean(S5_w)) if S5_w else 0.0
    share = BASE_FAILFAST_SHARE
    if w5 > w1 * 1.05:
        share = max(MIN_FAILFAST_SHARE, BASE_FAILFAST_SHARE - 0.20)
    elif w1 > w5 * 1.05:
        share = min(MAX_FAILFAST_SHARE, BASE_FAILFAST_SHARE + 0.10)
    return share


# ========================
# Candidate sets
# ========================
def predicted_S(engine: CAIBandit, arm_key: str, horizon: int) -> float:
    S = 1.0
    for t in range(1, horizon + 1):
        a, b = engine.posterior[arm_key][t]
        p = a / (a + b)
        S *= (1.0 - p)
    return S


def build_candidates(engine: CAIBandit):
    early = []
    for a in ARMS:
        k = a.key()
        a1, b1 = engine.posterior[k][1]
        h1 = a1 / (a1 + b1)
        pd = spot_price_delta(a.region, a.family)
        early.append((k, h1, pd))
    early.sort(key=lambda x: (-x[1], -abs(x[2])))
    F = [k for k, _, _ in early[:TOPK_FAIL]]

    longc = []
    for a in ARMS:
        k = a.key()
        S5 = predicted_S(engine, k, min(5, H))
        price, _src = spot_price_latest(a.region, a.family)
        longc.append((k, S5, -price))
    longc.sort(key=lambda x: (-x[1], x[2]))
    L = [k for k, _, _ in longc[:TOPK_LONG]]
    return L, F


# ========================
# Planning
# ========================
def _weighted_sample_without_replacement(items: List[str], scores: List[float], n: int) -> List[Tuple[str, float]]:
    eps = 1e-9
    chosen: List[Tuple[str, float]] = []
    pool = list(items)
    weights = np.array(scores, dtype=float)
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
    fail_share = compute_failfast_share(engine)
    n_fail = max(1, int(round(TOTAL_PROBES_PER_INTERVAL * fail_share)))
    n_long = max(1, TOTAL_PROBES_PER_INTERVAL - n_fail)

    L, F = build_candidates(engine)

    micro_items, micro_scores = [], []
    long_items, long_scores = [], []
    for k in set(L + F):
        info_sum, price, cov_pen = info_per_dollar_score(engine, k)
        micro_cost = price * (MICRO_PROBE_MIN / 60.0)
        long_cost = price * (LONG_PROBE_MIN / 60.0)
        cov_term = RHO_COVERAGE * cov_pen
        micro_score = info_sum - LAMBDA_COST * micro_cost - cov_term
        long_score = info_sum - LAMBDA_COST * long_cost - cov_term
        micro_items.append(k)
        micro_scores.append(micro_score)
        long_items.append(k)
        long_scores.append(long_score)

    micro_picks = _weighted_sample_without_replacement(micro_items, micro_scores, n_fail)
    long_picks = _weighted_sample_without_replacement(long_items, long_scores, n_long)

    chosen = []
    for k, pi in micro_picks:
        chosen.append(
            {
                "arm_key": k,
                "max_minutes": MICRO_PROBE_MIN,
                "probe_kind": "micro",
                "meta_policy": "fail_fast",
                "sampling_propensity": max(1e-6, float(pi)),
            }
        )
    for k, pi in long_picks:
        chosen.append(
            {
                "arm_key": k,
                "max_minutes": LONG_PROBE_MIN,
                "probe_kind": "long",
                "meta_policy": "long_run",
                "sampling_propensity": max(1e-6, float(pi)),
            }
        )
    return chosen


# ========================
# Metrics ingestion (simple)
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
        _dqL = deque([ts for ts in dqL if now - ts <= window_sec], maxlen=len(dqL))
        _dqF = deque([ts for ts in dqF if now - ts <= window_sec], maxlen=len(dqF))
        launches[k] = _dqL
        launch_failures[k] = _dqF
        tot = len(_dqL)
        fls = len(_dqF)
        fail_rate_by_arm[k] = (fls / tot) if tot > 0 else 0.0

        dqI = interrupts[k]
        _dqI = deque([ts for ts in dqI if now - ts <= window_sec], maxlen=len(dqI))
        interrupts[k] = _dqI
        eviction_rate_by_arm[k] = (len(_dqI) / tot) if tot > 0 else 0.0

        cp_by_arm[k] = 0.0

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
        alpha0=1.0,
        beta0=19.0,
        baseline_budget_per_hour=0,
        burst_size=0,
        shadow_burst=0,
        event_threshold=9999,
        neighbors_fn=neighbors_fn,
    )

    active_threads: List[threading.Thread] = []

    while True:
        t0 = datetime.datetime.utcnow()
        print(f"\n=== META-CAI cycle {t0.isoformat()}Z ===")
        ingest_simple_metrics_into_engine(engine)

        active_threads = [th for th in active_threads if th.is_alive()]
        free_total = nic_pool_status()
        print(f"Active probes still running: {len(active_threads)}")
        print(f"NIC pool status: {free_total}")

        if drain_enabled(args.drain):
            print("[DRAIN] Draining mode is ON — no new launches will be started.")
            if not active_threads:
                print("[DRAIN] All probes finished. Exiting scheduler.")
                return
            cai = engine.cai(H=H, mc_samples=800, risk_high_is_high=True)
            for k, s in cai.items():
                print(f"  {k:45s} CAI{H}={s['CAI']:3d}  risk_mean={s['risk_mean']:.3f}  CI80={s['risk_CI80']}")
            time.sleep(min(300, INTERVAL_MIN * 60))
            continue

        selection = plan(engine)
        print("Planned probes:", selection)

        for sel in selection:
            arm_key = sel["arm_key"]
            max_min = sel["max_minutes"]
            probe_kind = sel["probe_kind"]
            meta_pol = sel["meta_policy"]
            pi = sel["sampling_propensity"]

            arm = ARM_BY_KEY[arm_key]
            pred_h1, pred_risk5 = current_pred_h1_and_risk5(engine, arm_key)

            vm_name, st, price_at_launch, price_delta6, features, nic_id, err = launch_spot_probe(
                arm.region, arm.family
            )
            if vm_name and st and nic_id:
                th = threading.Thread(
                    target=monitor_probe,
                    args=(
                        arm.region,
                        arm.family,
                        vm_name,
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
                        pred_risk5,
                        features,
                        nic_id,
                    ),
                )
                th.daemon = True
                th.start()
                active_threads.append(th)
            else:
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
                    end_time=None,
                    duration_minutes=0.0,
                    interrupted=False,
                    interrupt_bin=None,
                    survived_hours=0,
                    spot_price_usd=price_at_launch,
                    price_delta_6h=price_delta6,
                    sampling_propensity=pi,
                    policy_version=POLICY_VERSION,
                    pred_h1_at_launch=pred_h1,
                    pred_risk_5h_at_launch=pred_risk5,
                    features_snapshot=features,
                )

        cai = engine.cai(H=H, mc_samples=800, risk_high_is_high=True)
        for k, s in cai.items():
            print(f"  {k:45s} CAI{H}={s['CAI']:3d}  risk_mean={s['risk_mean']:.3f}  CI80={s['risk_CI80']}")

        print(f"Sleeping {INTERVAL_MIN} minutes...")
        time.sleep(INTERVAL_MIN * 60)


if __name__ == "__main__":
    run_meta_scheduler()
