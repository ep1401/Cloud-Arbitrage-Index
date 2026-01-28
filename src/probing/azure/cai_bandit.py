"""
Cloud Arbitrage Index (CAI) — Bandit + Event-Burst Probing Framework
--------------------------------------------------------------------
Implements:
  - Arms = provider × region × instance family
  - Hourly baseline probes allocated by uncertainty (CI width of S(H))
  - Event detection (price jumps, launch failures, eviction rate, simple change-point)
  - Burst allocation on arms with active events (+ optional neighbor shadowing)
  - Online discrete-time hazard model with Beta-Binomial posteriors
  - CAI(H) computation and uncertainty intervals

Minimal dep: numpy
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Optional
import math
import numpy as np
from collections import deque, defaultdict

# ------------------------------
# Data structures
# ------------------------------

@dataclass(frozen=True)
class Arm:
    provider: str   # "aws" or "azure"
    region: str     # e.g., "us-east-1"
    family: str     # e.g., "m7g", "Standard_D4s_v5"
    def key(self) -> str:
        return f"{self.provider}:{self.region}:{self.family}"

@dataclass
class ProbeResult:
    arm: Arm
    survived_hours: int
    interrupted: bool
    interrupt_bin: Optional[int]  # hour bin (1..H) if interrupted, else None

# ------------------------------
# Rolling metric buffers
# ------------------------------

@dataclass
class RollingStats:
    """Maintain recent values and simple statistics for an arm's metric."""
    window: int
    values: deque = field(default_factory=deque)

    def push(self, x: float):
        self.values.append(float(x))
        while len(self.values) > self.window:
            self.values.popleft()

    def last(self, k: Optional[int] = None) -> List[float]:
        if not k or k >= len(self.values):
            return list(self.values)
        return list(self.values)[-k:]

    def mean(self) -> float:
        v = self.last()
        return sum(v)/len(v) if v else 0.0

    def std(self) -> float:
        v = self.last()
        if len(v) < 2: return 0.0
        m = self.mean()
        var = sum((xi - m)**2 for xi in v) / (len(v)-1)
        return math.sqrt(var)

    def zscore_latest(self) -> float:
        v = self.last()
        if len(v) < 2: return 0.0
        m, s = self.mean(), self.std()
        if s == 0: return 0.0
        return (v[-1] - m) / s

# ------------------------------
# Core CAI Bandit Engine
# ------------------------------

class CAIBandit:
    def __init__(
        self,
        arms: List[Arm],
        H: int = 5,
        alpha0: float = 1.0,
        beta0: float = 9.0,
        baseline_budget_per_hour: int = 0,
        burst_size: int = 4,
        shadow_burst: int = 1,
        burst_hours: int = 3,
        event_threshold: float = 2.0,  # z-threshold proxy
        neighbors_fn: Optional[Callable[[Arm], List[Arm]]] = None,
    ):
        self.arms = arms
        self.H = H
        self.alpha0, self.beta0 = alpha0, beta0
        self.budget = baseline_budget_per_hour or max(1, len(arms))  # ~1 per arm by default
        self.burst_size = burst_size
        self.shadow_burst = shadow_burst
        self.burst_hours = burst_hours
        self.event_threshold = event_threshold
        self.neighbors_fn = neighbors_fn or (lambda arm: [])

        # Posteriors: dict[arm_key][t] = (alpha, beta)
        self.posterior: Dict[str, Dict[int, Tuple[float, float]]] = {
            a.key(): {t: (self.alpha0, self.beta0) for t in range(1, H+1)}
            for a in self.arms
        }

        # Metric buffers per arm
        self.price_delta: Dict[str, RollingStats] = {a.key(): RollingStats(window=12) for a in self.arms}
        self.launch_fail_rate: Dict[str, RollingStats] = {a.key(): RollingStats(window=12) for a in self.arms}
        self.eviction_rate: Dict[str, RollingStats] = {a.key(): RollingStats(window=12) for a in self.arms}
        self.change_point: Dict[str, float] = {a.key(): 0.0 for a in self.arms}  # 0/1 proxy

        # Active bursts: arm_key -> hours remaining
        self.active_bursts: Dict[str, int] = defaultdict(int)

    # --------- Public API hooks you should call each hour ---------

    def ingest_metrics(
        self,
        price_delta_by_arm: Dict[str, float],
        launch_fail_rate_by_arm: Dict[str, float],
        eviction_rate_by_arm: Dict[str, float],
        change_point_by_arm: Optional[Dict[str, float]] = None,
    ):
        """Push latest metrics (one per hour) into rolling buffers."""
        for arm in self.arms:
            k = arm.key()
            if k in price_delta_by_arm: self.price_delta[k].push(price_delta_by_arm[k])
            if k in launch_fail_rate_by_arm: self.launch_fail_rate[k].push(launch_fail_rate_by_arm[k])
            if k in eviction_rate_by_arm: self.eviction_rate[k].push(eviction_rate_by_arm[k])
            if change_point_by_arm and k in change_point_by_arm:
                self.change_point[k] = float(change_point_by_arm[k])

    def plan_probes(self, mc_samples: int = 400) -> Dict[str, int]:
        """Compute allocation: baseline by CI width + event-based bursts."""
        # 1) CI width per arm for S(H)
        ci_width = {}
        for a in self.arms:
            k = a.key()
            S_samples = self._sample_S(k, mc_samples=mc_samples)
            lo, hi = np.percentile(S_samples, 10), np.percentile(S_samples, 90)
            ci_width[k] = float(hi - lo)  # uncertainty proxy

        # 2) Baseline proportional allocation
        total_unc = sum(ci_width.values())
        plan: Dict[str, int] = {k: 0 for k in ci_width}
        if total_unc == 0:
            # even allocation fallback
            keys = list(plan.keys())
            for i in range(min(self.budget, len(keys))):
                plan[keys[i]] += 1
        else:
            shares = {k: (ci_width[k] / total_unc) * self.budget for k in ci_width}
            assigned = 0
            for k, s in shares.items():
                n = int(math.floor(s))
                plan[k] += n
                assigned += n
            residual = self.budget - assigned
            if residual > 0:
                fracs = sorted([(k, shares[k] - math.floor(shares[k])) for k in shares], key=lambda x: -x[1])
                for i in range(residual):
                    plan[fracs[i][0]] += 1

        # 3) Event bursts
        for a in self.arms:
            k = a.key()
            score = self._event_score(k)
            if score >= self.event_threshold:
                self.active_bursts[k] = max(self.active_bursts[k], self.burst_hours)

        # Decrement active bursts and apply allocation
        for k, ttl in list(self.active_bursts.items()):
            if ttl > 0:
                plan[k] = plan.get(k, 0) + self.burst_size
                # neighbor shadowing
                arm = self._arm_from_key(k)
                for nb in self.neighbors_fn(arm):
                    plan[nb.key()] = plan.get(nb.key(), 0) + self.shadow_burst
                self.active_bursts[k] = ttl - 1
            else:
                del self.active_bursts[k]

        return plan  # {arm_key: num_probes}

    def update_posteriors(self, probe_results: List[ProbeResult]):
        """Tally per arm/bin and do Beta posterior updates."""
        # tallies[arm_key][t] = {'y': interrupts_in_bin_t, 'n': at_risk_in_bin_t}
        tallies: Dict[str, Dict[int, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {'y':0,'n':0}))

        for pr in probe_results:
            k = pr.arm.key()
            T = min(pr.survived_hours, self.H)
            for t in range(1, T+1):
                tallies[k][t]['n'] += 1
            if pr.interrupted and pr.interrupt_bin is not None and 1 <= pr.interrupt_bin <= self.H:
                tallies[k][pr.interrupt_bin]['y'] += 1

        # Posterior update
        for k in tallies:
            for t in range(1, self.H+1):
                y = tallies[k][t]['y']
                n = tallies[k][t]['n']
                a, b = self.posterior[k][t]
                self.posterior[k][t] = (a + y, b + n - y if (b + n - y) > 0 else b)

    # --------- Introspection / Reporting ---------

    def survival_summary(self, H: Optional[int] = None, mc_samples: int = 1000):
        """Return dict of arm_key -> (S_mean, (lo, hi)) for horizon H (default = self.H)."""
        horizon = H or self.H
        out = {}
        for a in self.arms:
            k = a.key()
            S_samples = self._sample_S(k, horizon, mc_samples)
            mean = float(np.mean(S_samples))
            lo, hi = np.percentile(S_samples, 10), np.percentile(S_samples, 90)
            out[k] = (mean, (float(lo), float(hi)))
        return out

    def cai(self, H: Optional[int] = None, mc_samples: int = 1000, risk_high_is_high: bool = True):
        """Compute CAI(H) in [0, 100], plus CI on risk = 1 - S(H)."""
        horizon = H or self.H
        out = {}
        for a in self.arms:
            k = a.key()
            S_samples = self._sample_S(k, horizon, mc_samples)
            risk_samples = 1.0 - np.array(S_samples)
            r_mean = float(np.mean(risk_samples))
            lo, hi = np.percentile(risk_samples, 10), np.percentile(risk_samples, 90)
            idx = int(round(100 * (r_mean if risk_high_is_high else (1 - r_mean))))
            out[k] = {
                "CAI": idx,
                "risk_mean": r_mean,
                "risk_CI80": (float(lo), float(hi))
            }
        return out

    # ------------------------------
    # Internals
    # ------------------------------

    def _arm_from_key(self, k: str) -> Arm:
        for a in self.arms:
            if a.key() == k: return a
        raise KeyError(k)

    def _event_score(self, k: str) -> float:
        # Compose a simple max-of-signals "score"
        z_price = self.price_delta[k].zscore_latest()
        # scale rates roughly into z-like range by multiplying by 10
        fail = (self.launch_fail_rate[k].last(1)[-1] if self.launch_fail_rate[k].last(1) else 0.0) * 10.0
        evic = (self.eviction_rate[k].last(1)[-1] if self.eviction_rate[k].last(1) else 0.0) * 10.0
        cp = self.change_point[k]  # 0/1
        return max(z_price, fail, evic, cp)

    def _sample_S(self, arm_key: str, horizon: Optional[int] = None, mc_samples: int = 400) -> np.ndarray:
        H = horizon or self.H
        S = np.ones(mc_samples, dtype=float)
        for t in range(1, H+1):
            a, b = self.posterior[arm_key][t]
            h = np.random.beta(a, b, size=mc_samples)  # hourly hazard samples
            S *= (1.0 - h)
        return S
