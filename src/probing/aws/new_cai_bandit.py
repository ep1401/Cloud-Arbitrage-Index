"""
Cloud Arbitrage Index (CAI) — Bandit + Event-Burst Probing Framework
(Ensemble + 1h Probes + Hedge weights + Option B persistence-friendly)

This version is the same logic you posted, with ONLY the changes needed so your
Option B scheduler persistence works cleanly:

Key updates (persistence compatibility):
  1) Adds public aliases expected by the scheduler snapshot code:
       - hedge_w    (normalized per-arm weights)  == hedge_weights
       - hedge_loss (per-arm Hedge log-weights)   == _hedge_logw
     So your scheduler can save/load without guessing attribute names.
  2) Adds a safe restore hook you can call after you load a snapshot:
       - apply_persisted_state(...)
     (Your scheduler already assigns attributes directly, but this makes it robust.)
  3) Keeps behavior identical for sampling, updating, and reporting.

Minimal dep: numpy
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Optional, Deque, Any
import math
import time
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
    # These fields are kept for backward compatibility, but for 1h probes we only use interrupted + launch_ts.
    survived_hours: int = 0
    interrupted: bool = False
    interrupt_bin: Optional[int] = None
    # launch timestamp (seconds since epoch, UTC). Needed for recent/TOD models and for weighting.
    launch_ts: Optional[float] = None


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
        return sum(v) / len(v) if v else 0.0

    def std(self) -> float:
        v = self.last()
        if len(v) < 2:
            return 0.0
        m = self.mean()
        var = sum((xi - m) ** 2 for xi in v) / (len(v) - 1)
        return math.sqrt(var)

    def zscore_latest(self) -> float:
        v = self.last()
        if len(v) < 2:
            return 0.0
        m, s = self.mean(), self.std()
        if s == 0:
            return 0.0
        return (v[-1] - m) / s


# ------------------------------
# Core CAI Bandit Engine (Ensemble + Hedge)
# ------------------------------

class CAIBandit:
    """
    Ensemble CAI engine for 1-hour probes.

    We model hour-1 hazard:
        h1 = P(interrupted within 1 hour)

    Experts:
      - GLOBAL posterior: global_posterior[arm_key][1] = (a,b)
      - RECENT posterior: recent_posterior[arm_key][1] = (a,b) rebuilt from last recent_window_sec
      - TOD posterior:    tod_posterior[block][arm_key][1] = (a,b)

    Per-arm weights (Hedge):
      - hedge_w[arm_key] (normalized weights): {"recent": wR, "tod": wT, "global": wG}
      - hedge_loss[arm_key] (log-weights accumulator): {"recent": logwR, ...}
        (This is what we persist/restore for exact continuity.)
    """

    def __init__(
        self,
        arms: List[Arm],
        H: int = 1,
        alpha0: float = 1.0,
        beta0: float = 19.0,
        baseline_budget_per_hour: int = 0,
        burst_size: int = 4,
        shadow_burst: int = 1,
        burst_hours: int = 3,
        event_threshold: float = 2.0,
        neighbors_fn: Optional[Callable[[Arm], List[Arm]]] = None,
        # Recent window length
        recent_window_hours: int = 6,
        # TOD block size
        tod_block_hours: int = 6,
        # TOD uses UTC hour
        tod_use_utc: bool = True,
        # Initial weights (prior mix)
        w_recent: float = 0.35,
        w_tod: float = 0.25,
        w_global: float = 0.40,
        # Hedge learning knobs
        hedge_eta: float = 0.06,
        hedge_decay: float = 0.995,
        weight_floor: float = 0.05,
    ):
        self.arms = arms
        self.H = max(1, int(H))
        self.alpha0, self.beta0 = float(alpha0), float(beta0)

        self.budget = baseline_budget_per_hour or max(1, len(arms))
        self.burst_size = int(burst_size)
        self.shadow_burst = int(shadow_burst)
        self.burst_hours = int(burst_hours)
        self.event_threshold = float(event_threshold)
        self.neighbors_fn = neighbors_fn or (lambda arm: [])

        self.recent_window_sec = int(max(1, recent_window_hours) * 3600)
        self.tod_block_hours = int(max(1, tod_block_hours))
        self.tod_use_utc = bool(tod_use_utc)

        self.hedge_eta = float(hedge_eta)
        self.hedge_decay = float(hedge_decay)
        self.weight_floor = float(weight_floor)

        # Normalize initial weights
        w_recent, w_tod, w_global = self._normalize_weights(w_recent, w_tod, w_global)
        self._init_weights = {"recent": w_recent, "tod": w_tod, "global": w_global}

        # --------------------------
        # Experts: posteriors for h1
        # --------------------------
        self.global_posterior: Dict[str, Dict[int, Tuple[float, float]]] = {
            a.key(): {1: (self.alpha0, self.beta0)} for a in self.arms
        }

        # RECENT window storage: per arm deque of (launch_ts, interrupted_bool)
        self._recent_events: Dict[str, Deque[Tuple[float, bool]]] = {a.key(): deque() for a in self.arms}
        self.recent_posterior: Dict[str, Dict[int, Tuple[float, float]]] = {
            a.key(): {1: (self.alpha0, self.beta0)} for a in self.arms
        }

        # TOD posterior: blocks -> arm_key -> bin1 posterior
        self._num_tod_blocks = max(1, int(math.ceil(24 / self.tod_block_hours)))
        self.tod_posterior: Dict[int, Dict[str, Dict[int, Tuple[float, float]]]] = {
            b: {a.key(): {1: (self.alpha0, self.beta0)} for a in self.arms}
            for b in range(self._num_tod_blocks)
        }

        # --------------------------
        # Per-arm Hedge state
        # --------------------------
        self._hedge_logw: Dict[str, Dict[str, float]] = {
            a.key(): {
                "recent": math.log(self._init_weights["recent"] + 1e-12),
                "tod": math.log(self._init_weights["tod"] + 1e-12),
                "global": math.log(self._init_weights["global"] + 1e-12),
            }
            for a in self.arms
        }
        self.hedge_weights: Dict[str, Dict[str, float]] = {a.key(): dict(self._init_weights) for a in self.arms}

        # >>> Persistence-friendly public aliases expected by the scheduler <<<
        # These names make Option B snapshotting/loading trivial.
        self.hedge_w = self.hedge_weights          # normalized weights
        self.hedge_loss = self._hedge_logw         # log-weights accumulator

        # --------------------------
        # Metric buffers per arm (unchanged)
        # --------------------------
        self.price_delta: Dict[str, RollingStats] = {a.key(): RollingStats(window=12) for a in self.arms}
        self.launch_fail_rate: Dict[str, RollingStats] = {a.key(): RollingStats(window=12) for a in self.arms}
        self.eviction_rate: Dict[str, RollingStats] = {a.key(): RollingStats(window=12) for a in self.arms}
        self.change_point: Dict[str, float] = {a.key(): 0.0 for a in self.arms}

        # Active bursts: arm_key -> hours remaining
        self.active_bursts: Dict[str, int] = defaultdict(int)

    # ------------------------------
    # Optional helper for robust restore
    # ------------------------------
    def apply_persisted_state(
        self,
        global_posterior: Optional[dict] = None,
        tod_posterior: Optional[dict] = None,
        hedge_w: Optional[dict] = None,
        hedge_loss: Optional[dict] = None,
    ) -> None:
        """
        If you load a snapshot and want a single hook to restore it, call this.
        Your scheduler already sets attributes directly; this just keeps invariants synced.
        """
        if isinstance(global_posterior, dict):
            self.global_posterior = global_posterior
        if isinstance(tod_posterior, dict):
            self.tod_posterior = tod_posterior
        if isinstance(hedge_w, dict):
            self.hedge_weights = hedge_w
            self.hedge_w = self.hedge_weights
        if isinstance(hedge_loss, dict):
            self._hedge_logw = hedge_loss
            self.hedge_loss = self._hedge_logw

    # ------------------------------
    # Public API: metrics
    # ------------------------------
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
            if k in price_delta_by_arm:
                self.price_delta[k].push(price_delta_by_arm[k])
            if k in launch_fail_rate_by_arm:
                self.launch_fail_rate[k].push(launch_fail_rate_by_arm[k])
            if k in eviction_rate_by_arm:
                self.eviction_rate[k].push(eviction_rate_by_arm[k])
            if change_point_by_arm and k in change_point_by_arm:
                self.change_point[k] = float(change_point_by_arm[k])

    # ------------------------------
    # Updating posteriors + weights
    # ------------------------------
    def update_from_probe_results(self, probe_results: List[ProbeResult], now_ts: Optional[float] = None):
        """
        Preferred update entrypoint for the scheduler.

        For each ProbeResult:
          1) Compute expert predictive probabilities p_e BEFORE updating posteriors.
          2) Hedge update weights with log-loss using outcome y in {0,1}.
          3) Update GLOBAL + TOD posteriors with the new observation.
          4) Append to RECENT event queue.
        Then:
          5) Rebuild RECENT posterior for all arms (prune+recount).
        """
        now = float(now_ts if now_ts is not None else time.time())

        for pr in probe_results:
            k = pr.arm.key()
            launch_ts = float(pr.launch_ts if pr.launch_ts is not None else now)
            y = 1 if pr.interrupted else 0

            # Expert predictions BEFORE updating posteriors
            p_recent = self._mean_h1_from_beta(self.recent_posterior[k][1])
            block = self._tod_block(launch_ts)
            p_tod = self._mean_h1_from_beta(self.tod_posterior[block][k][1])
            p_global = self._mean_h1_from_beta(self.global_posterior[k][1])

            # Hedge update
            self._hedge_update_one(k, y, p_recent, p_tod, p_global)

            # Update GLOBAL posterior
            a, b = self.global_posterior[k][1]
            self.global_posterior[k][1] = (a + y, b + (1 - y))

            # Update TOD posterior
            at, bt = self.tod_posterior[block][k][1]
            self.tod_posterior[block][k][1] = (at + y, bt + (1 - y))

            # Store recent event
            self._recent_events[k].append((launch_ts, bool(pr.interrupted)))

        # Rebuild RECENT posteriors
        self._rebuild_recent(now)

    def update_posteriors(self, probe_results: List[ProbeResult], now_ts: Optional[float] = None):
        """Backward-compatible alias."""
        self.update_from_probe_results(probe_results, now_ts=now_ts)

    # ------------------------------
    # Planning probes (optional baseline allocator)
    # ------------------------------
    def plan_probes(self, mc_samples: int = 400, now_ts: Optional[float] = None) -> Dict[str, int]:
        now = float(now_ts if now_ts is not None else time.time())

        ci_width: Dict[str, float] = {}
        for a in self.arms:
            k = a.key()
            h1_samples = self._sample_h1_mixture(k, mc_samples=mc_samples, now_ts=now)
            lo, hi = np.percentile(h1_samples, 10), np.percentile(h1_samples, 90)
            ci_width[k] = float(hi - lo)

        total_unc = sum(ci_width.values())
        plan: Dict[str, int] = {k: 0 for k in ci_width}
        if total_unc == 0:
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

        # Event bursts
        for a in self.arms:
            k = a.key()
            score = self._event_score(k)
            if score >= self.event_threshold:
                self.active_bursts[k] = max(self.active_bursts[k], self.burst_hours)

        for k, ttl in list(self.active_bursts.items()):
            if ttl > 0:
                plan[k] = plan.get(k, 0) + self.burst_size
                arm = self._arm_from_key(k)
                for nb in self.neighbors_fn(arm):
                    plan[nb.key()] = plan.get(nb.key(), 0) + self.shadow_burst
                self.active_bursts[k] = ttl - 1
            else:
                del self.active_bursts[k]

        return plan

    # ------------------------------
    # Introspection / Reporting
    # ------------------------------
    def cai(self, launch_ts: Optional[float] = None, mc_samples: int = 1000) -> Dict[str, Dict[str, Any]]:
        now = float(launch_ts if launch_ts is not None else time.time())
        out: Dict[str, Dict[str, Any]] = {}

        for a in self.arms:
            k = a.key()
            w = self.hedge_w.get(k, self._init_weights)  # use persistence-friendly alias
            h1_samples = self._sample_h1_mixture(
                k,
                mc_samples=mc_samples,
                now_ts=now,
                weights=(w["recent"], w["tod"], w["global"]),
            )
            mean = float(np.mean(h1_samples))
            lo, hi = np.percentile(h1_samples, 10), np.percentile(h1_samples, 90)
            idx = int(round(100 * mean))

            out[k] = {
                "CAI1": idx,
                "h1_mean": mean,
                "h1_CI80": (float(lo), float(hi)),
                "weights": {"recent": float(w["recent"]), "tod": float(w["tod"]), "global": float(w["global"])},
                "tod_block": self._tod_block(now),
            }

        return out

    # ------------------------------
    # Internals
    # ------------------------------
    def _normalize_weights(self, w_recent: float, w_tod: float, w_global: float) -> Tuple[float, float, float]:
        wr, wt, wg = float(w_recent), float(w_tod), float(w_global)
        wr = max(0.0, wr); wt = max(0.0, wt); wg = max(0.0, wg)
        s = wr + wt + wg
        if s <= 0:
            return 0.0, 0.0, 1.0
        return wr / s, wt / s, wg / s

    def _tod_block(self, ts: float) -> int:
        if self.tod_use_utc:
            hour = int(time.gmtime(ts).tm_hour)
        else:
            hour = int(time.localtime(ts).tm_hour)
        block = int(hour // self.tod_block_hours)
        return int(block % self._num_tod_blocks)

    def _rebuild_recent(self, now_ts: float):
        cutoff = now_ts - self.recent_window_sec
        for a in self.arms:
            k = a.key()
            dq = self._recent_events[k]
            while dq and dq[0][0] < cutoff:
                dq.popleft()
            y = sum(1 for _, interrupted in dq if interrupted)
            n = len(dq)
            self.recent_posterior[k][1] = (self.alpha0 + y, self.beta0 + (n - y))

    def _arm_from_key(self, k: str) -> Arm:
        for a in self.arms:
            if a.key() == k:
                return a
        raise KeyError(k)

    def _event_score(self, k: str) -> float:
        z_price = self.price_delta[k].zscore_latest()
        fail = (self.launch_fail_rate[k].last(1)[-1] if self.launch_fail_rate[k].last(1) else 0.0) * 10.0
        evic = (self.eviction_rate[k].last(1)[-1] if self.eviction_rate[k].last(1) else 0.0) * 10.0
        cp = self.change_point[k]
        return max(z_price, fail, evic, cp)

    @staticmethod
    def _mean_h1_from_beta(ab: Tuple[float, float]) -> float:
        a, b = float(ab[0]), float(ab[1])
        denom = a + b
        return float(a / denom) if denom > 0 else 0.5

    @staticmethod
    def _logloss(y: int, p: float, eps: float = 1e-9) -> float:
        p = min(1.0 - eps, max(eps, float(p)))
        return -math.log(p) if y == 1 else -math.log(1.0 - p)

    def _hedge_update_one(self, arm_key: str, y: int, p_recent: float, p_tod: float, p_global: float):
        eta = self.hedge_eta
        decay = self.hedge_decay

        losses = {
            "recent": self._logloss(y, p_recent),
            "tod": self._logloss(y, p_tod),
            "global": self._logloss(y, p_global),
        }

        # Update log-weights
        lw = self._hedge_logw[arm_key]
        for e, loss in losses.items():
            lw[e] = (decay * lw[e]) - (eta * float(loss))

        # Convert to positive weights via softmax over log-weights
        m = max(lw.values())
        raw = {e: math.exp(lw[e] - m) for e in lw}

        # Floor so experts never die
        floor = max(0.0, min(0.49, float(self.weight_floor)))
        raw = {e: (raw[e] + floor) for e in raw}

        s = sum(raw.values())
        if s <= 0:
            self.hedge_weights[arm_key] = dict(self._init_weights)
            self.hedge_w[arm_key] = dict(self._init_weights)  # keep alias in sync
            return

        w = {e: raw[e] / s for e in raw}

        neww = {"recent": w["recent"], "tod": w["tod"], "global": w["global"]}
        self.hedge_weights[arm_key] = neww
        self.hedge_w[arm_key] = neww          # keep alias in sync
        self.hedge_loss[arm_key] = lw         # keep alias in sync (same object, but explicit)

    def _sample_h1_mixture(
        self,
        arm_key: str,
        mc_samples: int = 400,
        now_ts: Optional[float] = None,
        weights: Optional[Tuple[float, float, float]] = None,  # (w_recent,w_tod,w_global)
    ) -> np.ndarray:
        now = float(now_ts if now_ts is not None else time.time())

        if weights is None:
            w = self.hedge_w.get(arm_key, self._init_weights)
            wr, wt, wg = w["recent"], w["tod"], w["global"]
        else:
            wr, wt, wg = weights
        wr, wt, wg = self._normalize_weights(wr, wt, wg)

        choices = np.random.choice(3, size=mc_samples, p=[wr, wt, wg])
        out = np.empty(mc_samples, dtype=float)

        a_r, b_r = self.recent_posterior[arm_key][1]
        block = self._tod_block(now)
        a_t, b_t = self.tod_posterior[block][arm_key][1]
        a_g, b_g = self.global_posterior[arm_key][1]

        idx_r = np.where(choices == 0)[0]
        if idx_r.size:
            out[idx_r] = np.random.beta(a_r, b_r, size=idx_r.size)

        idx_t = np.where(choices == 1)[0]
        if idx_t.size:
            out[idx_t] = np.random.beta(a_t, b_t, size=idx_t.size)

        idx_g = np.where(choices == 2)[0]
        if idx_g.size:
            out[idx_g] = np.random.beta(a_g, b_g, size=idx_g.size)

        return out
