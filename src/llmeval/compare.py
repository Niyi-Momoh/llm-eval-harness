"""Compare two runs case by case and decide whether the candidate may ship."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class GatePolicy:
    alpha: float = 0.05
    min_drop: float = 0.02
    latency_ratio: float = 1.5
    latency_floor_ms: float = 50.0
    cost_ratio: float = 1.1
    metrics: tuple[str, ...] = ("correct", "retrieval_hit", "tool_match")


def paired_pvalue(deltas: np.ndarray, samples: int = 5000, seed: int = 7) -> float:
    """One-sided sign-flip test that the candidate is worse than the baseline.

    Right/wrong scores use the exact binomial form, which matches McNemar's exact test.
    """
    values = np.asarray(deltas, dtype=float)
    nonzero = values[values != 0]
    if len(nonzero) == 0:
        return 1.0
    if np.isin(nonzero, (-1.0, 1.0)).all():
        total = len(nonzero)
        wins = int((nonzero > 0).sum())
        return sum(math.comb(total, count) for count in range(wins + 1)) / 2**total
    observed = nonzero.mean()
    signs = np.random.default_rng(seed).choice((-1.0, 1.0), (samples, len(nonzero)))
    simulated = (signs * nonzero).mean(axis=1)
    return float((1 + (simulated <= observed).sum()) / (samples + 1))


def delta_interval(deltas: np.ndarray, samples: int = 2000, seed: int = 7) -> tuple[float, float]:
    values = np.asarray(deltas, dtype=float)
    draws = np.random.default_rng(seed).integers(0, len(values), (samples, len(values)))
    low, high = np.percentile(values[draws].mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


def is_regression(deltas: np.ndarray, policy: GatePolicy) -> tuple[bool, float]:
    """A drop must be both large enough to matter and statistically credible."""
    p_value = paired_pvalue(deltas)
    return bool(deltas.mean() <= -policy.min_drop and p_value < policy.alpha), p_value


def compare_runs(baseline: dict, candidate: dict, policy: GatePolicy = GatePolicy()) -> dict:
    left, right = baseline["manifest"], candidate["manifest"]
    if left["dataset_sha256"] != right["dataset_sha256"] or left["case_ids"] != right["case_ids"]:
        raise ValueError("runs must use the same dataset and the same cases")
    reasons: list[str] = []
    metrics: dict[str, dict] = {}
    for metric in policy.metrics:
        pairs = [
            (a[metric], b[metric])
            for a, b in zip(baseline["cases"], candidate["cases"], strict=True)
            if a[metric] is not None and b[metric] is not None
        ]
        if not pairs:
            continue
        deltas = np.array([b - a for a, b in pairs], dtype=float)
        regressed, p_value = is_regression(deltas, policy)
        low, high = delta_interval(deltas)
        metrics[metric] = {
            "cases": len(pairs),
            "baseline": float(np.mean([a for a, _ in pairs])),
            "candidate": float(np.mean([b for _, b in pairs])),
            "delta": float(deltas.mean()),
            "delta_95": [low, high],
            "p_value": p_value,
            "regressed": regressed,
        }
        if regressed:
            reasons.append(f"{metric} fell by {-deltas.mean() * 100:.1f} points (p={p_value:.4f})")
    old_p95 = baseline["summary"]["latency_p95_ms"]
    new_p95 = candidate["summary"]["latency_p95_ms"]
    if new_p95 > policy.latency_floor_ms and new_p95 > old_p95 * policy.latency_ratio:
        reasons.append(f"p95 latency rose from {old_p95:.1f} ms to {new_p95:.1f} ms")
    old_cost = baseline["summary"]["total_cost_usd"]
    new_cost = candidate["summary"]["total_cost_usd"]
    if new_cost > old_cost * policy.cost_ratio and new_cost > 0:
        reasons.append(f"cost rose from ${old_cost:.4f} to ${new_cost:.4f}")
    if candidate["summary"]["errors"] > baseline["summary"]["errors"]:
        reasons.append("the candidate had more failed cases")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "metrics": metrics,
        "policy": asdict(policy),
    }
