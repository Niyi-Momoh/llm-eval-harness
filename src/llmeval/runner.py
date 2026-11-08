"""Run a pipeline over cases and record reproducible results."""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from llmeval.dataset import Case
from llmeval.pipelines import Pipeline
from llmeval.scoring import score_case

MEAN_METRICS = (
    "correct",
    "retrieval_hit",
    "reciprocal_rank",
    "token_f1",
    "faithfulness",
    "tool_match",
)


def _evaluate(pipeline: Pipeline, case: Case) -> dict:
    started = time.perf_counter()
    try:
        response = pipeline.run(case.question)
    except Exception as error:  # noqa: BLE001 - one failed case must not end the run
        return {
            "id": case.id,
            "category": case.category,
            "answer": "",
            "error": f"{type(error).__name__}: {error}",
            "correct": 0.0,
            "retrieval_hit": 0.0 if case.docs else None,
            "reciprocal_rank": 0.0 if case.docs else None,
            "token_f1": 0.0 if case.answerable else None,
            "faithfulness": None,
            "tool_match": 0.0 if case.tools else None,
            "abstained": False,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "cost_usd": 0.0,
        }
    latency = (time.perf_counter() - started) * 1000
    return {
        "id": case.id,
        "category": case.category,
        "answer": response.answer,
        "error": None,
        **score_case(case, response),
        "latency_ms": latency,
        "cost_usd": response.cost_usd,
    }


def interval(values: Sequence[float], samples: int = 2000, seed: int = 7) -> tuple[float, float]:
    """95% bootstrap confidence interval for a mean."""
    data = np.asarray(values, dtype=float)
    draws = np.random.default_rng(seed).integers(0, len(data), (samples, len(data)))
    means = data[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def summarise(cases: Sequence[dict]) -> dict:
    summary: dict = {"cases": len(cases), "errors": sum(bool(c["error"]) for c in cases)}
    for metric in MEAN_METRICS:
        values = [case[metric] for case in cases if case[metric] is not None]
        summary[metric] = float(np.mean(values)) if values else None
    low, high = interval([case["correct"] for case in cases])
    summary["correct_95"] = [low, high]
    latencies = [case["latency_ms"] for case in cases]
    summary["latency_p50_ms"] = float(np.percentile(latencies, 50))
    summary["latency_p95_ms"] = float(np.percentile(latencies, 95))
    summary["total_cost_usd"] = float(sum(case["cost_usd"] for case in cases))
    categories: dict[str, list[float]] = {}
    for case in cases:
        categories.setdefault(case["category"], []).append(case["correct"])
    summary["by_category"] = {name: float(np.mean(v)) for name, v in sorted(categories.items())}
    return summary


def run_suite(
    pipeline: Pipeline, cases: Sequence[Case], dataset_sha256: str, workers: int = 1
) -> dict:
    """Run every case. Output order always matches the input order."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda case: _evaluate(pipeline, case), cases))
    return {
        "manifest": {
            "pipeline": pipeline.name,
            "config": pipeline.config(),
            "dataset_sha256": dataset_sha256,
            "case_ids": [case.id for case in cases],
        },
        "summary": summarise(results),
        "cases": results,
    }
