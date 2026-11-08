"""Run the offline benchmark and the regression-gate power study.

Fixed before any result was seen:
  * The abstention threshold is tuned on development cases only.
  * Pipelines are compared on the test cases.
  * Gate targets: false alarms at most 5% at every suite size, and at least 80% detection
    of a 20-point accuracy drop with 100 cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from llmeval.compare import GatePolicy, compare_runs, is_regression
from llmeval.dataset import load_dataset
from llmeval.pipelines import offline_pipelines
from llmeval.runner import run_suite

SIZES = (25, 50, 100, 200, 400)
DROPS = (0.0, 0.05, 0.10, 0.20, 0.30)
SIMULATIONS = 2000
NOISE = 0.10


def simulate(base: np.ndarray, size: int, drop: float, rng: np.random.Generator) -> np.ndarray:
    """Return simulated score changes for a suite with a known true accuracy drop.

    A share of cases change in each direction regardless, because real model updates
    alter some answers without being better or worse overall.
    """
    baseline = rng.choice(base, size)
    deltas = np.zeros(size)
    right, wrong = np.flatnonzero(baseline == 1), np.flatnonzero(baseline == 0)
    changed = int(round(NOISE * size))
    ups = min(int(rng.binomial(changed, 0.5)), len(wrong), len(right))
    downs = min(changed - ups, len(right), len(wrong))
    if ups:
        deltas[rng.choice(wrong, ups, replace=False)] = 1
    if downs:
        deltas[rng.choice(right, downs, replace=False)] = -1
    remaining = np.flatnonzero((baseline == 1) & (deltas == 0))
    losses = rng.binomial(len(remaining), min(1.0, drop * size / max(len(remaining), 1)))
    if losses:
        deltas[rng.choice(remaining, losses, replace=False)] = -1
    return deltas


def power_study(base: np.ndarray) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(11)
    policy = GatePolicy()
    table: dict[str, dict[str, float]] = {}
    for size in SIZES:
        table[str(size)] = {}
        for drop in DROPS:
            flagged = sum(
                is_regression(simulate(base, size, drop, rng), policy)[0]
                for _ in range(SIMULATIONS)
            )
            table[str(size)][str(drop)] = flagged / SIMULATIONS
    return table


def main() -> None:
    dataset = load_dataset("data")
    pipelines = offline_pipelines(dataset.documents, dataset.split("dev"))
    test = dataset.split("test")
    runs = {name: run_suite(p, test, dataset.sha256) for name, p in pipelines.items()}
    comparisons = {
        name: compare_runs(runs["router_agent"], runs[name])
        for name in runs
        if name != "router_agent"
    }
    everything = run_suite(pipelines["router_agent"], dataset.cases, dataset.sha256)
    base = np.array([case["correct"] for case in everything["cases"]])
    output = {
        "dataset_sha256": dataset.sha256,
        "documents": len(dataset.documents),
        "cases": {"dev": len(dataset.split("dev")), "test": len(test)},
        "threshold": pipelines["bm25_abstain"].threshold,
        "summaries": {name: run["summary"] for name, run in runs.items()},
        "comparisons_against_router_agent": comparisons,
        "power": {
            "simulations": SIMULATIONS,
            "noise_share": NOISE,
            "base_accuracy": float(base.mean()),
            "regression_flag_rate": power_study(base),
        },
        "wrong_cases": {
            name: [case["id"] for case in run["cases"] if not case["correct"]]
            for name, run in runs.items()
        },
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/benchmark.json").write_text(json.dumps(output, indent=1))
    print(json.dumps({k: output[k] for k in ("threshold", "power", "wrong_cases")}, indent=1))
    for name, summary in output["summaries"].items():
        print(
            name,
            {
                k: summary[k]
                for k in (
                    "correct",
                    "correct_95",
                    "retrieval_hit",
                    "reciprocal_rank",
                    "tool_match",
                    "by_category",
                )
            },
        )
    for name, comparison in comparisons.items():
        print(name, comparison["passed"], comparison["reasons"])


if __name__ == "__main__":
    main()
