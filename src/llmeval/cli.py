"""Command line entry point: ``llm-eval --help``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llmeval.compare import compare_runs
from llmeval.dataset import load_dataset
from llmeval.llm import ChatPipeline, ResponseCache
from llmeval.pipelines import offline_pipelines
from llmeval.retrieval import Index
from llmeval.runner import run_suite


def _run(args) -> int:
    dataset = load_dataset(args.data)
    if args.pipeline == "chat":
        if not args.endpoint or not args.model:
            raise SystemExit("--endpoint and --model are required for the chat pipeline")
        pipeline = ChatPipeline(
            Index(dataset.documents),
            args.endpoint,
            args.model,
            price_per_million=(args.input_price, args.output_price),
            cache=ResponseCache(args.cache),
        )
    else:
        pipeline = offline_pipelines(dataset.documents, dataset.split("dev"))[args.pipeline]
    cases = dataset.split(args.split) if args.split != "all" else dataset.cases
    result = run_suite(pipeline, cases, dataset.sha256, args.workers)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1))
    print(json.dumps(result["summary"], indent=2))
    return 0


def _compare(args) -> int:
    baseline = json.loads(Path(args.baseline).read_text())
    candidate = json.loads(Path(args.candidate).read_text())
    try:
        result = compare_runs(baseline, candidate)
    except ValueError as error:
        raise SystemExit(f"Stopped: {error}") from error
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="llm-eval", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="evaluate a pipeline")
    run.add_argument(
        "--pipeline",
        default="bm25_abstain",
        choices=["overlap", "bm25", "bm25_abstain", "router_agent", "chat"],
    )
    run.add_argument("--data", default="data")
    run.add_argument("--split", default="test", choices=["dev", "test", "all"])
    run.add_argument("--out", default="runs/latest.json")
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--endpoint", help="for example http://localhost:11434")
    run.add_argument("--model")
    run.add_argument("--input-price", type=float, default=0.0, help="USD per million tokens")
    run.add_argument("--output-price", type=float, default=0.0, help="USD per million tokens")
    run.add_argument("--cache", default=".llm_cache")
    compare = commands.add_parser("compare", help="exit 1 if the candidate regressed")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    args = parser.parse_args(argv)
    sys.exit(_run(args) if args.command == "run" else _compare(args))


if __name__ == "__main__":
    main()
