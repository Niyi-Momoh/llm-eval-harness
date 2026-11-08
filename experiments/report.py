"""Write the README results block from results/benchmark.json."""

from __future__ import annotations

import json
import re
from pathlib import Path

START = "<!-- results:start -->"
END = "<!-- results:end -->"
LABELS = {
    "overlap": "Keyword overlap",
    "bm25": "BM25",
    "bm25_abstain": "BM25 with abstention",
    "router_agent": "Router agent (BM25, abstention, calculator)",
}
CATEGORIES = ("direct", "paraphrase", "unanswerable", "tool")


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def render(study: dict) -> str:
    cases = study["cases"]
    lines = [
        f"Measured on {cases['test']} held-out cases over {study['documents']} documents. "
        f"The abstention threshold ({study['threshold']:.2f}) was tuned on {cases['dev']} "
        "separate development cases.",
        "",
        "| Pipeline | Correct (95% interval) | Direct | Paraphrase | Unanswerable | Tool use "
        "| Retrieval hit@3 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in LABELS.items():
        item = study["summaries"][name]
        low, high = item["correct_95"]
        groups = item["by_category"]
        lines.append(
            f"| {label} | {percent(item['correct'])} ({percent(low)} to {percent(high)}) | "
            + " | ".join(percent(groups.get(category)) for category in CATEGORIES)
            + f" | {percent(item['retrieval_hit'])} | {item['reciprocal_rank']:.2f} |"
        )
    lines += [
        "",
        "**Regression gate, using the router agent as the baseline**",
        "",
        "| Candidate | Accuracy change | p-value | Decision |",
        "|---|---:|---:|---|",
    ]
    for name, result in study["comparisons_against_router_agent"].items():
        metric = result["metrics"]["correct"]
        decision = "Allowed" if result["passed"] else "Blocked"
        lines.append(
            f"| {LABELS[name]} | {metric['delta'] * 100:+.1f} points | "
            f"{metric['p_value']:.4f} | {decision} |"
        )
    power = study["power"]
    table = power["regression_flag_rate"]
    drops = list(next(iter(table.values())))
    heading = " | ".join(
        "No real change" if float(d) == 0 else f"{float(d) * 100:.0f}-point drop" for d in drops
    )
    lines += [
        "",
        f"**How often the gate blocks a change ({power['simulations']:,} simulations per cell)**",
        "",
        f"| Cases in suite | {heading} |",
        "|---:|" + "---:|" * len(drops),
    ]
    for size, row in table.items():
        lines.append(f"| {size} | " + " | ".join(f"{row[d] * 100:.1f}%" for d in drops) + " |")
    worst = max(row["0.0"] for row in table.values())
    detection = table["100"]["0.2"]
    verdict = lambda ok: "Pass" if ok else "Fail"  # noqa: E731
    lines += [
        "",
        "| Target, set before the study | Measured | Result |",
        "|---|---:|---|",
        f"| False alarms at most 5% at every suite size | {worst * 100:.1f}% at worst | "
        f"{verdict(worst <= 0.05)} |",
        f"| Detect a 20-point drop at least 80% of the time with 100 cases | "
        f"{detection * 100:.1f}% | {verdict(detection >= 0.8)} |",
    ]
    return "\n".join(lines)


def block(study: dict) -> str:
    return f"{START}\n{render(study)}\n{END}"


def main() -> None:
    study = json.loads(Path("results/benchmark.json").read_text())
    readme = Path("README.md")
    readme.write_text(
        re.sub(
            re.escape(START) + ".*?" + re.escape(END),
            lambda _: block(study),
            readme.read_text(),
            flags=re.S,
        )
    )


if __name__ == "__main__":
    main()
