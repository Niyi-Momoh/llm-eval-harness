"""Offline pipelines. They run without a model, so the benchmark can run in CI."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from llmeval.dataset import Case, Document
from llmeval.retrieval import Index, bm25_search, overlap_search
from llmeval.scoring import score_case
from llmeval.text import content_tokens, sentences

ABSTAIN_TEXT = "I don't know."
_ARITHMETIC = re.compile(
    r"(\d+(?:\.\d+)?)\s+(plus|minus|times|multiplied by|divided by)\s+(\d+(?:\.\d+)?)", re.I
)


@dataclass
class Response:
    answer: str
    contexts: list[Document] = field(default_factory=list)
    abstained: bool = False
    tool_calls: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class Pipeline(Protocol):
    name: str

    def config(self) -> dict: ...

    def run(self, question: str) -> Response: ...


class ExtractivePipeline:
    """Retrieve documents and return the sentence that best covers the question."""

    def __init__(
        self,
        name: str,
        index: Index,
        search: Callable[[Index, str, int], list[tuple[Document, float]]],
        k: int = 3,
        threshold: float | None = None,
    ):
        self.name = name
        self.index = index
        self.search = search
        self.k = k
        self.threshold = threshold

    def config(self) -> dict:
        return {"kind": "extractive", "k": self.k, "abstain_threshold": self.threshold}

    def _best_sentence(self, question: str, documents: Sequence[Document]) -> tuple[str, float]:
        terms = set(content_tokens(question))
        total = sum(self.index.idf(term) for term in terms) or 1.0
        best_text, best_score = "", -1.0
        for document in documents:
            for sentence in sentences(document.text):
                present = set(content_tokens(f"{document.title} {sentence}"))
                score = sum(self.index.idf(term) for term in terms & present) / total
                if score > best_score:
                    best_text, best_score = sentence, score
        return best_text, best_score

    def run(self, question: str) -> Response:
        found = [document for document, _ in self.search(self.index, question, self.k)]
        answer, confidence = self._best_sentence(question, found)
        if self.threshold is not None and confidence < self.threshold:
            return Response(ABSTAIN_TEXT, contexts=found, abstained=True, tool_calls=["search"])
        return Response(answer, contexts=found, tool_calls=["search"])


def calculate(left: float, operator: str, right: float) -> float:
    """A small calculator tool. It never evaluates arbitrary text."""
    operator = operator.lower()
    if operator == "plus":
        return left + right
    if operator == "minus":
        return left - right
    if operator in {"times", "multiplied by"}:
        return left * right
    if right == 0:
        raise ZeroDivisionError("division by zero")
    return left / right


class RouterAgent:
    """Send arithmetic to the calculator and everything else to retrieval."""

    name = "router_agent"

    def __init__(self, inner: Pipeline):
        self.inner = inner

    def config(self) -> dict:
        return {"kind": "router", "inner": self.inner.config()}

    def run(self, question: str) -> Response:
        match = _ARITHMETIC.search(question)
        if not match:
            return self.inner.run(question)
        value = calculate(float(match[1]), match[2], float(match[3]))
        text = str(int(value)) if value.is_integer() else str(round(value, 4))
        return Response(text, tool_calls=["calculator"])


def tune_threshold(index: Index, dev_cases: Sequence[Case]) -> float:
    """Choose the abstention threshold using development cases only."""
    best_threshold, best_score = 0.0, -1.0
    for step in range(5, 100, 5):
        threshold = step / 100
        pipeline = ExtractivePipeline("tuning", index, bm25_search, threshold=threshold)
        correct = sum(
            score_case(case, pipeline.run(case.question))["correct"] for case in dev_cases
        )
        if correct > best_score:
            best_threshold, best_score = threshold, correct
    return best_threshold


def offline_pipelines(
    documents: Sequence[Document], dev_cases: Sequence[Case]
) -> dict[str, Pipeline]:
    index = Index(documents)
    retrieval_dev = [case for case in dev_cases if case.category != "tool"]
    threshold = tune_threshold(index, retrieval_dev)
    abstaining = ExtractivePipeline("bm25_abstain", index, bm25_search, threshold=threshold)
    return {
        "overlap": ExtractivePipeline("overlap", index, overlap_search),
        "bm25": ExtractivePipeline("bm25", index, bm25_search),
        "bm25_abstain": abstaining,
        "router_agent": RouterAgent(abstaining),
    }
