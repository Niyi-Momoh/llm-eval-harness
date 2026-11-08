"""Score one response against one case."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from llmeval.dataset import Case
from llmeval.text import contains_phrase, content_tokens, tokens

if TYPE_CHECKING:
    from llmeval.pipelines import Response


def token_f1(answer: str, reference: str) -> float:
    predicted, expected = Counter(tokens(answer)), Counter(tokens(reference))
    shared = sum((predicted & expected).values())
    if not shared:
        return 0.0
    precision = shared / sum(predicted.values())
    recall = shared / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


def faithfulness(answer: str, context_text: str) -> float:
    """Share of the answer's content words that appear in the retrieved text.

    This is a cheap lexical check. It can flag invented wording, but it cannot prove
    that the answer's meaning is supported.
    """
    words = content_tokens(answer)
    if not words:
        return 1.0
    available = set(content_tokens(context_text))
    return sum(word in available for word in words) / len(words)


def score_case(case: Case, response: Response) -> dict:
    """Return per-case metrics. A metric that does not apply is ``None``."""
    context_ids = [document.id for document in response.contexts]
    context_text = " ".join(document.text for document in response.contexts)
    relevant = set(case.docs)
    retrieval_hit = None
    reciprocal_rank = None
    if relevant:
        ranks = [rank for rank, doc in enumerate(context_ids, start=1) if doc in relevant]
        retrieval_hit = float(bool(ranks))
        reciprocal_rank = 1 / ranks[0] if ranks else 0.0
    if case.answerable:
        correct = not response.abstained and any(
            contains_phrase(response.answer, answer) for answer in case.answers
        )
        f1 = max(token_f1(response.answer, answer) for answer in case.answers)
    else:
        correct = response.abstained
        f1 = None
    grounded = None
    if response.contexts and not response.abstained:
        grounded = faithfulness(response.answer, context_text)
    tool_match = float(list(case.tools) == response.tool_calls) if case.tools else None
    return {
        "correct": float(correct),
        "retrieval_hit": retrieval_hit,
        "reciprocal_rank": reciprocal_rank,
        "token_f1": f1,
        "faithfulness": grounded,
        "tool_match": tool_match,
        "abstained": response.abstained,
    }
