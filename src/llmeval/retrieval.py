"""Two retrievers: plain keyword overlap and BM25."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from llmeval.dataset import Document
from llmeval.text import content_tokens


class Index:
    """Holds term statistics for a fixed set of documents."""

    def __init__(self, documents: Sequence[Document]):
        self.documents = list(documents)
        self.terms = [Counter(content_tokens(f"{d.title} {d.text}")) for d in self.documents]
        self.lengths = [sum(term.values()) for term in self.terms]
        self.average_length = sum(self.lengths) / len(self.lengths)
        self.document_frequency: Counter[str] = Counter()
        for term in self.terms:
            self.document_frequency.update(term.keys())

    def idf(self, token: str) -> float:
        """Rare terms score high, and unseen terms score highest."""
        count = len(self.documents)
        seen = self.document_frequency[token]
        return math.log(1 + (count - seen + 0.5) / (seen + 0.5))


def overlap_search(index: Index, query: str, k: int = 3) -> list[tuple[Document, float]]:
    query_terms = set(content_tokens(query))
    scored = [
        (sum(term in terms for term in query_terms), position)
        for position, terms in enumerate(index.terms)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(index.documents[position], float(score)) for score, position in scored[:k]]


def bm25_search(
    index: Index, query: str, k: int = 3, k1: float = 1.5, b: float = 0.75
) -> list[tuple[Document, float]]:
    query_terms = content_tokens(query)
    scored = []
    for position, terms in enumerate(index.terms):
        scale = k1 * (1 - b + b * index.lengths[position] / index.average_length)
        score = sum(
            index.idf(term) * terms[term] * (k1 + 1) / (terms[term] + scale)
            for term in query_terms
            if term in terms
        )
        scored.append((score, position))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(index.documents[position], score) for score, position in scored[:k]]
