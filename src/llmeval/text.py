"""Text normalisation shared by retrieval and scoring."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "many",
        "much",
        "must",
        "of",
        "on",
        "or",
        "our",
        "the",
        "their",
        "there",
        "to",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    ]
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[a-z0-9]+(?:[.:'-][a-z0-9]+)*")


def tokens(text: str) -> list[str]:
    """Lower-case word tokens. Thousands separators are removed, so 1,200 matches 1200."""
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    return _TOKEN.findall(cleaned)


def stem(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") and token[-2] != "s" else token


def content_tokens(text: str) -> list[str]:
    return [stem(token) for token in tokens(text) if token not in STOPWORDS]


def sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE.split(text) if part.strip()]


def contains_phrase(text: str, phrase: str) -> bool:
    """Whether the words of ``phrase`` appear consecutively in ``text``."""
    haystack, needle = tokens(text), tokens(phrase)
    if not needle:
        return False
    return any(
        haystack[start : start + len(needle)] == needle
        for start in range(len(haystack) - len(needle) + 1)
    )
