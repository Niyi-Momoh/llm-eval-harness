"""Load and validate documents and evaluation cases."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


class DatasetError(ValueError):
    """Raised when the dataset is inconsistent."""


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    split: str
    category: str
    answers: tuple[str, ...] = ()
    docs: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    answerable: bool = True


@dataclass(frozen=True)
class Dataset:
    documents: tuple[Document, ...]
    cases: tuple[Case, ...]
    sha256: str = field(default="")

    def split(self, name: str) -> tuple[Case, ...]:
        return tuple(case for case in self.cases if case.split == name)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate(documents: list[Document], cases: list[Case]) -> None:
    doc_ids = [document.id for document in documents]
    case_ids = [case.id for case in cases]
    if len(set(doc_ids)) != len(doc_ids) or len(set(case_ids)) != len(case_ids):
        raise DatasetError("ids must be unique")
    known = set(doc_ids)
    for case in cases:
        if case.split not in {"dev", "test"}:
            raise DatasetError(f"{case.id}: split must be dev or test")
        if not known.issuperset(case.docs):
            raise DatasetError(f"{case.id}: refers to an unknown document")
        if case.answerable and not case.answers:
            raise DatasetError(f"{case.id}: answerable cases need answers")
        if not case.answerable and (case.answers or case.docs):
            raise DatasetError(f"{case.id}: unanswerable cases cannot have answers or documents")


def load_dataset(folder: str | Path) -> Dataset:
    root = Path(folder)
    raw_docs = (root / "documents.jsonl").read_bytes()
    raw_cases = (root / "cases.jsonl").read_bytes()
    documents = [Document(**row) for row in _read_jsonl(root / "documents.jsonl")]
    cases = [
        Case(
            id=row["id"],
            question=row["question"],
            split=row["split"],
            category=row["category"],
            answers=tuple(row.get("answers", ())),
            docs=tuple(row.get("docs", ())),
            tools=tuple(row.get("tools", ())),
            answerable=row.get("answerable", True),
        )
        for row in _read_jsonl(root / "cases.jsonl")
    ]
    validate(documents, cases)
    digest = hashlib.sha256(raw_docs + b"\0" + raw_cases).hexdigest()
    return Dataset(tuple(documents), tuple(cases), digest)
