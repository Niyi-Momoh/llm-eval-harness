"""RAG through any OpenAI-compatible chat endpoint, including a local Ollama server."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import requests

from llmeval.pipelines import ABSTAIN_TEXT, Response
from llmeval.retrieval import Index, bm25_search

SYSTEM_PROMPT = (
    "Answer using only the provided context. Give a short answer. "
    f"If the context does not contain the answer, reply exactly: {ABSTAIN_TEXT}"
)
RETRY_STATUS = {429, 500, 502, 503, 504}


class ResponseCache:
    """Avoids paying twice for an identical request."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def _path(self, body: dict) -> Path:
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        return self.folder / f"{key}.json"

    def get(self, body: dict) -> dict | None:
        path = self._path(body)
        return json.loads(path.read_text()) if path.exists() else None

    def put(self, body: dict, payload: dict) -> None:
        self._path(body).write_text(json.dumps(payload))


class ChatPipeline:
    name = "chat_rag"

    def __init__(
        self,
        index: Index,
        endpoint: str,
        model: str,
        k: int = 3,
        price_per_million: tuple[float, float] = (0.0, 0.0),
        session=None,
        cache: ResponseCache | None = None,
        retries: int = 3,
        sleep=time.sleep,
    ):
        self.index = index
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.k = k
        self.prices = price_per_million
        self.session = session or requests.Session()
        self.cache = cache
        self.retries = retries
        self.sleep = sleep

    def config(self) -> dict:
        return {"kind": "chat", "model": self.model, "k": self.k}

    def _request(self, body: dict) -> dict:
        if self.cache and (cached := self.cache.get(body)) is not None:
            return cached
        headers = {}
        if key := os.environ.get("LLM_API_KEY"):
            headers["Authorization"] = f"Bearer {key}"
        for attempt in range(self.retries + 1):
            response = self.session.post(
                f"{self.endpoint}/v1/chat/completions", json=body, headers=headers, timeout=120
            )
            if response.status_code in RETRY_STATUS and attempt < self.retries:
                self.sleep(2**attempt)
                continue
            response.raise_for_status()
            payload = response.json()
            if self.cache:
                self.cache.put(body, payload)
            return payload
        raise RuntimeError("no response")

    def run(self, question: str) -> Response:
        found = [document for document, _ in bm25_search(self.index, question, self.k)]
        context = "\n\n".join(f"[{d.title}] {d.text}" for d in found)
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        }
        payload = self._request(body)
        answer = payload["choices"][0]["message"]["content"].strip()
        usage = payload.get("usage", {})
        prompt, completion = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        cost = (prompt * self.prices[0] + completion * self.prices[1]) / 1_000_000
        return Response(
            answer,
            contexts=found,
            abstained="don't know" in answer.lower().replace("’", "'"),
            tool_calls=["search"],
            prompt_tokens=prompt,
            completion_tokens=completion,
            cost_usd=cost,
        )
