from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from llmeval.compare import GatePolicy, compare_runs, is_regression, paired_pvalue
from llmeval.dataset import Case, DatasetError, Document, load_dataset, validate
from llmeval.llm import ChatPipeline, ResponseCache
from llmeval.pipelines import (
    ABSTAIN_TEXT,
    ExtractivePipeline,
    Response,
    RouterAgent,
    calculate,
    offline_pipelines,
)
from llmeval.retrieval import Index, bm25_search, overlap_search
from llmeval.runner import run_suite, summarise
from llmeval.scoring import faithfulness, score_case, token_f1
from llmeval.text import contains_phrase, sentences, tokens

ROOT = Path(__file__).parents[1]
DATASET = load_dataset(ROOT / "data")
DOCS = [
    Document("a", "Battery", "Robots charge below 18 percent. Packs last 1,200 cycles."),
    Document("b", "Speed", "Aisles allow 0.8 metres per second."),
]


def case(**overrides) -> Case:
    values = {
        "id": "x",
        "question": "q",
        "split": "test",
        "category": "direct",
        "answers": ("18 percent",),
        "docs": ("a",),
    }
    return Case(**{**values, **overrides})


def test_bundled_dataset_is_valid_and_split():
    assert DATASET.split("dev") and DATASET.split("test")
    assert not {c.id for c in DATASET.split("dev")} & {c.id for c in DATASET.split("test")}
    assert len(DATASET.sha256) == 64


def test_every_expected_answer_appears_in_its_source_document():
    documents = {document.id: document.text for document in DATASET.documents}
    for item in DATASET.cases:
        for doc in item.docs:
            assert any(contains_phrase(documents[doc], answer) for answer in item.answers), item.id


@pytest.mark.parametrize(
    "bad",
    [
        [case(), case()],
        [case(split="train")],
        [case(docs=("missing",))],
        [case(answers=())],
        [case(answerable=False)],
    ],
)
def test_validation_rejects_bad_cases(bad):
    with pytest.raises(DatasetError):
        validate(DOCS, bad)


def test_text_helpers():
    assert tokens("1,200 cycles") == ["1200", "cycles"]
    assert tokens("0.8 metres at 18:00") == ["0.8", "metres", "at", "18:00"]
    assert contains_phrase("It takes 95 minutes.", "95 minutes")
    assert not contains_phrase("It takes 195 minutes.", "95 minutes")
    assert sentences("Limit is 0.5 metres. Next one.") == ["Limit is 0.5 metres.", "Next one."]


@pytest.mark.parametrize("search", [overlap_search, bm25_search])
def test_retrievers_rank_the_relevant_document_first(search):
    assert search(Index(DOCS), "battery charge percent", 2)[0][0].id == "a"


def test_bm25_prefers_rare_terms():
    docs = [
        Document("a", "t", "common common common word"),
        Document("b", "t", "common rare"),
        Document("c", "t", "common other"),
    ]
    assert bm25_search(Index(docs), "common rare", 1)[0][0].id == "b"


def test_scoring_answerable_and_abstaining():
    good = score_case(case(), Response("Below 18 percent.", contexts=[DOCS[0]]))
    assert good["correct"] == 1.0 and good["retrieval_hit"] == 1.0
    assert good["reciprocal_rank"] == 1.0
    abstain = score_case(case(), Response(ABSTAIN_TEXT, abstained=True))
    assert abstain["correct"] == 0.0


def test_scoring_unanswerable_rewards_only_abstention():
    item = case(answers=(), docs=(), answerable=False)
    assert score_case(item, Response(ABSTAIN_TEXT, abstained=True))["correct"] == 1.0
    assert score_case(item, Response("Probably 12."))["correct"] == 0.0


def test_reciprocal_rank_reflects_position():
    scored = score_case(case(docs=("b",)), Response("x", contexts=DOCS))
    assert scored["reciprocal_rank"] == 0.5


def test_token_f1_and_faithfulness():
    assert token_f1("95 minutes", "95 minutes") == 1.0
    assert token_f1("no match", "95 minutes") == 0.0
    assert faithfulness("Robots charge", "robots charge daily") == 1.0
    assert faithfulness("Robots fly", "robots charge") == 0.5


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is 140 multiplied by 3?", "420"),
        ("What is 7 divided by 2?", "3.5"),
        ("What is 10 minus 12?", "-2"),
    ],
)
def test_router_uses_calculator(question, expected):
    response = RouterAgent(ExtractivePipeline("x", Index(DOCS), bm25_search)).run(question)
    assert response.answer == expected
    assert response.tool_calls == ["calculator"]


def test_router_uses_search_otherwise():
    response = RouterAgent(ExtractivePipeline("x", Index(DOCS), bm25_search)).run("battery?")
    assert response.tool_calls == ["search"]


def test_division_by_zero_is_reported():
    with pytest.raises(ZeroDivisionError):
        calculate(1, "divided by", 0)


class Exploding:
    name = "exploding"

    def config(self):
        return {}

    def run(self, question):
        raise RuntimeError("boom")


class Echo:
    name = "echo"

    def __init__(self, answer="18 percent", contexts=None):
        self.answer = answer
        self.contexts = contexts or [DOCS[0]]

    def config(self):
        return {}

    def run(self, question):
        return Response(self.answer, contexts=self.contexts, tool_calls=["search"])


def test_a_failing_case_is_recorded_not_fatal():
    result = run_suite(Exploding(), [case()], "sha")
    assert result["summary"]["errors"] == 1
    assert result["cases"][0]["correct"] == 0.0


def test_results_keep_input_order_with_threads():
    cases = [case(id=str(number)) for number in range(30)]
    result = run_suite(Echo(), cases, "sha", workers=4)
    assert [item["id"] for item in result["cases"]] == [str(n) for n in range(30)]


def test_summary_ignores_metrics_that_do_not_apply():
    item = case(answers=(), docs=(), answerable=False)
    summary = run_suite(Echo(), [item], "sha")["summary"]
    assert summary["retrieval_hit"] is None
    assert summary["correct"] == 0.0


def test_exact_pvalue_matches_hand_calculation():
    assert paired_pvalue(np.array([-1.0, -1.0, -1.0, 0.0])) == pytest.approx(1 / 8)
    assert paired_pvalue(np.array([1.0, -1.0])) == pytest.approx(3 / 4)
    assert paired_pvalue(np.zeros(5)) == 1.0


def test_gate_needs_both_size_and_significance():
    policy = GatePolicy()
    assert is_regression(np.array([-1.0] * 8 + [0.0] * 12), policy)[0]
    assert not is_regression(np.array([-1.0] * 2 + [0.0] * 18), policy)[0]
    tiny = np.array([-1.0] * 6 + [0.0] * 994)
    assert not is_regression(tiny, policy)[0]


def make_run(scores: list[float], latency: float = 1.0, cost: float = 0.0, errors: int = 0) -> dict:
    cases = [
        {
            "correct": score,
            "retrieval_hit": None,
            "tool_match": None,
        }
        for score in scores
    ]
    return {
        "manifest": {"dataset_sha256": "s", "case_ids": list(range(len(scores)))},
        "summary": {"latency_p95_ms": latency, "total_cost_usd": cost, "errors": errors},
        "cases": cases,
    }


def test_gate_blocks_accuracy_regression():
    result = compare_runs(make_run([1.0] * 30), make_run([0.0] * 10 + [1.0] * 20))
    assert not result["passed"]
    assert "correct" in result["reasons"][0]


def test_gate_passes_identical_runs():
    run = make_run([1.0, 0.0] * 15)
    assert compare_runs(run, copy.deepcopy(run))["passed"]


def test_gate_blocks_latency_cost_and_errors():
    base = make_run([1.0] * 5, latency=100, cost=1.0)
    assert not compare_runs(base, make_run([1.0] * 5, latency=200, cost=1.0))["passed"]
    assert not compare_runs(base, make_run([1.0] * 5, latency=100, cost=2.0))["passed"]
    assert not compare_runs(base, make_run([1.0] * 5, latency=100, cost=1.0, errors=1))["passed"]


def test_latency_below_floor_is_ignored():
    assert compare_runs(make_run([1.0] * 5, latency=1), make_run([1.0] * 5, latency=9))["passed"]


def test_runs_on_different_cases_cannot_be_compared():
    other = make_run([1.0] * 5)
    other["manifest"]["dataset_sha256"] = "different"
    with pytest.raises(ValueError):
        compare_runs(make_run([1.0] * 5), other)


class FakeHttp:
    def __init__(self, statuses, content="18 percent"):
        self.statuses = list(statuses)
        self.content = content
        self.calls = 0
        self.urls = []

    def post(self, url, json, headers, timeout):
        self.calls += 1
        self.urls.append(url)
        status, content = self.statuses.pop(0), self.content

        class Reply:
            status_code = status

            def raise_for_status(self):
                if status >= 400:
                    raise RuntimeError(str(status))

            def json(self):
                return {
                    "choices": [{"message": {"content": content}}],
                    "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
                }

        return Reply()


def chat(http, **kwargs) -> ChatPipeline:
    return ChatPipeline(Index(DOCS), "http://x/", "m", session=http, sleep=lambda _: None, **kwargs)


def test_chat_pipeline_calculates_cost_and_retries():
    http = FakeHttp([503, 200])
    response = chat(http, price_per_million=(2.0, 10.0)).run("battery charge?")
    assert http.calls == 2
    assert http.urls[0] == "http://x/v1/chat/completions"
    assert response.cost_usd == pytest.approx(0.007)
    assert response.answer == "18 percent"
    assert not response.abstained


def test_chat_pipeline_detects_abstention():
    assert chat(FakeHttp([200], "I don't know.")).run("battery?").abstained


def test_chat_cache_avoids_a_second_call(tmp_path):
    http = FakeHttp([200, 200])
    pipeline = chat(http, cache=ResponseCache(tmp_path))
    first = pipeline.run("battery?")
    second = pipeline.run("battery?")
    assert http.calls == 1
    assert first.answer == second.answer


def test_failed_chat_call_becomes_a_recorded_error():
    result = run_suite(chat(FakeHttp([400])), [case()], "sha")
    assert result["summary"]["errors"] == 1


def test_threshold_is_tuned_without_test_cases():
    dev = DATASET.split("dev")
    first = offline_pipelines(DATASET.documents, dev)["bm25_abstain"].threshold
    assert 0 < first < 1


def test_summary_interval_contains_the_mean():
    items = [
        {
            "error": None,
            "correct": float(n % 2),
            "category": "c",
            "latency_ms": 1.0,
            "cost_usd": 0.0,
            "retrieval_hit": None,
            "reciprocal_rank": None,
            "token_f1": None,
            "faithfulness": None,
            "tool_match": None,
        }
        for n in range(40)
    ]
    summary = summarise(items)
    assert summary["correct_95"][0] <= summary["correct"] <= summary["correct_95"][1]


def test_readme_results_match_the_benchmark_file():
    spec = importlib.util.spec_from_file_location("report", ROOT / "experiments" / "report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    study = json.loads((ROOT / "results" / "benchmark.json").read_text())
    assert module.block(study) in (ROOT / "README.md").read_text()
