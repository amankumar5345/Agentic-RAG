"""Offline tests for the arm interface and ablations."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from bench.draft_cases import split_corpus
from litho_eval.arms import (
    AgentArm,
    ArmOutput,
    ArmSpec,
    CitationAgentArm,
    ClosedBookArm,
    RetrievalArm,
    build_arm,
    load_arms,
)
from litho_eval.index import BM25Index, HybridRetriever, RerankRetriever, RetrievedChunk, get_index
from src.core.citations import RetrievedChunk as CitationChunk
from src.core.graph import GraphConfig, RoleMaker, build_graph


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = 0

    def search(self, query, k):
        self.calls += 1
        return self.chunks[:k]


class FakeGenerator:
    def __init__(self):
        self.calls = 0

    def answer(self, question, context):
        self.calls += 1
        return "mock answer", SimpleNamespace(
            cost_usd=0.0,
            prompt_tokens=1,
            completion_tokens=1,
        )


def chunk(identifier, text):
    return RetrievedChunk(id=identifier, text=text, doc="d.pdf")


def test_closed_book_never_retrieves():
    arm = ClosedBookArm(FakeGenerator())
    output = arm.answer("what is lithography?")

    assert output.answer == "mock answer"
    assert output.retrieved == []
    assert output.retrieval_calls == 0


def test_retrieval_arms_end_to_end():
    a, b, c = chunk("a", "alpha beta"), chunk("b", "beta gamma"), chunk("c", "gamma delta")

    for name in ("A1", "A2"):
        arm = RetrievalArm(name, FakeRetriever([a, b]), FakeGenerator(), top_k=2)
        output = arm.answer("question")
        assert output.answer == "mock answer"
        assert len(output.retrieved) == 2
        assert output.citations
        assert output.retrieval_calls == 1

    hybrid = HybridRetriever(
        FakeRetriever([a, b]),
        FakeRetriever([b, c]),
        top_k=2,
        candidate_k=2,
    )
    output = RetrievalArm("A3", hybrid, FakeGenerator(), top_k=2).answer("question")
    assert {item.id for item in output.retrieved} == {"a", "b"}

    reranked = RerankRetriever(
        hybrid,
        top_k=2,
        candidate_k=3,
        reranker=lambda query, candidates: list(reversed(candidates)),
    )
    output = RetrievalArm("A4", reranked, FakeGenerator(), top_k=2).answer("question")
    assert len(output.retrieved) == 2
    assert output.retrieved[0].id != output.retrieved[1].id


def test_hybrid_rrf_order():
    a, b, c = chunk("a", "alpha"), chunk("b", "beta"), chunk("c", "gamma")
    hybrid = HybridRetriever(
        FakeRetriever([a, b]),
        FakeRetriever([b, c]),
        top_k=3,
        candidate_k=2,
    )

    assert [item.id for item in hybrid.search("q")] == ["b", "a", "c"]


def test_bm25_arm_on_a_tiny_corpus():
    corpus = """--- Source: a.pdf ---
## Safety
laser safety goggles are required.
--- Source: b.pdf ---
## Process
the wafer is etched.
"""
    chunks = split_corpus(corpus)
    arm = RetrievalArm("A2", BM25Index(chunks), FakeGenerator(), top_k=1)

    output = arm.answer("goggles")

    assert "goggles" in output.retrieved[0].text


def test_one_index_per_chunk_setting_is_reused(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("--- Source: a.pdf ---\nalpha beta gamma\n", encoding="utf-8")

    first = get_index(corpus, 500, 75, "test-model", cache_root=tmp_path / "idx")
    second = get_index(corpus, 500, 75, "test-model", cache_root=tmp_path / "idx")

    assert first is second


class FakeRails:
    def __init__(self):
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        return {"content": "QUERY_PASSED"}


class FakeRouter:
    def __init__(self):
        self.calls = 0

    def invoke(self, query):
        self.calls += 1
        return RoleMaker(role="Chatbot")


class FakeOrchestrator:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content="final answer")


class FakeEvaluator:
    def __call__(self, state):
        return {"eval_results": {}}


def make_agent(guardrail: bool, router: bool):
    rails, router_llm, orchestrator = FakeRails(), FakeRouter(), FakeOrchestrator()
    llms = {
        "orchestrator": orchestrator,
        "router": router_llm,
        "judge": None,
        "rails": rails,
        "evaluator": FakeEvaluator(),
    }
    graph = build_graph(
        GraphConfig(),
        llms=llms,
        enable_guardrail=guardrail,
        enable_router=router,
    ).compile()
    return AgentArm("arm", graph), rails, router_llm, orchestrator


def test_agent_ablations():
    a5, rails, router, orchestrator = make_agent(True, True)
    assert a5.answer("q").answer == "final answer"
    assert (rails.calls, router.calls, orchestrator.calls) == (1, 1, 1)

    a6, rails, router, orchestrator = make_agent(True, False)
    assert a6.answer("q").answer == "final answer"
    assert (rails.calls, router.calls, orchestrator.calls) == (1, 0, 1)

    a7, rails, router, orchestrator = make_agent(False, True)
    assert a7.answer("q").answer == "final answer"
    assert (rails.calls, router.calls, orchestrator.calls) == (0, 1, 1)


def test_arms_config_has_all_arms():
    config = load_arms()
    assert set(config.arms) == {f"A{index}" for index in range(9)}
    assert config.arms["A6"].router is False
    assert config.arms["A7"].guardrail is False
    assert config.arms["A4"].kind == "hybrid_rerank"
    assert config.arms["A8"].kind == "agent_citations"


def test_build_arm_rejects_unknown_kind():
    with pytest.raises(ValueError):
        build_arm(ArmSpec(name="AX", kind="nope"))


class FakeBaseArm:
    def answer(self, question):
        return ArmOutput(
            answer="Wear goggles when using the laser.",
            retrieved=[],
            citations=[],
            retrieval_calls=1,
            stage_latencies={"total_s": 0.1},
            cost_usd=0.0,
        )


def test_citation_agent_wraps_a8_and_abstains_or_cites():
    chunks = [CitationChunk(id="C1", text="Wear goggles.", doc="laser.pdf", section="PPE")]

    def generator(question, context, feedback):
        return "Wear goggles when using the laser [C1]."

    judge = SimpleNamespace(supports=lambda sentence, chunk: True)

    arm = CitationAgentArm("A8", FakeBaseArm(), lambda: chunks, generator, judge)
    output = arm.answer("What PPE?")

    assert output.answer.endswith("[C1].")
    assert output.citations[0]["id"] == "C1"
    assert output.retrieval_calls == 1
