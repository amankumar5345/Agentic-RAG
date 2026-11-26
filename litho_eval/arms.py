"""One interface for every RAG arm: A0 through A7."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml
from langchain_core.messages import HumanMessage, ToolMessage

from litho_eval.client import LLMClient, LLMResponse, assert_different_family
from litho_eval.index import (
    HybridRetriever,
    IndexBundle,
    RerankRetriever,
    RetrievedChunk,
)
from litho_eval.metrics import parse_json
from src.core.citations import (
    CITATION_RULE,
    enforce_citations,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARMS_CONFIG = REPO_ROOT / "configs" / "arms.yaml"

RAG_PROMPT = """You are a technical assistant for semiconductor photolithography
and cleanroom operations. Answer the question using only the retrieved context.
If the context does not contain the answer, say "Not answerable from the provided corpus."

Question:
{question}

Retrieved context:
{context}
"""

CLOSED_BOOK_PROMPT = """You are a technical assistant for semiconductor
photolithography and cleanroom operations. Answer the question from your own
knowledge. If you are not confident, say "Not answerable from the provided corpus."

Question:
{question}
"""


@dataclass
class ArmOutput:
    answer: str
    retrieved: list[RetrievedChunk]
    citations: list[dict]
    retrieval_calls: int
    stage_latencies: dict[str, float]
    cost_usd: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    citation_validation: dict | None = None


class Arm(Protocol):
    name: str

    def answer(self, question: str) -> ArmOutput: ...


class Generator:
    """The single generator used by A0-A4."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def answer(self, question: str, context: str | None) -> tuple[str, LLMResponse]:
        if context is None:
            prompt = CLOSED_BOOK_PROMPT.format(question=question)
        else:
            prompt = RAG_PROMPT.format(question=question, context=context)
        response = self.client.complete(prompt)
        return response.text, response


def _citations(chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "doc": chunk.doc,
            "section": chunk.section,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
        }
        for chunk in chunks
    ]


class ClosedBookArm:
    def __init__(self, generator: Generator, name: str = "A0") -> None:
        self.name = name
        self.generator = generator

    def answer(self, question: str) -> ArmOutput:
        started = time.perf_counter()
        text, response = self.generator.answer(question, None)
        return ArmOutput(
            answer=text,
            retrieved=[],
            citations=[],
            retrieval_calls=0,
            stage_latencies={"generation_s": time.perf_counter() - started, "total_s": time.perf_counter() - started},
            cost_usd=response.cost_usd,
            prompt_tokens=response.prompt_tokens or 0,
            completion_tokens=response.completion_tokens or 0,
        )


class RetrievalArm:
    def __init__(
        self,
        name: str,
        retriever,
        generator: Generator,
        top_k: int = 7,
    ) -> None:
        self.name = name
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def answer(self, question: str) -> ArmOutput:
        started = time.perf_counter()
        chunks = self.retriever.search(question, self.top_k)
        retrieval_s = time.perf_counter() - started

        context = "\n\n".join(chunk.text for chunk in chunks)
        generation_started = time.perf_counter()
        text, response = self.generator.answer(question, context)
        generation_s = time.perf_counter() - generation_started

        return ArmOutput(
            answer=text,
            retrieved=chunks,
            citations=_citations(chunks),
            retrieval_calls=1,
            stage_latencies={
                "retrieval_s": retrieval_s,
                "generation_s": generation_s,
                "total_s": time.perf_counter() - started,
            },
            cost_usd=response.cost_usd,
            prompt_tokens=response.prompt_tokens or 0,
            completion_tokens=response.completion_tokens or 0,
        )


def _extract_context(content: str) -> str:
    marker = "Retrieved context:"
    if marker not in content:
        return content.strip()
    blob = content.split(marker, 1)[1]
    for stop in (
        "Answer the user's question using only the retrieved context.",
        "Answer the user's question",
    ):
        if stop in blob:
            blob = blob.split(stop, 1)[0]
    return blob.strip()


class AgentArm:
    def __init__(self, name: str, graph) -> None:
        self.name = name
        self.graph = graph

    def answer(self, question: str) -> ArmOutput:
        started = time.perf_counter()
        state = {
            "messages": [HumanMessage(content=question)],
            "role": "",
            "is_blocked": False,
            "eval_results": None,
            "rag_calls": 0,
        }
        result = self.graph.invoke(state)
        messages = result.get("messages", [])

        retrieved = []
        for index, message in enumerate(messages):
            if isinstance(message, ToolMessage) and getattr(message, "name", None) == "load_prompt":
                retrieved.append(
                    RetrievedChunk(
                        id=f"tool-{index}",
                        text=_extract_context(str(message.content)),
                        doc="graph",
                    )
                )

        return ArmOutput(
            answer=messages[-1].content if messages else "",
            retrieved=retrieved,
            citations=_citations(retrieved),
            retrieval_calls=len(retrieved),
            stage_latencies={"total_s": time.perf_counter() - started},
        )


@dataclass(frozen=True)
class ArmSpec:
    name: str
    kind: str
    top_k: int = 7
    candidate_k: int = 25
    guardrail: bool = True
    router: bool = True


CITATION_ANSWER_PROMPT = """You are a technical assistant for semiconductor
photolithography and cleanroom operations. Answer using only the retrieved context.
{rules}
{feedback}
Question:
{question}

Retrieved context:
{context}
"""

SUPPORT_PROMPT = """Does the passage support the claim? Answer with only JSON:
{{"supported": true, "reason": "one short sentence"}}

Passage:
{chunk}

Claim:
{claim}
"""


class CitationGenerator:
    """Answer/repair callable used by enforce_citations."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def __call__(self, question: str, context: str, feedback: str | None) -> str:
        prompt = CITATION_ANSWER_PROMPT.format(
            rules=CITATION_RULE,
            feedback=f"Fix these citation issues: {feedback}" if feedback else "",
            question=question,
            context=context,
        )
        return self.client.complete(prompt).text


class CitationJudge:
    """Rule-adjacent LLM support check; judge family must differ from generator."""

    def __init__(
        self,
        client: LLMClient,
        judge_model: str,
        generator_model: str,
    ) -> None:
        assert_different_family(judge_model, generator_model)
        self.client = client

    def supports(self, sentence: str, chunk: str) -> bool:
        prompt = SUPPORT_PROMPT.format(chunk=chunk, claim=sentence)
        try:
            return bool(parse_json(self.client.complete(prompt).text).get("supported"))
        except Exception:
            return False


class CitationAgentArm:
    """A5 + citation enforcement/repair/abstention."""

    def __init__(
        self,
        name: str,
        base_arm,
        chunks_fn,
        generator: CitationGenerator,
        judge: CitationJudge,
    ) -> None:
        self.name = name
        self.base_arm = base_arm
        self.chunks_fn = chunks_fn
        self.generator = generator
        self.judge = judge

    def answer(self, question: str) -> ArmOutput:
        base = self.base_arm.answer(question)
        chunks = list(self.chunks_fn())
        started = time.perf_counter()
        result = enforce_citations(
            question,
            chunks,
            self.generator,
            self.judge.supports,
        )
        citation_s = time.perf_counter() - started
        latencies = dict(base.stage_latencies)
        latencies["citation_s"] = citation_s
        latencies["total_s"] = latencies.get("total_s", 0.0) + citation_s
        return ArmOutput(
            answer=result.text,
            retrieved=chunks,
            citations=result.citations,
            retrieval_calls=base.retrieval_calls,
            stage_latencies=latencies,
            cost_usd=base.cost_usd,
            prompt_tokens=base.prompt_tokens,
            completion_tokens=base.completion_tokens,
            citation_validation={
                "ok": result.validation.ok if result.validation else False,
                "citation_precision": (
                    result.validation.citation_precision if result.validation else 0.0
                ),
                "coverage": result.validation.coverage if result.validation else 0.0,
                "abstained": result.abstained,
                "issues": result.validation.issues if result.validation else [],
            },
        )


@dataclass
class ArmsConfig:
    generator_model: str
    judge_model: str
    temperature: float
    seed: int
    index: dict
    sweeps: dict
    arms: dict[str, ArmSpec]


def load_arms(path: str | Path = DEFAULT_ARMS_CONFIG) -> ArmsConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    arms = {}
    for name, raw in data.get("arms", {}).items():
        arms[name] = ArmSpec(
            name=name,
            kind=raw["kind"],
            top_k=int(raw.get("top_k", 7)),
            candidate_k=int(raw.get("candidate_k", 25)),
            guardrail=bool(raw.get("guardrail", True)),
            router=bool(raw.get("router", True)),
        )
    return ArmsConfig(
        generator_model=data.get("generator_model", "openai/gpt-oss-120b"),
        judge_model=data.get("judge_model", "groq/qwen/qwen3.8-27b"),
        temperature=float(data.get("temperature", 0.0)),
        seed=int(data.get("seed", 42)),
        index=data.get("index", {}),
        sweeps=data.get("sweeps", {"enabled": False}),
        arms=arms,
    )


def build_arm(
    spec: ArmSpec,
    generator: Generator | None = None,
    index: IndexBundle | None = None,
    graph=None,
):
    if spec.kind == "closed_book":
        if generator is None:
            raise ValueError("closed_book needs a generator")
        return ClosedBookArm(generator, spec.name)

    if spec.kind in {"dense", "bm25", "hybrid", "hybrid_rerank"}:
        if generator is None or index is None:
            raise ValueError(f"{spec.kind} needs a generator and an index")
        if spec.kind == "dense":
            retriever = index.dense
        elif spec.kind == "bm25":
            retriever = index.bm25
        elif spec.kind == "hybrid":
            retriever = HybridRetriever(
                index.dense,
                index.bm25,
                top_k=spec.top_k,
                candidate_k=spec.candidate_k,
            )
        else:
            retriever = RerankRetriever(
                HybridRetriever(
                    index.dense,
                    index.bm25,
                    top_k=spec.candidate_k,
                    candidate_k=spec.candidate_k,
                ),
                top_k=spec.top_k,
                candidate_k=spec.candidate_k,
            )
        return RetrievalArm(spec.name, retriever, generator, spec.top_k)

    if spec.kind == "agent":
        if graph is None:
            raise ValueError("agent arms need a compiled graph")
        return AgentArm(spec.name, graph)

    raise ValueError(f"unknown arm kind {spec.kind!r}")
