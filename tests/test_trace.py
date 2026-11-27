"""Trace schema, redaction and non-interference with evaluation keys."""

from __future__ import annotations

import json

from litho_eval.client import LLMClient
from litho_eval.runner import (
    DEFAULT_ARMS_CONFIG,
    DEFAULT_CORPUS,
    DEFAULT_DATA_DIR,
    DEFAULT_JUDGE_PROMPT,
    DEFAULT_PRICES,
    RunConfig,
    config_hash,
)
from src.core.retriever import retrieve_with_scores
from src.core.trace import (
    NOT_RECORDED,
    TRACE_SCHEMA_VERSION,
    _dotenv_values,
    load_traces,
    normalise_trace,
    redact_text,
    redact_trace,
    secret_values,
    trace_from_result_case,
)


def _case() -> dict:
    return {
        "id": "qa-1",
        "input": "What PPE is required?",
        "output": "Wear goggles [C1].",
        "retrieved": [
            {
                "id": "gold-0",
                "doc": "laser.pdf",
                "section": "PPE",
                "text": "Wear goggles.",
                "score": 0.0,
            }
        ],
        "citations": [{"doc": "laser.pdf", "section": "PPE"}],
        "retrieval_calls": 1,
        "latencies": {"total_s": 0.2},
        "tokens": {"prompt": 10, "completion": 5},
        "cost_usd": 0.0,
        "mock": True,
    }


def test_trace_from_result_case_is_versioned_and_does_not_synthesize():
    trace = trace_from_result_case(_case(), source="results/A5.json", sample=True)

    assert trace["schema_version"] == TRACE_SCHEMA_VERSION
    assert trace["sample"] is True
    assert trace["source"] == "results/A5.json"
    assert trace["retrieval"]["candidates"] == NOT_RECORDED
    assert trace["guardrail"]["verdict"] == NOT_RECORDED
    assert trace["router"]["role"] == NOT_RECORDED
    assert trace["chunks"][0]["id"] == "gold-0"
    assert "id" not in trace["citations"][0]
    assert trace["citation_check"]["result"] == NOT_RECORDED


def test_normalise_trace_labels_missing_fields_without_inference():
    trace = normalise_trace(
        {
            "schema_version": TRACE_SCHEMA_VERSION,
            "question": "q",
            "answer": "a",
        }
    )

    assert trace["guardrail"]["verdict"] == NOT_RECORDED
    assert trace["retrieval"]["candidates"] == NOT_RECORDED
    assert trace["chunks"] == NOT_RECORDED
    assert trace["citation_check"]["result"] == NOT_RECORDED
    assert trace["mock"] is False


def test_trace_redacts_api_keys_and_dotenv_values(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_" + "a" * 24)
    secrets = secret_values()
    trace = {
        "answer": "token gsk_" + "a" * 24,
        "guardrail": {"verdict": "passed"},
    }

    redacted = redact_trace(trace, secrets)

    assert "gsk_" not in json.dumps(redacted)
    assert "[REDACTED]" in redacted["answer"]
    assert redact_text("Bearer abcdefghijklmnopqrstuvwxyz", secrets).endswith("[REDACTED]")

    dotenv = tmp_path / ".env"
    fake_key = "sk-" + "z" * 24
    dotenv.write_text(f"DEEPSEEK_API_KEY={fake_key}\n", encoding="utf-8")
    assert fake_key in _dotenv_values(dotenv)


def test_sample_trace_is_committed_and_loads():
    traces = load_traces()

    assert traces, "expected at least one recorded trace in results/traces/"
    sample = next(trace for trace in traces if trace.get("sample"))
    assert sample["schema_version"] == TRACE_SCHEMA_VERSION
    assert sample["mock"] is True
    assert sample["question"]
    assert sample["retrieval"]["selected"] is not None


def test_trace_env_does_not_change_run_config_hash(monkeypatch):
    config = RunConfig(
        arm="A1",
        split="test",
        generator_model="openai/gpt-oss-120b",
        judge_model="groq/qwen/qwen3.8-27b",
        top_k=7,
        temperature=0.0,
        seed=42,
        limit=None,
        mock=True,
        data_dir=str(DEFAULT_DATA_DIR),
        corpus_path=str(DEFAULT_CORPUS),
        prices_path=str(DEFAULT_PRICES),
        judge_prompt_path=str(DEFAULT_JUDGE_PROMPT),
        arms_config_path=str(DEFAULT_ARMS_CONFIG),
        chunk_size=1000,
        chunk_overlap=150,
        embedding_model="bge-m3:latest",
    )

    monkeypatch.delenv("LITHO_TRACE", raising=False)
    before = config_hash(config)
    monkeypatch.setenv("LITHO_TRACE", "1")
    after = config_hash(config)

    assert before == after


def test_trace_env_does_not_change_llm_cache_key(monkeypatch):
    client = LLMClient("openai/gpt-oss-120b", mock=True)
    messages = [{"role": "user", "content": "same prompt"}]

    monkeypatch.delenv("LITHO_TRACE", raising=False)
    before = client._cache_key(messages, 512)
    monkeypatch.setenv("LITHO_TRACE", "1")
    after = client._cache_key(messages, 512)

    assert before == after


def test_retrieve_with_scores_captures_25_candidates_and_7_selected():
    from langchain_core.documents import Document

    class FakeBase:
        def invoke(self, query):
            return [
                Document(
                    page_content=f"chunk {index}",
                    metadata={"id": f"K{index}", "doc": "d.pdf", "section": "s"},
                )
                for index in range(1, 26)
            ]

    class FakeCompressor:
        def compress_documents(self, documents, query):
            selected = []
            for index, document in enumerate(documents[:7], start=1):
                selected.append(
                    Document(
                        page_content=document.page_content,
                        metadata={
                            **document.metadata,
                            "id": f"C{index}",
                            "relevance_score": 1.0 - index / 10,
                        },
                    )
                )
            return selected

    class FakeRetriever:
        base_retriever = FakeBase()
        base_compressor = FakeCompressor()

    captured = retrieve_with_scores("question", retriever=FakeRetriever())

    assert len(captured["candidates"]) == 25
    assert len(captured["selected"]) == 7
    assert captured["selected"][0]["score"] == 0.9
    assert captured["selected"][0]["id"] == "C1"
