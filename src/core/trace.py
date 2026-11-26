"""Structured answer traces for the UI and demo replay."""

from __future__ import annotations

import time


def demo_trace(case: dict) -> dict:
    chunks = []
    for index, item in enumerate(case.get("retrieved", []), start=1):
        chunks.append(
            {
                "id": f"C{index}",
                "text": item.get("text", ""),
                "doc": item.get("doc", ""),
                "section": item.get("section", ""),
                "score": item.get("score", 0.0),
            }
        )

    citations = []
    for index, citation in enumerate(case.get("citations", []), start=1):
        snippet = citation.get("snippet", citation.get("text", ""))
        if not snippet and index <= len(chunks):
            snippet = chunks[index - 1].get("text", "")[:240]
        citations.append(
            {
                "id": f"C{index}",
                "doc": citation.get("doc", ""),
                "section": citation.get("section", ""),
                "snippet": snippet,
            }
        )

    trace = {
        "guardrail": {"decision": "not recorded", "category": "not recorded"},
        "role": "not recorded",
        "retrieval_calls": case.get("retrieval_calls", 0),
        "chunks": chunks,
        "citations": citations,
        "citation_validation": _structural_validation(case, chunks),
        "stage_latencies": case.get("latencies", {}),
        "tokens": case.get("tokens", {}),
        "cost_usd": case.get("cost_usd"),
        "mock": bool(case.get("mock")),
    }
    return {"answer": case.get("output", ""), "trace": trace, "geometry": None, "case": case}


def _structural_validation(case: dict, chunks: list[dict]) -> dict:
    from src.core.citations import RetrievedChunk, validate_citations

    objects = [
        RetrievedChunk(
            id=chunk["id"],
            text=chunk.get("text", ""),
            doc=chunk.get("doc", ""),
            section=chunk.get("section", ""),
        )
        for chunk in chunks
    ]
    validation = validate_citations(
        case.get("output", ""),
        objects,
        lambda sentence, chunk: True,
    )
    return {
        "result": "passed (structural)" if validation.ok else "failed (structural)",
        "detail": (
            f"coverage={validation.coverage:.3f}, "
            f"precision={validation.citation_precision:.3f}; judge support not recorded"
        ),
    }


def arm_trace(question: str, arm) -> dict:
    output = arm.answer(question)
    chunks = [
        {
            "id": chunk.id,
            "text": chunk.text,
            "doc": chunk.doc,
            "section": chunk.section,
            "score": getattr(chunk, "score", 0.0),
        }
        for chunk in output.retrieved
    ]
    trace = {
        "guardrail": {"decision": "live", "category": "not recorded in arm output"},
        "role": "A8 citation arm",
        "retrieval_calls": output.retrieval_calls,
        "chunks": chunks,
        "citations": output.citations,
        "citation_validation": {
            "result": (
                "passed"
                if (output.citation_validation or {}).get("ok")
                else "failed or not recorded"
            ),
            "detail": str((output.citation_validation or {}).get("issues", "")),
        },
        "stage_latencies": output.stage_latencies,
        "tokens": {
            "prompt": output.prompt_tokens,
            "completion": output.completion_tokens,
        },
        "cost_usd": output.cost_usd,
        "mock": False,
    }
    return {"answer": output.answer, "trace": trace, "geometry": None, "case": None}


def live_trace(question: str, graph_chain=None) -> dict:
    from langchain_core.messages import HumanMessage

    chain = graph_chain
    if chain is None:
        from src.core.graph import chain

    started = time.perf_counter()
    result = chain.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "rag_calls": 0,
            "is_blocked": False,
            "eval_results": None,
            "role": "",
        }
    )
    elapsed = time.perf_counter() - started
    messages = result.get("messages", [])
    answer = messages[-1].content if messages else ""
    trace = {
        "guardrail": {
            "decision": "blocked" if result.get("is_blocked") else "passed",
            "category": "guardrail",
        },
        "role": result.get("role") or "not recorded",
        "retrieval_calls": result.get("rag_calls", 0),
        "chunks": [],
        "citations": [],
        "citation_validation": {
            "result": "not recorded",
            "detail": "live trace does not capture structured chunks yet",
        },
        "stage_latencies": {"total_s": elapsed},
        "tokens": {},
        "cost_usd": None,
        "mock": False,
    }
    return {"answer": answer, "trace": trace, "geometry": None, "case": None}
