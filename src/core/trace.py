"""Versioned, redacted answer traces.

A trace is a recorded account of how one answer was produced. It is a
presentation artefact: tracing is off by default, nothing in the evaluation
arms imports it, and it never changes a cache key or a result file.

Missing values are written as ``"not recorded"``. They are never inferred from
other fields.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

TRACE_SCHEMA_VERSION = "1.0"
NOT_RECORDED = "not recorded"
TRACE_DIR = Path(__file__).resolve().parents[2] / "results" / "traces"

SECRET_PATTERNS = (
    re.compile(r"gsk_[A-Za-z0-9]{16,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"lsv2_[A-Za-z0-9]{10,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{16,}"),
)
REDACTED = "[REDACTED]"


def _recorded(value: Any) -> Any:
    """Return a value, or the explicit missing marker. Never infer a value."""
    if value is None:
        return NOT_RECORDED
    if isinstance(value, str) and not value.strip():
        return NOT_RECORDED
    if isinstance(value, dict) and not value:
        return NOT_RECORDED
    return value


def _dotenv_values(path: Path | None = None) -> set[str]:
    path = path or Path(__file__).resolve().parents[2] / ".env"
    values: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        _, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        if len(value) >= 8:
            values.add(value)
    return values


def secret_values() -> set[str]:
    """Collect values that must never appear in a trace.

    Values are read from secret-looking environment variables and from
    ``.env``. The set itself is never written to disk.
    """
    values = _dotenv_values()
    markers = ("KEY", "TOKEN", "SECRET", "PASSWORD")
    for name, value in os.environ.items():
        if any(marker in name.upper() for marker in markers) and len(value) >= 8:
            values.add(value)
    return values


def redact_text(text: str, secrets: Iterable[str] | None = None) -> str:
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    for secret in secrets or ():
        if secret:
            text = text.replace(secret, REDACTED)
    return text


def redact_trace(value: Any, secrets: Iterable[str] | None = None) -> Any:
    """Recursively redact strings in a trace structure."""
    secrets = list(secrets or ())
    if isinstance(value, str):
        return redact_text(value, secrets)
    if isinstance(value, list):
        return [redact_trace(item, secrets) for item in value]
    if isinstance(value, dict):
        return {
            redact_text(str(key), secrets): redact_trace(item, secrets)
            for key, item in value.items()
        }
    return value


def _trace_id(question: str, source: str) -> str:
    digest = hashlib.sha256(f"{source}|{question}".encode("utf-8")).hexdigest()[:12]
    return f"trace-{digest}"


def _chunks(items: Sequence[dict]) -> list[dict]:
    chunks = []
    for item in items:
        chunks.append(
            {
                "id": _recorded(item.get("id")),
                "doc": _recorded(item.get("doc")),
                "section": _recorded(item.get("section")),
                "score": _recorded(item.get("score")),
            }
        )
    return chunks


def _citation_check(validation: dict | None) -> dict:
    if not validation:
        return {
            "result": NOT_RECORDED,
            "coverage": NOT_RECORDED,
            "precision": NOT_RECORDED,
            "abstained": NOT_RECORDED,
            "issues": NOT_RECORDED,
        }
    return {
        "result": "passed" if validation.get("ok") else "failed",
        "coverage": _recorded(validation.get("coverage")),
        "precision": _recorded(validation.get("citation_precision")),
        "abstained": _recorded(validation.get("abstained")),
        "issues": _recorded(validation.get("issues")),
    }


def trace_from_result_case(
    case: dict,
    *,
    source: str,
    sample: bool = False,
    mock: bool | None = None,
) -> dict:
    """Build a trace from one recorded result case without inventing data."""
    chunks = _chunks(case.get("retrieved") or [])
    selected = [
        {"id": chunk["id"], "doc": chunk["doc"], "score": chunk["score"]}
        for chunk in chunks
    ]
    question = case.get("input", "")
    trace = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "trace_id": _trace_id(question, source),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "sample": sample,
        "source": source,
        "mock": bool(case.get("mock")) if mock is None else mock,
        "question": question,
        "answer": case.get("output", ""),
        "guardrail": case.get("guardrail") or {
            "verdict": NOT_RECORDED,
            "category": NOT_RECORDED,
        },
        "router": {"role": _recorded(case.get("role"))},
        "retrieval": {
            "calls": _recorded(case.get("retrieval_calls")),
            "candidates": NOT_RECORDED,
            "selected": selected,
        },
        "chunks": chunks,
        "citations": case.get("citations") or [],
        "citation_check": _citation_check(case.get("citation_validation")),
        "stage_latencies": _recorded(case.get("latencies")),
        "tokens": _recorded(case.get("tokens")),
        "cost_usd": _recorded(case.get("cost_usd")),
    }
    return normalise_trace(trace)


def trace_from_arm_output(
    question: str,
    output,
    *,
    guardrail: dict | None = None,
    router: dict | None = None,
    retrieval: dict | None = None,
    source: str = "live run",
) -> dict:
    chunks = [
        {
            "id": _recorded(getattr(chunk, "id", None)),
            "doc": _recorded(getattr(chunk, "doc", None)),
            "section": _recorded(getattr(chunk, "section", None)),
            "score": _recorded(getattr(chunk, "score", None)),
        }
        for chunk in output.retrieved
    ]
    selected = [
        {"id": chunk["id"], "doc": chunk["doc"], "score": chunk["score"]}
        for chunk in chunks
    ]
    trace = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "trace_id": _trace_id(question, source),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "sample": False,
        "source": source,
        "mock": False,
        "question": question,
        "answer": output.answer,
        "guardrail": guardrail or {
            "verdict": NOT_RECORDED,
            "category": NOT_RECORDED,
        },
        "router": router or {"role": NOT_RECORDED},
        "retrieval": retrieval
        or {
            "calls": output.retrieval_calls,
            "candidates": NOT_RECORDED,
            "selected": selected,
        },
        "chunks": chunks,
        "citations": output.citations,
        "citation_check": _citation_check(output.citation_validation),
        "stage_latencies": output.stage_latencies,
        "tokens": {
            "prompt": output.prompt_tokens,
            "completion": output.completion_tokens,
        },
        "cost_usd": _recorded(output.cost_usd),
    }
    return normalise_trace(trace)


def normalise_trace(trace: dict) -> dict:
    """Fill required presentation keys with the missing marker.

    This only adds explicit labels; it never derives a value from another
    field or from the model.
    """
    trace = dict(trace)
    trace.setdefault("schema_version", TRACE_SCHEMA_VERSION)
    trace.setdefault("sample", False)
    trace.setdefault("source", NOT_RECORDED)
    trace.setdefault("mock", False)
    trace.setdefault("question", NOT_RECORDED)
    trace.setdefault("answer", NOT_RECORDED)
    trace.setdefault("guardrail", {"verdict": NOT_RECORDED, "category": NOT_RECORDED})
    trace.setdefault("router", {"role": NOT_RECORDED})
    trace.setdefault("retrieval", {"calls": NOT_RECORDED, "candidates": NOT_RECORDED, "selected": NOT_RECORDED})
    trace.setdefault("chunks", NOT_RECORDED)
    trace.setdefault("citations", NOT_RECORDED)
    trace.setdefault("citation_check", {"result": NOT_RECORDED})
    trace.setdefault("stage_latencies", NOT_RECORDED)
    trace.setdefault("tokens", NOT_RECORDED)
    trace.setdefault("cost_usd", NOT_RECORDED)

    guardrail = trace.get("guardrail")
    if not isinstance(guardrail, dict):
        guardrail = {"verdict": NOT_RECORDED, "category": NOT_RECORDED}
        trace["guardrail"] = guardrail
    guardrail.setdefault("verdict", NOT_RECORDED)
    guardrail.setdefault("category", NOT_RECORDED)
    router = trace.get("router")
    if not isinstance(router, dict):
        router = {"role": NOT_RECORDED}
        trace["router"] = router
    router.setdefault("role", NOT_RECORDED)
    citation_check = trace.get("citation_check")
    if not isinstance(citation_check, dict):
        trace["citation_check"] = {"result": NOT_RECORDED}
    else:
        citation_check.setdefault("result", NOT_RECORDED)
    return trace


def load_traces(directory: str | Path | None = None) -> list[dict]:
    """Load committed trace files; invalid files are skipped."""
    root = Path(directory) if directory else TRACE_DIR
    traces = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or "schema_version" not in payload:
            continue
        payload.setdefault("file", str(path.name))
        traces.append(normalise_trace(payload))
    return traces


def trace_for_question(question: str, traces: Sequence[dict] | None = None) -> dict | None:
    for trace in traces if traces is not None else load_traces():
        if trace.get("question") == question:
            return trace
    return None


def trace_result(trace: dict | None) -> dict:
    if trace is None:
        return {
            "answer": "No recorded trace for this question in demo mode.",
            "trace": None,
            "geometry": None,
            "case": None,
        }
    return {
        "answer": trace.get("answer", NOT_RECORDED),
        "trace": trace,
        "geometry": trace.get("geometry"),
        "case": None,
    }


def demo_trace(case: dict) -> dict:
    """Backwards-compatible wrapper around a recorded result case."""
    trace = trace_from_result_case(
        case,
        source="recorded result case",
        sample=False,
        mock=bool(case.get("mock")),
    )
    return trace_result(trace)


def arm_trace(question: str, arm) -> dict:
    output = arm.answer(question)
    trace = trace_from_arm_output(question, output, source="live A8 run")
    return trace_result(trace)


def live_trace(question: str, graph_chain=None) -> dict:
    """Run the graph once and return a minimally populated trace.

    Structured chunks and rerank scores are only available through
    ``scripts/record_trace.py``; this function records what the graph returns
    and leaves everything else explicitly unrecorded.
    """
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
        "schema_version": TRACE_SCHEMA_VERSION,
        "source": "live graph run",
        "sample": False,
        "mock": False,
        "question": question,
        "answer": answer,
        "guardrail": {
            "verdict": "blocked" if result.get("is_blocked") else "passed",
            "category": NOT_RECORDED,
        },
        "router": {"role": _recorded(result.get("role"))},
        "retrieval": {
            "calls": result.get("rag_calls", 0),
            "candidates": NOT_RECORDED,
            "selected": NOT_RECORDED,
        },
        "chunks": NOT_RECORDED,
        "citations": NOT_RECORDED,
        "citation_check": {"result": NOT_RECORDED},
        "stage_latencies": {"total_s": elapsed},
        "tokens": NOT_RECORDED,
        "cost_usd": NOT_RECORDED,
    }
    return trace_result(normalise_trace(trace))
