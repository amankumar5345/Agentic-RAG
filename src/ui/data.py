"""Cached result and dataset loaders with schema checks."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
NOT_RUN = "NOT YET RUN"

RESULT_FILES = {
    "comparison": "comparison.json",
    "guardrails": "guardrails.json",
    "geometry": "geometry.json",
    "e2e_latency": "e2e_latency.json",
    "injection": "injection.json",
    "citations": "citations.json",
    "judge_validation": "judge_validation.json",
}


@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float) -> dict | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if "schema_version" not in payload:
        return None
    return payload


def load_result(name: str) -> dict | None:
    filename = RESULT_FILES.get(name, name)
    path = RESULTS / filename
    if not path.exists():
        return None
    return _read_json(str(path), path.stat().st_mtime)


def load_arm(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        return None
    return _read_json(str(path), path.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _read_text(path: str, mtime: float) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""


def load_doc(relative: str) -> str:
    path = REPO_ROOT / relative
    if not path.exists():
        return ""
    return _read_text(str(path), path.stat().st_mtime)


def is_mock(payload: dict | None) -> bool:
    return bool(
        payload
        and (payload.get("mock") is True or payload.get("status") == "mock")
    )


def status_label(payload: dict | None) -> str:
    if payload is None:
        return NOT_RUN
    if payload.get("status") == "blocked":
        return NOT_RUN
    if payload.get("mock") or payload.get("status") == "mock":
        return "MOCK"
    return "OK"


@st.cache_data(show_spinner=False)
def _cases(mtime: float) -> list[dict]:
    from bench.schema import read_cases

    return [case.model_dump(mode="json") for case in read_cases()]


def load_cases() -> list[dict]:
    path = REPO_ROOT / "bench" / "data"
    mtimes = [p.stat().st_mtime for p in path.glob("*.jsonl") if p.exists()]
    return _cases(max(mtimes) if mtimes else 0.0)


@st.cache_data(show_spinner=False)
def _injection_cases(mtime: float) -> list[dict]:
    from bench.injection.cases import read_injection_cases

    return [case.model_dump(mode="json") for case in read_injection_cases()]


def load_injection_cases() -> list[dict]:
    path = REPO_ROOT / "bench" / "injection" / "cases.jsonl"
    return _injection_cases(path.stat().st_mtime if path.exists() else 0.0)


def difficulty_view(case: dict) -> tuple[str, str]:
    """Return (value, source) for the UI.

    Stored values win. Unset cases may show a heuristic bucket, explicitly
    tagged as derived; the value is never written back to the dataset and is
    never used in README, resume numbers or benchmark metrics.
    """
    stored = case.get("difficulty")
    if stored in {"easy", "medium", "hard"}:
        return stored, "stored"
    subtype = case.get("subtype") or ""
    if subtype in {"multi_hop", "numeric"}:
        return "hard", "derived (heuristic)"
    if subtype == "single_hop":
        return "medium", "derived (heuristic)"
    return "not set", "not set"


def derived_difficulty(case: dict) -> str:
    """Backward-compatible bucket value; see ``difficulty_view``."""
    return difficulty_view(case)[0]


def gold_text(case: dict) -> str:
    sources = case.get("gold_sources") or []
    if not sources:
        return ""
    corpus_path = REPO_ROOT / "our_database" / "all_contents.txt"
    if not corpus_path.exists():
        return ""
    corpus = corpus_path.read_text(encoding="utf-8", errors="replace")
    return "\n\n".join(
        corpus[source.get("char_start", 0):source.get("char_end", 0)]
        for source in sources
    )


def results_manifest() -> list[dict]:
    rows = []
    for name in RESULT_FILES:
        payload = load_result(name)
        rows.append({"result": name, "status": status_label(payload), "mock": is_mock(payload)})
    return rows


@st.cache_resource(show_spinner=False)
def get_chain():
    from src.core.graph import chain

    return chain


@st.cache_resource(show_spinner=False)
def get_index_bundle(
    corpus_path: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
):
    from litho_eval.index import get_index

    return get_index(corpus_path, chunk_size, chunk_overlap, embedding_model)


@st.cache_resource(show_spinner=False)
def get_live_arm(arm: str):
    from litho_eval.arms import load_arms
    from litho_eval.runner import (
        DEFAULT_ARMS_CONFIG,
        DEFAULT_CORPUS,
        DEFAULT_DATA_DIR,
        DEFAULT_JUDGE_PROMPT,
        DEFAULT_PRICES,
        RunConfig,
        build_real_arm,
    )

    arms = load_arms(DEFAULT_ARMS_CONFIG)
    spec = arms.arms[arm]
    index = arms.index or {}
    config = RunConfig(
        arm=arm,
        split="test",
        generator_model=arms.generator_model,
        judge_model=arms.judge_model,
        top_k=spec.top_k,
        temperature=0.0,
        seed=arms.seed,
        limit=None,
        mock=False,
        data_dir=str(DEFAULT_DATA_DIR),
        corpus_path=str(DEFAULT_CORPUS),
        prices_path=str(DEFAULT_PRICES),
        judge_prompt_path=str(DEFAULT_JUDGE_PROMPT),
        arms_config_path=str(DEFAULT_ARMS_CONFIG),
        chunk_size=int(index.get("chunk_size", 1000)),
        chunk_overlap=int(index.get("chunk_overlap", 150)),
        embedding_model=str(index.get("embedding_model", "bge-m3:latest")),
    )
    return build_real_arm(config, spec, str(REPO_ROOT / ".cache" / "litho_eval"))

