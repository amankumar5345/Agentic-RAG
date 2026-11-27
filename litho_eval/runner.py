"""Run one LithoBench arm and write a reproducible results file.

    python -m litho_eval.run --arm A4 --split test --out results/A4.json
    python -m litho_eval.run --arm A4 --mock --limit 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from bench.schema import QACase, case_json, load_cases
from litho_eval.arms import (
    AgentArm,
    ArmOutput,
    CitationAgentArm,
    CitationGenerator,
    CitationJudge,
    Generator,
    build_arm,
    load_arms,
)
from litho_eval.client import DEFAULT_CACHE_DIR, DEFAULT_PRICES, REPO_ROOT, LLMClient
from litho_eval.index import RetrievedChunk, get_index
from litho_eval.metrics import (
    Judge,
    abstention_metrics,
    cost_per_correct,
    faithfulness,
    latency_summary,
    numeric_match,
    retrieval_metrics,
)
from litho_eval.stats import bootstrap_ci, proportion_ci

SCHEMA_VERSION = "1.0"
DEFAULT_JUDGE_PROMPT = REPO_ROOT / "configs" / "judge_prompt.md"
DEFAULT_ARMS_CONFIG = REPO_ROOT / "configs" / "arms.yaml"
DEFAULT_DATA_DIR = REPO_ROOT / "bench" / "data"
DEFAULT_CORPUS = REPO_ROOT / "our_database" / "all_contents.txt"


@dataclass(frozen=True)
class RunConfig:
    arm: str
    split: str
    generator_model: str
    judge_model: str
    top_k: int
    temperature: float
    seed: int
    limit: int | None
    mock: bool
    data_dir: str
    corpus_path: str
    prices_path: str
    judge_prompt_path: str
    arms_config_path: str
    chunk_size: int
    chunk_overlap: int
    embedding_model: str


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def config_dict(config: RunConfig) -> dict:
    return {
        **asdict(config),
        "judge_prompt_sha256": _sha256_file(Path(config.judge_prompt_path)),
        "prices_sha256": _sha256_file(Path(config.prices_path)),
        "arms_config_sha256": _sha256_file(Path(config.arms_config_path)),
    }


def config_hash(config: RunConfig) -> str:
    payload = json.dumps(config_dict(config), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dataset_hash(cases: Sequence[QACase]) -> str:
    payload = "\n".join(case_json(case) for case in sorted(cases, key=lambda c: c.id))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def load_qa_cases(
    data_dir: Path,
    split: str,
    limit: int | None,
    include_draft: bool = False,
) -> list[QACase]:
    cases = [
        case
        for case in load_cases(data_dir=data_dir, include_draft=include_draft)
        if isinstance(case, QACase) and case.split == split
    ]
    cases.sort(key=lambda case: case.id)
    return cases[:limit] if limit else cases


def resolve_gold_texts(case: QACase, corpus: str) -> list[str]:
    return [corpus[source.char_start:source.char_end] for source in case.gold_sources]


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def build_agent_graph(spec, arms_config):
    from src.core.graph import GraphConfig, build_graph, default_llms

    config = GraphConfig(
        orchestrator_model=arms_config.generator_model,
        router_model=arms_config.generator_model,
        judge_model=arms_config.judge_model,
        temperature=0.0,
        rails_temperature=0.0,
        max_rag_calls=3,
    )
    llms = default_llms(config)
    llms["evaluator"] = lambda state: {"eval_results": {}}
    return build_graph(
        config,
        llms=llms,
        enable_guardrail=spec.guardrail,
        enable_router=spec.router,
    ).compile()


def a8_cache_dir(cache_dir: str | Path) -> Path:
    """A8 has its own cache namespace so it cannot reuse A0-A7 entries."""
    return Path(cache_dir) / "A8"


def build_citation_arm(config: RunConfig, cache_dir: str):
    from langchain_core.tools import tool

    from src.core.citations import CITATION_RULE, format_chunks
    from src.core.graph import GraphConfig, build_graph, default_llms
    from src.core.retriever import retrieve_chunks
    from src.core.tools import spawn_geometry

    captured: list = []

    @tool
    def load_prompt(user_input: str) -> str:
        """Retrieve documents and return [C1]..[Cn] chunks with metadata."""
        chunks = retrieve_chunks(user_input)
        captured.clear()
        captured.extend(chunks)
        return f"""Answer using the retrieved context.
{CITATION_RULE}
User question:
{user_input}

Retrieved context:
{format_chunks(chunks)}

Answer the user's question using only the retrieved context."""

    graph_config = GraphConfig(
        orchestrator_model=config.generator_model,
        router_model=config.generator_model,
        judge_model=config.judge_model,
        temperature=0.0,
        rails_temperature=0.0,
        max_rag_calls=3,
    )
    tools = [load_prompt, spawn_geometry]
    llms = default_llms(graph_config, tools)
    llms["evaluator"] = lambda state: {"eval_results": {}}
    graph = build_graph(graph_config, llms=llms, tools=tools).compile()

    generator_client = LLMClient(
        config.generator_model,
        prices_path=config.prices_path,
        cache_dir=str(a8_cache_dir(cache_dir) / "generator"),
        seed=config.seed,
        mock=False,
    )
    judge_client = LLMClient(
        config.judge_model,
        prices_path=config.prices_path,
        cache_dir=str(a8_cache_dir(cache_dir) / "judge"),
        seed=config.seed,
        mock=False,
    )
    return CitationAgentArm(
        "A8",
        AgentArm("A8", graph),
        lambda: captured,
        CitationGenerator(generator_client),
        CitationJudge(judge_client, config.judge_model, config.generator_model),
    )


def build_real_arm(config: RunConfig, spec, cache_dir: str):
    if spec.kind == "agent_citations":
        return build_citation_arm(config, cache_dir)

    if spec.kind == "agent":
        return build_arm(spec, graph=build_agent_graph(spec, load_arms(config.arms_config_path)))

    generator_client = LLMClient(
        config.generator_model,
        prices_path=config.prices_path,
        cache_dir=cache_dir,
        seed=config.seed,
        mock=False,
    )
    generator = Generator(generator_client)

    if spec.kind == "closed_book":
        return build_arm(spec, generator=generator)

    index = get_index(
        config.corpus_path,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        embedding_model=config.embedding_model,
        cache_root=Path(cache_dir) / "indexes",
    )
    return build_arm(spec, generator=generator, index=index)


def mock_output(case: QACase, corpus: str) -> ArmOutput:
    gold_texts = resolve_gold_texts(case, corpus)
    if case.subtype == "unanswerable":
        return ArmOutput(
            answer="Not answerable from the provided corpus.",
            retrieved=[],
            citations=[],
            retrieval_calls=0,
            stage_latencies={"retrieval_s": 0.0, "generation_s": 0.0, "total_s": 0.0},
            cost_usd=0.0,
        )

    chunks = [
        RetrievedChunk(
            id=f"gold-{index}",
            text=text,
            doc=case.gold_sources[index].doc,
            section=case.gold_sources[index].section,
            char_start=case.gold_sources[index].char_start,
            char_end=case.gold_sources[index].char_end,
        )
        for index, text in enumerate(gold_texts)
    ]
    return ArmOutput(
        answer=case.gold_answer,
        retrieved=chunks,
        citations=[
            {
                "doc": chunk.doc,
                "section": chunk.section,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
            }
            for chunk in chunks
        ],
        retrieval_calls=1,
        stage_latencies={"retrieval_s": 0.0, "generation_s": 0.0, "total_s": 0.0},
        cost_usd=0.0,
    )


def evaluate_case(
    case: QACase,
    arm,
    judge: Judge,
    k: int,
    corpus: str,
    mock: bool,
) -> dict:
    gold_texts = resolve_gold_texts(case, corpus)
    output = mock_output(case, corpus) if mock else arm.answer(case.question)

    judge_started = time.perf_counter()
    correctness = None
    faithfulness_payload = None
    judge_responses = []
    retrieval = None

    if case.subtype != "unanswerable":
        retrieval = retrieval_metrics(
            [chunk.text for chunk in output.retrieved],
            gold_texts,
            k=k,
        )

        if case.subtype == "numeric":
            correct = numeric_match(
                output.answer,
                case.gold_answer,
                case.numeric_tolerance,
            )
            correctness = {"correct": correct, "reason": "numeric exact match"}
        else:
            result = judge.correctness(case.question, case.gold_answer, output.answer)
            correctness = {"correct": result.correct, "reason": result.reason}
            if result.response is not None:
                judge_responses.append(result.response)

        faith = faithfulness(
            judge,
            output.answer,
            [chunk.text for chunk in output.retrieved],
        )
        faithfulness_payload = {
            "score": faith.score,
            "claims": faith.claims,
            "supported": faith.supported,
        }
        judge_responses.extend(faith.responses)

    judge_s = time.perf_counter() - judge_started
    prompt_tokens = output.prompt_tokens + sum(
        response.prompt_tokens or 0 for response in judge_responses
    )
    completion_tokens = output.completion_tokens + sum(
        response.completion_tokens or 0 for response in judge_responses
    )
    judge_cost = sum(response.cost_usd for response in judge_responses)
    cost = None if output.cost_usd is None else output.cost_usd + judge_cost

    stage_latencies = dict(output.stage_latencies)
    stage_latencies["judge_s"] = judge_s
    stage_latencies["total_s"] = stage_latencies.get("total_s", 0.0) + judge_s

    return {
        "id": case.id,
        "type": case.type,
        "subtype": case.subtype,
        "split": case.split,
        "input": case.question,
        "output": output.answer,
        "retrieved": [
            {
                "id": chunk.id,
                "text": chunk.text,
                "doc": chunk.doc,
                "section": chunk.section,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "score": chunk.score,
            }
            for chunk in output.retrieved
        ],
        "citations": output.citations,
        "retrieval_calls": output.retrieval_calls,
        "gold_sources": [source.model_dump(mode="json") for source in case.gold_sources],
        "retrieval": retrieval,
        "correctness": correctness,
        "faithfulness": faithfulness_payload,
        "judge_responses": [response.text for response in judge_responses],
        "citation_validation": output.citation_validation,
        "latencies": stage_latencies,
        "tokens": {"prompt": prompt_tokens, "completion": completion_tokens},
        "cost_usd": cost,
        "mock": mock,
    }


def aggregate(records: Sequence[dict], k: int) -> dict:
    answerable = [record for record in records if record["subtype"] != "unanswerable"]
    correctness_values = [
        1.0 if record["correctness"]["correct"] else 0.0
        for record in answerable
        if record["correctness"] is not None
    ]
    faithfulness_values = [
        record["faithfulness"]["score"]
        for record in answerable
        if record["faithfulness"] is not None
    ]
    recall_values = [
        record["retrieval"]["recall_at_k"]
        for record in answerable
        if record["retrieval"] is not None
        and record["retrieval"]["recall_at_k"] is not None
    ]
    mrr_values = [
        record["retrieval"]["mrr"]
        for record in answerable
        if record["retrieval"] is not None and record["retrieval"]["mrr"] is not None
    ]
    hit_values = [
        record["retrieval"]["hit_at_k"]
        for record in answerable
        if record["retrieval"] is not None
        and record["retrieval"]["hit_at_k"] is not None
    ]
    latencies = [record["latencies"]["total_s"] for record in records]

    total_cost = sum(record["cost_usd"] for record in records if record["cost_usd"] is not None)
    cost_tracked = bool(records) and all(record["cost_usd"] is not None for record in records)
    correct_count = sum(1 for value in correctness_values if value == 1.0)

    correctness = _mean(correctness_values)
    correctness_ci = None
    if correctness_values:
        if len(correctness_values) < 50:
            correctness_ci = list(proportion_ci(correct_count, len(correctness_values)))
        elif len(correctness_values) > 1:
            correctness_ci = list(bootstrap_ci(correctness_values))

    faithfulness_mean = _mean(faithfulness_values)
    faithfulness_ci = (
        list(bootstrap_ci(faithfulness_values)) if len(faithfulness_values) > 1 else None
    )
    recall_mean = _mean(recall_values)
    recall_ci = list(bootstrap_ci(recall_values)) if len(recall_values) > 1 else None

    return {
        "n_cases": len(records),
        "n_answerable": len(answerable),
        "n_unanswerable": len(records) - len(answerable),
        "retrieval": {
            "k": k,
            "recall_at_k": recall_mean,
            "recall_at_k_ci95": recall_ci,
            "hit_at_k": _mean(hit_values),
            "mrr": _mean(mrr_values),
        },
        "answer_correctness": correctness,
        "answer_correctness_ci95": correctness_ci,
        "faithfulness": faithfulness_mean,
        "faithfulness_ci95": faithfulness_ci,
        "abstention": abstention_metrics(
            [
                (record["subtype"] == "unanswerable", record["output"])
                for record in records
            ]
        ),
        "latency_s": latency_summary(latencies),
        "cost_usd": {
            "total": total_cost if cost_tracked else None,
            "per_correct_answer": (
                cost_per_correct(total_cost, correct_count) if cost_tracked else None
            ),
            "tracked": cost_tracked,
        },
        "tokens": {
            "prompt": sum(record["tokens"]["prompt"] for record in records),
            "completion": sum(record["tokens"]["completion"] for record in records),
        },
    }


def build_result(
    config: RunConfig,
    cases: Sequence[QACase],
    records: Sequence[dict],
    arm_kind: str,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "arm": config.arm,
        "arm_kind": arm_kind,
        "split": config.split,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "config_hash": config_hash(config),
        "config": config_dict(config),
        "models": {
            "generator": config.generator_model,
            "judge": config.judge_model,
        },
        "seed": config.seed,
        "mock": config.mock,
        "dataset_hash": dataset_hash(cases),
        "aggregates": aggregate(records, config.top_k),
        "cases": list(records),
    }


def run(
    config: RunConfig,
    cases: Sequence[QACase],
    cache_dir: str,
    spec,
    previous: dict | None = None,
) -> list[dict]:
    corpus = ""
    try:
        corpus = Path(config.corpus_path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        pass

    judge_client = LLMClient(
        config.judge_model,
        prices_path=config.prices_path,
        cache_dir=cache_dir,
        seed=config.seed,
        mock=config.mock,
    )
    judge = Judge(
        judge_client,
        config.judge_model,
        config.generator_model,
        config.judge_prompt_path,
    )

    arm = None if config.mock else build_real_arm(config, spec, cache_dir)

    existing = {record["id"]: record for record in (previous or {}).get("cases", [])}
    records: list[dict] = []
    for case in cases:
        if case.id in existing:
            records.append(existing[case.id])
            continue
        records.append(
            evaluate_case(case, arm, judge, config.top_k, corpus, config.mock)
        )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one LithoBench arm")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--generator-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--prices", default=str(DEFAULT_PRICES))
    parser.add_argument("--judge-prompt", default=str(DEFAULT_JUDGE_PROMPT))
    parser.add_argument("--arms-config", default=str(DEFAULT_ARMS_CONFIG))
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="shared disk cache for this run",
    )
    args = parser.parse_args(argv)

    arms_config = load_arms(args.arms_config)
    if args.arm not in arms_config.arms:
        raise SystemExit(f"unknown arm {args.arm!r}; known: {', '.join(arms_config.arms)}")
    spec = arms_config.arms[args.arm]
    if args.top_k is not None:
        spec = replace(spec, top_k=args.top_k)

    index_defaults = arms_config.index or {}
    config = RunConfig(
        arm=args.arm,
        split=args.split,
        generator_model=args.generator_model or arms_config.generator_model,
        judge_model=args.judge_model or arms_config.judge_model,
        top_k=spec.top_k,
        temperature=0.0,
        seed=args.seed,
        limit=args.limit,
        mock=args.mock,
        data_dir=args.data_dir,
        corpus_path=args.corpus,
        prices_path=args.prices,
        judge_prompt_path=args.judge_prompt,
        arms_config_path=args.arms_config,
        chunk_size=args.chunk_size or int(index_defaults.get("chunk_size", 1000)),
        chunk_overlap=args.chunk_overlap or int(index_defaults.get("chunk_overlap", 150)),
        embedding_model=args.embedding_model
        or str(index_defaults.get("embedding_model", "bge-m3:latest")),
    )

    out_path = Path(args.out) if args.out else REPO_ROOT / "results" / f"{args.arm}.json"
    cases = load_qa_cases(
        Path(config.data_dir), config.split, config.limit, include_draft=config.mock
    )
    if not cases:
        raise SystemExit(
            f"no verified QA cases for split {config.split!r} in {config.data_dir}; "
            "draft and review the dataset first"
        )

    previous = None
    if args.resume and out_path.exists():
        previous = json.loads(out_path.read_text(encoding="utf-8"))
        if previous.get("config_hash") != config_hash(config):
            raise SystemExit("cannot resume: config hash differs from the existing results file")
        if previous.get("dataset_hash") != dataset_hash(cases):
            raise SystemExit("cannot resume: dataset hash differs")

    records = run(config, cases, args.cache_dir, spec, previous)
    result = build_result(config, cases, records, "mock" if config.mock else spec.kind)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp = out_path.with_suffix(out_path.suffix + ".tmp")
    temp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(temp, out_path)

    aggregates = result["aggregates"]
    print(
        f"{config.arm}: {aggregates['n_cases']} cases, "
        f"correctness={aggregates['answer_correctness']}, "
        f"faithfulness={aggregates['faithfulness']}, "
        f"recall@k={aggregates['retrieval']['recall_at_k']}"
    )
    print(f"wrote {out_path}")
    return 0
