"""Record a redacted, versioned answer trace.

Tracing is off by default. Use ``--from-results`` to record a trace from an
existing result file (the committed sample was produced this way), or opt in
to a live run with ``LITHO_TRACE=1`` and ``--live``.

Examples:
    python scripts/record_trace.py --from-results results/A5.json --index 0 \
        --out results/traces/sample_trace.json --sample
    LITHO_TRACE=1 python scripts/record_trace.py --live \
        --question "What PPE is required for HF?" --out results/traces/live_trace.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.core.trace import (  # noqa: E402
    redact_trace,
    secret_values,
    trace_from_arm_output,
    trace_from_result_case,
)


def _load_case(path: Path, case_id: str | None, index: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = [case for case in payload.get("cases", []) if case.get("type", "qa") == "qa"]
    if case_id:
        for case in cases:
            if case.get("id") == case_id:
                return case
        raise SystemExit(f"case {case_id!r} not found in {path}")
    if not cases:
        raise SystemExit(f"no QA cases in {path}")
    if index < 0 or index >= len(cases):
        raise SystemExit(f"index {index} out of range (0..{len(cases) - 1})")
    return cases[index]


def _record_from_results(args) -> dict:
    path = Path(args.from_results)
    if not path.exists():
        raise SystemExit(f"result file not found: {path}")
    case = _load_case(path, args.case_id, args.index)
    source = f"{path.as_posix()} ({'mock replay' if case.get('mock') else 'recorded result'})"
    return trace_from_result_case(
        case,
        source=source,
        sample=args.sample,
        mock=bool(case.get("mock")),
    )


def _record_live(args) -> dict:
    if os.environ.get("LITHO_TRACE") != "1":
        raise SystemExit(
            "tracing is off by default; set LITHO_TRACE=1 to record a live trace"
        )
    if not args.question:
        raise SystemExit("--live requires --question")

    from litho_eval.arms import load_arms
    from litho_eval.client import DEFAULT_CACHE_DIR
    from litho_eval.runner import (
        DEFAULT_ARMS_CONFIG,
        DEFAULT_CORPUS,
        DEFAULT_DATA_DIR,
        DEFAULT_JUDGE_PROMPT,
        DEFAULT_PRICES,
        RunConfig,
        build_real_arm,
    )
    from src.core.graph import GraphConfig, default_llms, run_guardrail
    from src.core.retriever import retrieve_with_scores

    arms = load_arms(DEFAULT_ARMS_CONFIG)
    spec = arms.arms["A8"]
    index = arms.index or {}
    config = RunConfig(
        arm="A8",
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

    graph_config = GraphConfig(
        orchestrator_model=arms.generator_model,
        router_model=arms.generator_model,
        judge_model=arms.judge_model,
    )
    decision = run_guardrail(args.question)
    router = default_llms(graph_config)["router"].invoke(args.question)
    arm = build_real_arm(config, spec, str(DEFAULT_CACHE_DIR))
    output = arm.answer(args.question)
    retrieval = retrieve_with_scores(args.question)

    return trace_from_arm_output(
        args.question,
        output,
        guardrail={
            "verdict": "blocked" if decision.blocked else "passed",
            "category": "not recorded",
        },
        router={"role": router.role},
        retrieval={
            "calls": output.retrieval_calls,
            "candidates": retrieval["candidates"],
            "selected": retrieval["selected"],
        },
        source="live A8 run",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record a versioned answer trace")
    parser.add_argument("--from-results", default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--question", default=None)
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "traces" / "sample_trace.json"))
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args(argv)

    if args.live and args.from_results:
        raise SystemExit("choose either --live or --from-results, not both")
    if args.live:
        trace = _record_live(args)
    elif args.from_results:
        trace = _record_from_results(args)
    else:
        raise SystemExit("nothing to record: pass --from-results or --live")

    trace["recorded_at"] = datetime.now(timezone.utc).isoformat()
    trace["redacted"] = True
    trace = redact_trace(trace, secret_values())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(out.suffix + ".tmp")
    temp.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    os.replace(temp, out)
    print(f"wrote {out} ({'sample, ' if args.sample else ''}mock={trace.get('mock')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
