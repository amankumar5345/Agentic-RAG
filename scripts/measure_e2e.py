"""Measure end-to-end latency from text prompt to Unreal acknowledgement.

Fake mode uses tools/fake_unreal.py and is the only mode CI can run. Real
Unreal measurements require --real-unreal and a running UE5 project.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from litho_eval.geometry_eval import (
    latency_block,
    load_geometry_cases,
    run_agent_timed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "1.0"


def blocked(reason: str, out: Path, ue_mode: str) -> int:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "reason": reason,
        "ue_mode": ue_mode,
        "speech_to_text_s": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"blocked: {reason}")
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure text/voice to Unreal ack latency")
    parser.add_argument("--real-unreal", action="store_true")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--out", default="results/e2e_latency.json")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)

    if args.mock and args.real_unreal:
        raise SystemExit("choose --mock or --real-unreal, not both")

    ue_mode = "real" if args.real_unreal else "fake"
    out = Path(args.out)

    if args.mock:
        runs = args.runs
        guardrail = [0.01 + index * 0.0001 for index in range(runs)]
        routing = [0.02 + index * 0.0001 for index in range(runs)]
        orchestration = [0.20 + index * 0.001 for index in range(runs)]
        tool = [0.01 + index * 0.0001 for index in range(runs)]
        packet_write = [0.001 for _ in range(runs)]
        ack = [args.delay for _ in range(runs)]
        total = [
            guardrail[index] + routing[index] + orchestration[index]
            + tool[index] + packet_write[index] + ack[index]
            for index in range(runs)
        ]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "mock": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ue_mode": "fake",
            "runs": runs,
            "completed": runs,
            "speech_to_text_used": False,
            "latency": latency_block(
                {
                    "guardrail_s": guardrail,
                    "routing_s": routing,
                    "orchestration_s": orchestration,
                    "tool_call_s": tool,
                    "packet_write_s": packet_write,
                    "ack_s": ack,
                    "total_s": total,
                },
                "fake",
            ),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("MOCK e2e run; these numbers are not benchmark results.")
        print(f"wrote {out}")
        return 0

    cases = load_geometry_cases()
    if not cases:
        return blocked("no verified geometry cases; run bench.review", out, ue_mode)

    from src.core.graph import GraphConfig, build_graph, default_llms
    from src.core.retriever import load_prompt
    from src.core.tools import spawn_geometry
    from src.services.ue_ack import wait_for_ack
    from tools.fake_unreal import process_once, read_packet

    mailbox = Path(os.environ.get("LITHO_MAILBOX_DIR", "./runtime/unreal_mailbox"))
    if not args.real_unreal:
        temp = tempfile.TemporaryDirectory()
        mailbox = Path(temp.name)
        os.environ["LITHO_MAILBOX_DIR"] = str(mailbox)

    config = GraphConfig()
    tools = [load_prompt, spawn_geometry]
    llms = default_llms(config, tools)
    llms["evaluator"] = lambda state: {"eval_results": {}}
    graph = build_graph(config, llms=llms).compile()

    stage_values: dict[str, list[float]] = defaultdict(list)
    completed = 0

    for index in range(args.runs):
        case = cases[index % len(cases)]
        started = time.perf_counter()
        _, timings, _ = run_agent_timed(graph, case.utterance)

        packet = read_packet(mailbox)
        ack_s = None
        if packet and packet.get("command_id"):
            if not args.real_unreal:
                process_once(mailbox, args.delay)
            ack_started = time.perf_counter()
            ack = wait_for_ack(
                packet["command_id"],
                timeout_s=10.0,
                mailbox_dir=mailbox,
                poll_interval_s=0.05,
            )
            ack_s = time.perf_counter() - ack_started
            if ack is not None:
                completed += 1

        total_s = time.perf_counter() - started
        stages = {
            "speech_to_text_s": None,
            "guardrail_s": timings.get("guardrail", 0.0),
            "routing_s": timings.get("role_decide", 0.0),
            "orchestration_s": timings.get("orchestrator", 0.0),
            "tool_call_s": timings.get("tools", 0.0),
            "packet_write_s": 0.0,
            "ack_s": ack_s or 0.0,
            "total_s": total_s,
        }
        for stage, value in stages.items():
            if value is not None:
                stage_values[stage].append(value)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ue_mode": ue_mode,
        "runs": args.runs,
        "completed": completed,
        "speech_to_text_used": False,
        "latency": latency_block(stage_values, ue_mode),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{ue_mode}: {completed}/{args.runs} acks; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
