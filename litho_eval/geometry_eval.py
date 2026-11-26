"""Evaluate the text-to-geometry bridge through the A5 agent."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from bench.schema import GeometryCase, QACase, load_cases
from litho_eval.stats import bootstrap_ci
from src.core.geometry import replay, validate_commands, write_packet

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "1.0"

SPAWN_STUB_DOC = """Dispatch drawing commands to Unreal Engine.
Args:
    commands: list of {shape, size, angle} dictionaries.
"""


def make_spawn_stub(captured: list):
    from langchain_core.tools import tool

    @tool
    def spawn_geometry(commands: list[dict]) -> str:
        """Dispatch drawing commands to Unreal Engine (test stub, captures only)."""
        captured.append(commands)
        if any(command.get("size") is None for command in commands):
            return json.dumps({"status": "needs_clarification", "reason": "missing_size"})
        try:
            validate_commands(commands)
        except Exception as exc:
            return json.dumps({"status": "invalid", "error": str(exc)})
        return json.dumps({"status": "ok"})

    return spawn_geometry


def make_load_prompt_stub():
    from langchain_core.tools import tool

    @tool
    def load_prompt(user_input: str) -> str:
        """Disabled retrieval stub for over-trigger measurement."""
        return "N/A"

    return load_prompt


def run_agent_timed(graph, question: str) -> tuple[str, dict[str, float], list]:
    """Run a compiled graph and return answer, per-node seconds and messages."""
    from langchain_core.messages import AIMessage, HumanMessage

    state = {
        "messages": [HumanMessage(content=question)],
        "role": "",
        "is_blocked": False,
        "eval_results": None,
        "rag_calls": 0,
    }
    timings: dict[str, float] = defaultdict(float)
    messages = []
    last = time.perf_counter()

    for update in graph.stream(state, stream_mode="updates"):
        now = time.perf_counter()
        for node, payload in update.items():
            timings[node] += now - last
            last = now
            if isinstance(payload, dict) and "messages" in payload:
                messages.extend(payload["messages"])

    answer = ""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            answer = message.content or ""
            break
    return answer, dict(timings), messages


def normalise(commands: Sequence[dict]) -> list[dict]:
    return [
        {
            "shape": str(command.get("shape")),
            "size": round(float(command["size"]), 6) if command.get("size") is not None else None,
            "angle": round(float(command.get("angle", 0.0)) % 360.0, 6),
        }
        for command in commands
    ]


def commands_equal(prediction: Sequence[dict], gold: Sequence[dict]) -> bool:
    return normalise(prediction) == normalise(gold)


def replay_distance(prediction: Sequence[dict], gold: Sequence[dict]) -> float | None:
    try:
        predicted = replay(prediction)
        expected = replay(gold)
    except Exception:
        return None

    if len(predicted["vertices"]) != len(expected["vertices"]):
        return math.inf
    if len(predicted["circles"]) != len(expected["circles"]):
        return math.inf

    distances = [
        math.dist(pred, gold_point)
        for pred, gold_point in zip(predicted["vertices"], expected["vertices"])
    ]
    for pred, expected_circle in zip(predicted["circles"], expected["circles"]):
        distances.append(math.dist(pred["center"], expected_circle["center"]))
        distances.append(abs(pred["radius"] - expected_circle["radius"]))
    return max(distances, default=0.0)


def asks_for_size(text: str) -> bool:
    return bool(
        re.search(
            r"\b(size|how large|how big|dimension|length|radius|measurement)\b",
            text or "",
            re.IGNORECASE,
        )
    )


def metric_block(values: Sequence[float]) -> dict:
    if not values:
        return {"rate": None, "ci95": None, "n": 0}
    return {
        "rate": sum(values) / len(values),
        "ci95": list(bootstrap_ci(values)) if len(values) > 1 else None,
        "n": len(values),
    }


def summarise(case_results: Sequence[dict], over_trigger_values: Sequence[float] | None = None) -> dict:
    drawable = [result for result in case_results if not result["expect_clarification"]]
    clarification = [result for result in case_results if result["expect_clarification"]]

    schema = [1.0 if result["schema_valid"] else 0.0 for result in drawable]
    exact = [1.0 if result["exact_match"] else 0.0 for result in drawable]
    geometric = [1.0 if result["geometric_match"] else 0.0 for result in drawable]
    clarify = [1.0 if result["clarification_correct"] else 0.0 for result in clarification]

    over_values = list(over_trigger_values or [])

    return {
        "schema_valid": metric_block(schema),
        "exact_match": metric_block(exact),
        "geometric_match": metric_block(geometric),
        "clarification_correctness": metric_block(clarify),
        "over_trigger": metric_block(over_values),
        "n_cases": len(case_results),
        "n_drawable": len(drawable),
        "n_clarification": len(clarification),
    }


def by_category(case_results: Sequence[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for result in case_results:
        groups[result["category"]].append(result)
    return {category: summarise(results) for category, results in sorted(groups.items())}


def latency_block(stage_values: dict[str, list[float]], ue_mode: str) -> dict:
    from litho_eval.metrics import latency_summary

    return {
        "ue_mode": ue_mode,
        "stages": {stage: latency_summary(values) for stage, values in stage_values.items()},
        "total": latency_summary(stage_values.get("total_s", [])),
    }


def blocked_payload(reason: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "reason": reason,
        "ue_mode": "fake",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def load_geometry_cases(include_draft: bool = False) -> list[GeometryCase]:
    return [
        case
        for case in load_cases(include_draft=include_draft)
        if isinstance(case, GeometryCase)
    ]


def _packet_id() -> str:
    return str(uuid.uuid4())


def write_failures(failures: Sequence[dict], path: str | Path, mock: bool = False) -> None:
    lines = ["# Geometry failures", ""]
    if mock:
        lines += ["_MOCK run; not benchmark data._", ""]
    lines += [
        "| Input | Gold | Prediction | Replay diff |",
        "| :--- | :--- | :--- | ---: |",
    ]
    for result in failures[:10]:
        lines.append(
            f"| {result['input']} | `{json.dumps(result['gold'])}` | "
            f"`{json.dumps(result['prediction'])}` | {result['replay_distance']} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def mock_case_results(cases: Sequence[GeometryCase]) -> tuple[list[dict], dict[str, list[float]], list[dict]]:
    """Deterministic, network-free mock predictions for a smoke run."""
    results = []
    stage_values: dict[str, list[float]] = defaultdict(list)
    failures = []

    for index, case in enumerate(cases):
        if case.expect_clarification:
            packet = None
            answer = "What size should the shape be?"
            clarification = True
            schema_valid = exact = geometric = False
            distance = None
        else:
            packet = copy.deepcopy(case.gold_commands)
            if index % 3 == 0:
                packet[0]["size"] = float(packet[0]["size"]) * 1.5 + 5.0
            answer = "Drawing now."
            clarification = False
            try:
                validate_commands(packet)
                schema_valid = True
            except Exception:
                schema_valid = False
            exact = commands_equal(packet, case.gold_commands)
            distance = replay_distance(packet, case.gold_commands)
            geometric = bool(
                distance is not None and distance <= case.tolerance
            )

        base = 0.001 * (index + 1)
        stages = {
            "guardrail_s": base,
            "routing_s": base,
            "orchestration_s": base,
            "tool_call_s": base,
            "packet_write_s": 0.0,
            "ack_s": 0.0,
            "speech_to_text_s": None,
            "total_s": base * 4,
        }
        for stage, value in stages.items():
            if value is not None:
                stage_values[stage].append(value)

        result = {
            "id": case.id,
            "category": case.category,
            "input": case.utterance,
            "gold": case.gold_commands,
            "prediction": packet,
            "expect_clarification": case.expect_clarification,
            "schema_valid": schema_valid,
            "exact_match": exact,
            "geometric_match": geometric,
            "replay_distance": (
                None if distance is None or math.isinf(distance) else distance
            ),
            "clarification_correct": clarification,
            "over_trigger": None,
            "answer": answer,
            "latency": stages,
        }
        results.append(result)
        if (
            (packet is None and not case.expect_clarification)
            or (packet is not None and not geometric)
            or (case.expect_clarification and not clarification)
        ):
            failures.append(result)

    failures.sort(
        key=lambda result: (
            result["replay_distance"]
            if result["replay_distance"] is not None
            else math.inf
        ),
        reverse=True,
    )
    return results, stage_values, failures


def run_mock(cases: Sequence[GeometryCase], out: Path, failures_path: Path) -> int:
    results, stage_values, failures = mock_case_results(cases)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "mock": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ue_mode": "fake",
        "metrics": summarise(results, [0.0]),
        "by_category": by_category(results),
        "latency": latency_block(stage_values, "fake"),
        "cases": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_failures(failures, failures_path, mock=True)
    print("MOCK geometry run; these numbers are not benchmark results.")
    print(json.dumps(payload["metrics"], indent=2))
    print(f"wrote {out} and {failures_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the geometry bridge")
    parser.add_argument("--out", default="results/geometry.json")
    parser.add_argument("--failures", default="results/geometry_failures.md")
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--include-draft", action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)

    cases = load_geometry_cases(args.include_draft or args.mock)
    if not cases:
        payload = blocked_payload("no verified geometry cases; run bench.review")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        Path(args.failures).write_text(
            "# Geometry failures\n\nNot run: there are no verified geometry cases.\n",
            encoding="utf-8",
        )
        print("blocked: no verified geometry cases")
        return 2

    if args.mock:
        return run_mock(cases, Path(args.out), Path(args.failures))

    from src.core.graph import GraphConfig, build_graph, default_llms
    from src.core.retriever import load_prompt
    from src.services.ue_ack import wait_for_ack
    from tools.fake_unreal import process_once

    captured: list = []
    stub = make_spawn_stub(captured)
    graph_config = GraphConfig()
    tools = [load_prompt, stub]
    llms = default_llms(graph_config, tools)
    llms["evaluator"] = lambda state: {"eval_results": {}}
    graph = build_graph(graph_config, llms=llms, tools=tools).compile()

    case_results = []
    stage_values: dict[str, list[float]] = defaultdict(list)
    failures = []

    for case in cases:
        captured.clear()
        started = time.perf_counter()
        answer, timings, _ = run_agent_timed(graph, case.utterance)
        total_s = time.perf_counter() - started
        packet = captured[-1] if captured else None

        schema_valid = False
        if packet is not None:
            try:
                validate_commands(packet)
                schema_valid = True
            except Exception:
                schema_valid = False

        exact = bool(packet is not None and commands_equal(packet, case.gold_commands))
        distance = replay_distance(packet, case.gold_commands) if packet is not None else None
        geometric = bool(
            distance is not None
            and distance <= case.tolerance
            and not case.expect_clarification
        )

        clarification_correct = False
        if case.expect_clarification:
            drew = bool(packet is not None and all(c.get("size") is not None for c in packet))
            clarification_correct = asks_for_size(answer) and not drew

        packet_write_s = None
        ack_s = None
        if packet is not None and not case.expect_clarification:
            with tempfile.TemporaryDirectory() as mailbox:
                command_id = _packet_id()
                write_started = time.perf_counter()
                write_packet(
                    {"schema_version": "1.0", "command_id": command_id,
                     "created_at": datetime.now(timezone.utc).isoformat(),
                     "commands": packet},
                    mailbox,
                )
                packet_write_s = time.perf_counter() - write_started
                process_once(mailbox, args.delay)
                ack_started = time.perf_counter()
                wait_for_ack(command_id, timeout_s=2.0, mailbox_dir=mailbox, poll_interval_s=0.01)
                ack_s = time.perf_counter() - ack_started

        stages = {
            "guardrail_s": timings.get("guardrail", 0.0) if "guardrail" in timings else 0.0,
            "routing_s": timings.get("role_decide", 0.0) if "role_decide" in timings else 0.0,
            "orchestration_s": timings.get("orchestrator", 0.0) if "orchestrator" in timings else 0.0,
            "tool_call_s": timings.get("tools", 0.0) if "tools" in timings else 0.0,
            "packet_write_s": packet_write_s or 0.0,
            "ack_s": ack_s or 0.0,
            "speech_to_text_s": None,
            "total_s": total_s,
        }
        for stage, value in stages.items():
            if value is not None:
                stage_values[stage].append(value)

        result = {
            "id": case.id,
            "category": case.category,
            "input": case.utterance,
            "gold": case.gold_commands,
            "prediction": packet,
            "expect_clarification": case.expect_clarification,
            "schema_valid": schema_valid,
            "exact_match": exact,
            "geometric_match": geometric,
            "replay_distance": (
                None if distance is None or math.isinf(distance) else distance
            ),
            "clarification_correct": clarification_correct,
            "over_trigger": None,
            "answer": answer,
            "latency": stages,
        }
        case_results.append(result)
        if (
            (packet is None and not case.expect_clarification)
            or (packet is not None and (not geometric or not schema_valid))
            or (case.expect_clarification and not clarification_correct)
        ):
            failures.append(result)

    failures.sort(
        key=lambda result: (
            result["replay_distance"]
            if result["replay_distance"] is not None
            else math.inf
        ),
        reverse=True,
    )

    over_values: list[float] = []
    over_cases: list[dict] = []
    qa_cases = [case for case in load_cases() if isinstance(case, QACase)]
    if qa_cases:
        over_captured: list = []
        over_tools = [make_load_prompt_stub(), make_spawn_stub(over_captured)]
        over_llms = default_llms(graph_config, over_tools)
        over_llms["evaluator"] = lambda state: {"eval_results": {}}
        over_graph = build_graph(
            graph_config,
            llms=over_llms,
            tools=over_tools,
        ).compile()
        for case in qa_cases:
            over_captured.clear()
            run_agent_timed(over_graph, case.question)
            triggered = bool(over_captured)
            over_values.append(1.0 if triggered else 0.0)
            over_cases.append({"id": case.id, "over_trigger": triggered})

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ue_mode": "fake",
        "metrics": summarise(case_results, over_values),
        "by_category": by_category(case_results),
        "latency": latency_block(stage_values, "fake"),
        "over_trigger_cases": over_cases,
        "cases": case_results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# Geometry failures", "", "| Input | Gold | Prediction | Replay diff |", "| :--- | :--- | :--- | ---: |"]
    for result in failures[:10]:
        lines.append(
            f"| {result['input']} | `{json.dumps(result['gold'])}` | "
            f"`{json.dumps(result['prediction'])}` | {result['replay_distance']} |"
        )
    Path(args.failures).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload["metrics"], indent=2))
    print(f"wrote {out} and {args.failures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
