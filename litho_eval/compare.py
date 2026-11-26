"""Compare every results/<arm>.json against a baseline arm.

    python -m litho_eval.compare --baseline A0
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from litho_eval.stats import paired_bootstrap_ci

SCHEMA_VERSION = "1.0"
METRICS = {
    "answer_correctness": lambda record: (
        None
        if record.get("correctness") is None
        else 1.0 if record["correctness"]["correct"] else 0.0
    ),
    "faithfulness": lambda record: (
        None
        if record.get("faithfulness") is None
        else record["faithfulness"]["score"]
    ),
    "recall_at_k": lambda record: (
        None
        if record.get("retrieval") is None
        else record["retrieval"]["recall_at_k"]
    ),
    "hit_at_k": lambda record: (
        None
        if record.get("retrieval") is None
        else record["retrieval"]["hit_at_k"]
    ),
    "mrr": lambda record: (
        None if record.get("retrieval") is None else record["retrieval"]["mrr"]
    ),
    "latency_s": lambda record: record["latencies"]["total_s"],
    "cost_usd": lambda record: record["cost_usd"],
}


def load_results(results_dir: Path) -> dict[str, dict]:
    results = {}
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "comparison.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "arm" not in payload:
            continue
        results[payload["arm"]] = payload
    return results


def paired_metric(
    arm_cases: dict[str, dict],
    baseline_cases: dict[str, dict],
    extractor,
) -> dict:
    arm_values = []
    baseline_values = []
    for case_id in sorted(set(arm_cases) & set(baseline_cases)):
        arm_value = extractor(arm_cases[case_id])
        baseline_value = extractor(baseline_cases[case_id])
        if arm_value is None or baseline_value is None:
            continue
        arm_values.append(float(arm_value))
        baseline_values.append(float(baseline_value))

    if not arm_values:
        return {"n": 0, "arm": None, "baseline": None, "delta": None, "ci95": None}

    arm_mean = sum(arm_values) / len(arm_values)
    baseline_mean = sum(baseline_values) / len(baseline_values)
    delta = arm_mean - baseline_mean
    ci = (
        list(paired_bootstrap_ci(arm_values, baseline_values))
        if len(arm_values) > 1
        else None
    )
    return {
        "n": len(arm_values),
        "arm": arm_mean,
        "baseline": baseline_mean,
        "delta": delta,
        "ci95": ci,
    }


def compare(results: dict[str, dict], baseline_name: str) -> dict:
    if baseline_name not in results:
        raise SystemExit(f"baseline arm {baseline_name!r} has no results file")

    baseline = results[baseline_name]
    baseline_cases = {case["id"]: case for case in baseline["cases"]}
    arms = {}

    for name, payload in results.items():
        if name == baseline_name:
            continue
        if payload.get("dataset_hash") != baseline.get("dataset_hash"):
            raise SystemExit(
                f"{name} and {baseline_name} were run on different datasets "
                f"({payload.get('dataset_hash')} != {baseline.get('dataset_hash')})"
            )
        arm_cases = {case["id"]: case for case in payload["cases"]}
        retrieval_arm = payload.get("arm_kind") in {
            "dense",
            "bm25",
            "hybrid",
            "hybrid_rerank",
        }
        metrics = {}
        for metric, extractor in METRICS.items():
            if metric in {"recall_at_k", "hit_at_k", "mrr"} and not retrieval_arm:
                continue
            metrics[metric] = paired_metric(arm_cases, baseline_cases, extractor)
        arms[name] = {
            "mock": payload.get("mock", False),
            "metrics": metrics,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "baseline": baseline_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_hash": baseline.get("dataset_hash"),
        "split": baseline.get("split"),
        "mock": any(payload.get("mock", False) for payload in results.values()),
        "arms": arms,
    }


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def markdown(payload: dict) -> str:
    lines = [
        "# LithoBench arm comparison",
        "",
        f"- Baseline: `{payload['baseline']}`",
        f"- Split: `{payload['split']}`",
        f"- Dataset: `{payload['dataset_hash']}`",
        f"- Generated: {payload['generated_at']}",
        "",
        "_Retrieval metrics are meaningful for A1-A4; A0 and agent arms report n/a._",
        "",
    ]

    for arm, data in payload["arms"].items():
        lines.append(f"## {arm}")
        lines.append("")
        lines.append("| Metric | Baseline | Arm | Delta | 95% CI | n |")
        lines.append("| :--- | ---: | ---: | ---: | :--- | ---: |")
        for metric, values in data["metrics"].items():
            ci = values["ci95"]
            ci_text = "n/a" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            lines.append(
                f"| {metric} | {_format(values['baseline'])} | "
                f"{_format(values['arm'])} | {_format(values['delta'])} | "
                f"{ci_text} | {values['n']} |"
            )
        if data.get("mock"):
            lines.append("")
            lines.append("_Mock run; not a benchmark result._")
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare LithoBench arms")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--baseline", default="A1")
    parser.add_argument("--out-json", default="results/comparison.json")
    parser.add_argument("--out-md", default="results/comparison.md")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    results = load_results(results_dir)
    if not results:
        raise SystemExit(f"no results/*.json files in {results_dir}")

    payload = compare(results, args.baseline)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_md = Path(args.out_md)
    out_md.write_text(markdown(payload), encoding="utf-8")

    print(f"wrote {out_json} and {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
