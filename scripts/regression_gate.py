"""Fail when a fresh cached run drops a headline metric beyond tolerance.

Generate the fresh file with the normal runner so LLMClient serves cached
responses; this script never calls an API itself.

    python scripts/regression_gate.py --fresh results/A4.json --baseline results/baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def lookup(payload: dict, path: str):
    value = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regression gate")
    parser.add_argument("--fresh", default="results/A4.json")
    parser.add_argument("--baseline", default="results/baseline.json")
    parser.add_argument("--config", default="configs/gate.yaml")
    args = parser.parse_args(argv)

    fresh_path, baseline_path = Path(args.fresh), Path(args.baseline)
    if not fresh_path.exists():
        raise SystemExit(f"fresh results not found: {fresh_path}")
    if not baseline_path.exists():
        raise SystemExit(f"baseline results not found: {baseline_path}")

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    tolerances = config.get("tolerance", {})
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    failures = []
    for metric in config.get("metrics", []):
        fresh_value = lookup(fresh, metric)
        baseline_value = lookup(baseline, metric)
        if fresh_value is None or baseline_value is None:
            failures.append(f"{metric}: missing in fresh or baseline")
            continue
        drop = baseline_value - fresh_value
        tolerance = float(tolerances.get(metric, 0.0))
        status = "OK" if drop <= tolerance else "DROP"
        print(
            f"{status} {metric}: baseline={baseline_value:.4f} "
            f"fresh={fresh_value:.4f} drop={drop:.4f} tolerance={tolerance:.4f}"
        )
        if drop > tolerance:
            failures.append(metric)

    if failures:
        print("regression gate failed: " + ", ".join(failures))
        return 1
    print("regression gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
