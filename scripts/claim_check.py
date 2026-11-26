"""Fail when README/docs quote a metric that is absent from results/.

Only percentages and decimals in a claim-like context are checked, so Python
versions, dates, code fences and file sizes are ignored.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = [REPO_ROOT / "README.md", REPO_ROOT / "docs"]
RESULTS_DIR = REPO_ROOT / "results"
EXEMPT = {REPO_ROOT / "docs" / "CLAIMS_TO_VERIFY.md"}

CLAIM_WORDS = (
    "accuracy",
    "score",
    "rate",
    "latency",
    "cost",
    "precision",
    "recall",
    "asr",
    "faithfulness",
    "relevancy",
    "correct",
    "refusal",
    "f1",
    "mrr",
    "hit@",
    "p50",
    "p95",
)
VERSION_CONTEXT = re.compile(r"(python|unreal|ue|version|v)\s*$", re.IGNORECASE)


def collect_numbers(value, out: list[float]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        out.append(float(value))
    elif isinstance(value, dict):
        for item in value.values():
            collect_numbers(item, out)
    elif isinstance(value, list):
        for item in value:
            collect_numbers(item, out)


def result_numbers(results_dir: Path) -> set[float]:
    numbers: list[float] = []
    if results_dir.exists():
        for path in results_dir.rglob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if payload.get("mock") is True:
                continue
            collect_numbers(payload, numbers)
    return set(numbers)


def present(value: float, numbers: set[float]) -> bool:
    for number in numbers:
        if abs(number - value) <= 1e-6:
            return True
        if abs(number - value / 100) <= 1e-6:
            return True
        if abs(number * 100 - value) <= 1e-4:
            return True
    return False


def strip_code(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return re.sub(r"`[^`]*`", "", text)


def check_text(path: Path, text: str, numbers: set[float]) -> list[str]:
    errors = []
    in_fence = False
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if raw.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        line = strip_code(raw)
        lowered = line.lower()
        claim_context = any(word in lowered for word in CLAIM_WORDS) or "%" in line

        for match in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)\s*%", line):
            value = float(match.group(1))
            if not present(value, numbers):
                errors.append(f"{path}:{line_no}: {value}% not found in results/")

        if claim_context:
            for match in re.finditer(r"(?<![\w.])(\d+\.\d+)(?![\w.])", line):
                prefix = line[:match.start()].strip()
                if VERSION_CONTEXT.search(prefix[-20:]):
                    continue
                value = float(match.group(1))
                if not present(value, numbers):
                    errors.append(f"{path}:{line_no}: {value} not found in results/")
    return errors


def check(paths: list[Path], results_dir: Path = RESULTS_DIR) -> list[str]:
    numbers = result_numbers(results_dir)
    errors = []
    for path in paths:
        if path.name == "CLAIMS_TO_VERIFY.md" or path in EXEMPT or not path.exists():
            continue
        if path.is_dir():
            for child in sorted(path.rglob("*.md")):
                if child.name == "CLAIMS_TO_VERIFY.md" or child in EXEMPT:
                    continue
                errors += check_text(child, child.read_text(encoding="utf-8"), numbers)
        else:
            errors += check_text(path, path.read_text(encoding="utf-8"), numbers)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check quantified claims against results/")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("paths", nargs="*", default=None)
    args = parser.parse_args(argv)

    paths = [Path(path) for path in args.paths] if args.paths else DEFAULT_PATHS
    errors = check(paths, Path(args.results_dir))
    if errors:
        print("Claims without a results/ source:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("All quantified claims are backed by results/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
