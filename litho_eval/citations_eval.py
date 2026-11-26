"""Citation metrics and the A8 vs A5 comparison."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from litho_eval.stats import paired_bootstrap_ci
from src.core.citations import CitationAnswer

SCHEMA_VERSION = "1.0"
REPO_ROOT = Path(__file__).resolve().parents[1]


def citation_summary(answers: Sequence[CitationAnswer]) -> dict:
    if not answers:
        return {
            "citation_precision": None,
            "citation_coverage": None,
            "abstention_rate": None,
            "n": 0,
        }
    return {
        "citation_precision": sum(
            answer.validation.citation_precision if answer.validation else 0.0
            for answer in answers
        ) / len(answers),
        "citation_coverage": sum(
            answer.validation.coverage if answer.validation else 0.0
            for answer in answers
        ) / len(answers),
        "abstention_rate": sum(1 for answer in answers if answer.abstained) / len(answers),
        "n": len(answers),
    }


def paired_correctness(a8_correct: Sequence[float], a5_correct: Sequence[float]) -> dict:
    if len(a8_correct) != len(a5_correct):
        raise ValueError("A8 and A5 correctness must be paired")
    delta = (
        sum(a8_correct) / len(a8_correct) - sum(a5_correct) / len(a5_correct)
        if a8_correct
        else None
    )
    return {
        "n": len(a8_correct),
        "a8": (sum(a8_correct) / len(a8_correct)) if a8_correct else None,
        "a5": (sum(a5_correct) / len(a5_correct)) if a5_correct else None,
        "delta": delta,
        "ci95": (
            list(paired_bootstrap_ci(list(a8_correct), list(a5_correct)))
            if len(a8_correct) > 1
            else None
        ),
    }


def blocked(reason: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "reason": reason,
        "a8_vs_a5": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def mock_payload() -> dict:
    from src.core.citations import CitationAnswer, CitationValidation

    answers = []
    for index in range(10):
        valid = index < 8
        answers.append(
            CitationAnswer(
                text="mock answer [C1]" if valid else "mock answer",
                citations=[],
                abstained=index == 9,
                validation=CitationValidation(
                    ok=valid,
                    citation_precision=1.0 if valid else 0.0,
                    coverage=1.0 if valid else 0.0,
                ),
            )
        )
    a8 = [1.0] * 9 + [0.0]
    a5 = [1.0] * 8 + [0.0, 0.0]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "mock",
        "mock": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "citations": citation_summary(answers),
        "a8_vs_a5": paired_correctness(a8, a5),
    }


def citation_metrics_from_cases(cases: Sequence[dict]) -> dict:
    validations = [
        case.get("citation_validation")
        for case in cases
        if case.get("citation_validation")
    ]
    if not validations:
        return {
            "citation_precision": None,
            "citation_coverage": None,
            "abstention_rate": None,
            "n": 0,
        }
    return {
        "citation_precision": sum(
            item.get("citation_precision", 0.0) for item in validations
        ) / len(validations),
        "citation_coverage": sum(item.get("coverage", 0.0) for item in validations)
        / len(validations),
        "abstention_rate": sum(1 for item in validations if item.get("abstained"))
        / len(validations),
        "n": len(validations),
    }


def _load_result(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate citation enforcement")
    parser.add_argument("--out", default="results/citations.json")
    parser.add_argument("--a5", default="results/A5.json")
    parser.add_argument("--a8", default="results/A8.json")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.mock:
        payload = mock_payload()
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("MOCK citation run; these numbers are not benchmark results.")
        print(f"wrote {out}")
        return 0

    a5_payload = _load_result(Path(args.a5))
    a8_payload = _load_result(Path(args.a8))
    if not a5_payload or not a8_payload:
        payload = blocked("results/A5.json or results/A8.json is missing")
    elif a5_payload.get("mock") or a8_payload.get("mock"):
        payload = blocked("A5/A8 results are mock; run real arms first")
    else:
        a5_cases = {case["id"]: case for case in a5_payload.get("cases", [])}
        a8_cases = {case["id"]: case for case in a8_payload.get("cases", [])}
        common = sorted(set(a5_cases) & set(a8_cases))
        paired = [
            case_id
            for case_id in common
            if a5_cases[case_id].get("correctness") is not None
        ]
        a5_correct = [
            1.0 if a5_cases[case_id]["correctness"]["correct"] else 0.0
            for case_id in paired
        ]
        a8_correct = [
            1.0 if a8_cases[case_id]["correctness"]["correct"] else 0.0
            for case_id in paired
        ]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "mock": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "citations": citation_metrics_from_cases(
                [a8_cases[case_id] for case_id in common]
            ),
            "a8_vs_a5": paired_correctness(a8_correct, a5_correct),
        }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}: {payload['status']}")
    return 0 if payload.get("status") == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
