"""Render README/docs result blocks from results/*.json."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
DOCS = [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").glob("*.md"))]
BLOCK_RE = re.compile(
    r"<!-- RESULTS:([a-z_]+) -->.*?<!-- /RESULTS:\1 -->",
    re.DOTALL,
)
NOT_RUN = "_NOT YET RUN._"


def load(name: str) -> dict | None:
    path = RESULTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def arms_block() -> str:
    comparison = load("comparison.json")
    if comparison is None:
        return NOT_RUN
    for path in RESULTS.glob("A*.json"):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("mock"):
                return NOT_RUN
        except json.JSONDecodeError:
            continue
    rows = []
    for arm, data in comparison.get("arms", {}).items():
        metric = data.get("metrics", {}).get("answer_correctness", {})
        ci = metric.get("ci95")
        rows.append(
            f"| {arm} | {metric.get('delta')} | {ci} | {metric.get('n')} |"
        )
    return "\n".join(
        [
            f"Baseline: `{comparison.get('baseline')}`",
            "",
            "| Arm | Correctness delta | 95% CI | n |",
            "| :--- | ---: | :--- | ---: |",
            *rows,
        ]
    )


def guardrails_block() -> str:
    payload = load("guardrails.json")
    if not payload or payload.get("mock") or payload.get("status") != "ok":
        return NOT_RUN
    rows = [
        f"| {name} | {data.get('asr')} | {data.get('false_refusal')} |"
        for name, data in payload.get("systems", {}).items()
    ]
    return "\n".join(
        ["| System | ASR | False refusal |", "| :--- | ---: | ---: |", *rows]
    )


def geometry_block() -> str:
    payload = load("geometry.json")
    if not payload or payload.get("mock") or payload.get("status") != "ok":
        return NOT_RUN
    metrics = payload.get("metrics", {})
    return "\n".join(
        f"- {name}: {value.get('rate')} (95% CI {value.get('ci95')})"
        for name, value in metrics.items()
    )


def latency_block() -> str:
    payload = load("e2e_latency.json")
    if (
        not payload
        or payload.get("mock")
        or payload.get("status") != "ok"
        or payload.get("ue_mode") != "real"
    ):
        return NOT_RUN
    total = payload.get("latency", {}).get("total", {})
    return f"p50 `{total.get('p50')}` s, p95 `{total.get('p95')}` s, `ue_mode: real`"


def injection_block() -> str:
    payload = load("injection.json")
    if not payload or payload.get("mock") or payload.get("status") != "ok":
        return NOT_RUN
    rows = [
        f"| {name} | {data.get('overall', {}).get('rate')} |"
        for name, data in payload.get("defences", {}).items()
    ]
    return "\n".join(["| Defence | ASR |", "| :--- | ---: |", *rows])


def citations_block() -> str:
    payload = load("citations.json")
    if not payload or payload.get("mock") or payload.get("status") != "ok":
        return NOT_RUN
    return json.dumps(payload.get("citations"), indent=2)


def judge_block() -> str:
    payload = load("judge_validation.json")
    if not payload or payload.get("n", 0) < 50:
        return NOT_RUN
    return (
        f"Agreement `{payload.get('agreement')}`, "
        f"Cohen's kappa `{payload.get('kappa')}` "
        f"(95% CI `{payload.get('kappa_ci95')}`), n=`{payload.get('n')}`"
    )


def limitations_block() -> str:
    path = REPO_ROOT / "docs" / "LIMITATIONS.md"
    if not path.exists():
        return NOT_RUN
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if lines and lines[0].startswith("#"):
        lines = lines[1:]
    return "\n".join(lines).strip()


BLOCKS = {
    "arms": arms_block,
    "guardrails": guardrails_block,
    "geometry": geometry_block,
    "e2e_latency": latency_block,
    "injection": injection_block,
    "citations": citations_block,
    "judge_validation": judge_block,
    "limitations": limitations_block,
}


def render(text: str) -> str:
    def replace(match: re.Match) -> str:
        name = match.group(1)
        body = BLOCKS.get(name, lambda: NOT_RUN)()
        return f"<!-- RESULTS:{name} -->\n{body}\n<!-- /RESULTS:{name} -->"

    return BLOCK_RE.sub(replace, text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build README result blocks")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    changed = []
    for path in DOCS:
        if not path.exists():
            continue
        current = path.read_text(encoding="utf-8")
        if "<!-- RESULTS:" not in current:
            continue
        expected = render(current)
        if args.check:
            if expected != current:
                changed.append(path)
        else:
            path.write_text(expected, encoding="utf-8")

    if args.check and changed:
        print("README results are stale:")
        for path in changed:
            print(f"- {path}")
        return 1
    print("README results are up to date." if args.check else "README results updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
