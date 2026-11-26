"""Export a stratified judge-validation sample without the judge label."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_cases(results_path: Path) -> list[dict]:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    if payload.get("mock") is True:
        raise SystemExit("mock results cannot be used for judge validation")
    return [
        case
        for case in payload.get("cases", [])
        if case.get("type") == "qa"
        and case.get("subtype") != "unanswerable"
        and case.get("correctness") is not None
    ]


def stratified_sample(cases: list[dict], count: int, seed: int = 42) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        groups[case.get("subtype", "unknown")].append(case)
    if len(cases) < count:
        raise SystemExit(f"need {count} answerable cases, found {len(cases)}")

    rng = random.Random(seed)
    selected = []
    total = len(cases)
    for subtype, group in sorted(groups.items()):
        take = max(1, round(count * len(group) / total))
        rng.shuffle(group)
        selected.extend(group[:take])
    selected = selected[:count]
    if len(selected) < count:
        remaining = [case for case in cases if case not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
    return selected


def write_sample(cases: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "subtype", "question", "context", "answer", "human_label"],
        )
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "case_id": case["id"],
                    "subtype": case["subtype"],
                    "question": case["input"],
                    "context": "\n\n".join(item["text"] for item in case.get("retrieved", [])),
                    "answer": case["output"],
                    "human_label": "",
                }
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a judge sample")
    parser.add_argument("--results", default=str(REPO_ROOT / "results" / "A5.json"))
    parser.add_argument("--out", default=str(REPO_ROOT / "bench" / "judge_sample.csv"))
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    cases = load_cases(Path(args.results))
    sample = stratified_sample(cases, args.n, args.seed)
    write_sample(sample, Path(args.out))
    print(f"wrote {len(sample)} rows to {args.out}; fill human_label by hand")
    return 0


if __name__ == "__main__":
    sys.exit(main())
