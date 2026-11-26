"""Validate LithoBench: schemas, targets, leakage, duplicates and multi-hop.

    python -m bench.validate                 # verified cases only
    python -m bench.validate --include-draft # smoke only, never for results
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

from bench.schema import (
    ADVERSARIAL_TARGETS,
    DATA_DIR,
    QA_TARGETS,
    ROLE_NAMES,
    TARGETS,
    AdversarialCase,
    GeometryCase,
    QACase,
    RouterCase,
    read_cases,
    source_key,
)


def check_counts(cases, targets=TARGETS) -> list[str]:
    errors: list[str] = []

    qa_counts = Counter(case.subtype for case in cases if isinstance(case, QACase))
    for subtype, target in targets["qa"].items():
        if qa_counts[subtype] != target:
            errors.append(f"QA {subtype}: {qa_counts[subtype]} != {target}")

    adv_counts = Counter(case.category for case in cases if isinstance(case, AdversarialCase))
    for category, target in targets["adversarial"].items():
        if adv_counts[category] != target:
            errors.append(f"adversarial {category}: {adv_counts[category]} != {target}")

    geometry_count = sum(isinstance(case, GeometryCase) for case in cases)
    if geometry_count != targets["geometry"]:
        errors.append(f"geometry: {geometry_count} != {targets['geometry']}")

    router_counts = Counter(case.gold_role for case in cases if isinstance(case, RouterCase))
    for role, target in targets["router"].items():
        if router_counts[role] != target:
            errors.append(f"router {role}: {router_counts[role]} != {target}")

    return errors


def check_split_leakage(cases) -> list[str]:
    passage_splits: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    for case in cases:
        if not isinstance(case, QACase):
            continue
        for source in case.gold_sources:
            passage_splits[source_key(source)].add(case.split)

    return [
        f"split leakage: {doc}:{start}-{end} appears in {sorted(splits)}"
        for (doc, start, end), splits in passage_splits.items()
        if len(splits) > 1
    ]


def check_duplicates(cases) -> list[str]:
    by_text: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        if isinstance(case, QACase):
            text = case.question
        elif isinstance(case, AdversarialCase):
            text = case.prompt
        elif isinstance(case, GeometryCase):
            text = case.utterance
        elif isinstance(case, RouterCase):
            text = case.query
        else:  # pragma: no cover - exhaustive union
            continue
        by_text[" ".join(text.lower().split())].append(case.id)

    id_counts = Counter(case.id for case in cases)
    errors = [
        f"duplicate text: {ids}"
        for ids in by_text.values()
        if len(ids) > 1
    ]
    errors += [
        f"duplicate id: {case_id} appears {count} times"
        for case_id, count in id_counts.items()
        if count > 1
    ]
    return errors


def check_multihop(cases) -> list[str]:
    errors = []
    for case in cases:
        if isinstance(case, QACase) and case.subtype == "multi_hop":
            passages = {source_key(source) for source in case.gold_sources}
            if len(passages) < 2:
                errors.append(f"multi_hop {case.id} cites {len(passages)} passage(s), needs >= 2")
    return errors


def check_numeric(cases) -> list[str]:
    return [
        f"numeric {case.id} has no tolerance"
        for case in cases
        if isinstance(case, QACase)
        and case.subtype == "numeric"
        and case.numeric_tolerance is None
    ]


def check_human_ratio(cases) -> list[str]:
    qa_cases = [case for case in cases if isinstance(case, QACase)]
    if not qa_cases:
        return []
    human = sum(case.origin in {"human", "llm_edited"} for case in qa_cases)
    if human / len(qa_cases) < 0.5:
        return [
            f"QA human/edited ratio {human}/{len(qa_cases)} < 50% "
            "(at least half must be human-written or substantially human-edited)"
        ]
    return []


def validate_cases(cases, targets=TARGETS) -> list[str]:
    errors: list[str] = []
    errors += check_counts(cases, targets)
    errors += check_split_leakage(cases)
    errors += check_duplicates(cases)
    errors += check_multihop(cases)
    errors += check_numeric(cases)
    errors += check_human_ratio(cases)
    return errors


def _groups(cases):
    qa = [case for case in cases if isinstance(case, QACase)]
    for subtype in QA_TARGETS:
        yield f"qa/{subtype}", [case for case in qa if case.subtype == subtype]
    adv = [case for case in cases if isinstance(case, AdversarialCase)]
    for category in ADVERSARIAL_TARGETS:
        yield f"adversarial/{category}", [case for case in adv if case.category == category]
    yield "geometry", [case for case in cases if isinstance(case, GeometryCase)]
    router = [case for case in cases if isinstance(case, RouterCase)]
    for role in ROLE_NAMES:
        yield f"router/{role}", [case for case in router if case.gold_role == role]


def counts_table(cases) -> str:
    lines = [
        "| Group | dev | test | human | llm_draft | llm_edited | draft | verified | total |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, group in _groups(cases):
        origins = Counter(case.origin for case in group)
        statuses = Counter(case.status for case in group)
        lines.append(
            f"| {name} | "
            f"{sum(case.split == 'dev' for case in group)} | "
            f"{sum(case.split == 'test' for case in group)} | "
            f"{origins['human']} | {origins['llm_draft']} | {origins['llm_edited']} | "
            f"{statuses['draft']} | {statuses['verified']} | {len(group)} |"
        )

    origins = Counter(case.origin for case in cases)
    statuses = Counter(case.status for case in cases)
    lines.append(
        "| **total** | "
        f"{sum(case.split == 'dev' for case in cases)} | "
        f"{sum(case.split == 'test' for case in cases)} | "
        f"{origins['human']} | {origins['llm_draft']} | {origins['llm_edited']} | "
        f"{statuses['draft']} | {statuses['verified']} | {len(cases)} |"
    )
    return "\n".join(lines)


def run(data_dir: Path, include_draft: bool) -> int:
    try:
        cases = read_cases(data_dir)
    except ValueError as exc:
        print(f"schema error: {exc}")
        return 2

    selected = cases if include_draft else [case for case in cases if case.status == "verified"]
    if include_draft:
        print("SMOKE: drafts included; these counts are not benchmark results.\n")

    print(counts_table(selected))
    errors = validate_cases(selected)
    if errors:
        print(f"\n{len(errors)} validation error(s):")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nAll checks passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate LithoBench")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument(
        "--include-draft",
        action="store_true",
        help="include drafts (smoke tests only; never for results)",
    )
    args = parser.parse_args(argv)
    return run(Path(args.data_dir), args.include_draft)


if __name__ == "__main__":
    sys.exit(main())
