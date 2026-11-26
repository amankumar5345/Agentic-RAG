"""Compute human/judge agreement and Cohen's kappa from a filled CSV."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

REPO_ROOT = Path(__file__).resolve().parents[1]
MIN_LABELS = 50
TRUTHY = {"1", "true", "yes", "correct", "y"}


def read_labels(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def judge_labels(results_path: Path) -> dict[str, bool]:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    return {
        case["id"]: bool(case["correctness"]["correct"])
        for case in payload.get("cases", [])
        if case.get("correctness") is not None
    }


def compute(rows: list[dict], judge: dict[str, bool]) -> dict:
    human = []
    machine = []
    for row in rows:
        label = (row.get("human_label") or "").strip().lower()
        if not label:
            continue
        if row["case_id"] not in judge:
            continue
        human.append(label in TRUTHY)
        machine.append(judge[row["case_id"]])

    n = len(human)
    if n < MIN_LABELS:
        raise SystemExit(f"need at least {MIN_LABELS} labels, found {n}")

    agreement = sum(1 for left, right in zip(human, machine, strict=True) if left == right) / n
    kappa = cohen_kappa_score(human, machine)
    if kappa != kappa:  # NaN when only one class is present
        kappa = 0.0

    rng = random.Random(42)
    values = []
    for _ in range(1000):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_human = [human[index] for index in indices]
        sample_machine = [machine[index] for index in indices]
        value = cohen_kappa_score(sample_human, sample_machine)
        if value == value:  # skip NaN
            values.append(value)
    values.sort()
    ci = (values[int(0.025 * len(values))], values[int(0.975 * len(values))]) if values else (None, None)

    return {
        "n": n,
        "agreement": agreement,
        "kappa": kappa,
        "kappa_ci95": list(ci),
        "min_labels": MIN_LABELS,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Judge agreement")
    parser.add_argument("--labels", default=str(REPO_ROOT / "bench" / "judge_sample.csv"))
    parser.add_argument("--results", default=str(REPO_ROOT / "results" / "A5.json"))
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "judge_validation.json"))
    args = parser.parse_args(argv)

    rows = read_labels(Path(args.labels))
    judge = judge_labels(Path(args.results))
    payload = compute(rows, judge)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"agreement={payload['agreement']:.3f} kappa={payload['kappa']:.3f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
