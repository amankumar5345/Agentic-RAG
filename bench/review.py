"""Interactive review loop for LithoBench drafts.

    python -m bench.review --reviewer "Your Name"

Accept keeps origin='llm_draft'; edit sets origin='llm_edited'; reject removes
the case (and appends it to data/rejected.jsonl). Both accept and edit set
status='verified', reviewer and reviewed_at (UTC).
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from bench.injection.cases import (
    CASES_PATH,
    INJECTION_ADAPTER,
    read_injection_cases,
    write_injection_cases,
)
from bench.schema import CASE_ADAPTER, DATA_DIR, QACase, read_cases, write_cases

DEFAULT_CORPUS = Path("our_database/all_contents.txt")


def source_text(case, corpus: str) -> str | None:
    if not isinstance(case, QACase) or not case.gold_sources:
        return None
    source = case.gold_sources[0]
    text = corpus[source.char_start:source.char_end]
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    suffix = "" if digest == source.text_hash else "\n[WARNING: text_hash mismatch]"
    return f"{source.doc} :: {source.section}\n{text}{suffix}"


def apply_decision(
    case,
    decision: str,
    reviewer: str,
    now: datetime | None = None,
    edited=None,
):
    """Return the updated case, or None when the case is rejected."""
    if decision not in {"accept", "edit", "reject", "skip"}:
        raise ValueError(f"unknown decision {decision!r}")

    if decision == "skip":
        return case
    if decision == "reject":
        return None

    stamp = now or datetime.now(timezone.utc)
    updated = edited if decision == "edit" and edited is not None else case

    content_fields = {"origin", "status", "reviewer", "reviewed_at", "split"}
    changed = updated.model_dump(exclude=content_fields) != case.model_dump(exclude=content_fields)
    origin = "llm_edited" if changed else case.origin

    return updated.model_copy(
        update={
            "status": "verified",
            "reviewer": reviewer,
            "reviewed_at": stamp,
            "origin": origin,
        }
    )


def edit_interactively(case, adapter=CASE_ADAPTER):
    """Open the case JSON in $EDITOR and return the edited, validated case."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(case.model_dump_json(indent=2))
        path = Path(handle.name)

    editor = os.environ.get("EDITOR") or ("notepad" if os.name == "nt" else "vi")
    try:
        subprocess.run(shlex.split(editor) + [str(path)], check=False)
        return adapter.validate_json(path.read_text(encoding="utf-8"))
    finally:
        path.unlink(missing_ok=True)


def _append_rejected(case, reviewer: str, data_dir: Path) -> None:
    path = data_dir / "rejected.jsonl"
    record = case.model_dump(mode="json")
    record["reviewer"] = reviewer
    record["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def run_review(
    data_dir: Path,
    reviewer: str,
    corpus_path: Path,
    input_fn=input,
    print_fn=print,
) -> int:
    cases = read_cases(data_dir)
    drafts = [case for case in cases if case.status == "draft"]
    if not drafts:
        print_fn("No draft cases found.")
        return 0

    corpus = corpus_path.read_text(encoding="utf-8", errors="replace")
    print_fn(f"Reviewing {len(drafts)} draft cases as {reviewer!r}.")

    for position, case in enumerate(drafts, start=1):
        print_fn(f"\n=== [{position}/{len(drafts)}] {case.id} ({case.type}) ===")
        print_fn(case.model_dump_json(indent=2))
        passage = source_text(case, corpus)
        if passage:
            print_fn("\nSource passage:\n" + passage)

        choice = input_fn("[a]ccept [e]dit [r]eject [s]kip [q]uit > ").strip().lower()
        if choice == "q":
            return 0
        if choice not in {"a", "e", "r", "s"}:
            print_fn("Unknown choice; skipping.")
            continue

        decision = {"a": "accept", "e": "edit", "r": "reject", "s": "skip"}[choice]
        edited = None
        if decision == "edit":
            try:
                edited = edit_interactively(case)
            except Exception as exc:
                print_fn(f"Edit rejected: {exc}")
                continue

        updated = apply_decision(case, decision, reviewer, edited=edited)

        if decision == "reject":
            _append_rejected(case, reviewer, data_dir)
            cases = [item for item in cases if item.id != case.id]
        elif decision == "skip":
            continue
        else:
            cases = [updated if item.id == case.id else item for item in cases]
        write_cases(cases, data_dir)

    print_fn("Review loop finished.")
    return 0


def run_injection_review(
    cases_path: Path,
    reviewer: str,
    input_fn=input,
    print_fn=print,
) -> int:
    cases = read_injection_cases(cases_path)
    drafts = [case for case in cases if case.status == "draft"]
    if not drafts:
        print_fn("No draft injection cases found.")
        return 0

    print_fn(f"Reviewing {len(drafts)} injection cases as {reviewer!r}.")
    for position, case in enumerate(drafts, start=1):
        print_fn(f"\n=== [{position}/{len(drafts)}] {case.id} ({case.attack_type}) ===")
        print_fn(f"Question: {case.question}")
        print_fn("Injected passage:\n" + case.injected_passage)
        choice = input_fn("[a]ccept [e]dit [r]eject [s]kip [q]uit > ").strip().lower()
        if choice == "q":
            return 0
        if choice not in {"a", "e", "r", "s"}:
            print_fn("Unknown choice; skipping.")
            continue

        decision = {"a": "accept", "e": "edit", "r": "reject", "s": "skip"}[choice]
        edited = None
        if decision == "edit":
            try:
                edited = edit_interactively(case, INJECTION_ADAPTER)
            except Exception as exc:
                print_fn(f"Edit rejected: {exc}")
                continue

        updated = apply_decision(case, decision, reviewer, edited=edited)
        if decision == "reject":
            _append_rejected(case, reviewer, Path(cases_path).parent)
            cases = [item for item in cases if item.id != case.id]
        elif decision == "skip":
            continue
        else:
            cases = [updated if item.id == case.id else item for item in cases]
        write_injection_cases(cases, cases_path)

    print_fn("Injection review finished.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review LithoBench drafts")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--reviewer", default=os.environ.get("LITHO_REVIEWER"))
    parser.add_argument("--injection", action="store_true")
    parser.add_argument("--cases", default=str(CASES_PATH))
    args = parser.parse_args(argv)

    reviewer = args.reviewer or getpass.getuser()
    if args.injection:
        return run_injection_review(Path(args.cases), reviewer)
    return run_review(Path(args.data_dir), reviewer, Path(args.corpus))


if __name__ == "__main__":
    sys.exit(main())
