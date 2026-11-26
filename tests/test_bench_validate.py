"""Offline tests for LithoBench validation checks."""

from datetime import datetime, timezone

from bench.schema import (
    ADVERSARIAL_TARGETS,
    ROLE_NAMES,
    GoldSource,
    QACase,
)
from bench.validate import (
    check_duplicates,
    check_human_ratio,
    check_multihop,
    check_split_leakage,
    counts_table,
    validate_cases,
)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)

SMALL_TARGETS = {
    "qa": {"single_hop": 1, "multi_hop": 0, "numeric": 0, "hazard": 0, "unanswerable": 0},
    "adversarial": {category: 0 for category in ADVERSARIAL_TARGETS},
    "geometry": 0,
    "router": {role: 0 for role in ROLE_NAMES},
}


def source(doc="a.pdf", start=0, end=None):
    if end is None:
        end = start + 10
    return GoldSource(doc=doc, section="s", char_start=start, char_end=end, text_hash="h")


def make_qa(
    case_id,
    *,
    subtype="single_hop",
    split="test",
    origin="human",
    question=None,
    sources=None,
    tolerance=None,
):
    return QACase(
        id=case_id,
        split=split,
        origin=origin,
        status="verified",
        reviewer="r",
        reviewed_at=NOW,
        question=question or f"question {case_id}",
        gold_answer="answer",
        gold_sources=sources or [source(start=int(case_id.split("-")[-1]) * 100)],
        subtype=subtype,
        numeric_tolerance=tolerance,
    )


def test_small_dataset_passes():
    cases = [make_qa("qa-1")]
    assert validate_cases(cases, SMALL_TARGETS) == []


def test_count_mismatch_is_reported():
    errors = validate_cases([], SMALL_TARGETS)
    assert any("QA single_hop: 0 != 1" in error for error in errors)


def test_split_leakage_is_detected():
    shared = source("shared.pdf")
    cases = [
        make_qa("qa-1", split="dev", sources=[shared]),
        make_qa("qa-2", split="test", sources=[shared], question="other question"),
    ]

    errors = check_split_leakage(cases)

    assert len(errors) == 1
    assert "shared.pdf" in errors[0]


def test_duplicate_questions_are_detected():
    cases = [
        make_qa("qa-1", question="Same  Question?"),
        make_qa("qa-2", question="same question?"),
    ]

    assert len(check_duplicates(cases)) == 1


def test_multihop_needs_two_passages():
    case = make_qa("qa-1", subtype="multi_hop", sources=[source()])
    assert len(check_multihop([case])) == 1

    two = make_qa(
        "qa-2",
        subtype="multi_hop",
        sources=[source("a.pdf", 0, 10), source("b.pdf", 0, 10)],
    )
    assert check_multihop([two]) == []


def test_human_ratio_rule():
    drafts = [make_qa("qa-1", origin="llm_draft"), make_qa("qa-2", origin="llm_draft")]
    assert len(check_human_ratio(drafts)) == 1

    edited = [make_qa("qa-3", origin="llm_edited")]
    assert check_human_ratio(edited) == []


def test_counts_table_has_total_row():
    table = counts_table([make_qa("qa-1")])
    assert "| **total** |" in table
    assert table.count("\n") >= 3
