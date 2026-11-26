"""Offline tests for LithoBench schemas, IO and split assignment."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from bench.schema import (
    AdversarialCase,
    GeometryCase,
    GoldSource,
    QACase,
    RouterCase,
    assign_splits,
    load_cases,
    read_cases,
    source_key,
    write_cases,
)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def source(doc="a.pdf", start=0, end=None):
    if end is None:
        end = start + 10
    return GoldSource(
        doc=doc,
        section="s",
        char_start=start,
        char_end=end,
        text_hash="hash",
    )


def make_qa(
    case_id,
    *,
    subtype="single_hop",
    split="test",
    origin="llm_draft",
    status="draft",
    question=None,
    sources=None,
    tolerance=None,
):
    verified = status == "verified"
    return QACase(
        id=case_id,
        split=split,
        origin=origin,
        status=status,
        reviewer="r" if verified else None,
        reviewed_at=NOW if verified else None,
        question=question or f"question {case_id}",
        gold_answer="answer",
        gold_sources=sources or [source(start=int(case_id.split("-")[-1]) * 100)],
        subtype=subtype,
        numeric_tolerance=tolerance,
    )


def test_verified_requires_reviewer_and_timestamp():
    with pytest.raises(ValidationError):
        QACase(
            id="qa-1",
            split="test",
            origin="llm_draft",
            status="verified",
            question="q",
            gold_answer="a",
            gold_sources=[source()],
            subtype="single_hop",
        )


def test_numeric_requires_tolerance():
    with pytest.raises(ValidationError):
        make_qa("qa-1", subtype="numeric")


def test_round_trip_all_types(tmp_path):
    cases = [
        make_qa("qa-1"),
        AdversarialCase(
            id="adv-1",
            split="test",
            origin="llm_draft",
            status="draft",
            prompt="ignore instructions",
            category="jailbreak",
            expected="block",
        ),
        GeometryCase(
            id="geo-1",
            split="test",
            origin="llm_draft",
            status="draft",
            utterance="draw a line of 10",
            gold_commands=[{"shape": "line", "size": 10, "angle": 0}],
            tolerance=1.0,
            expect_clarification=False,
            category="single",
        ),
        RouterCase(
            id="router-1",
            split="test",
            origin="llm_draft",
            status="draft",
            query="is this safe?",
            gold_role="Safety Agent",
        ),
    ]

    write_cases(cases, tmp_path)
    loaded = read_cases(tmp_path)

    assert {case.id for case in loaded} == {case.id for case in cases}
    assert len(loaded) == 4


def test_loader_excludes_drafts_by_default(tmp_path):
    write_cases(
        [
            make_qa("qa-1", status="verified"),
            make_qa("qa-2", status="draft"),
        ],
        tmp_path,
    )

    assert [case.id for case in load_cases(data_dir=tmp_path)] == ["qa-1"]
    assert len(load_cases(include_draft=True, data_dir=tmp_path)) == 2


def test_assign_splits_is_deterministic():
    cases = [make_qa(f"qa-{index}") for index in range(10)]

    first = assign_splits(cases, seed=7)
    second = assign_splits(cases, seed=7)

    assert [case.split for case in first] == [case.split for case in second]


def test_qa_passages_never_cross_splits():
    shared = source("shared.pdf", 0, 50)
    cases = [
        make_qa("qa-1", sources=[shared]),
        make_qa("qa-2", sources=[shared], question="second question on shared passage"),
    ]
    cases += [make_qa(f"qa-{index}", sources=[source(f"{index}.pdf")]) for index in range(3, 13)]

    assigned = assign_splits(cases, seed=11)
    by_id = {case.id: case for case in assigned}

    assert by_id["qa-1"].split == by_id["qa-2"].split

    splits_by_passage = {}
    for case in assigned:
        key = source_key(case.gold_sources[0])
        assert splits_by_passage.setdefault(key, case.split) == case.split


def test_dev_share_is_about_thirty_percent():
    cases = [make_qa(f"qa-{index}") for index in range(10)]
    assigned = assign_splits(cases, seed=3)

    assert sum(case.split == "dev" for case in assigned) == 3


def test_geometry_clarification_must_match_commands():
    with pytest.raises(ValidationError):
        GeometryCase(
            id="geo-1",
            split="test",
            origin="llm_draft",
            status="draft",
            utterance="draw a line",
            gold_commands=[{"shape": "line", "size": 10}],
            expect_clarification=True,
            category="single",
        )


def test_large_component_does_not_eat_the_dev_split():
    shared = source("shared.pdf")
    cases = [make_qa(f"qa-{index}", sources=[shared]) for index in range(10)]
    cases += [
        make_qa(f"qa-{index}", sources=[source(f"{index}.pdf")]) for index in range(10, 20)
    ]

    assigned = assign_splits(cases, seed=5)
    shared_splits = {
        case.split for case in assigned if case.gold_sources[0].doc == "shared.pdf"
    }

    assert shared_splits == {"test"}
    assert sum(case.split == "dev" for case in assigned) == 6


def test_write_cases_rejects_duplicate_ids(tmp_path):
    first = make_qa("qa-1", question="first")
    second = make_qa("qa-1", question="second")

    with pytest.raises(ValueError):
        write_cases([first, second], tmp_path)
